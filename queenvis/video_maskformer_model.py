# Copyright (c) 2026, Arian Kheirandish. All rights reserved.
# Modified by Arian Kheirandish for QueenVIS.
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Tuple
import einops

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.modeling.criterion import VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher
from mask2former_video.utils.memory import retry_if_cuda_oom

from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# Memory-bank implementation adapted from VISAGE (Apache-2.0):
# https://github.com/KimHanjung/VISAGE
# See NOTICE and THIRD_PARTY_LICENSES.md for attribution and citation details.
class Memorybank(object):
    def __init__(self, num_queries, hidden_dim, bank_size=3, device='cuda'):
        self.bank_size = bank_size
        self.num_queries = num_queries
        self.size = 0
        self.device = device

        self.memory_bank = torch.zeros(self.bank_size, num_queries, hidden_dim).to(device)
        self.scores = torch.zeros(self.bank_size, num_queries).to(device)

    def reset(self):
        self.memory_bank.zero_()
        self.scores.zero_()
        self.size = 0

    def update(self, embeddings, max_scores):
        '''
        Args:
            embeddings: (Q, C)
        '''
        self.memory_bank = self.memory_bank.roll(1, dims=0)
        self.memory_bank[0] = embeddings
        self.scores = self.scores.roll(1, dims=0)
        self.scores[0] = max_scores
        self.size = min(self.size + 1, self.bank_size)

    def get(self):
        temporal_weight = torch.arange(0, 1+1e-6, 1 / self.size, device=self.memory_bank.device)[1:]
        score_weight = self.scores[:self.size]
        weight = temporal_weight[:, None] + score_weight
        temporal_embedding = (self.memory_bank[:self.size] * weight[..., None]).sum(0) / weight.sum(0)[:, None]

        return temporal_embedding

@META_ARCH_REGISTRY.register()
class VideoMaskFormer_frame(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        window_inference,
        #passing the pred_features 
        pred_features,
        #for Memory Bank
        hidden_dim,
        #memory bank size for inference
        memory_bank_size: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        #Adding the passed pred_features
        self.pred_features = pred_features
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.window_inference = window_inference

        #making the memory bank
        self.hidden_dim = hidden_dim
        self.memory_bank = Memorybank(num_queries, hidden_dim, bank_size=memory_bank_size, device=self.device)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        center_weight = cfg.MODEL.MASK_FORMER.CENTER_WEIGHT
        features_weight = cfg.MODEL.MASK_FORMER.FEATURES_WEIGHT
        align_weight = cfg.MODEL.MASK_FORMER.ALIGN_WEIGHT
        #Which preds to predict
        pred_features = cfg.MODEL.SEM_SEG_HEAD.PRED_FEATURES
        #the hidden_dim for memory bank
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_features=features_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_center": center_weight, "loss_features" : features_weight, "loss_align": align_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "center", "features", "align"]
        aux_losses = ["labels", "masks", "center", "features"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            aux_losses=aux_losses,  
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            #returning it for use
            "pred_features": pred_features,
            #returning hidden_dim for memory bank
            "hidden_dim": hidden_dim,
            #memory bank size for inference
            "memory_bank_size": cfg.MODEL.MASK_FORMER.MEMORY_BANK_SIZE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor)
        elif not self.training:
            outputs = self.run_window_inference_frame_By_frame(images.tensor)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            #Added features to the target
            # Training: no memory bank
            targets = self.prepare_targets(batched_inputs, images, features)

            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            #uses memory bank in post_processing
            outputs = self.post_processing(outputs)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            #returning the pred embeds for similarity checks
            embd_results = outputs['pred_embds']
            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            embd_result = embd_results[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])
            #Also returning the embed results
            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size)

    def frame_decoder_loss_reshape(self, outputs, targets):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        #Rearranging for Added Centers
        if 'pred_centers' in outputs:
            outputs['pred_centers'] = einops.rearrange(outputs['pred_centers'], 'b q t c -> (b t) q c')
        if 'pred_feats' in outputs:
            outputs['pred_feats'] = einops.rearrange(outputs['pred_feats'], 'b q t c -> (b t) q c')
        if 'pred_embds' in outputs:
            outputs['pred_embds'] = einops.rearrange(outputs['pred_embds'], 'b c t q -> (b t) q c')
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> (b t) q () h w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                )
                #Rearranging for Added Centers in aux_outputs
                if 'pred_centers' in outputs['aux_outputs'][i]:
                    outputs['aux_outputs'][i]['pred_centers'] = einops.rearrange(
                        outputs['aux_outputs'][i]['pred_centers'], 'b q t c -> (b t) q c'
                    )
                if 'pred_feats' in outputs['aux_outputs'][i]:
                    outputs['aux_outputs'][i]['pred_feats'] = einops.rearrange(
                        outputs['aux_outputs'][i]['pred_feats'], 'b q t c -> (b t) q c'
                    )                

        gt_instances = []
        for targets_per_video in targets:
            # labels: N (num instances)
            # ids: N, num_labeled_frames
            # masks: N, num_labeled_frames, H, W
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]
                #setting Targets for Centers and Features
                if ('centers' in targets_per_video) and ('features' in targets_per_video):
                    centers = targets_per_video['centers'][:, f, :]
                    features = targets_per_video['features'][:, f, :]
                    gt_instances.append({"labels": labels, "ids": ids, "masks": masks, "centers": centers, "features":features})
                else:
                    gt_instances.append({"labels": labels, "ids": ids, "masks": masks})

        return outputs, gt_instances

    def match_from_embds(self, tgt_embds, cur_embds):

        tgt_embds = tgt_embds.to(cur_embds.device)
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices

    def post_processing(self, outputs):
        self.memory_bank.reset()
        pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embds']

        # pred_logits: 1 t q c
        # pred_masks: 1 q t h w
        pred_logits = pred_logits[0]
        pred_masks = einops.rearrange(pred_masks[0], 'q t h w -> t q h w')
        pred_embds = einops.rearrange(pred_embds[0], 'c t q -> t q c')

        pred_logits = list(torch.unbind(pred_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))

        out_logits = []
        out_masks = []
        out_embds = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        out_embds.append(pred_embds[0])

        #first frame in Memory Bank
        max_scores, _ = torch.max(out_logits[-1].softmax(dim=-1)[:, :-1], dim=-1)
        self.memory_bank.update(pred_embds[0], max_scores)

        #without memory bank
        # for i in range(1, len(pred_logits)):
        #     indices = self.match_from_embds(out_embds[-1], pred_embds[i])

        #     out_logits.append(pred_logits[i][indices, :])
        #     out_masks.append(pred_masks[i][indices, :, :])
        #     out_embds.append(pred_embds[i][indices, :])

        #with Memory Bank
        for i in range(1, len(pred_logits)):
            tgt_embds = self.memory_bank.get()
            indices = self.match_from_embds(tgt_embds, pred_embds[i])

            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_embds.append(pred_embds[i][indices, :])

            # Calculate scores and update memory bank
            max_scores, _ = torch.max(out_logits[-1].softmax(dim=-1)[:, :-1], dim=-1)
            self.memory_bank.update(pred_embds[i][indices, :], max_scores)


        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w
        out_embds = torch.stack(out_embds, dim=1)


        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)
        out_embds = out_embds.unsqueeze(0)
                                        
        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks
        outputs['pred_embds'] = out_embds
        return outputs

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2).detach()

        return outputs
    def run_window_inference_frame_By_frame(self, images_tensor, window_size=30):
        T = len(images_tensor)
        out_list = []
        prev_queries = None   # [Q, 1, C]
        prev_scores  = None   # [1, Q]
        for t in range(T):
            frame = images_tensor[t:t+1]
            features = self.backbone(frame)

            # Replicate MaskFormerHead internals so we can inject prev_* into the predictor
            out = self.sem_seg_head(features, mask=None, prev_queries=prev_queries, prev_scores=prev_scores)

            # Update propagation state for next frame
            prev_queries = out['final_queries']                  # already detached
            logits       = out['pred_logits']                    # [1, 1, Q, C+1]
            scores       = F.softmax(logits, dim=-1)[..., :-1]
            max_scores,_ = scores.max(dim=-1)                    # [1, 1, Q]
            prev_scores  = max_scores.squeeze(1).detach()        # [1, Q]

            # Free memory like the original did
            for k in ('res2', 'res3', 'res4', 'res5'):
                features.pop(k, None)
            for j in range(len(out['aux_outputs'])):
                out['aux_outputs'][j].pop('pred_masks',  None)
                out['aux_outputs'][j].pop('pred_logits', None)
            out_list.append(out)

        outputs = {
            'pred_logits': torch.cat([x['pred_logits'] for x in out_list], dim=1).detach(),
            'pred_masks':  torch.cat([x['pred_masks']  for x in out_list], dim=2).detach(),
            'pred_embds':  torch.cat([x['pred_embds']  for x in out_list], dim=2).detach(),
        }
        return outputs

    def prepare_targets(self, targets, images, features):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        # Calculating the total feautre channles for features dim
        features_dim = 0
        for key in features.keys():
            # we customize the features we want to predict using pred_features
            if key in self.pred_features:
                features_dim += features[key].shape[1]
        # added batch support
        frame_ptr = 0
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            # Adding the Gt centers
            gt_centers_per_video = torch.zeros((_num_instance, self.num_frames, 2), dtype=torch.float32, device=self.device)
            # Adding the Gt features
            gt_features_per_video = torch.zeros((_num_instance, self.num_frames, features_dim), dtype=torch.float32, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size
                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                if hasattr(targets_per_frame.gt_masks, "tensor"):
                    raw_masks = targets_per_frame.gt_masks.tensor
                else:
                    raw_masks = targets_per_frame.gt_masks
                gt_masks_per_video[:, f_i, :h, :w] = raw_masks
                
                # Adding the Gt features = avg polling(features * masks) for each features[key] and then concat them
                pooled_frame_features = []
                # Get Gt masks for this frame: [N, H, W]
                current_masks = raw_masks.float()
                # Unsqueeze to [N, 1, H, W] for interpolation
                current_masks = current_masks.unsqueeze(1)
                # calculating for each key in features
                for key in self.pred_features:
                    # we customize the features we want to predict using pred_features
                    if key in features.keys():
                        # C = features[key].shape[1], N = _num_instance
                        feat_map = features[key][frame_ptr] #shape[C, H_feat, W_feat]
                        # Detaching the feature map so the model wouldn't cheat with gradient descent on the backbone
                        feat_map = feat_map.detach()
                        target_size = feat_map.shape[-2:]
                        # resize/downsample GT mask to feature map size
                        resized_masks = F.interpolate(current_masks, size=target_size, mode='bilinear', align_corners=False)
                        binary_masks = (resized_masks > 0).float() # shape [N, 1, H_feat, W_feat]
                        masked_feat = feat_map.unsqueeze(0) * binary_masks # shape [N, C, H, W]
                        sum_feat = masked_feat.sum(dim=(-2, -1)) # shape [N, C]
                        mask_area = binary_masks.sum(dim=(-2, -1)) # shape [N, 1]
                        mask_area = torch.clamp(mask_area, min=1e-6)#Avoid division by Zero
                        avg_feat = sum_feat / mask_area #shape [N, C]
                        pooled_frame_features.append(avg_feat)
                gt_features_per_video[:, f_i, :] = torch.cat(pooled_frame_features, dim = 1) #shape [n, features_dim]
                frame_ptr += 1


                # Adding the Gt centers
                centers = targets_per_frame.gt_centers 
                gt_centers_per_video[:, f_i] = centers

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

            #Adding the Gt centers
            gt_centers_per_video = gt_centers_per_video[valid_idx]
            gt_instances[-1].update({"centers": gt_centers_per_video})

            #Adding the Gt features
            gt_features_per_video = gt_features_per_video[valid_idx]
            gt_instances[-1].update({"features": gt_features_per_video})

        return gt_instances
    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(len(pred_cls), 1).flatten(0, 1)
            # keep top k predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.sem_seg_head.num_classes, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]

            interim_mask_soft = pred_masks.sigmoid()
            interim_mask_hard = interim_mask_soft > 0.5

            numerator   = (interim_mask_soft.flatten(1) * interim_mask_hard.flatten(1)).sum(1)
            denominator = interim_mask_hard.flatten(1).sum(1)

            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            scores_per_image *= (numerator / (denominator + 1e-6))
            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output
