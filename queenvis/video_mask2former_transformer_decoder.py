# Copyright (c) 2026, Arian Kheirandish. All rights reserved.
# Modified by Arian Kheirandish for QueenVIS.
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder, MLP
import einops

@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder_frame(VideoMultiScaleMaskedTransformerDecoder):

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # video related
        num_frames,
        #for features prediction
        features_dim: int,
        #for training free query propagation
        propagate_queries: bool,
        prop_alpha: float,
        prop_threshold: float,
    ):
        super().__init__(
            in_channels=in_channels, 
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            num_frames=num_frames,
        )

        # use 2D positional embedding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        # Adding the MLP for predicting centers
        self.center_embed = nn.Linear(hidden_dim, 2)
        # Adding the MLP for predicting features 
        self.feat_embed = MLP(hidden_dim, hidden_dim, features_dim, 3)
        # Inference-only query propagation
        self.propagate_queries = propagate_queries
        self.prop_alpha = prop_alpha
        self.prop_threshold = prop_threshold

    def forward(self, x, mask_features, mask = None, prev_queries=None, prev_scores=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        init_output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        if (self.propagate_queries and prev_queries is not None and prev_scores is not None):
            gate    = (prev_scores > self.prop_threshold).float()       # [B, Q]
            gate    = gate.transpose(0, 1).unsqueeze(-1)                 # [Q, B, 1]
            blended = self.prop_alpha * prev_queries + (1.0 - self.prop_alpha) * init_output
            output  = gate * blended + (1.0 - gate) * init_output
        else:
            output = init_output
        predictions_class = []
        predictions_mask = []
        #Arryas for center predictions and feature predictions
        predictions_center = []
        predictions_feat = []
        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, output_center, output_feat = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        #Appending the output results for center and features
        predictions_center.append(output_center)
        predictions_feat.append(output_feat)
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, output_center, output_feat = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            #Appending the output results for center and features
            predictions_center.append(output_center)
            predictions_feat.append(output_feat)

        assert len(predictions_class) == self.num_layers + 1

        assert len(predictions_center) == len(predictions_mask)
        # expand BT to B, T  
        bt = predictions_mask[-1].shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)

        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(predictions_class[i], '(b t) q c -> b t q c', t=t)

        if self.training:
            #changing the dimensions to make them suitable
            for i in range(len(predictions_center)):
                predictions_center[i] = einops.rearrange(predictions_center[i], '(b t) q c -> b q t c', t=t)
            #changing the dimensions to make them suitable
            for i in range(len(predictions_feat)):
                predictions_feat[i] = einops.rearrange(predictions_feat[i], '(b t) q c -> b q t c', t=t)

        pred_embds = self.decoder_norm(output)
        final_queries = pred_embds
        pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b c t q', t=t)

        out = {
            'final_queries': final_queries.detach(),   # [Q, B*T, C]
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            #added the centers here for loss
            #added the feat here for loss
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_center, predictions_feat
            ),
            'pred_embds': pred_embds,
        }
        if self.training:
            # returning the centers and features predicted
            out['pred_centers'] = predictions_center[-1]
            out['pred_feats'] = predictions_feat[-1]
        
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        if self.training:
            ##Adding centers prediction-head in forward(Added sigmoid for normalization)
            output_center = torch.sigmoid(self.center_embed(decoder_output))
            #Adding the features prediction-head in forward
            output_feat = self.feat_embed(decoder_output)
        else:
            output_center = None
            output_feat = None
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, output_center, output_feat
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_centers, outputs_feats):
        if self.training:
            # keep TorchScript-friendly homogeneous dicts per layer
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b, "pred_centers": c, "pred_feats": d}
                    for a, b, c, d in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_centers[:-1], outputs_feats[:-1])
                ]
            else:
                return [
                    {"pred_masks": b, "pred_centers": c, "pred_feats": d}
                    for b, c, d in zip(outputs_seg_masks[:-1], outputs_centers[:-1], outputs_feats[:-1])
                ]
        else:
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [
                    {"pred_masks": b}
                    for b in outputs_seg_masks[:-1]
                ]
