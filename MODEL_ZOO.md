# QueenVIS Model Zoo

Official QueenVIS checkpoints will be published here.

## Training setting

Every model listed on this page uses the complete training split for its dataset:

* **Training data:** 100% of the available training frames
* **Training regime:** image-only (`INPUT.SAMPLING_FRAME_NUM: 1`)
* **Video supervision:** none
* **Temporally linked identity supervision:** none

There are no official QueenVIS model-zoo entries for reduced data percentages.

## ResNet-50

| Dataset | Training | Reported AP | Configuration | Checkpoint |
|---|---|---:|---|---|
| YouTube-VIS 2019 | 100% frames, image-only | 51.8 | [config](configs/youtubevis_2019/video_maskformer2_R50_bs32_8ep_frame.yaml) | [queenvis_ytvis19_r50.pth](https://huggingface.co/ArianKheir/QueenVIS/blob/main/queenvis_ytvis19_R50.pth) |
| YouTube-VIS 2021 | 100% frames, image-only | 50.9 | [config](configs/youtubevis_2021/video_maskformer2_R50_bs32_8ep_frame.yaml) | [queenvis_ytvis21_r50.pth](https://huggingface.co/ArianKheir/QueenVIS/blob/main/queenvis_ytvis21_R50.pth) |
| OVIS | 100% frames, image-only | 29.8 | [config](configs/ovis/video_maskformer2_R50_bs32_8ep_frame.yaml) | [queenvis_ovis_r50.pth](https://huggingface.co/ArianKheir/QueenVIS/blob/main/queenvis_ovis_r50.pth) |

## Swin-L

| Dataset | Training | Reported AP | Configuration | Checkpoint |
|---|---|---:|---|---|
| YouTube-VIS 2019 | 100% frames, image-only | 63.2 | [config](configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame.yaml) | [queenvis_ytvis19_swin.pth](https://huggingface.co/ArianKheir/QueenVIS/blob/main/queenvis_ytvis19_swin.pth) |
| YouTube-VIS 2021 | 100% frames, image-only | 59.8 | [config](configs/youtubevis_2021/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame.yaml) | [queenvis_ytvis21_swin.pth](https://huggingface.co/ArianKheir/QueenVIS/blob/main/queenvis_ytvis21_swin.pth) |
| OVIS | 100% frames, image-only | 41.0 | [config](configs/ovis/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame.yaml) | [queenvis_ovis_swin.pth](https://huggingface.co/ArianKheir/QueenVIS/blob/main/queenvis_ovis_swin.pth) |

## Using a released checkpoint

You can pass the Hugging Face direct download link directly into Detectron2's evaluation script:

```bash
python train_net_video.py \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs32_8ep_frame.yaml \
  --eval-only MODEL.WEIGHTS path/to/weights.pth
```
