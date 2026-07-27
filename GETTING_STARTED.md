# Getting Started with QueenVIS

QueenVIS trains on independent images and performs temporal association only at video inference time. The official full-data ResNet-50 configurations use `INPUT.SAMPLING_FRAME_NUM: 1`.

For general framework usage, also consult the [Detectron2 documentation](https://detectron2.readthedocs.io/).

## 1. Prepare the environment

Follow [INSTALL.md](INSTALL.md), including compilation of the MSDeformAttn CUDA operator.

## 2. Prepare datasets

Follow [datasets/README.md](datasets/README.md) for YouTube-VIS 2019/2021 and OVIS directory layouts.

QueenVIS uses 100% of the selected dataset's available training frames as independent images. It does not train on video clips or temporally linked track identities.

## 3. Download image-pretrained weights

Download the appropriate COCO-pretrained Mask2Former instance-segmentation checkpoint:

- [ResNet-50](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl)
- [Swin-L](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl)

Place the checkpoint at the path expected by the selected YAML file, or override it with `MODEL.WEIGHTS` on the command line.

## 4. Train

Example: train the ResNet-50 model on YouTube-VIS 2021 using independent frames:

```bash
python train_net_video.py --num-gpus 8 \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs32_8ep_frame.yaml
```

To override the initialization checkpoint:

```bash
python train_net_video.py --num-gpus 8 \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs32_8ep_frame.yaml \
  MODEL.WEIGHTS /path/to/model_final_3c8ec9.pkl
```

The supplied batch sizes and learning rates were selected for their original hardware setup. If you change the GPU count or total batch size, adjust optimization settings deliberately rather than assuming linear scaling.

## 5. Evaluate

```bash
python train_net_video.py \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs32_8ep_frame.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint.pth
```

For additional command-line options:

```bash
python train_net_video.py -h
```

## 6. Visualize video predictions

Choose a config and checkpoint, then run:

```bash
cd demo_video
python demo.py \
  --config-file ../configs/youtubevis_2021/video_maskformer2_R50_bs32_8ep_frame.yaml \
  --input /path/to/video/frames \
  --output /path/to/output/folder \
  --opts MODEL.WEIGHTS /path/to/checkpoint.pth
```

`--input` should point to a directory containing the frames of one video as image files.

## Model Zoo

See [MODEL_ZOO.md](MODEL_ZOO.md). Only 100%-data, image-trained official models are listed; checkpoint URLs are placeholders until release.
