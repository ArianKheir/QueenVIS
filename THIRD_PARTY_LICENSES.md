# Third-Party Software and Attribution

This file records the principal upstream projects represented in QueenVIS. It is informational and does not replace the full license texts or per-file notices.

## License precedence

QueenVIS is derived substantially from MinVIS. Unless a file explicitly states separate terms, the repository is governed by the [NVIDIA Source Code License-NC](LICENSE), including its non-commercial research/evaluation restriction. A QueenVIS copyright or modification notice covers only the stated modifications; it does not relicense upstream code.

When a file contains third-party code under separate terms, that file's preserved notices and the corresponding upstream license also apply.

## Components

| Project | Use in this repository | License | Upstream source |
|---|---|---|---|
| MinVIS | Primary repository and VIS baseline | NVIDIA Source Code License-NC | [NVlabs/MinVIS](https://github.com/NVlabs/MinVIS) |
| VISAGE | `convert_coco2ytvis.py`, `queenvis/data_video/` lineage, and memory-bank implementation | Apache-2.0 | [KimHanjung/VISAGE](https://github.com/KimHanjung/VISAGE) |
| Mask2Former | Image segmentation architecture and vendored components | MIT | [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former) |
| VITA | Video instance segmentation code lineage | Apache-2.0 | [sukjunhwang/VITA](https://github.com/sukjunhwang/VITA) |
| Detectron2 | Framework dependency and derived utilities/demo code | Apache-2.0 | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) |
| IFC | Dataset-loading and video code lineage retained in file headers | Apache-2.0 | [sukjunhwang/IFC](https://github.com/sukjunhwang/IFC) |
| Deformable-DETR | Multi-scale deformable-attention components | Apache-2.0 | [fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) |
| Swin-Transformer-Semantic-Segmentation | Swin backbone lineage | Apache-2.0, with the Microsoft Swin component under MIT | [SwinTransformer/Swin-Transformer-Semantic-Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) |
| YouTube-VOS COCO API | YTVOS dataset API/evaluation lineage | BSD-2-Clause-style license | [youtubevos/cocoapi](https://github.com/youtubevos/cocoapi) |

## Included license copies

- [Apache License 2.0](LICENSES/Apache-2.0.txt): VISAGE, VITA, Detectron2, IFC, Deformable-DETR, and Apache-licensed Swin project components.
- [Mask2Former MIT License](LICENSES/Mask2Former-MIT.txt).
- [Microsoft Swin MIT License](LICENSES/Microsoft-Swin-MIT.txt).
- [YouTube-VOS COCO API BSD License](LICENSES/YouTubeVOS-BSD-2-Clause.txt).
- [NVIDIA Source Code License-NC](LICENSE): MinVIS and the default repository terms.

## VISAGE attribution and citation

The appropriate source notice is:

> `convert_coco2ytvis.py`, portions of `queenvis/data_video/`, and the memory-bank implementation in `queenvis/video_maskformer_model.py` are adapted from VISAGE, distributed under the Apache License, Version 2.0.

Academic citation:

```bibtex
@misc{kim2024visage,
  title        = {VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement},
  author       = {Hanjung Kim and Jaehyun Kang and Miran Heo and Sukjun Hwang and Seoung Wug Oh and Seon Joo Kim},
  year         = {2024},
  eprint       = {2312.04885},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```

Citation is an academic attribution request; the binding redistribution conditions are those in the applicable license text.
