# -*- coding: utf-8 -*-
# Modified by Arian Kheirandish for QueenVIS.
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_queenvis_config(cfg):
    cfg.DATASETS.DATASET_RATIO = []
    cfg.INPUT.SAMPLING_FRAME_RATIO = 1.0
    cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE = False
    cfg.MODEL.MASK_FORMER.PROPAGATE_QUERIES = True
    cfg.MODEL.MASK_FORMER.PROP_ALPHA = 0.25
    cfg.MODEL.MASK_FORMER.PROP_THRESHOLD = 0.80
    cfg.MODEL.MASK_FORMER.MEMORY_BANK_SIZE = 3

    # Pseudo Data Use
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)
    cfg.INPUT.PSEUDO.COPY_PASTE = False
    
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0
