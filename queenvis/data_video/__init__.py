# Adapted from VISAGE (Apache-2.0): https://github.com/KimHanjung/VISAGE
# See NOTICE and THIRD_PARTY_LICENSES.md for attribution and license details.
#
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

from .dataset_mapper import YTVISDatasetMapper, CocoClipDatasetMapper
from .build import *
from .combined_loader import *

from .datasets import *
from .ytvis_eval import YTVISEvaluator
