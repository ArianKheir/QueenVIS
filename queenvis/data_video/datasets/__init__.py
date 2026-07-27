# Adapted from VISAGE (Apache-2.0): https://github.com/KimHanjung/VISAGE
# See NOTICE and THIRD_PARTY_LICENSES.md for attribution and license details.
#
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

from . import builtin  # ensure the builtin datasets are registered

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
