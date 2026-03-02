#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for normalization with mismatched stat shapes."""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS


def test_mean_std_length_mismatch():
    # create dummy stats where mean/std have length 128 but actual tensor
    # has length 48.  The normalization step should crop without error.
    stats = {
        OBS_LANGUAGE_TOKENS: {
            "mean": torch.zeros(1, 128),
            "std": torch.ones(1, 128),
        }
    }
    features = {OBS_LANGUAGE_TOKENS: PolicyFeature(type=FeatureType.LANGUAGE, shape=(48,))}

    processor = NormalizerProcessorStep(
        features=features,
        norm_map={FeatureType.LANGUAGE: FeatureType.LANGUAGE},
        stats=stats,
    )

    # build fake observation dims [batch, seq]
    tensor = torch.randn(2, 48)
    normalized = processor._apply_transform(tensor, OBS_LANGUAGE_TOKENS, FeatureType.LANGUAGE)
    assert normalized.shape == tensor.shape
    # mean=0 std=1 -> normalization should be identity-ish
    assert torch.allclose(normalized, tensor)
