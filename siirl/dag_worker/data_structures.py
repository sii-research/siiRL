# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from siirl.data_coordinator import DataProto


@dataclass
class ValidationResult:
    """A structured container for a single validation sample's results."""

    input_text: str
    output_text: str
    score: float
    data_source: str
    reward_tensor: torch.Tensor
    extra_rewards: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationPayload:
    """A lightweight, serializable container for validation metrics for efficient gathering."""

    input_text: str
    score: float
    data_source: str
    extra_rewards: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeOutput:
    """A standardized return object for all node execution functions."""

    batch: DataProto
    metrics: Dict[str, Any] = field(default_factory=dict)
