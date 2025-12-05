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


import torch
from statistics import mean
from typing import List, Optional, Union, Any, Dict
from dataclasses import dataclass
from tensordict import TensorDict


@dataclass
class Metric:
    name: None
    value: Any
    world_size: None

def MetricFunc(name: str):
    if "min" in name:
        return MinMetric
    elif "max" in name:
        return MaxMetric
    elif "sum" in name or "total" in name:
        return SumMetric
    else:
        return MeanMetric

def SumMetric(metrics: List[Metric]):
    value = [v
        for metric in metrics
        for v in (metric.value if isinstance(metric.value, list) else [metric.value])]
    return sum(value)

def MeanMetric(metrics: List[Metric]):
    value = [v
        for metric in metrics
        for v in (metric.value if isinstance(metric.value, list) else [metric.value])]
    return mean(value)


def MaxMetric(metrics: List[Metric]):
    value = [v
        for metric in metrics
        for v in (metric.value if isinstance(metric.value, list) else [metric.value])]
    return max(value)

def MinMetric(metrics: List[Metric]):
    value = [v
        for metric in metrics
        for v in (metric.value if isinstance(metric.value, list) else [metric.value])]
    return min(value)