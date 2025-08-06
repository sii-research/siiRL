# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from .performance import GPUMemoryLogger, log_gpu_memory_usage
from siirl.utils.extras.import_utils import is_nvtx_available
from siirl.utils.extras.device import is_npu_available
from siirl.utils.debug.profile import DistProfiler, DistProfilerExtension 

if is_nvtx_available():
    pass
    # todo
elif is_npu_available:
    from .mstx_profile import NPUProfiler as DistProfiler
    from .mstx_profile import mark_annotate, mark_end_range, mark_start_range
else:
    pass
    # todo
# 文档todo

__all__ = [
    "GPUMemoryLogger",
    "log_gpu_memory_usage",
    "DistProfiler",
    "DistProfilerExtension",
    "mark_annotate",
    "mark_end_range",
    "mark_start_range",
]
