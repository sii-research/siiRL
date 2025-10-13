# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
# Copyright 2025, Infrawaves. All rights reserved.
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


from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import torch
import torch.distributed as dist

from siirl.utils.extras.device import get_device_id, get_device_name



class _ReduceOp(Enum):
    """Enumeration for supported reduction operations."""

    SUM = dist.ReduceOp.SUM
    MAX = dist.ReduceOp.MAX
    MIN = dist.ReduceOp.MIN


# Configuration for metrics that require mean, max, and min aggregation.
# Format: { "key_in_local_data": "final_metric_prefix" }
METRIC_CONFIG_FULL = {
    "score": "critic/score",
    "rewards": "critic/rewards",
    "advantages": "critic/advantages",
    "returns": "critic/returns",
    "values": "critic/values",
    "response_length": "response/length",
    "prompt_length": "prompt/length",
    "correct_response_length": "response/correct_length",
    "wrong_response_length": "response/wrong_length",
}

# Configuration for metrics that only require mean aggregation.
# Format: { "key_in_local_data": "final_metric_prefix" }
METRIC_CONFIG_MEAN_ONLY = {
    "response_clip_ratio": "response/clip_ratio",
    "prompt_clip_ratio": "prompt/clip_ratio",
}

class DistributedMetricAggregator:
    """
    A helper class to encapsulate the logic for aggregating metrics
    in a distributed environment.
    """

    def __init__(
        self, local_metrics: Dict[str, Union[float, List[float], torch.Tensor]], group: Optional[dist.ProcessGroup]
    ):
        """
        Initializes the aggregator and prepares metrics for reduction.

        Args:
            local_metrics: The dictionary of metrics on the local rank.
            group: The process group for distributed communication.
        """
        self.group = group
        device_name = get_device_name()
        if device_name in ["cuda", "npu"]:
            self.device = f"{device_name}:{get_device_id()}"
        else:
            self.device = "cpu"
        self.op_buckets = self._bucket_local_metrics(local_metrics)

    def _bucket_local_metrics(self, metrics: Dict, expected_keys: set = None) -> defaultdict:
        """
        Parses local metrics and groups them by the required reduction operation.
        This step also performs local pre-aggregation on lists and tensors.
        This version correctly handles multi-element tensors as input.
        
        For Pipeline Parallel (PP), different stages may have different metrics.
        This method ensures all ranks have the same set of keys by adding missing
        metrics with default values (0.0) to avoid tensor shape mismatch in all_reduce.

        Args:
            metrics: Local metrics dictionary
            expected_keys: Optional set of all expected metric keys across all ranks
            
        Returns:
            A defaultdict containing keys and pre-aggregated values,
            grouped by reduction operation type (_ReduceOp).
        """
        buckets = defaultdict(list)
        
        # If expected_keys is provided, ensure all ranks have the same metrics
        if expected_keys:
            # define metrics that should be excluded from non-computing ranks
            # these are training-specific metrics that only the last PP stage should contribute
            training_metrics = {
                'actor/pg_loss', 'actor/kl_loss', 'actor/entropy_loss', 'actor/ppo_kl',
                'actor/pg_clipfrac', 'actor/pg_clipfrac_lower', 'actor/kl_coef',
                'critic/vf_loss', 'critic/clipfrac'
            }

            # Token counting metrics should only be contributed by PP rank 0 to avoid double counting
            token_counting_metrics = {
                'perf/total_num_tokens/mean'
            }
            
            for key in expected_keys:
                if key not in metrics:
                    # for training metrics: use None to indicate this rank shouldn't contribute
                    # for other metrics: use 0.0 as default
                    if any(key.startswith(prefix) for prefix in training_metrics) or key in token_counting_metrics:
                        # mark as None - will be handled specially in aggregation
                        metrics[key] = None
                    else:
                        # performance metrics get default value 0.0
                        metrics[key] = 0.0
        
        for key in sorted(metrics.keys()):
            value = metrics[key]
            
            # Skip None values (training metrics from non-contributing ranks)
            if value is None:
                # for training metrics that this rank (those ranks that are not the last PP stage) shouldn't contribute to,
                # add with count=0 so it doesn't affect the average
                buckets[_ReduceOp.SUM].append((key, (0.0, 0)))
                continue

            # Determine if the value is a list or a tensor that needs aggregation
            is_list = isinstance(value, list)
            is_tensor = isinstance(value, torch.Tensor)

            if "_max" in key:
                op_type = _ReduceOp.MAX
                if is_tensor:
                    # Use torch.max for tensors, get the scalar value
                    local_val = torch.max(value).item() if value.numel() > 0 else 0.0
                elif is_list:
                    local_val = max(value) if value else 0.0
                else: # Is a scalar float
                    local_val = value
                buckets[op_type].append((key, local_val))

            elif "_min" in key:
                op_type = _ReduceOp.MIN
                if is_tensor:
                    local_val = torch.min(value).item() if value.numel() > 0 else 0.0
                elif is_list:
                    local_val = min(value) if value else 0.0
                else:
                    local_val = value
                buckets[op_type].append((key, local_val))

            else:  # Default to mean calculation (SUM operation).
                op_type = _ReduceOp.SUM
                if is_tensor:
                    local_sum = torch.sum(value).item()
                    local_count = value.numel()
                elif is_list:
                    local_sum = sum(value) if value else 0.0
                    local_count = len(value)
                else: # Is a scalar float
                    local_sum = value
                    local_count = 1
                buckets[op_type].append((key, (local_sum, local_count)))
        return buckets

    def aggregate_and_get_results(self) -> Dict[str, float]:
        """
        Performs the distributed all_reduce operations and composes the final
        metrics dictionary.

        Returns:
            A dictionary with the globally aggregated metrics.
        """
        final_metrics = {}
        for op_type, data in self.op_buckets.items():
            if not data:
                continue

            keys, values = zip(*data)

            if op_type == _ReduceOp.SUM:
                sums, counts = zip(*values)
                sum_tensor = torch.tensor(sums, dtype=torch.float32, device=self.device)
                count_tensor = torch.tensor(counts, dtype=torch.float32, device=self.device)

                if self.group is not None:
                    dist.all_reduce(sum_tensor, op=op_type.value, group=self.group)
                    dist.all_reduce(count_tensor, op=op_type.value, group=self.group)

                global_sums = sum_tensor.cpu().numpy()
                global_counts = count_tensor.cpu().numpy()

                for i, key in enumerate(keys):
                    final_metrics[key] = global_sums[i] / global_counts[i] if global_counts[i] > 0 else 0.0
            else:  # MAX or MIN operations
                value_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
                if self.group is not None:
                    dist.all_reduce(value_tensor, op=op_type.value, group=self.group)

                global_values = value_tensor.cpu().numpy()
                for i, key in enumerate(keys):
                    final_metrics[key] = global_values[i]

        return final_metrics


