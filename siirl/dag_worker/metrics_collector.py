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

"""
MetricsCollector: Orchestrates metrics aggregation for DAG worker training steps.

This module provides the MetricsCollector class which handles the collection,
aggregation, and formatting of training metrics across distributed workers.
"""

import os
import psutil
import torch
import torch.distributed as dist
from typing import Dict, Callable, Any
from torch.distributed import ProcessGroup
from tensordict import TensorDict, NonTensorData

from siirl.execution.dag import TaskGraph
from siirl.execution.dag.node import Node, NodeRole
from siirl.dag_worker.metric_aggregator import METRIC_CONFIG_FULL, METRIC_CONFIG_MEAN_ONLY
from siirl.dag_worker.dag_utils import (
    timer,
    prepare_local_batch_metrics,
    reduce_and_broadcast_metrics,
    remove_prefix_from_dataproto,
    add_prefix_to_dataproto,
    add_prefix_to_metrics,
)
from siirl.utils.extras.device import get_device_name, get_device_id
from siirl.utils.metrics.metric_utils import compute_throughout_metrics, compute_timing_metrics
from loguru import logger

class MetricsCollector:
    """
    Collects and aggregates training metrics across distributed workers.

    This class orchestrates the final metrics collection for each training step,
    including batch metrics, performance metrics, and timing information. It uses
    the existing four-layer metrics architecture:
    - Layer 1: torch.distributed (communication)
    - Layer 2: DistributedMetricAggregator (aggregation engine)
    - Layer 3: dag_utils functions (utility layer)
    - Layer 4: MetricsCollector (orchestration layer)
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        gather_group: ProcessGroup,
        taskgraph: TaskGraph,
        first_rollout_node: Node,
        get_node_dp_info_fn: Callable,
        multi_agent: bool = False,
        enable_perf: bool = False,
    ):
        """
        Initialize MetricsCollector with explicit parameters.

        Args:
            rank: Current process rank
            world_size: Total number of processes
            gather_group: Process group for metric aggregation
            taskgraph: DAG task graph for node iteration
            first_rollout_node: Reference rollout node for configuration
            get_node_dp_info_fn: Function to get node DP/TP/PP info
            multi_agent: Whether in multi-agent mode
            enable_perf: Whether to enable performance timing
        """
        self.rank = rank
        self.world_size = world_size
        self.gather_group = gather_group
        self.taskgraph = taskgraph
        self.first_rollout_node = first_rollout_node
        self.get_node_dp_info_fn = get_node_dp_info_fn
        self.multi_agent = multi_agent
        self.enable_perf = enable_perf

    def collect_final_metrics(self, batch: TensorDict, timing_raw: dict) -> Dict[str, float]:
        """
        Orchestrates the collection and computation of all metrics for a training step
        using a highly efficient, all_reduce-based aggregation strategy.

        This function replaces the old `compute -> reduce -> finalize` pipeline.

        Args:
            batch: Final batch data (TensorDict) containing all computed values
            timing_raw: Dictionary of raw timing measurements

        Returns:
            Dictionary of aggregated metrics
        """
        device_name = get_device_name()
        if device_name == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif device_name == "npu":
            torch.npu.reset_peak_memory_stats()

        final_metrics = {}

        # --- 1. Prepare all local metric data ---
        use_critic = any(node.node_role == NodeRole.CRITIC for node in self.taskgraph.nodes.values())
        local_data = prepare_local_batch_metrics(batch, use_critic=use_critic)

        # --- 2. Build the dictionary for our generic, high-performance aggregator ---
        # We want mean, max, and min for most standard metrics.
        metrics_to_aggregate = {}

        # Process metrics requiring mean, max, and min
        for key, prefix in METRIC_CONFIG_FULL.items():
            if key in local_data:
                # The aggregator determines the operation from the key.
                # We provide the same raw tensor for mean, max, and min calculations.
                metrics_to_aggregate[f"{prefix}/mean"] = local_data[key]
                metrics_to_aggregate[f"{prefix}_max"] = local_data[key]
                metrics_to_aggregate[f"{prefix}_min"] = local_data[key]

        # Process metrics requiring only mean
        for key, prefix in METRIC_CONFIG_MEAN_ONLY.items():
            if key in local_data:
                metrics_to_aggregate[f"{prefix}/mean"] = local_data[key]

        representative_actor_node = next(
            (n for n in self.taskgraph.nodes.values() if n.node_role == NodeRole.ACTOR), self.first_rollout_node
        )
        _, _, _, _, pp_rank_in_group, _ = self.get_node_dp_info_fn(representative_actor_node)
        # (1) For TP: we have already taken TP into account when we set global_token_num in compute_reward.
        # see: siirl/workers/dag_worker/mixins/node_executors_mixin.py:compute_reward
        # (2) For PP: only PP rank 0 contributes to avoid double counting within PP groups
        # The aggregation will average across DP groups and multiply by world size to get global estimate
        if pp_rank_in_group == 0:
            local_token_sum = sum(batch["global_token_num"])
            metrics_to_aggregate["perf/total_num_tokens/mean"] = float(local_token_sum)

        # --- 3. Perform the aggregated, distributed reduction ---
        with timer(self.enable_perf, "metrics_aggregation", timing_raw):
            aggregated_metrics = reduce_and_broadcast_metrics(metrics_to_aggregate, self.gather_group)

        # Post-process keys and values for the final output
        for key, value in aggregated_metrics.items():
            if "_max" in key and "mem" not in key:
                final_metrics[key.replace("_max", "/max")] = value
            elif "_min" in key:
                final_metrics[key.replace("_min", "/min")] = value
            else:
                final_metrics[key] = value

        # Special handling for total_num_tokens to convert mean back to sum
        if "perf/total_num_tokens/mean" in final_metrics:
            final_metrics["perf/total_num_tokens"] = final_metrics.pop(
                "perf/total_num_tokens/mean"
            ) * dist.get_world_size(self.gather_group)

        # --- 4. Handle special cases like Explained Variance ---
        if use_critic:
            # Determine the correct device for distributed operations
            device_name = get_device_name()
            if device_name in ["cuda", "npu"]:
                device = f"{device_name}:{get_device_id()}"
            else:
                # Fallback to the device of an existing tensor. If it's CPU, all_reduce will fail,
                # which is the original problem, indicating a deeper issue.
                device = local_data["returns"].device
            # These components only need to be summed. We can do a direct all_reduce.
            components_to_sum = {k: v for k, v in local_data.items() if k.endswith("_comp")}
            for tensor in components_to_sum.values():
                if self.gather_group is not None:
                    dist.all_reduce(tensor.to(device), op=dist.ReduceOp.SUM, group=self.gather_group)

            # Now all ranks have the global sums and can compute the final value.
            N = local_data["returns"].numel()
            total_N_tensor = torch.tensor([N], dtype=torch.int64, device=local_data["returns"].device)
            if self.gather_group is not None:
                dist.all_reduce(total_N_tensor.to(device), op=dist.ReduceOp.SUM, group=self.gather_group)
            global_N = total_N_tensor.item()

            if global_N > 0:
                global_returns_sum = final_metrics["critic/returns/mean"] * global_N
                global_returns_sq_sum = components_to_sum["returns_sq_sum_comp"].item()
                global_error_sum = components_to_sum["error_sum_comp"].item()
                global_error_sq_sum = components_to_sum["error_sq_sum_comp"].item()

                mean_returns = global_returns_sum / global_N
                var_returns = (global_returns_sq_sum / global_N) - (mean_returns**2)

                mean_error = global_error_sum / global_N
                var_error = (global_error_sq_sum / global_N) - (mean_error**2)

                final_metrics["critic/vf_explained_var"] = 1.0 - var_error / (var_returns + 1e-8)
            else:
                final_metrics["critic/vf_explained_var"] = 0.0

        # --- 5. Add timing and other rank-0-only metrics ---
        # Only rank 0 needs to compute these for logging.
        if self.rank == 0:
            batch["global_token_num"] = NonTensorData([final_metrics.get("perf/total_num_tokens", 0)])
            final_metrics.update(compute_throughout_metrics(batch, timing_raw, dist.get_world_size()))
            final_metrics["perf/process_cpu_mem_used_gb"] = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
            timing_metrics = compute_timing_metrics(batch, timing_raw)
            for key, value in timing_metrics.items():
                if key.startswith("timing_s/"):
                    final_metrics[key.replace("timing_s/", "perf/delta_time/")] = value

            # Calculate rollout and actor log probs difference statistics
            if "rollout_log_probs" in batch and "old_log_probs" in batch:
                rollout_probs = torch.exp(batch["rollout_log_probs"])
                actor_probs = torch.exp(batch["old_log_probs"])
                rollout_probs_diff = torch.masked_select(
                    torch.abs(rollout_probs.cpu() - actor_probs),
                    batch["response_mask"].bool().cpu()
                )
                if rollout_probs_diff.numel() > 0:
                    final_metrics.update({
                        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).item(),
                        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).item(),
                        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).item()
                    })

        # All ranks return the final metrics. Ranks other than 0 can use them if needed,
        # or just ignore them. This is cleaner than returning an empty dict.
        return final_metrics

    def collect_multi_agent_final_metrics(self, batch: TensorDict, ordered_metrics: list, timing_raw: dict) -> list:
        """
        Collects final metrics for multi-agent mode by iterating through agent rollout nodes.

        Args:
            batch: Final batch data (TensorDict) with multi-agent prefixes
            ordered_metrics: List of (key, value) tuples to extend
            timing_raw: Dictionary of raw timing measurements

        Returns:
            Extended list of ordered metrics
        """
        node_queue = self.taskgraph.get_entry_nodes()
        visited_nodes = set()
        while node_queue:
            cur_node = node_queue.pop(0)
            if cur_node.node_id in visited_nodes:
                continue
            if cur_node.node_role != NodeRole.ROLLOUT:
                break
            batch = remove_prefix_from_dataproto(batch, cur_node)
            final_metrics = self.collect_final_metrics(batch, timing_raw)
            final_metrics = add_prefix_to_metrics(final_metrics, cur_node)
            if final_metrics:
                ordered_metrics.extend(sorted(final_metrics.items()))
            if next_nodes := self.taskgraph.get_downstream_nodes(cur_node.node_id):
                for n in next_nodes:
                    if n.node_id not in visited_nodes:
                        node_queue.append(n)
            batch = add_prefix_to_dataproto(batch, cur_node)
        return ordered_metrics
