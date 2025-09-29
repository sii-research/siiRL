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

import asyncio
import csv
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import hashlib
import numpy as np
import psutil
import ray
import torch
import torch.distributed as dist
from loguru import logger
from tensordict import TensorDict
from zoneinfo import ZoneInfo

from siirl.data_coordinator.dataloader import DataLoaderNode
from siirl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from siirl.utils.extras.device import device_synchronize, get_device_id, get_device_name
from siirl.utils.metrics.metric_utils import compute_throughout_metrics, compute_timing_metrics
from siirl.global_config.params import SiiRLArguments
from siirl.engine.base_worker import Worker
from siirl.execution.dag import TaskGraph
from siirl.execution.dag.node import Node, NodeRole, NodeType
from siirl.data_coordinator import DataProto
from siirl.dag_worker.dag_utils import add_prefix_to_dataproto, remove_prefix_from_dataproto, add_prefix_to_metrics
from siirl.data_coordinator.protocol import collate_fn


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


def sort_by_other_list(objects, reference_list):
    """
    Sort the target list according to the order of the reference list

    Args:
    target_list: The target list to be sorted
    reference_list: The reference list used as the basis for sorting

    Returns:
    The sorted target list
    """
    combined = [(val, i, obj) for i, (val, obj) in enumerate(zip(reference_list, objects))]
    combined.sort()
    sorted_objects = [item[2] for item in combined]
    return sorted_objects


def consistent_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)
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


class UtilitiesMixin:
    """A collection of utility methods for the DAGWorker, including I/O, logging, and metrics."""

    enable_perf: bool
    taskgraph: TaskGraph
    config: SiiRLArguments
    global_steps: int
    _gather_group: Optional[dist.ProcessGroup]
    _rank: int
    workers: Dict[str, Worker]
    first_rollout_node: Node
    dataloader: DataLoaderNode
    validate_tokenizer: Any
    internal_data_cache: Dict[str, DataProto]
    data_buffers: List["ray.actor.ActorHandle"]
    taskgraph_execute_finished: bool

    generate: Any
    compute_ref_log_prob: Any
    compute_old_log_prob: Any
    train_actor: Any
    compute_value: Any
    train_critic: Any
    _generate_node_worker_key: Any
    _get_node_dp_info: Any
    postprocess_sampling: Any

    @contextmanager
    def _timer(self, name: str, timing_dict: dict):
        """A context manager to measure execution time of a code block."""
        if self.enable_perf:
            device_synchronize()
        start_time = time.perf_counter()
        yield
        if self.enable_perf:
            device_synchronize()
        end_time = time.perf_counter()
        timing_dict[name] = timing_dict.get(name, 0) + end_time - start_time

    def _set_node_executables(self):
        """Maps node roles to their corresponding execution methods."""
        ROLE_METHOD_MAPPING = {
            (NodeRole.ROLLOUT, False): self.generate,
            (NodeRole.REFERENCE, False): self.compute_ref_log_prob,
            (NodeRole.ACTOR, True): self.compute_old_log_prob,
            (NodeRole.ACTOR, False): self.train_actor,
            (NodeRole.CRITIC, True): self.compute_value,
            (NodeRole.CRITIC, False): self.train_critic,
            (NodeRole.POSTPROCESS_SAMPLING, False): self.postprocess_sampling,
        }
        for node in self.taskgraph.nodes.values():
            if node.node_role in [NodeRole.REWARD, NodeRole.ADVANTAGE]:
                continue
            key = (node.node_role, node.only_forward_compute)
            if executable_func := ROLE_METHOD_MAPPING.get(key):
                node.executable = executable_func

    def _save_checkpoint(self):
        """
        Saves a checkpoint in a fully distributed, robust, and multi-agent compatible manner.
        - Each agent's state is saved to a unique, agent-specific subdirectory.
        - A barrier ensures all file writes are complete before committing.
        - Only Rank 0 updates the tracker file, effectively "committing" the checkpoint atomically.
        """
        from siirl.execution.dag.node import NodeType

        step_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        os.makedirs(step_dir, exist_ok=True)
        dist.barrier(self._gather_group)

        logger.info(f"Rank {self._rank}: Saving checkpoint for global_step {self.global_steps} to {step_dir}")

        # --- 1. All ranks save their sharded data ---
        # Save states for trainable models (Actor and Critic) for all agents.
        saved_worker_keys = set()
        for node in self.taskgraph.nodes.values():
            if node.node_type == NodeType.MODEL_TRAIN and node.node_role in [NodeRole.ACTOR, NodeRole.CRITIC]:
                node_worker_key = self._generate_node_worker_key(node)
                if node_worker_key in saved_worker_keys:
                    continue

                worker = self.workers[node_worker_key]

                # Create an agent-specific subdirectory name to prevent different agents
                # from overwriting each other's checkpoints.
                sub_dir_name = f"{node.node_role.name.lower()}_agent_{node.agent_group}"
                checkpoint_path = os.path.join(step_dir, sub_dir_name)

                # The config key for max checkpoints is still role-based (e.g., max_actor_ckpt_to_keep).
                role_name_for_config = node.node_role.name.lower()
                max_ckpt_keep = getattr(self.config.trainer, f"max_{role_name_for_config}_ckpt_to_keep", 10)

                worker.save_checkpoint(
                    local_path=checkpoint_path, global_step=self.global_steps, max_ckpt_to_keep=max_ckpt_keep
                )
                saved_worker_keys.add(node_worker_key)

        # In each DP group, only TP rank 0 saves the DataLoader state to avoid redundancy.
        _, dp_rank, tp_rank, _, pp_rank, _ = self._get_node_dp_info(self.first_rollout_node)
        if tp_rank == 0 and pp_rank == 0:
            # The filename is based on the DP rank to distinguish different data partitions.
            dataloader_path = os.path.join(step_dir, f"data_dp_rank_{dp_rank}.pt")
            dataloader_state = self.dataloader.state_dict()
            torch.save(dataloader_state, dataloader_path)
            logger.debug(f"Rank {self._rank} (DP_Rank {dp_rank}, TP_Rank {tp_rank}, PP_Rank {pp_rank}): Saved dataloader state.")

        # --- 2. All ranks wait for I/O to complete ---
        # This barrier ensures all data is written BEFORE committing the checkpoint via the tracker file.
        logger.debug(f"Rank {self._rank}: All data saved. Waiting at barrier before committing checkpoint.")
        dist.barrier(self._gather_group)

        # --- 3. Only Rank 0 commits the checkpoint by writing the tracker file ---
        if self._rank == 0:
            tracker_file = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
            with open(tracker_file, "w") as f:
                f.write(str(self.global_steps))
            logger.info(f"Rank 0: Checkpoint for step {self.global_steps} successfully committed.")

        # Final barrier to ensure the tracker file is visible before any rank proceeds.
        dist.barrier(self._gather_group)
        logger.info(f"Rank {self._rank}: Finished saving and committing checkpoint for step {self.global_steps}.")

    def _load_checkpoint(self):
        """
        Loads a checkpoint in a fully distributed and consistent manner.
        - It relies on Rank 0 to be the single source of truth for which checkpoint to load and broadcasts that
          decision. This is essential to prevent inconsistencies from filesystem latency.
        - It constructs agent-specific paths to load the correct state for each agent.
        """
        from siirl.execution.dag.node import NodeType

        if self.config.trainer.resume_mode == "disable":
            if self._rank == 0:
                logger.info("Checkpoint loading is disabled. Starting from scratch.")
            self.global_steps = 0
            return

        # --- 1. Only Rank 0 determines the path to load ---
        checkpoint_path_container = [None]
        if self._rank == 0:
            checkpoint_dir = self.config.trainer.default_local_dir
            resume_from_path = self.config.trainer.resume_from_path

            path_to_load = None
            if self.config.trainer.resume_mode == "auto":
                # This now reads from an atomically-written tracker file.
                latest_path = find_latest_ckpt_path(checkpoint_dir)
                if latest_path:
                    logger.info(f"Rank 0: Auto-found latest checkpoint at {latest_path}")
                    path_to_load = latest_path
            elif self.config.trainer.resume_mode == "resume_path" and resume_from_path:
                logger.info(f"Rank 0: Attempting to load from specified path: {resume_from_path}")
                path_to_load = resume_from_path

            if path_to_load and os.path.exists(path_to_load):
                checkpoint_path_container[0] = path_to_load
            else:
                logger.warning(
                    f"Rank 0: Checkpoint path not found or invalid: '{path_to_load}'. Starting from scratch."
                )

        # --- 2. Rank 0 broadcasts the decision to all other ranks ---
        # This is the crucial step for ensuring consistency.
        dist.broadcast_object_list(checkpoint_path_container, src=0)
        global_step_folder = checkpoint_path_container[0]

        # --- 3. All ranks act on the broadcasted decision ---
        if global_step_folder is None:
            if self._rank == 0:
                logger.info("No valid checkpoint to load. Training will start from step 0.")
            self.global_steps = 0
            dist.barrier(self._gather_group)
            return

        try:
            self.global_steps = int(os.path.basename(global_step_folder).split("global_step_")[-1])
            logger.info(f"Rank {self._rank}: Resuming from checkpoint. Setting global_steps to {self.global_steps}.")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Could not parse global step from checkpoint path: {global_step_folder}") from e

        # Load sharded model states for all agents.
        loaded_worker_keys = set()
        for node in self.taskgraph.nodes.values():
            if node.node_type == NodeType.MODEL_TRAIN and node.node_role in [NodeRole.ACTOR, NodeRole.CRITIC]:
                node_worker_key = self._generate_node_worker_key(node)
                if node_worker_key in loaded_worker_keys:
                    continue

                worker = self.workers[node_worker_key]

                # Construct the agent-specific subdirectory name to load from.
                sub_dir_name = f"{node.node_role.name.lower()}_agent_{node.agent_group}"
                checkpoint_path = os.path.join(global_step_folder, sub_dir_name)

                if os.path.exists(checkpoint_path):
                    worker.load_checkpoint(
                        local_path=checkpoint_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
                    )
                    loaded_worker_keys.add(node_worker_key)
                else:
                    logger.warning(
                        f"Rank {self._rank}: Checkpoint for agent {node.agent_group}'s {node.node_role.name} not found "
                        f"at {checkpoint_path}. Weights will be from initialization."
                        f"If has multi-agent, will share the same checkpoint in agents"
                    )

        # Load dataloader state. All ranks in a DP group load from the same file.
        _, dp_rank, _, _, _, _ = self._get_node_dp_info(self.first_rollout_node)
        dataloader_path = os.path.join(global_step_folder, f"data_dp_rank_{dp_rank}.pt")
        if os.path.exists(dataloader_path):
            dataloader_state = torch.load(dataloader_path, map_location="cpu")
            self.dataloader.load_state_dict(dataloader_state)
        else:
            logger.warning(
                f"Rank {self._rank} (DP_Rank {dp_rank}): Dataloader checkpoint not found at {dataloader_path}. Sampler "
                f"state will not be restored, which may lead to data inconsistency."
            )

        # Barrier to ensure all ranks are synchronized after loading.
        dist.barrier(self._gather_group)
        logger.info(f"Rank {self._rank}: Finished loading all checkpoint components.")

    def _log_metrics_to_console(self, ordered_metrics: List[Tuple[str, Any]], step: int):
        """Logs a formatted string of metrics to the console on rank 0."""
        if self._rank != 0:
            return
        log_parts = [f"step:{step}"]
        log_parts.extend([f"{k}:{v:.4f}" if isinstance(v, float) else f"{k}:{v}" for k, v in ordered_metrics])
        logger.info(" | ".join(log_parts))

    def _reduce_and_broadcast_metrics(
        self, local_metrics: Dict[str, Union[float, List[float], torch.Tensor]], group: dist.ProcessGroup
    ) -> Dict[str, float]:
        """
        Aggregates metrics in a distributed environment using a dedicated helper class.
        For Pipeline Parallel setups, ensures all ranks have the same metric keys to avoid
        tensor shape mismatch during all_reduce operations.

        Args:
            local_metrics: A dictionary of metrics on each rank.
            group: The process group for the aggregation.

        Returns:
            A dictionary with the globally aggregated metrics, available on all ranks.
        """
        if not isinstance(local_metrics, dict) or not local_metrics:
            return {}

        world_size = dist.get_world_size(group)
        if world_size <= 1:
            # If not in a distributed setting, perform local aggregation only.
            aggregator = DistributedMetricAggregator(local_metrics, group=None)
            # The bucketed values are already the final values in a non-distributed case.
            final_metrics = {}
            for op_type, data in aggregator.op_buckets.items():
                for key, value in data:
                    if op_type == _ReduceOp.SUM: # value is a (sum, count) tuple
                        final_metrics[key] = value[0] / value[1] if value[1] > 0 else 0.0
                    else: # value is a float
                        final_metrics[key] = float(value)
            return final_metrics

        # In Megatron with Pipeline Parallel:
        # 1. First gather all metric keys from all ranks to ensure consistency
        local_keys = set(local_metrics.keys())
        all_keys_list = [None] * world_size
        dist.all_gather_object(all_keys_list, local_keys, group=group)
        
        # 2. Union all keys to get the complete set of expected metrics
        all_expected_keys = set()
        for keys_set in all_keys_list:
            all_expected_keys.update(keys_set)
        
        # 3. Use the aggregator with unified keys to perform communication
        aggregator = DistributedMetricAggregator(local_metrics, group)
        # NOTE(Ping Zhang): Ensure all ranks have the same metrics by adding missing ones with default values
        aggregator.op_buckets = aggregator._bucket_local_metrics(local_metrics, all_expected_keys)
        return aggregator.aggregate_and_get_results()

    def _prepare_local_batch_metrics(self, batch: DataProto, use_critic: bool = True) -> Dict[str, torch.Tensor]:
        """
        Prepares a dictionary of raw local metric tensors from a batch.
        This function DOES NOT pre-aggregate values (like sum, max, min).
        It provides the raw data needed for a more efficient `all_reduce` aggregation.

        Args:
            batch: The local data shard for the current rank.
            use_critic: Flag to include critic-related metric components.

        Returns:
            A dictionary of tensors representing local, raw metric values.
        """
        from siirl.utils.metrics.metric_utils import _compute_response_info

        response_info = _compute_response_info(batch)
        response_mask = response_info["response_mask"].bool()
        device = batch.batch["advantages"].device
        max_response_length = batch.batch["responses"].shape[-1]
        response_lengths = response_info["response_length"].to(device)
        prompt_lengths = response_info["prompt_length"].to(device)
        # Components for correct/wrong response length metrics
        correct_threshold = 0.5
        rewards_per_response = batch.batch["token_level_rewards"].sum(-1)
        correct_mask = rewards_per_response > correct_threshold
        # Components for prompt clip ratio
        prompt_attn_mask = batch.batch["attention_mask"][:, :-max_response_length]
        max_prompt_length = prompt_attn_mask.size(-1)

        # Prepare a dictionary to hold all local raw values
        local_data = {
            "score": batch.batch["token_level_scores"].sum(-1),
            "rewards": batch.batch["token_level_rewards"].sum(-1),
            "advantages": torch.masked_select(batch.batch["advantages"], response_mask),
            "returns": torch.masked_select(batch.batch["returns"], response_mask),
            "response_length": response_info["response_length"].to(device),
            "prompt_length": response_info["prompt_length"].to(device),
            "correct_response_length": response_lengths[correct_mask],
            "wrong_response_length": response_lengths[~correct_mask],
            "response_clip_ratio": torch.eq(response_info["response_length"], max_response_length).float(),
            "prompt_clip_ratio": torch.eq(prompt_lengths, max_prompt_length).float(),
        }

        if use_critic:
            valid_values = torch.masked_select(batch.batch["values"], response_mask)
            error = local_data["returns"] - valid_values

            critic_data = {
                "values": valid_values,
                # Special components for explained variance. These will be summed globally.
                "returns_sq_sum_comp": torch.sum(torch.square(local_data["returns"])),
                "error_sum_comp": torch.sum(error),
                "error_sq_sum_comp": torch.sum(torch.square(error)),
            }
            local_data.update(critic_data)

        return local_data

    def _collect_final_metrics(self, batch: DataProto, timing_raw: dict) -> Dict[str, float]:
        """
        Orchestrates the collection and computation of all metrics for a training step
        using a highly efficient, all_reduce-based aggregation strategy.

        This function replaces the old `compute -> reduce -> finalize` pipeline.
        """
        device_name = get_device_name()
        if device_name == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif device_name == "npu":
            torch.npu.reset_peak_memory_stats()

        final_metrics = {}

        # --- 1. Prepare all local metric data ---
        use_critic = any(node.node_role == NodeRole.CRITIC for node in self.taskgraph.nodes.values())
        local_data = self._prepare_local_batch_metrics(batch, use_critic=use_critic)

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
        _, _, _, _, pp_rank_in_group, _ = self._get_node_dp_info(representative_actor_node)
        # (1) For TP: we have already taken TP into account when we set global_token_num in compute_reward.
        # see: siirl/workers/dag_worker/mixins/node_executors_mixin.py:compute_reward
        # (2) For PP: only PP rank 0 contributes to avoid double counting within PP groups
        # The aggregation will average across DP groups and multiply by world size to get global estimate
        if pp_rank_in_group == 0:
            local_token_sum = sum(batch.meta_info.get("global_token_num", [0]))
            metrics_to_aggregate["perf/total_num_tokens/mean"] = float(local_token_sum)

        # --- 3. Perform the aggregated, distributed reduction ---
        with self._timer("metrics_aggregation", timing_raw):
            aggregated_metrics = self._reduce_and_broadcast_metrics(metrics_to_aggregate, self._gather_group)

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
            ) * dist.get_world_size(self._gather_group)

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
                if self._gather_group is not None:
                    dist.all_reduce(tensor.to(device), op=dist.ReduceOp.SUM, group=self._gather_group)

            # Now all ranks have the global sums and can compute the final value.
            N = local_data["returns"].numel()
            total_N_tensor = torch.tensor([N], dtype=torch.int64, device=local_data["returns"].device)
            if self._gather_group is not None:
                dist.all_reduce(total_N_tensor.to(device), op=dist.ReduceOp.SUM, group=self._gather_group)
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
        if self._rank == 0:
            batch.meta_info["global_token_num"] = [final_metrics.get("perf/total_num_tokens", 0)]
            final_metrics.update(compute_throughout_metrics(batch, timing_raw, dist.get_world_size()))
            final_metrics["perf/process_cpu_mem_used_gb"] = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
            timing_metrics = compute_timing_metrics(batch, timing_raw)
            for key, value in timing_metrics.items():
                if key.startswith("timing_s/"):
                    final_metrics[key.replace("timing_s/", "perf/delta_time/")] = value

        # All ranks return the final metrics. Ranks other than 0 can use them if needed,
        # or just ignore them. This is cleaner than returning an empty dict.
        return final_metrics


    def _collect_multi_final_metrics(self, batch: DataProto, ordered_metrics: dict, timing_raw: dict) -> Dict[str, float]:
        node_queue = self.taskgraph.get_entry_nodes()
        visited_nodes = set()
        while node_queue:
            cur_node = node_queue.pop(0)
            if cur_node.node_id in visited_nodes:
                continue
            if cur_node.node_role !=  NodeRole.ROLLOUT:
                break
            batch = remove_prefix_from_dataproto(batch, cur_node)        
            final_metrics = self._collect_final_metrics(batch, timing_raw)
            final_metrics = add_prefix_to_metrics(final_metrics, cur_node)
            if final_metrics:
                ordered_metrics.extend(sorted(final_metrics.items()))
            if next_nodes := self.taskgraph.get_downstream_nodes(cur_node.node_id):
                for n in next_nodes:
                    if n.node_id not in visited_nodes:
                        node_queue.append(n)
            batch = add_prefix_to_dataproto(batch, cur_node)
        return ordered_metrics

    def put_data_to_buffers(
        self, key: str, data: DataProto, source_dp_size: int, dest_dp_size: int, enforce_buffer: bool, timing_raw: Dict[str, float]
    ):
        """Puts data into shared Ray plasma store for consumption by downstream nodes."""
        try:
            logger.debug(f"Rank {self._rank}: Starting put_data_to_buffers for key '{key}', source_dp_size={source_dp_size}, dest_dp_size={dest_dp_size}")
            
            data.meta_info["padding_values"] = {"input_ids": self.validate_tokenizer.pad_token_id, "responses": self.validate_tokenizer.pad_token_id, "labels": -100, "attention_mask": 0, "response_mask": 0}
            data.meta_info["padding_side"] = self.validate_tokenizer.padding_side

            if (not enforce_buffer) and source_dp_size == dest_dp_size:
                with self._timer(f"put_intern_data_{key}", timing_raw):
                    logger.debug(f"Rank {self._rank}: DP size match ({source_dp_size}). Storing data for key '{key}' in local cache.")
                    self.internal_data_cache[key] = data
                    logger.debug(f"Rank {self._rank}: Successfully stored data for key '{key}' in local cache.")
            else:
                logger.debug(f"Rank {self._rank}: DP size mismatch (source={source_dp_size}, dest={dest_dp_size}). Using Ray buffers for key '{key}'.")
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    logger.debug(f"Rank {self._rank}: Creating new event loop for key '{key}'")
                    loop = asyncio.new_event_loop()

                with self._timer(f"put_ray_proto_data_{key}", timing_raw):
                    chunks = data.chunk(chunks=len(self.data_buffers))
                    logger.debug(f"Rank {self._rank}: Created {len(chunks)} chunks for key '{key}'")
                    put_futures = [buf.put.remote(key, chunk) for buf, chunk in zip(self.data_buffers, chunks)]
                
                with self._timer(f"put_proto_data_{key}", timing_raw):
                    try:
                        loop.run_until_complete(asyncio.gather(*put_futures))
                        logger.debug(f"Rank {self._rank}: Successfully stored all chunks for key '{key}' in Ray buffers")
                    except Exception as e:
                        logger.error(f"Rank {self._rank}: Failed to store chunks for key '{key}' in Ray buffers: {e}")
                        raise
        except Exception as e:
            logger.error(f"Rank {self._rank}: Unexpected error in put_data_to_buffers for key '{key}': {e}")
            raise  # Re-raise the exception to maintain the original behavior

    def get_data_from_buffers(self, key: str, my_current_dp_rank: int, my_current_dp_size: int, timing_raw: Dict[str, float]) -> Optional[DataProto]:
        """Gets data from shared buffers that was produced by an upstream node."""
        try:
            # First, check the high-speed internal cache.
            with self._timer(f"get_intern_data_{key}", timing_raw):
                if key in self.internal_data_cache:
                    logger.debug(f"Rank {self._rank}: Found data for key '{key}' in local cache. Bypassing Ray.")
                    return self.internal_data_cache.pop(key)

            # If not in the local cache, fall back to remote Ray buffers.
            logger.debug(f"Rank {self._rank}: Data for key '{key}' not in local cache. Fetching from remote buffers.")
            if not self.data_buffers:
                logger.error(f"Rank {self._rank}: data_buffers is None, cannot get data for key '{key}'")
                return None

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                logger.debug(f"Rank {self._rank}: Creating new event loop for key '{key}'")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            try:
                logger.debug(f"Rank {self._rank}: Attempting Ray remote call for key '{key}'")
                first_item = loop.run_until_complete(self.data_buffers[0].get.remote(key, my_current_dp_rank, my_current_dp_size))
                logger.debug(f"Rank {self._rank}: Completed Ray remote call for key '{key}', got result type: {type(first_item)}")
            except Exception as e:
                logger.error(f"Rank {self._rank}: Error getting data from Ray buffer for key '{key}': {e}")
                return None

            if first_item is None:
                logger.error(f"Rank {self._rank}: first_item is None for key '{key}'")
                return None

            if isinstance(first_item, ray.ObjectRef):
                with self._timer(f"get_ref_data_{key}", timing_raw):
                    try:
                        return loop.run_until_complete(first_item)
                    except Exception as e:
                        logger.error(f"Rank {self._rank}: Error resolving Ray ObjectRef for key '{key}': {e}")
                        return None
            elif isinstance(first_item, DataProto):
                try:
                    # If data was chunked, retrieve all chunks and concatenate
                    with self._timer(f"get_proto_data_{key}", timing_raw):
                        other_chunks_futures = [b.get.remote(key, my_current_dp_rank, my_current_dp_size) for b in self.data_buffers[1:]]
                        other_chunks = loop.run_until_complete(asyncio.gather(*other_chunks_futures))
                    with self._timer(f"get_proto_data_concat_chunks_{key}", timing_raw):
                        return DataProto.concat([first_item] + other_chunks)
                except Exception as e:
                    logger.error(f"Rank {self._rank}: Error concatenating chunks for key '{key}': {e}")
                    return None
            logger.error(f"Rank {self._rank}: first_item type {type(first_item)} is neither ray.ObjectRef nor DataProto for key '{key}'")
            return None

        except Exception as e:
            logger.error(f"Rank {self._rank}: Unexpected error in get_data_from_buffers for key '{key}': {e}")
            return None

    def reset_data_buffer(self, all_keys: List[str]):
        """
        Reset the data buffer for a given list of keys.
        """
        if self._rank == 0:
            loop = asyncio.get_event_loop()
            for data_buffer in self.data_buffers:
                loop.run_until_complete(data_buffer.reset.remote())

    def taskgroup_have_finish(self) -> bool:
        """
        Check if the taskgroup has finished.
        """
        return self.taskgraph_execute_finished

    def format_metrics_by_group(
        self, metrics: Dict[str, Any], group_order: List[str], float_precision: int = 3, delimiter: str = " - "
    ) -> Dict[str, Any]:
        """
        A flexible helper function that formats metrics based on a predefined group order
        and alphabetical order within groups. It supports extracting specific keys from
        a group to be placed elsewhere in the sequence.
        """
        if not metrics:
            return {}

        ordered_dict = {}
        processed_keys = set()

        # Pre-identify all explicitly mentioned full keys to exclude them from group processing.
        explicitly_mentioned_keys = {key for key in group_order if key in metrics}

        # 1. Process metrics according to the defined group/key order.
        for pattern in group_order:
            # First, check if the pattern is a full key that should be processed now.
            if pattern in explicitly_mentioned_keys and pattern not in processed_keys:
                ordered_dict[pattern] = metrics[pattern]
                processed_keys.add(pattern)
            else:
                # Otherwise, treat the pattern as a group prefix.
                group_prefix = f"{pattern}/"

                # Find all keys belonging to this group, excluding any that are already processed
                # or explicitly mentioned elsewhere in the order. Then sort them alphabetically.
                keys_in_group = sorted(
                    [
                        key
                        for key in metrics
                        if key.startswith(group_prefix)
                        and key not in processed_keys
                        and key not in explicitly_mentioned_keys
                    ]
                )

                for key in keys_in_group:
                    ordered_dict[key] = metrics[key]
                    processed_keys.add(key)

        # 2. Process all remaining keys that were not matched by any rule.
        remaining_keys = sorted([key for key in metrics if key not in processed_keys])
        if remaining_keys:
            for key in remaining_keys:
                ordered_dict[key] = metrics[key]

        return ordered_dict

    @staticmethod
    def _get_time_now(time_zone: str = "Asia/Shanghai") -> datetime:
        """
        Returns the current time in Shanghai timezone.
        """
        return datetime.now(tz=ZoneInfo(time_zone))

    def _try_to_get_model_name_from_path(self) -> str:
        """
        Attempts to extract the model name from the model path in the configuration.
        """
        model_path = self.config.actor_rollout_ref.model.path
        return os.path.basename(os.path.normpath(model_path))

    def _aggregate_and_write_performance_metrics(self, metrics: Dict[str, Any]):
        """
        Gathers performance metrics from all ranks to rank 0 and writes them to a CSV file.
        Each row corresponds to a metric key COMMON to all ranks, and each column to a rank.
        This function is called only if performance profiling is enabled.
        """
        # Gather all metrics dictionaries to rank 0
        world_size = dist.get_world_size()
        gathered_metrics = [None] * world_size if self._rank == 0 else None
        dist.gather_object(metrics, gathered_metrics, dst=0, group=self._gather_group)

        if self._rank == 0:
            if not gathered_metrics:
                logger.warning("No metrics gathered on rank 0. Skipping performance CSV write.")
                return

            # Filter out any non-dict items and find the intersection of keys
            valid_metrics = [m for m in gathered_metrics if isinstance(m, dict) and m]
            if not valid_metrics:
                logger.warning("No valid metric dictionaries received on rank 0. Skipping CSV write.")
                return

            # Start with keys from the first valid dict, then find the intersection with the rest
            common_keys = set(valid_metrics[0].keys())
            for rank_metrics in valid_metrics[1:]:
                common_keys.intersection_update(rank_metrics.keys())

            sorted_keys = sorted(list(common_keys))

            if not sorted_keys:
                logger.warning(
                    f"No common metric keys found across all ranks for step {self.global_steps}. Skipping CSV write."
                )
                return

            # Define output directory and create it if it doesn't exist
            ts = self._get_time_now().strftime("%Y-%m-%d-%H-%M-%S")
            try:
                # Try to get model name from model path config
                model_name = self._try_to_get_model_name_from_path()
                output_dir = os.path.join("performance_logs", model_name, ts)
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create performance log directory {output_dir}: {e}")
                return

            filename = os.path.join(output_dir, f"world_{world_size}_step_{self.global_steps}_common_metrics.csv")

            # Write data to the CSV file
            try:
                with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)

                    # Write the header row: ['metric', 'rank_0', 'rank_1', ...]
                    header = (
                        ["metric"]
                        + [f"rank_{i}" for i in range(world_size)]
                        + ["max", "min", "delta_max_min", "delta_max_rank_0"]
                    )
                    writer.writerow(header)

                    # Write one row for each common metric key
                    for key in sorted_keys:
                        # The first element of the row is the metric name
                        row = [key]
                        # Append the value for this key from each rank's metrics
                        for i in range(world_size):
                            rank_metrics = gathered_metrics[i]
                            # Since we know the key is common, we can access it directly,
                            # but add a check for robustness in case an object wasn't a dict.
                            if isinstance(rank_metrics, dict):
                                value = rank_metrics.get(key, "Error: Key Missing")
                            else:
                                value = "N/A: Invalid Data"
                            row.append(value)
                        # Calculate max and min values for the row
                        row_max = max([x for x in row[1:] if isinstance(x, (int, float))], default="N/A")
                        row_min = min([x for x in row[1:] if isinstance(x, (int, float))], default="N/A")
                        row_delta_max = (
                            row_max - row_min
                            if isinstance(row_max, (int, float)) and isinstance(row_min, (int, float))
                            else "N/A"
                        )
                        row_delta_rank0 = row_max - row[1] if isinstance(row[1], (int, float)) else "N/A"
                        row.extend([row_max, row_min, row_delta_max, row_delta_rank0])
                        writer.writerow(row)

                logger.info(
                    f"Common performance metrics for step {self.global_steps} successfully written to {filename}"
                )

            except OSError as e:
                logger.error(f"Failed to write performance metrics to CSV file {filename}: {e}")

    def _log_core_performance_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Logs a formatted, easy-to-read summary of core performance metrics on rank 0.
        This provides a clear, separate view of the most important indicators.
        """
        if self._rank != 0:
            return

        # Helper to safely get metric values, returning 'N/A' if the key is not found
        def get_metric(key, precision=3):
            val = metrics.get(key)
            if val is None:
                return "N/A"
            if isinstance(val, (float, np.floating)):
                return f"{val:.{precision}f}"
            return val

        # --- Build the log string ---
        log_str = f"\n\n{'=' * 25} RANK({self._rank}): Core Performance Metrics (Step: {step}) {'=' * 25}\n"

        # --- Overall Performance ---
        log_str += "\n--- ⏱️  Overall Performance ---\n"
        log_str += f"  {'Step Time':<28}: {get_metric('perf/time_per_step', 3)} s\n"
        log_str += f"  {'Throughput (tokens/s)':<28}: {get_metric('perf/throughput', 2)}\n"
        log_str += f"  {'Total Tokens in Step':<28}: {get_metric('perf/total_num_tokens', 0)}\n"

        # --- Algorithm-Specific Metrics ---
        log_str += "\n--- 📈 Algorithm Metrics ---\n"
        log_str += f"  {'Actor Entropy':<28}: {get_metric('actor/entropy_loss', 4)}\n"
        log_str += (
            f"  {'Critic Rewards (Mean/Min/Max)':<28}: {get_metric('critic/rewards/mean', 3)} / "
            f"{get_metric('critic/rewards/min', 3)} / {get_metric('critic/rewards/max', 3)}\n"
        )
        log_str += (
            f"  {'Critic Scores (Mean/Min/Max)':<28}: {get_metric('critic/score/mean', 3)} / "
            f"{get_metric('critic/score/min', 3)} / {get_metric('critic/score/max', 3)}\n"
        )

        if self.enable_perf:
            # --- Module-wise Timings (Single Column) ---
            log_str += "\n--- ⏳ Module-wise Timings (s) ---\n"
            # Dynamically find all delta_time metrics except the total step time
            timing_keys = sorted(
                [k for k in metrics.keys() if k.startswith("perf/delta_time/") and k != "perf/delta_time/step"]
            )

            ref_key = "perf/delta_time/ref"
            reference_key = "perf/delta_time/reference"
            if ref_key in timing_keys and reference_key in timing_keys:
                timing_keys.remove(reference_key)

            if timing_keys:
                # Find the maximum label length across all keys for clean alignment
                max_label_len = 0
                if timing_keys:
                    max_label_len = max(
                        len(k.replace("perf/delta_time/", "").replace("_", " ").title()) for k in timing_keys
                    )

                for key in timing_keys:
                    label = key.replace("perf/delta_time/", "").replace("_", " ").title()
                    value = get_metric(key, 3)
                    log_str += f"  {label:<{max_label_len}} : {value}s\n"
            else:
                log_str += "  No detailed timing metrics available.\n"

        # --- Model Flops Utilization (MFU) ---
        log_str += "\n--- 🔥 Model Flops Utilization (MFU) ---\n"
        log_str += f"  {'Mean MFU':<28}: {get_metric('perf/mfu/mean', 3)}\n"
        log_str += f"  {'Actor Training MFU':<28}: {get_metric('perf/mfu/actor', 3)}\n"
        # log_str += f"  {'Rollout MFU':<28}: {get_metric('perf/mfu/rollout', 3)}\n"
        log_str += f"  {'Reference Policy MFU':<28}: {get_metric('perf/mfu/ref', 3)}\n"
        log_str += f"  {'Actor LogProb MFU':<28}: {get_metric('perf/mfu/actor_log_prob', 3)}\n"

        # --- Memory Usage ---
        log_str += "\n--- 💾 Memory Usage ---\n"
        log_str += f"  {'Max GPU Memory Allocated':<28}: {get_metric('perf/max_memory_allocated_gb', 2)} GB\n"
        log_str += f"  {'Max GPU Memory Reserved':<28}: {get_metric('perf/max_memory_reserved_gb', 2)} GB\n"
        log_str += f"  {'CPU Memory Used':<28}: {get_metric('perf/cpu_memory_used_gb', 2)} GB\n"

        # --- Sequence Lengths ---
        log_str += "\n--- 📏 Sequence Lengths ---\n"
        log_str += (
            f"  {'Prompt Length (Mean/Max)':<28}: {get_metric('prompt/length/mean', 1)} / "
            f"{get_metric('prompt/length/max', 0)}\n"
        )
        log_str += (
            f"  {'Response Length (Mean/Max)':<28}: {get_metric('response/length/mean', 1)} / "
            f"{get_metric('response/length/max', 0)}\n"
        )
        log_str += f"  {'Response Clip Ratio':<28}: {get_metric('response/clip_ratio/mean', 4)}\n"
        log_str += f"  {'Prompt Clip Ratio':<28}: {get_metric('prompt/clip_ratio/mean', 4)}\n"
        log_str += (
            f"  {'Correct Resp Len (Mean/Max)':<28}: {get_metric('response/correct_length/mean', 1)} / "
            f"{get_metric('response/correct_length/max', 0)}\n"
        )
        log_str += (
            f"  {'Wrong Resp Len (Mean/Max)':<28}: {get_metric('response/wrong_length/mean', 1)} / "
            f"{get_metric('response/wrong_length/max', 0)}\n"
        )

        log_str += "\n" + "=" * 82 + "\n"

        logger.info(log_str)

    def _whether_put_data(self, is_current_last_pp_tp_rank0, next_dp_size, cur_dp_size, cur_node, next_node) -> bool:
        # Determine whether to put data into buffer based on node configuration
        result = False
        reason = "No condition met"
        
        if is_current_last_pp_tp_rank0:
            result = True
            reason = "Current last PP rank's TP rank 0"
        elif next_dp_size == cur_dp_size:
            if next_node.node_type in [NodeType.COMPUTE, NodeType.MODEL_TRAIN]:
                result = True
                reason = f"DP sizes match and next node is {next_node.node_type}"
        elif cur_node.node_role == next_node.node_role and cur_node.node_role == NodeRole.ROLLOUT:
            result = True
            reason = "Both nodes are ROLLOUT"
            
        logger.debug(f"Rank {self._rank}: _whether_put_data decision for {cur_node.node_id}->{next_node.node_id}: {result} ({reason}). "
                    f"is_current_last_pp_tp_rank0={is_current_last_pp_tp_rank0}, next_dp_size={next_dp_size}, cur_dp_size={cur_dp_size}, "
                    f"cur_node_type={cur_node.node_type}, next_node_type={next_node.node_type}, "
                    f"cur_node_role={cur_node.node_role}, next_node_role={next_node.node_role}")
        return result


    def check_spmd_mode(self):
        return self.rollout_mode == 'sync' and self._multi_agent == False
    

    def multi_agent_put_log(self, key: str, data: DataProto, agent_group: int, next_dp_size: int, timing_raw):
        def uuid_hex_to_bucket(uuid_hex: str, num_buckets: int = 8) -> int:
            return consistent_hash(uuid_hex) % num_buckets
        data_size = len(data)
        put_futures = []
        meta_info = data.meta_info
        with self._timer(f"put_ray_proto_data_{key}", timing_raw):
            for i in range(data_size):
                cur_data = data[i]
                request_id = cur_data.non_tensor_batch[f"agent_group_{agent_group}_request_id"]
                next_dp_rank = uuid_hex_to_bucket(request_id, next_dp_size)
                cur_key = key + f"_{next_dp_rank}"
                buf = random.choice(self.data_buffers)
                # slice of DataProto is DataProtoItem, will loss meta_info, need to recompute when use
                cur_data = collate_fn([cur_data])
                cur_data.meta_info = meta_info
                put_futures.append(buf.put.remote(cur_key, cur_data))
        with self._timer(f"put_proto_data_{key}", timing_raw):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncio.gather(*put_futures))

        
    
    def multi_agent_get_log(self, key: str, cur_dp_rank: int, agent_group: int, timing_raw):
        loop = asyncio.get_event_loop()
        key = key + f"_{cur_dp_rank}"
        prefix_key = f"agent_group_{agent_group}_"
        with self._timer(f"get_ref_data_{key}", timing_raw):
            tasks = [buf.get.remote(key) for buf in self.data_buffers]
            temp_data =loop.run_until_complete(asyncio.gather(*tasks))
            # temp_data = self.data_buffers.get(key) 
            datas = [item for t in temp_data if t is not None for item in t]
            sorted_datas = sorted(datas, key = lambda x : (x.non_tensor_batch[prefix_key + 'request_id'], -x.non_tensor_batch[prefix_key + 'traj_step']))
            meta_info = sorted_datas[0].meta_info
            dataproto = collate_fn(sorted_datas)
            dataproto.meta_info = meta_info
        return dataproto
