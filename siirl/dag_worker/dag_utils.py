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
Utility functions for DAG worker operations.
"""

import os
import ray
import torch
import inspect
import json
import time
import csv
import hashlib
import numpy as np
import torch.distributed as dist
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from collections import deque
from tensordict import TensorDict
from typing import Dict, Optional, Type, List, Any, Tuple, Union
from loguru import logger
from tensordict import TensorDict

from siirl.data_coordinator import DataProto
from siirl.execution.dag.node import Node, NodeType, NodeRole
from siirl.execution.dag import TaskGraph
from siirl.utils.extras.device import get_device_name, device_synchronize
from siirl.engine.base_worker import Worker
from siirl.utils.import_string import import_string
from siirl.dag_worker.constants import DAGConstants
from siirl.dag_worker.data_structures import ValidationResult
from siirl.dag_worker.metric_aggregator import (
    DistributedMetricAggregator,
    _ReduceOp
)


# ==========================================================================================
# Section 1: Performance & Timing
# ==========================================================================================

@contextmanager
def timer(enable_perf: bool, name: str, timing_dict: dict):
    """Measures execution time of a code block and stores in timing_dict."""
    if enable_perf:
        device_synchronize()
    start_time = time.perf_counter()
    yield
    if enable_perf:
        device_synchronize()
    end_time = time.perf_counter()
    timing_dict[name] = timing_dict.get(name, 0) + end_time - start_time


def add_prefix_to_dataproto(tensordict: TensorDict, node: Node):
    """
    Adds a prefix to all keys in the TensorDict.
    The prefix is formatted as f"agent_group_{node.agent_group}_".
    Only keys that do not already have a prefix will be modified.

    Args:
        data_proto (TensorDict): The TensorDict instance.
        node (Node): The node containing the agent_group.
    """
    prefix = f"agent_group_{node.agent_group}_"
    prefix_agent_group = "agent_group_"

    # Process tensor batch
    if tensordict is not None:
        new_batch = {}
        for key, value in tensordict.items():
            if not key.startswith(prefix_agent_group):
                new_key = prefix + key
                new_batch[new_key] = value
            else:
                new_batch[key] = value
        tensordict = TensorDict(new_batch, batch_size=tensordict.batch_size)
    return tensordict


def remove_prefix_from_dataproto(tensordict, node: Node):
    """
    Removes the prefix from all keys in the TensorDict.
    Only keys with a matching prefix will have the prefix removed.

    Args:
        data_proto (TensorDict): The TensorDict instance.
        node (Node): The node containing the agent_group to identify the prefix.
    """
    prefix = f"agent_group_{node.agent_group}_"
    prefix_len = len(prefix)

    # Process tensor batch
    if tensordict is not None:
        new_batch = {}
        for key, value in tensordict.items():
            if key.startswith(prefix):
                new_key = key[prefix_len:]
                new_batch[new_key] = value
            else:
                new_batch[key] = value
        tensordict = TensorDict(new_batch, batch_size=tensordict.batch_size)

    return tensordict


def add_prefix_to_metrics(metrics: dict, node: Node) -> dict:
    """Adds agent prefix to all metric keys for multi-agent isolation."""
    prefix = f"agent_{node.agent_group}_"
    prefix_agent_group = "agent_"
    if metrics:
        new_metrics = {}
        for key, value in metrics.items():
            if not key.startswith(prefix_agent_group):
                new_key = prefix + key
                new_metrics[new_key] = value
            else:
                new_metrics[key] = value
        metrics = new_metrics
    return metrics


# ==========================================================================================
# Section 3: Initialization & Setup
# ==========================================================================================

def get_and_validate_rank() -> int:
    """Retrieves and validates worker rank from RANK environment variable."""
    rank_str = os.environ.get("RANK")
    if rank_str is None:
        raise ValueError("Environment variable 'RANK' is not set. This is required for distributed setup.")
    try:
        return int(rank_str)
    except ValueError as e:
        raise ValueError(f"Invalid RANK format: '{rank_str}'. Must be an integer.") from e


def get_taskgraph_for_rank(rank: int, taskgraph_mapping: Dict[int, TaskGraph]) -> TaskGraph:
    """Retrieves TaskGraph for current rank from mapping."""
    if rank not in taskgraph_mapping:
        raise ValueError(f"Rank {rank} not found in the provided taskgraph_mapping.")
    taskgraph = taskgraph_mapping[rank]

    if not isinstance(taskgraph, TaskGraph):
        raise TypeError(f"Object for rank {rank} must be a TaskGraph, but got {type(taskgraph).__name__}.")
    logger.info(f"Rank {rank} assigned to TaskGraph with ID {taskgraph.graph_id}.")
    return taskgraph


def log_ray_actor_info(rank: int):
    """Logs Ray actor context information for debugging."""
    try:
        ctx = ray.get_runtime_context()
        logger.debug(
            f"Ray Actor Context for Rank {rank}: ActorID={ctx.get_actor_id()}, JobID={ctx.get_job_id()}, "
            f"NodeID={ctx.get_node_id()}, PID={os.getpid()}"
        )
    except RuntimeError:
        logger.warning(f"Rank {rank}: Not running in a Ray actor context.")


def log_role_worker_mapping(role_worker_mapping: Dict[NodeRole, Type[Worker]]):
    """Logs role-to-worker class mapping for verification."""
    if not role_worker_mapping:
        logger.error("Role-to-worker mapping is empty after setup. This will cause execution failure.")
        return

    logger.debug("--- [Role -> Worker Class] Mapping ---")
    max_len = max((len(r.name) for r in role_worker_mapping.keys()), default=0)
    for role, worker_cls in sorted(role_worker_mapping.items(), key=lambda item: item[0].name):
        logger.debug(
            f"  {role.name:<{max_len}} => {worker_cls.__name__} (from {inspect.getmodule(worker_cls).__name__})"
        )
    logger.debug("--------------------------------------")


# ==========================================================================================
# Section 4: Worker Management
# ==========================================================================================

def find_first_non_compute_ancestor(taskgraph: TaskGraph, start_node_id: str) -> Optional[Node]:
    """Finds first ancestor node that is not COMPUTE type using BFS."""
    start_node = taskgraph.get_node(start_node_id)
    if not start_node:
        logger.warning(f"Could not find start node '{start_node_id}' in the graph.")
        return None

    if start_node.node_type != NodeType.COMPUTE:
        return start_node

    queue = deque(start_node.dependencies)
    visited = set(start_node.dependencies)
    node_id = start_node_id

    while queue:
        logger.debug(f"try find dependency node with ID '{node_id}' during upward search")
        node_id = queue.popleft()
        node = taskgraph.get_node(node_id)

        if not node:
            logger.warning(f"Could not find dependency node with ID '{node_id}' during upward search.")
            continue

        if node.node_type != NodeType.COMPUTE:
            return node

        for dep_id in node.dependencies:
            if dep_id not in visited:
                visited.add(dep_id)
                queue.append(dep_id)
    return None


def should_create_worker(role_worker_mapping: Dict[NodeRole, Type[Worker]], node: Node) -> bool:
    """Determines if worker instance should be created for a given node."""
    if node.agent_options and node.agent_options.share_instance:
        # Worker already initialized in target agent node
        return False
    return node.node_type in [NodeType.MODEL_TRAIN, NodeType.MODEL_INFERENCE] and node.node_role in role_worker_mapping


def generate_node_worker_key(node: Node) -> str:
    """Generates unique key for node's worker instance."""
    return f"{node.agent_group}_{node.node_type.value}_{node.node_role.value}"


def setup_sharding_manager(
    config,
    agent_group_process_group: Dict,
    agent_group: int,
    worker_dict: Dict[NodeRole, Worker]
):
    """Configures sharding manager to sync weights between training and inference backends."""
    actor_worker = worker_dict[NodeRole.ACTOR]
    rollout_worker = worker_dict[NodeRole.ROLLOUT]
    rollout_pg = agent_group_process_group[agent_group][NodeRole.ROLLOUT]
    
    if config.actor_rollout_ref.model.model_type == "embodied":
        if hasattr(actor_worker, "actor_module_fsdp"):
            rollout_worker.rollout.model = actor_worker.actor_module_fsdp
            logger.info(f"[Embodied] Set module for EmbodiedHFRollout for agent group {agent_group}.")
        else:
            logger.error(f"[Embodied] Actor worker for agent group {agent_group} does not have 'actor_module_fsdp'.")

    rollout_pg = agent_group_process_group[agent_group][NodeRole.ROLLOUT]
    
    parallel_config = {
        "rollout_parallel_size": rollout_worker.config.rollout.tensor_model_parallel_size,
        "rollout_world_size": dist.get_world_size(rollout_pg),
        "rollout_rank": dist.get_rank(rollout_pg),
    }

    device_name = get_device_name()
    layer_name_mapping = {
        "qkv_layer_name": "self_attention.linear_qkv.",
        "gate_proj_layer_name": "linear_fc1.weight",
    }

    # Lazy import and deferred execution mapping
    sharding_manager_map = {
        ("fsdp", "hf"): (
                "siirl.engine.sharding_manager.fsdp_hf.FSDPHFShardingManager",
                lambda: {
                    "module": actor_worker.actor_module_fsdp,
                    "rollout": rollout_worker.rollout,
                    "offload_param": getattr(actor_worker, "_is_offload_param", False),
                    "offload_embedding": (
                        getattr(rollout_worker.config, "embodied", None) is not None and
                        getattr(rollout_worker.config.embodied, "embedding_model_offload", False)),
            },
        ),
        ("fsdp", "vllm"): (
            "siirl.engine.sharding_manager.fsdp_vllm.MultiAgentFSDPVLLMShardingManager",
            lambda: {
                "module": actor_worker.actor_module_fsdp,
                "inference_engine": rollout_worker.rollout.inference_engine,
                "model_config": actor_worker.actor_model_config,
                "parallel_config": parallel_config,
                "full_params": "hf" in rollout_worker.config.rollout.load_format,
                "offload_param": getattr(actor_worker, "_is_offload_param", False),
            },
        ),
        ("fsdp", "sglang"): (
            "siirl.engine.sharding_manager.fsdp_sglang.MultiAgentFSDPSGLangShardingManager",
            lambda: {
                "module": actor_worker.actor_module_fsdp,
                "inference_engine": rollout_worker.rollout.inference_engine,
                "model_config": actor_worker.actor_model_config,
                "device_mesh": torch.distributed.init_device_mesh(
                    device_name,
                    mesh_shape=(
                        parallel_config.get("rollout_world_size") // parallel_config.get("rollout_parallel_size"),
                        parallel_config.get("rollout_parallel_size"),
                    ),
                    mesh_dim_names=["dp", "infer_tp"],
                ),
                "rollout_config": rollout_worker.config.rollout,
                "full_params": "hf" in rollout_worker.config.rollout.load_format,
                "offload_param": getattr(actor_worker, "_is_offload_param", False),
                "multi_stage_wake_up": rollout_worker.config.rollout.multi_stage_wake_up,
            },
        ),
        ("megatron", "vllm"): (
            "siirl.engine.sharding_manager.megatron_vllm.MultiAgentMegatronVLLMShardingManager",
            lambda: {
                "actor_module": actor_worker.actor_module,
                "inference_engine": rollout_worker.rollout.inference_engine,
                "model_config": actor_worker.actor_model_config,
                "rollout_config": rollout_worker.config.rollout,
                "transformer_config": actor_worker.tf_config,
                "layer_name_mapping": layer_name_mapping,
                "weight_converter": get_mcore_weight_converter(actor_worker.actor_model_config, actor_worker.dtype),
                "device_mesh": rollout_worker.device_mesh,
                "offload_param": actor_worker._is_offload_param,
                "bridge": actor_worker.bridge,
            },
        ),
        ("megatron", "sglang"): (
            "siirl.engine.sharding_manager.megatron_sglang.MultiAgentMegatronSGLangShardingManager",
            lambda: {
                "actor_module": actor_worker.actor_module,
                "inference_engine": rollout_worker.rollout.inference_engine,
                "model_config": actor_worker.actor_model_config,
                "rollout_config": rollout_worker.config.rollout,
                "transformer_config": actor_worker.tf_config,
                "layer_name_mapping": layer_name_mapping,
                "weight_converter": get_mcore_weight_converter(actor_worker.actor_model_config, actor_worker.dtype),
                "device_mesh": torch.distributed.init_device_mesh(
                    device_name,
                    mesh_shape=(
                        parallel_config.get("rollout_world_size") // parallel_config.get("rollout_parallel_size"),
                        parallel_config.get("rollout_parallel_size"),
                    ),
                    mesh_dim_names=["dp", "infer_tp"],
                ),
                "offload_param": getattr(actor_worker, "_is_offload_param", False),
                "bridge": actor_worker.bridge,
            },
        ),
    }

    strategy = actor_worker.config.actor.strategy.lower()
    if strategy == DAGConstants.MEGATRON_STRATEGY:
        from siirl.models.mcore import get_mcore_weight_converter
    rollout_name = config.actor_rollout_ref.rollout.name.lower()
    if (strategy, rollout_name) not in sharding_manager_map:
        raise NotImplementedError(f"Unsupported sharding manager configuration: {strategy=}, {rollout_name=}")

    sharding_manager_cls_str, kwargs_builder = sharding_manager_map[(strategy, rollout_name)]
    sharding_manager_cls = import_string(sharding_manager_cls_str)
    sharding_manager = sharding_manager_cls(**kwargs_builder())
    rollout_worker.set_rollout_sharding_manager(sharding_manager)
    logger.debug(f"Set up {sharding_manager_cls.__name__}  for agent group {agent_group}.")


def get_worker_classes(config, strategy: str) -> Dict[NodeRole, Type[Worker]]:
    """Dynamically imports worker classes based on specified training strategy."""
    if strategy in DAGConstants.FSDP_STRATEGIES:
        from siirl.engine.fsdp_workers import (
            ActorRolloutRefWorker,
            AsyncActorRolloutRefWorker,
            CriticWorker,
            RewardModelWorker,
        )

        actor_cls = (
            AsyncActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.mode == "async"
            else ActorRolloutRefWorker
        )
        return {
            NodeRole.ACTOR: actor_cls,
            NodeRole.ROLLOUT: actor_cls,
            NodeRole.REFERENCE: actor_cls,
            NodeRole.CRITIC: CriticWorker,
            NodeRole.REWARD: RewardModelWorker,
        }
    elif strategy in DAGConstants.MEGATRON_STRATEGYS:
        from siirl.engine.megatron_workers import (
            ActorWorker,
            RolloutWorker,
            AsyncRolloutWorker,
            ReferenceWorker,
            CriticWorker,
            RewardModelWorker
        )

        is_async_mode = config.actor_rollout_ref.rollout.mode == "async"

        return {
            NodeRole.ACTOR: ActorWorker,
            NodeRole.ROLLOUT: AsyncRolloutWorker if is_async_mode else RolloutWorker,
            NodeRole.REFERENCE: ReferenceWorker,
            NodeRole.CRITIC: CriticWorker,
            NodeRole.REWARD: RewardModelWorker
        }
    raise NotImplementedError(f"Strategy '{strategy}' is not supported.")


def get_parallelism_config(reference_node: Node) -> tuple[int, int]:
    """Extracts tensor parallel (TP) and pipeline parallel (PP) sizes from node config."""
    tp_size = 1
    pp_size = 1

    if intern_config := reference_node.config.get(DAGConstants.INTERN_CONFIG):
        if reference_node.node_type == NodeType.MODEL_INFERENCE:
            # Rollout nodes: only TP supported (PP not typically used for inference)
            tp_size = intern_config.rollout.tensor_model_parallel_size
            pp_size = 1

        elif reference_node.node_type == NodeType.MODEL_TRAIN:
            # Extract strategy from config
            strategy = 'fsdp'  # default

            if hasattr(intern_config, 'actor') and hasattr(intern_config.actor, 'strategy'):
                strategy = intern_config.actor.strategy
            elif hasattr(intern_config, 'strategy'):
                strategy = intern_config.strategy

            if strategy in DAGConstants.MEGATRON_STRATEGYS:
                # Megatron supports both TP and PP
                if hasattr(intern_config, 'actor') and hasattr(intern_config.actor, 'megatron'):
                    tp_size = intern_config.actor.megatron.tensor_model_parallel_size
                    pp_size = intern_config.actor.megatron.pipeline_model_parallel_size
                elif hasattr(intern_config, 'megatron'):
                    tp_size = intern_config.megatron.tensor_model_parallel_size
                    pp_size = intern_config.megatron.pipeline_model_parallel_size
            else:
                # FSDP: no TP/PP, keep TP=PP=1
                tp_size = 1
                pp_size = 1

    return tp_size, pp_size


def prepare_generation_batch(batch: TensorDict) -> TensorDict:
    """Pops keys from a batch to isolate data needed for sequence generation."""
    keys_to_pop = ["input_ids", "attention_mask", "position_ids", "raw_prompt_ids"]
    if "multi_modal_inputs" in batch:
        keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
    if "tools_kwargs" in batch:
        keys_to_pop.append("tools_kwargs")
    if "raw_prompt" in batch:
        keys_to_pop.append("raw_prompt")
    if "interaction_kwargs" in batch:
        keys_to_pop.append("interaction_kwargs")
    return batch.pop(
    )


def prepare_local_batch_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, torch.Tensor]:
    """Extracts raw metric tensors from batch for distributed aggregation."""
    from siirl.utils.metrics.metric_utils import _compute_response_info

    response_info = _compute_response_info(batch)
    response_mask = response_info["response_mask"].bool()
    device = batch["advantages"].device
    max_response_length = batch["responses"].shape[-1]
    response_lengths = response_info["response_length"].to(device)
    prompt_lengths = response_info["prompt_length"].to(device)

    # Components for correct/wrong response length metrics
    correct_threshold = 0.5
    rewards_per_response = batch["token_level_rewards"].sum(-1)
    correct_mask = rewards_per_response > correct_threshold

    # Components for prompt clip ratio
    prompt_attn_mask = batch["attention_mask"][:, :-max_response_length]
    max_prompt_length = prompt_attn_mask.size(-1)

    # Prepare raw metric values
    local_data = {
        "score": batch["token_level_scores"].sum(-1),
        "rewards": batch["token_level_rewards"].sum(-1),
        "advantages": torch.masked_select(batch["advantages"], response_mask),
        "returns": torch.masked_select(batch["returns"], response_mask),
        "response_length": response_info["response_length"].to(device),
        "prompt_length": response_info["prompt_length"].to(device),
        "correct_response_length": response_lengths[correct_mask],
        "wrong_response_length": response_lengths[~correct_mask],
        "response_clip_ratio": torch.eq(response_info["response_length"], max_response_length).float(),
        "prompt_clip_ratio": torch.eq(prompt_lengths, max_prompt_length).float(),
    }

    if use_critic:
        valid_values = torch.masked_select(batch["values"], response_mask)
        error = local_data["returns"] - valid_values

        critic_data = {
            "values": valid_values,
            # Special components for explained variance (summed globally)
            "returns_sq_sum_comp": torch.sum(torch.square(local_data["returns"])),
            "error_sum_comp": torch.sum(error),
            "error_sq_sum_comp": torch.sum(torch.square(error)),
        }
        local_data.update(critic_data)

    return local_data


def whether_put_data(rank, is_current_last_pp_tp_rank0, next_dp_size, cur_dp_size, cur_node, next_node) -> bool:
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
        
    logger.debug(f"Rank {rank}: _whether_put_data decision for {cur_node.node_id}->{next_node.node_id}: {result} ({reason}). "
                f"is_current_last_pp_tp_rank0={is_current_last_pp_tp_rank0}, next_dp_size={next_dp_size}, cur_dp_size={cur_dp_size}, "
                f"cur_node_type={cur_node.node_type}, next_node_type={next_node.node_type}, "
                f"cur_node_role={cur_node.node_role}, next_node_role={next_node.node_role}")
    return result


# ==========================================================================================
# Section 6: Metrics Collection & Aggregation
# ==========================================================================================

def reduce_and_broadcast_metrics(
    local_metrics: Dict[str, Union[float, List[float], torch.Tensor]],
    group: dist.ProcessGroup
) -> Dict[str, float]:
    """Aggregates metrics across all ranks using all_reduce operations."""
    if not isinstance(local_metrics, dict) or not local_metrics:
        return {}

    world_size = dist.get_world_size(group)
    if world_size <= 1:
        # Non-distributed case: perform local aggregation only
        aggregator = DistributedMetricAggregator(local_metrics, group=None)
        final_metrics = {}
        for op_type, data in aggregator.op_buckets.items():
            for key, value in data:
                if op_type == _ReduceOp.SUM:  # value is a (sum, count) tuple
                    final_metrics[key] = value[0] / value[1] if value[1] > 0 else 0.0
                else:  # value is a float
                    final_metrics[key] = float(value)
        return final_metrics

    # Pipeline Parallel: ensure all ranks have same metric keys
    # 1. Gather all metric keys from all ranks
    local_keys = set(local_metrics.keys())
    all_keys_list = [None] * world_size
    dist.all_gather_object(all_keys_list, local_keys, group=group)

    # 2. Union all keys to get complete set
    all_expected_keys = set()
    for keys_set in all_keys_list:
        all_expected_keys.update(keys_set)

    # 3. Aggregate with unified keys
    aggregator = DistributedMetricAggregator(local_metrics, group)
    aggregator.op_buckets = aggregator._bucket_local_metrics(local_metrics, all_expected_keys)
    return aggregator.aggregate_and_get_results()


def format_metrics_by_group(metrics: Dict[str, Any], group_order: List[str]) -> Dict[str, Any]:
    """Reorders metrics by group prefixes and alphabetically within groups."""
    if not metrics:
        return {}

    ordered_dict = {}
    processed_keys = set()

    # Pre-identify explicitly mentioned full keys
    explicitly_mentioned_keys = {key for key in group_order if key in metrics}

    # Process metrics according to group/key order
    for pattern in group_order:
        # Check if pattern is a full key
        if pattern in explicitly_mentioned_keys and pattern not in processed_keys:
            ordered_dict[pattern] = metrics[pattern]
            processed_keys.add(pattern)
        else:
            # Treat as group prefix
            group_prefix = f"{pattern}/"

            # Find all keys in this group and sort alphabetically
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

    # Process remaining keys
    remaining_keys = sorted([key for key in metrics if key not in processed_keys])
    if remaining_keys:
        for key in remaining_keys:
            ordered_dict[key] = metrics[key]

    return ordered_dict


# ==========================================================================================
# Section 7: Logging & Output
# ==========================================================================================

def log_metrics_to_console(rank: int, ordered_metrics: List[Tuple[str, Any]], step: int):
    """Logs formatted metrics string to console (rank 0 only)."""
    if rank != 0:
        return
    log_parts = [f"step:{step}"]
    log_parts.extend([f"{k}:{v:.4f}" if isinstance(v, float) else f"{k}:{v}" for k, v in ordered_metrics])
    logger.info(" | ".join(log_parts))


def dump_validation_generations(
    config,
    global_steps: int,
    rank: int,
    results: List[ValidationResult]
):
    """Dumps validation generation results to rank-specific JSON file."""
    dump_path_str = config.trainer.rollout_data_dir
    if not dump_path_str:
        return
    dump_path = Path(dump_path_str)

    try:
        dump_path.mkdir(parents=True, exist_ok=True)

        filename = dump_path / f"step_{global_steps}_rank_{rank}.json"

        # Collect entries
        entries = []
        for res in results:
            entry = {
                "rank": rank,
                "global_step": global_steps,
                "data_source": res.data_source,
                "input": res.input_text,
                "output": res.output_text,
                "score": res.score,
            }
            if res.extra_rewards:
                entry.update(res.extra_rewards)
            entries.append(entry)

        # Write with pretty formatting
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=4)

        if rank == 0:
            logger.info(f"Validation generations are being dumped by all ranks to: {dump_path.resolve()}")
        logger.debug(f"Rank {rank}: Dumped {len(results)} validation generations to {filename}")

    except (OSError, IOError) as e:
        logger.error(f"Rank {rank}: Failed to write validation dump file to {dump_path}: {e}")
    except Exception as e:
        logger.error(f"Rank {rank}: An unexpected error occurred during validation dumping: {e}", exc_info=True)


def aggregate_and_write_performance_metrics(
    gather_group,
    rank,
    global_steps,
    config,
    metrics: Dict[str, Any]):
    """
    Gathers performance metrics from all ranks to rank 0 and writes them to a CSV file.
    Each row corresponds to a metric key COMMON to all ranks, and each column to a rank.
    This function is called only if performance profiling is enabled.
    """
    # Gather all metrics dictionaries to rank 0
    world_size = dist.get_world_size()
    gathered_metrics = [None] * world_size if rank == 0 else None
    dist.gather_object(metrics, gathered_metrics, dst=0, group=gather_group)

    if rank == 0:
        if not gathered_metrics:
            logger.warning("No metrics gathered on rank 0. Skipping performance CSV write.")
            return

        valid_metrics = [m for m in gathered_metrics if isinstance(m, dict) and m]
        if not valid_metrics:
            logger.warning("No valid metric dictionaries received on rank 0. Skipping CSV write.")
            return

        common_keys = set(valid_metrics[0].keys())
        for rank_metrics in valid_metrics[1:]:
            common_keys.intersection_update(rank_metrics.keys())

        sorted_keys = sorted(list(common_keys))

        if not sorted_keys:
            logger.warning(
                f"No common metric keys found across all ranks for step {global_steps}. Skipping CSV write."
            )
            return

        ts = get_time_now().strftime("%Y-%m-%d-%H-%M-%S")
        try:
            # Try to get model name from model path config
            model_name = os.path.basename(os.path.normpath(config.actor_rollout_ref.model.path))
            output_dir = os.path.join("performance_logs", model_name, ts)
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create performance log directory {output_dir}: {e}")
            return

        filename = os.path.join(output_dir, f"world_{world_size}_step_{global_steps}_common_metrics.csv")

        try:
            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                header = (
                    ["metric"]
                    + [f"rank_{i}" for i in range(world_size)]
                    + ["max", "min", "delta_max_min", "delta_max_rank_0"]
                )
                writer.writerow(header)

                for key in sorted_keys:
                    row = [key]
                    for i in range(world_size):
                        rank_metrics = gathered_metrics[i]

                        if isinstance(rank_metrics, dict):
                            value = rank_metrics.get(key, "Error: Key Missing")
                        else:
                            value = "N/A: Invalid Data"
                        row.append(value)

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
                f"Common performance metrics for step {global_steps} successfully written to {filename}"
            )

        except OSError as e:
            logger.error(f"Failed to write performance metrics to CSV file {filename}: {e}")


def log_core_performance_metrics(rank: int, enable_perf: bool, metrics: Dict[str, Any], step: int):
    """
    Logs a formatted, easy-to-read summary of core performance metrics on rank 0.
    This provides a clear, separate view of the most important indicators.
    """
    if rank != 0:
        return

    def get_metric(key, precision=3):
        val = metrics.get(key)
        if val is None:
            return "N/A"
        if isinstance(val, (float, np.floating)):
            return f"{val:.{precision}f}"
        return val

    # --- Build the log string ---
    log_str = f"\n\n{'=' * 25} RANK({rank}): Core Performance Metrics (Step: {step}) {'=' * 25}\n"

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

    if enable_perf:
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


# ==========================================================================================
# Section 8: General Utilities
# ==========================================================================================

@staticmethod
def get_time_now(time_zone: str = "Asia/Shanghai") -> datetime:
    """Returns current time in specified timezone."""
    return datetime.now(tz=ZoneInfo(time_zone))


def consistent_hash(s: str) -> int:
    """Returns consistent hash of string using MD5."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)
