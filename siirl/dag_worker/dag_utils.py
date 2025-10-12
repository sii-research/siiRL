import os
import ray
import torch
import inspect
from collections import deque
from tensordict import TensorDict
from typing import Dict, Optional, Type
from loguru import logger
import torch.distributed as dist


from siirl.data_coordinator import DataProto
from siirl.execution.dag.node import Node, NodeType, NodeRole
from siirl.execution.dag import TaskGraph
from siirl.utils.extras.device import get_device_name
from siirl.engine.base_worker import Worker
from siirl.utils.import_string import import_string
from siirl.dag_worker.constants import DAGConstants



def add_prefix_to_dataproto(data_proto: DataProto, node: Node):
    """
    Adds a prefix to all keys in the DataProto.
    The prefix is formatted as f"agent_group_{node.agent_group}_".
    Only keys that do not already have a prefix will be modified.

    Args:
        data_proto (DataProto): The DataProto instance.
        node (Node): The node containing the agent_group.
    """
    prefix = f"agent_group_{node.agent_group}_"
    prefix_agent_group = "agent_group_"

    # Process tensor batch
    if data_proto.batch is not None:
        new_batch = {}
        for key, value in data_proto.batch.items():
            if not key.startswith(prefix_agent_group):
                new_key = prefix + key
                new_batch[new_key] = value
            else:
                new_batch[key] = value
        data_proto.batch = TensorDict(new_batch, batch_size=data_proto.batch.batch_size)

    # Process non_tensor_batch
    if data_proto.non_tensor_batch is not None:
        new_non_tensor = {}
        for key, value in data_proto.non_tensor_batch.items():
            if not key.startswith(prefix_agent_group):
                new_key = prefix + key
                new_non_tensor[new_key] = value
            else:
                new_non_tensor[key] = value
        data_proto.non_tensor_batch = new_non_tensor

    # Process meta_info
    if data_proto.meta_info is not None:
        new_meta = {}
        for key, value in data_proto.meta_info.items():
            if not key.startswith(prefix_agent_group):
                new_key = prefix + key
                new_meta[new_key] = value
            else:
                new_meta[key] = value
        data_proto.meta_info = new_meta
    return data_proto


def remove_prefix_from_dataproto(data_proto, node: Node):
    """
    Removes the prefix from all keys in the DataProto.
    Only keys with a matching prefix will have the prefix removed.

    Args:
        data_proto (DataProto): The DataProto instance.
        node (Node): The node containing the agent_group to identify the prefix.
    """
    prefix = f"agent_group_{node.agent_group}_"
    prefix_len = len(prefix)

    # Process tensor batch
    if data_proto.batch is not None:
        new_batch = {}
        for key, value in data_proto.batch.items():
            if key.startswith(prefix):
                new_key = key[prefix_len:]
                new_batch[new_key] = value
            else:
                new_batch[key] = value
        data_proto.batch = TensorDict(new_batch, batch_size=data_proto.batch.batch_size)

    # Process non_tensor_batch
    if data_proto.non_tensor_batch is not None:
        new_non_tensor = {}
        for key, value in data_proto.non_tensor_batch.items():
            if key.startswith(prefix):
                new_key = key[prefix_len:]
                new_non_tensor[new_key] = value
            else:
                new_non_tensor[key] = value
        data_proto.non_tensor_batch = new_non_tensor

    # Process meta_info
    if data_proto.meta_info is not None:
        new_meta = {}
        for key, value in data_proto.meta_info.items():
            if key.startswith(prefix):
                new_key = key[prefix_len:]
                new_meta[new_key] = value
            else:
                new_meta[key] = value
        data_proto.meta_info = new_meta

    return data_proto


def add_prefix_to_metrics(metrics: dict, node: Node):
    """
    Adds a prefix to all keys in the metrics.
    The prefix is formatted as f"agent_group_{node.agent_group}_".
    Only keys that do not already have a prefix will be modified.

    Args:
        metrics (Dict): The metrics instance.
        node (Node): The node containing the agent_group.
    """
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


def get_and_validate_rank() -> int:
    """Retrieves and validates the worker's rank from the environment."""
    rank_str = os.environ.get("RANK")
    if rank_str is None:
        raise ValueError("Environment variable 'RANK' is not set. This is required for distributed setup.")
    try:
        return int(rank_str)
    except ValueError as e:
        raise ValueError(f"Invalid RANK format: '{rank_str}'. Must be an integer.") from e
    

def get_taskgraph_for_rank(rank, taskgraph_mapping: Dict[int, "TaskGraph"]) -> "TaskGraph":
    """Retrieves the TaskGraph for the current rank from the provided mapping."""
    if rank not in taskgraph_mapping:
        raise ValueError(f"Rank {rank} not found in the provided taskgraph_mapping.")
    taskgraph = taskgraph_mapping[rank]

    if not isinstance(taskgraph, TaskGraph):
        raise TypeError(f"Object for rank {rank} must be a TaskGraph, but got {type(taskgraph).__name__}.")
    logger.info(f"Rank {rank} assigned to TaskGraph with ID {taskgraph.graph_id}.")
    return taskgraph


def log_ray_actor_info(rank):
    """Logs detailed information about the Ray actor's context for debugging."""
    try:
        ctx = ray.get_runtime_context()
        logger.debug(
            f"Ray Actor Context for Rank {rank}: ActorID={ctx.get_actor_id()}, JobID={ctx.get_job_id()}, "
            f"NodeID={ctx.get_node_id()}, PID={os.getpid()}"
        )
    except RuntimeError:
        logger.warning(f"Rank {rank}: Not running in a Ray actor context.")

       
def log_role_worker_mapping(role_worker_mapping):
    """Logs the final role-to-worker mapping for setup verification."""
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


def find_first_non_compute_ancestor(taskgraph, start_node_id: str) -> Optional[Node]:
    """
    Traverses upwards from a starting node to find the first ancestor
    that is not of type COMPUTE.

    Uses a Breadth-First Search (BFS) strategy to prioritize finding the
    closest ancestor by level.
    """
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


def should_create_worker(role_worker_mapping, node: Node) -> bool:
    """Determines if a worker instance should be created for a given graph node."""
    if node.agent_options and node.agent_options.share_instance:
        # has been initialized in target agent node
        return False
    return node.node_type in [NodeType.MODEL_TRAIN, NodeType.MODEL_INFERENCE] and node.node_role in role_worker_mapping


def generate_node_worker_key(node: Node) -> str:
    """Generates a unique string key for a node's worker instance."""
    return f"{node.agent_group}_{node.node_type.value}_{node.node_role.value}"


def generate_agent_group_key(node: Node) -> str:
    """Generates a unique key for an agent group, used for caching (e.g., tokenizers)."""
    return f"group_key_{node.agent_group}"


def setup_sharding_manager(config, agent_group_process_group, agent_group: int, worker_dict: Dict[NodeRole, Worker]):
    """Configures the sharding manager to sync weights between training backend and inference backend."""
    actor_worker = worker_dict[NodeRole.ACTOR]
    rollout_worker = worker_dict[NodeRole.ROLLOUT]
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

    # Use lazy import and defer execution.
    sharding_manager_map = {
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
        # TODO(Ping Zhang): update for SGLang later
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
    """Dynamically imports worker classes based on the specified strategy."""
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
        """
        Extract tensor parallel and pipeline parallel sizes based on backend strategy.
        Currently, only FSDP and Megatron backends are supported, in which Megatron supports PP.
        
        Args:
            reference_node: The node to extract parallelism config from
            
        Returns:
            tuple: (tp_size, pp_size)
        """
        tp_size = 1
        pp_size = 1
        
        if intern_config := reference_node.config.get(DAGConstants.INTERN_CONFIG):
            if reference_node.node_type == NodeType.MODEL_INFERENCE:
                # For rollout nodes, only TP is supported currently.
                # Pipeline parallelism is not typically used for inference

                # TODO(Ping Zhang): support PP for rollout nodes, which will be used for very large models
                # that need multi-server inference.
                tp_size = intern_config.rollout.tensor_model_parallel_size
                pp_size = 1

            elif reference_node.node_type == NodeType.MODEL_TRAIN:
                # Extract strategy based on the specific config type
                strategy = 'fsdp'  # default
                
                if hasattr(intern_config, 'actor') and hasattr(intern_config.actor, 'strategy'):
                    # For ActorRolloutRefArguments, strategy is in actor
                    strategy = intern_config.actor.strategy
                elif hasattr(intern_config, 'strategy'):
                    # For CriticArguments, RefArguments, RewardModelArguments, strategy is direct attribute
                    strategy = intern_config.strategy
                
                if strategy in DAGConstants.MEGATRON_STRATEGYS:
                    # Megatron backend supports both TP and PP
                    if hasattr(intern_config, 'actor') and hasattr(intern_config.actor, 'megatron'):
                        # ActorRolloutRefArguments case
                        tp_size = intern_config.actor.megatron.tensor_model_parallel_size
                        pp_size = intern_config.actor.megatron.pipeline_model_parallel_size
                    elif hasattr(intern_config, 'megatron'):
                        # CriticArguments, RefArguments, RewardModelArguments cases
                        tp_size = intern_config.megatron.tensor_model_parallel_size
                        pp_size = intern_config.megatron.pipeline_model_parallel_size
                else:
                    # FSDP's ZeRO-like parallelism is essentially DP; therefore,
                    # For MODEL_TRAIN type, we should keep TP=PP=1.
                    tp_size = 1
                    pp_size = 1

        return tp_size, pp_size
    

def prepare_generation_batch(batch: DataProto) -> DataProto:
    """Pops keys from a batch to isolate data needed for sequence generation."""
    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
    if "multi_modal_inputs" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
    if "tools_kwargs" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("tools_kwargs")
    if "raw_prompt" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("raw_prompt")
    if "interaction_kwargs" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("interaction_kwargs")
    return batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
    )