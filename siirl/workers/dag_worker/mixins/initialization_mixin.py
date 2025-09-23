# Copyright (c) 2025, Shanghai Innovation Institute. All rights reserved.
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

import inspect
import os
from typing import Dict, Type

import ray
import torch
import torch.distributed as dist
from loguru import logger

from siirl.dataloader import DataLoaderNode
from siirl.models.loader import load_tokenizer
from siirl.scheduler.enums import AlgorithmType
from siirl.scheduler.reward import create_reward_manager
from siirl.utils.debug import DistProfiler
from siirl.utils.extras.device import get_device_name, get_nccl_backend
from siirl.workers.base_worker import Worker
from siirl.workers.dag.node import NodeRole, NodeType
from siirl.workers.dag_worker.constants import DAGConstants
from siirl.utils.import_string import import_string

device_name = get_device_name()


class InitializationMixin:
    """Handles the initialization and setup logic for the DAGWorker."""

    from typing import Any, Dict, List, Optional, Type

    from torch.distributed import ProcessGroup

    from siirl.models.loader import TokenizerModule
    from siirl.scheduler.process_group_manager import ProcessGroupManager
    from siirl.utils.logger.tracking import Tracking
    from siirl.utils.params import SiiRLArguments
    from siirl.workers.base_worker import Worker
    from siirl.workers.dag import TaskGraph
    from siirl.workers.dag.node import Node, NodeRole

    # Attributes from DAGWorker's __init__
    config: SiiRLArguments
    process_group_manager: ProcessGroupManager
    taskgraph_mapping: Dict[int, TaskGraph]
    data_buffers: List["ray.actor.ActorHandle"]
    enable_perf: bool
    workers: Dict[str, Worker]
    agent_group_worker: Dict[int, Dict[NodeRole, Worker]]
    agent_group_process_group: Dict[int, Dict[NodeRole, ProcessGroup]]
    process_groups: Dict[str, ProcessGroup]
    tokenizer_mapping: Dict[str, TokenizerModule]
    logger: Optional[Tracking]
    _multi_agent: bool
    # Attributes initialized within this mixin
    _rank: int
    taskgraph: TaskGraph
    _gather_group: Optional[ProcessGroup]
    first_rollout_node: Node
    dataloader: "DataLoaderNode"
    val_reward_fn: Any
    reward_fn: Any
    kl_ctrl_in_reward: Optional[Any]
    validate_tokenizer: Any
    role_worker_mapping: Dict[NodeRole, Type[Worker]]
    _profiler: DistProfiler
    postsampling_masters_group: Optional[ProcessGroup] = None

    multi_agent_loop: Any

    def _initialize_worker(self):
        """Orchestrates the ordered initialization of all worker components."""
        self._rank = self._get_and_validate_rank()
        self.taskgraph = self._get_taskgraph_for_rank(self.taskgraph_mapping)
        self._setup_distributed_environment()
        self._initialize_core_components()
        self._initialize_node_workers()
        self._profiler = DistProfiler(rank=self._rank, config=self.config.profiler)

        if self._rank == 0:
            logger.info("Rank 0: Initializing tracking logger...")
            from siirl.utils.logger.tracking import Tracking

            self.logger = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=self.config.to_dict(),
            )
            if self.enable_perf:
                logger.warning("Performance tracking is enabled. This may impact training speed.")

    def _get_and_validate_rank(self) -> int:
        """Retrieves and validates the worker's rank from the environment."""
        rank_str = os.environ.get("RANK")
        if rank_str is None:
            raise ValueError("Environment variable 'RANK' is not set. This is required for distributed setup.")
        try:
            return int(rank_str)
        except ValueError as e:
            raise ValueError(f"Invalid RANK format: '{rank_str}'. Must be an integer.") from e

    def _get_taskgraph_for_rank(self, taskgraph_mapping: Dict[int, "TaskGraph"]) -> "TaskGraph":
        """Retrieves the TaskGraph for the current rank from the provided mapping."""
        if self._rank not in taskgraph_mapping:
            raise ValueError(f"Rank {self._rank} not found in the provided taskgraph_mapping.")
        taskgraph = taskgraph_mapping[self._rank]
        from siirl.workers.dag import TaskGraph

        if not isinstance(taskgraph, TaskGraph):
            raise TypeError(f"Object for rank {self._rank} must be a TaskGraph, but got {type(taskgraph).__name__}.")
        logger.info(f"Rank {self._rank} assigned to TaskGraph with ID {taskgraph.graph_id}.")
        return taskgraph

    def _setup_distributed_environment(self):
        """Initializes the default process group and all required subgroups."""
        # gloo_socket_ifname = 'bond0'
        # os.environ["GLOO_SOCKET_IFNAME"] = gloo_socket_ifname
        # os.environ["GLOO_LOG_LEVEL"] = "DEBUG"
        import torch.distributed as dist

        if not dist.is_initialized():
            backend = (
                f"{get_nccl_backend()}"
                if self.world_size >= self.config.dag.backend_threshold
                else f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}"
            )
            logger.info(
                f"Rank {self._rank}: Initializing world size {self.world_size} default process group with '{backend}' "
                f"backend."
            )
            dist.init_process_group(backend=backend)

        if device_name == "npu":
            # For NPU, metrics aggregation requires the hccl backend for device-to-device communication.
            # This group is created regardless of world size for NPU environments.
            gather_backend = get_nccl_backend()
            self._gather_group = dist.new_group(backend=gather_backend)
        else:
            # For GPU, the original logic is preserved for backward compatibility.
            # The gather group is only created if world_size < backend_threshold.
            self._gather_group = dist.new_group(
                backend="gloo") if self.world_size < self.config.dag.backend_threshold else None
        self._build_all_process_groups()
        self._resolve_taskgraph_process_groups()

        # try create post sampling process_groups for dapo
        if self.config.algorithm.algorithm_name == AlgorithmType.DAPO.value:
            self._create_postsampling_masters_group()

        # Ensure all ranks have finished group creation before proceeding.
        dist.barrier(self._gather_group)
        logger.info(f"Rank {self._rank}: Distributed environment setup complete.")

    def _create_postsampling_masters_group(self):
        """
        Creates a dedicated process group containing only master ranks (tp_rank=0).
        This group is used for post-sampling rebalancing logic to prevent deadlocks.
        """
        logger.info(f"Rank {self._rank}: Attempting to create dedicated process group for post-sampling masters...")
        try:
            # To create the group, we need the tensor parallel size (tp_size).
            # We derive it from the first rollout node, as this setting governs the rebalancing logic.
            rollout_nodes = [n for n in self.taskgraph.nodes.values() if n.node_type == NodeType.MODEL_INFERENCE]
            if not rollout_nodes:
                logger.warning("No MODEL_INFERENCE nodes found. Skipping creation of post-sampling masters group.")
                self.postsampling_masters_group = None
                return

            first_rollout_node = rollout_nodes[0]
            tp_size = first_rollout_node.config[DAGConstants.INTERN_CONFIG].rollout.tensor_model_parallel_size

            # The group is only necessary for distributed training with tensor parallelism.
            if self.world_size > 1 and tp_size > 1:
                all_ranks = list(range(self.world_size))
                master_ranks = [rank for rank in all_ranks if (rank % tp_size) == 0]
                self.postsampling_masters_group = dist.new_group(ranks=master_ranks)
                logger.success(
                    f"Rank {self._rank}: Successfully created 'postsampling_masters_group' with ranks: {master_ranks}"
                )
            else:
                logger.info(
                    f"Rank {self._rank}: No need to create 'postsampling_masters_group' (world_size={self.world_size}, "
                    f"tp_size={tp_size})."
                )
                self.postsampling_masters_group = None

        except (AttributeError, KeyError) as e:
            logger.error(
                f"Failed to create post-sampling masters group due to missing config. Error: {e}", exc_info=True
            )
            self.postsampling_masters_group = None

    def _build_all_process_groups(self):
        """Builds all process groups defined in the ProcessGroupManager."""
        import torch.distributed as dist

        group_specs = self.process_group_manager.get_all_specs()
        if not group_specs:
            logger.warning("No process group specifications found in ProcessGroupManager.")
            return

        for name, spec in group_specs.items():
            if not isinstance(spec, dict) or not (ranks := spec.get("ranks")):
                logger.warning(f"Skipping group '{name}' due to invalid spec or missing 'ranks'.")
                continue
            self.process_groups[name] = dist.new_group(ranks=ranks)
        logger.debug(f"Rank {self._rank}: Created {len(self.process_groups)} custom process groups.")

    def _resolve_taskgraph_process_groups(self):
        """Identifies and caches process groups relevant to this worker's TaskGraph."""
        self.inference_group_name_set = self.process_group_manager.get_process_group_for_node_type_in_subgraph(
            self.taskgraph.graph_id, NodeType.MODEL_INFERENCE.value
        )
        self.train_group_name_set = self.process_group_manager.get_process_group_for_node_type_in_subgraph(
            self.taskgraph.graph_id, NodeType.MODEL_TRAIN.value
        )

    def _initialize_core_components(self):
        """Initializes shared components like tokenizers, data loaders, and reward functions."""
        self._setup_tokenizers()
        self._setup_dataloader_and_reward()
        self._setup_role_worker_mapping()

    def _setup_tokenizers(self):
        """Initializes and caches tokenizers for all models in the task graph."""
        model_nodes = [
            node
            for node in self.taskgraph.nodes.values()
            if node.node_type in [NodeType.MODEL_TRAIN, NodeType.MODEL_INFERENCE]
        ]
        if not model_nodes:
            logger.warning("No model nodes found in the task graph. Tokenizer setup will be skipped.")
            return

        for node in model_nodes:
            agent_key = self._generate_agent_group_key(node)
            if agent_key not in self.tokenizer_mapping:
                # Add robust check for missing configuration.
                intern_config = node.config.get(DAGConstants.INTERN_CONFIG)
                if not intern_config or not (model_dict := getattr(intern_config, "model", None)):
                    logger.warning(f"Node {node.node_id} is missing model config. Skipping tokenizer setup for it.")
                    continue

                tokenizer_module = load_tokenizer(model_args=model_dict)
                if tokenizer := tokenizer_module.get("tokenizer"):
                    tokenizer.padding_side = "left"  # Required for most causal LM generation
                self.tokenizer_mapping[agent_key] = tokenizer_module
        logger.info(f"Rank {self._rank}: Initialized {len(self.tokenizer_mapping)} tokenizer(s).")

    def _setup_dataloader_and_reward(self):
        """Initializes the data loader and reward functions."""
        rollout_nodes = [n for n in self.taskgraph.nodes.values() if n.node_type == NodeType.MODEL_INFERENCE]
        if not rollout_nodes:
            raise ValueError("At least one MODEL_INFERENCE node is required for dataloader and reward setup.")
        self.first_rollout_node = rollout_nodes[0]

        pg_assignment = self.process_group_manager.get_node_assignment(self.first_rollout_node.node_id)
        if not (process_group_name := pg_assignment.get("process_group_name")):
            raise ValueError(
                f"Process group name not found for the first rollout node {self.first_rollout_node.node_id}."
            )

        self.dataloader_process_group = self.process_groups.get(process_group_name)
        if self.dataloader_process_group is None:
            raise ValueError(f"Could not find process group '{process_group_name}' in the created groups.")

        self.dataloader_tensor_model_parallel_size = self.first_rollout_node.config[
            DAGConstants.INTERN_CONFIG
        ].rollout.tensor_model_parallel_size

        self.dataloader = DataLoaderNode(
            node_id="dataloader",
            global_config=self.config,
            config={
                "group_world_size": dist.get_world_size(self.dataloader_process_group),
                "group_rank": dist.get_rank(self.dataloader_process_group),
                "group_parallel_size": self.dataloader_tensor_model_parallel_size,
                "num_loader_workers": self.config.data.num_loader_workers,
                "auto_repeat": self.config.data.auto_repeat,
            },
        )

        self.validate_tokenizer = next(iter(self.tokenizer_mapping.values()), {}).get("tokenizer")
        if not self.validate_tokenizer:
            logger.warning("No tokenizer loaded; reward functions might fail or use a default one.")

        self.val_reward_fn = create_reward_manager(
            self.config,
            self.validate_tokenizer,
            num_examine=1,
            max_resp_len=self.config.data.max_response_length,
            overlong_buffer_cfg=self.config.reward_model.overlong_buffer,
        )
        self.reward_fn = create_reward_manager(
            self.config,
            self.validate_tokenizer,
            num_examine=0,
            max_resp_len=self.config.data.max_response_length,
            overlong_buffer_cfg=self.config.reward_model.overlong_buffer,
            **self.config.reward_model.reward_kwargs,
        )

        if self.config.algorithm.use_kl_in_reward:
            from siirl.workers.dag_worker import core_algos

            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # TODO: support multi-agent environment

    def _get_worker_classes(self, strategy: str) -> Dict[NodeRole, Type[Worker]]:
        """Dynamically imports worker classes based on the specified strategy."""
        if strategy in DAGConstants.FSDP_STRATEGIES:
            from siirl.workers.fsdp_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
                CriticWorker,
                RewardModelWorker,
            )

            actor_cls = (
                AsyncActorRolloutRefWorker
                if self.config.actor_rollout_ref.rollout.mode == "async"
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
            from siirl.workers.megatron_workers import (
                ActorWorker, 
                RolloutWorker, 
                AsyncRolloutWorker, 
                ReferenceWorker, 
                CriticWorker, 
                RewardModelWorker
            )

            is_async_mode = self.config.actor_rollout_ref.rollout.mode == "async"
            
            return {
                NodeRole.ACTOR: ActorWorker,
                NodeRole.ROLLOUT: AsyncRolloutWorker if is_async_mode else RolloutWorker,
                NodeRole.REFERENCE: ReferenceWorker,
                NodeRole.CRITIC: CriticWorker,
                NodeRole.REWARD: RewardModelWorker
            }
        raise NotImplementedError(f"Strategy '{strategy}' is not supported.")

    def _setup_role_worker_mapping(self):
        """Creates a mapping from NodeRole to the corresponding Worker implementation class."""
        self.role_worker_mapping: Dict[NodeRole, Type[Worker]] = {}
        # Actor/Ref/Rollout/Critic workers
        actor_strategy = self.config.actor_rollout_ref.actor.strategy
        self.role_worker_mapping.update(self._get_worker_classes(actor_strategy))

        # Reward model worker (if enabled)
        if self.config.reward_model.enable:
            reward_strategy = self.config.reward_model.strategy
            reward_workers = self._get_worker_classes(reward_strategy)
            if NodeRole.REWARD in reward_workers:
                self.role_worker_mapping[NodeRole.REWARD] = reward_workers[NodeRole.REWARD]
            else:
                logger.warning(
                    f"Reward model is enabled, but no worker found for role REWARD with strategy {reward_strategy}."
                )

        self._log_role_worker_mapping()

    def _log_role_worker_mapping(self):
        """Logs the final role-to-worker mapping for setup verification."""
        if not self.role_worker_mapping:
            logger.error("Role-to-worker mapping is empty after setup. This will cause execution failure.")
            return

        logger.debug("--- [Role -> Worker Class] Mapping ---")
        max_len = max((len(r.name) for r in self.role_worker_mapping.keys()), default=0)
        for role, worker_cls in sorted(self.role_worker_mapping.items(), key=lambda item: item[0].name):
            logger.debug(
                f"  {role.name:<{max_len}} => {worker_cls.__name__} (from {inspect.getmodule(worker_cls).__name__})"
            )
        logger.debug("--------------------------------------")

    def _initialize_node_workers(self):
        """Instantiates worker objects for all nodes in the task graph."""
        for node in self.taskgraph.nodes.values():
            if not self._should_create_worker(node):
                continue

            worker_cls = self.role_worker_mapping.get(node.node_role)
            if not worker_cls:
                logger.warning(f"No worker class found for role {node.node_role.name}. Skipping node {node.node_id}.")
                continue

            node_worker_key = self._generate_node_worker_key(node)
            if node_worker_key in self.workers:
                continue

            try:
                node_process_group = self._get_node_process_group(node)
                config = node.config.get(DAGConstants.INTERN_CONFIG)
                if hasattr(config, "actor") and hasattr(config.actor, "optim"):
                    config.actor.optim.total_training_steps = self.dataloader.total_training_steps
                elif hasattr(config, "optim"):
                    config.optim.total_training_steps = self.dataloader.total_training_steps
                worker_args = {"config": config, "process_group": node_process_group}
               
                # For separated workers (Megatron backend), no role parameter is needed
                # Only legacy ActorRolloutRefWorker needs the role parameter
                if hasattr(worker_cls, '__name__') and 'ActorRolloutRefWorker' in worker_cls.__name__:
                    if node.node_role in DAGConstants.WORKER_ROLE_MAPPING:
                        worker_args["role"] = DAGConstants.WORKER_ROLE_MAPPING[node.node_role]
                if node.agent_options and node.agent_options.share_instance:
                    # cur agent share same critic with target agent
                    self.agent_group_worker[node.agent_group][node.node_role] = self.agent_group_worker[node.agent_options.share_instance][node.node_role]
                else:
                    worker_instance = worker_cls(**worker_args)
                    self.workers[node_worker_key] = worker_instance
                    self.agent_group_worker[node.agent_group][node.node_role] = worker_instance
                    self.agent_group_process_group[node.agent_group][node.node_role] = node_process_group
                    logger.success(
                        f"Rank {self._rank}: Successfully created worker '{worker_cls.__name__}' for node: {node.node_id}"
                    )

            except Exception as e:
                #  Explicitly log the failing node and worker class, then re-raise
                # the exception to prevent silent failures.
                logger.error(
                    f"Failed to create worker for node {node.node_id} with class {worker_cls.__name__}.", exc_info=True
                )
                raise RuntimeError(f"Worker instantiation failed for node {node.node_id}") from e
        
        if len(self.agent_group_worker) > 1:
            self._multi_agent = True
    def _generate_node_worker_key(self, node: Node) -> str:
        """Generates a unique string key for a node's worker instance."""
        return f"{node.agent_group}_{node.node_type.value}_{node.node_role.value}"

    def _generate_agent_group_key(self, node: Node) -> str:
        """Generates a unique key for an agent group, used for caching (e.g., tokenizers)."""
        return f"group_key_{node.agent_group}"

    def _should_create_worker(self, node: Node) -> bool:
        """Determines if a worker instance should be created for a given graph node."""
        if node.agent_options and node.agent_options.share_instance:
            # has been initialized in target agent node
            return False
        return node.node_type in [NodeType.MODEL_TRAIN, NodeType.MODEL_INFERENCE] and node.node_role in self.role_worker_mapping

    def _get_node_process_group(self, node: Node) -> ProcessGroup:
        """Retrieves the PyTorch ProcessGroup assigned to a specific graph node."""
        assignment = self.process_group_manager.get_node_assignment(node.node_id)
        if not (assignment and (name := assignment.get("process_group_name"))):
            raise ValueError(f"Process group assignment or name not found for node {node.node_id}.")

        pg = self.process_groups.get(name)
        if pg is None:
            raise ValueError(f"Process group '{name}' for node {node.node_id} was not created or found.")
        return pg

    def _get_node(self, role: NodeRole, agent_group: int) -> Node:
        """
        Finds and returns a specific node from the task graph based on its role
        and agent group.
        """
        found_node = next(
            (
                node
                for node in self.taskgraph.nodes.values()
                if node.node_role == role and node.agent_group == agent_group
            ),
            None,
        )

        if found_node is None:
            raise RuntimeError(f"Could not find a node with role {role.name} for agent_group {agent_group}")
        return found_node

    def _get_node_dp_info(self, node: Node) -> tuple[int, int, int, int, int, int]:
        """
        Calculates Data Parallel (DP), Tensor Parallel (TP), and Pipeline Parallel (PP) info for a node.
        
        Returns:
            tuple: (dp_size, dp_rank, tp_rank, tp_size, pp_rank, pp_size)
        """
        reference_node = node
        if node.node_type == NodeType.COMPUTE:
            # If the node is a COMPUTE type, find its true data source ancestor.
            ancestor = self._find_first_non_compute_ancestor(node.node_id)
            if ancestor:
                reference_node = ancestor
            else:
                # If no non-COMPUTE ancestor is found, it's a critical error.
                raise RuntimeError(f"Could not find any non-COMPUTE ancestor for COMPUTE node '{node.node_id}'. Please check your DAG graph configuration.")

        if reference_node.node_type == NodeType.COMPUTE:
            group_world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
            group_rank = dist.get_rank()
        else:
            process_group = self._get_node_process_group(reference_node)
            group_world_size = dist.get_world_size(process_group)
            group_rank = dist.get_rank(process_group)

        # Get parallelism configuration based on backend strategy
        tp_size, pp_size = self._get_parallelism_config(reference_node)
        
        # Calculate total parallel size (TP * PP)
        total_parallel_size = tp_size * pp_size
        
        if group_world_size % total_parallel_size != 0:
            raise ValueError(f"Configuration error for node {node.node_id}: Group world size ({group_world_size}) is not divisible by total parallel size (TP={tp_size} * PP={pp_size} = {total_parallel_size}). Check your parallel configuration.")
        
        dp_size = group_world_size // total_parallel_size
        
        # Calculate ranks within the data parallel group
        dp_rank = group_rank // total_parallel_size
        
        # Calculate position within the TP-PP grid
        local_rank_in_tp_pp_group = group_rank % total_parallel_size
        
        # For 2D parallelism: ranks are arranged as [PP0_TP0, PP0_TP1, ..., PP0_TP(tp_size-1), PP1_TP0, ...]
        pp_rank = local_rank_in_tp_pp_group // tp_size
        tp_rank = local_rank_in_tp_pp_group % tp_size
        
        return dp_size, dp_rank, tp_rank, tp_size, pp_rank, pp_size
    
    def _get_parallelism_config(self, reference_node: Node) -> tuple[int, int]:
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

    def log_ray_actor_info(self):
        """Logs detailed information about the Ray actor's context for debugging."""
        try:
            ctx = ray.get_runtime_context()
            logger.debug(
                f"Ray Actor Context for Rank {self._rank}: ActorID={ctx.get_actor_id()}, JobID={ctx.get_job_id()}, "
                f"NodeID={ctx.get_node_id()}, PID={os.getpid()}"
            )
        except RuntimeError:
            logger.warning(f"Rank {self._rank}: Not running in a Ray actor context.")

    def init_model(self):
        """Initializes models for all workers and sets up sharding managers where applicable."""
        logger.info("Initializing models for all worker nodes...")
        have_init_workers = set()
        for node in self.taskgraph.nodes.values():
            if self._should_create_worker(node):
                node_worker = self.workers[self._generate_node_worker_key(node)]
                if not isinstance(node_worker, Worker):
                    raise TypeError(f"Invalid worker type for node {node.node_id}: {type(node_worker).__name__}")
                if self._generate_node_worker_key(node) in have_init_workers:
                    logger.warning(
                        f"Rank {self._rank}: Worker {self._generate_node_worker_key(node)} for node {node.node_id} "
                        f"already initialized. Skipping."
                    )
                    continue
                node_worker.init_model()
                have_init_workers.add(self._generate_node_worker_key(node))
                if node.node_role == NodeRole.ROLLOUT and node.config["intern_config"].rollout.mode == "async":
                    self.rollout_mode = "async"
                    self.zmq_address = node_worker.get_zeromq_address()
        logger.success("All worker models initialized.")

        logger.info(f"Setting up sharding managers {self.config.actor_rollout_ref.rollout.name} ...")
        for agent_group, worker_dict in self.agent_group_worker.items():
            if NodeRole.ACTOR in worker_dict and NodeRole.ROLLOUT in worker_dict:
                try:
                    self._setup_sharding_manager(agent_group, worker_dict)
                except Exception as e:
                    logger.error(f"Failed to set up sharding manager for agent group {agent_group}: {e}", exc_info=True)
                    raise
        logger.info("All models and sharding managers initialized successfully.")
        if self._multi_agent:
            from siirl.workers.multi_agent.multiagent_generate import MultiAgentLoop
            self.multi_agent_loop =  MultiAgentLoop(self, config = self.config.actor_rollout_ref, node_workers = self.workers, local_dag = self.taskgraph, databuffer = self.data_buffers, placement_mode = 'colocate')
        
    def _setup_sharding_manager(self, agent_group: int, worker_dict: Dict[NodeRole, Worker]):
        """Configures the sharding manager to sync weights between training backend and inference backend."""
        actor_worker = worker_dict[NodeRole.ACTOR]
        rollout_worker = worker_dict[NodeRole.ROLLOUT]
        rollout_pg = self.agent_group_process_group[agent_group][NodeRole.ROLLOUT]

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
                "siirl.workers.sharding_manager.fsdp_vllm.MultiAgentFSDPVLLMShardingManager",
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
                "siirl.workers.sharding_manager.fsdp_sglang.MultiAgentFSDPSGLangShardingManager",
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
                "siirl.workers.sharding_manager.megatron_vllm.MultiAgentMegatronVLLMShardingManager",
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
                "siirl.workers.sharding_manager.megatron_sglang.MultiAgentMegatronSGLangShardingManager",
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
        rollout_name = self.config.actor_rollout_ref.rollout.name.lower()
        if (strategy, rollout_name) not in sharding_manager_map:
            raise NotImplementedError(f"Unsupported sharding manager configuration: {strategy=}, {rollout_name=}")

        sharding_manager_cls_str, kwargs_builder = sharding_manager_map[(strategy, rollout_name)]
        sharding_manager_cls = import_string(sharding_manager_cls_str)
        sharding_manager = sharding_manager_cls(**kwargs_builder())
        rollout_worker.set_rollout_sharding_manager(sharding_manager)
        logger.debug(f"Set up {sharding_manager_cls.__name__}  for agent group {agent_group}.")

    def init_graph(self):
        # this is needed by async rollout manager
        self._set_node_executables()
        self.init_model()
        self._load_checkpoint()
        # Ensure all models are initialized and checkpoints are loaded before starting.
        dist.barrier(self._gather_group)

    def set_async_rollout_manager(self, async_rollout_manager):
        self._async_rollout_manager = async_rollout_manager

    def get_zeromq_address(self):
        return self.zmq_address
