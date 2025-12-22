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

import time
from typing import Any, Dict, List
import ray
from loguru import logger

from siirl.workers.dag import Node, NodeRole, NodeType, TaskGraph

from siirl.utils.params import ActorRolloutRefArguments, ActorArguments, RefArguments, RolloutArguments, CriticArguments, RewardModelArguments, SiiRLArguments
from siirl.scheduler.enums import AdvantageEstimator

from .process_group_manager import ProcessGroupManager
from .ray_actor_manager import RayActorManager
from .graph_updater import INTERN_CONFIG
from .resource_manager import ResourcePoolManager


class RayTrainer:
    """
    The main orchestrator for a distributed training session using Ray.

    This class is responsible for:
    1.  Validating the configurations for all components in the task graphs.
    2.  Managing hardware resources (GPUs) across nodes.
    3.  Initializing and managing the lifecycle of Ray actors (DAGWorkers).
    4.  Starting the training process and monitoring its execution until completion or failure.
    """

    def __init__(self, config: SiiRLArguments, process_group_manager: ProcessGroupManager, rank_taskgraph_mapping: Dict[int, "TaskGraph"], unique_graphs_map: Dict[str, "TaskGraph"], data_buffer_handles: List["ray.actor.ActorHandle"], device_name="cuda"):
        """
        Initializes the RayTrainer.

        Args:
            config: The main SiiRLArguments object containing all configuration parameters.
            process_group_manager: Manages communication groups for distributed training.
            rank_taskgraph_mapping: A mapping from a global rank to its assigned TaskGraph.
            unique_graphs_map: A mapping of unique graph IDs to their TaskGraph objects.
            data_buffer_handles: A list of Ray actor handles for data buffers.
        """
        # Store essential configuration and management objects.
        self.base_config = config
        self.process_group_manager = process_group_manager
        self.rank_taskgraph_mapping = rank_taskgraph_mapping
        self.data_buffer_handles = data_buffer_handles
        self.unique_graphs_map = unique_graphs_map

        # Calculate the total number of GPUs available for the training job.
        self.total_gpu = self.base_config.trainer.n_gpus_per_node * self.base_config.trainer.nnodes

        # Determine whether a critic model is needed based on the chosen advantage estimator algorithm.
        # GAE requires a critic for value estimation. Other listed methods do not.
        if self.base_config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.base_config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.CPGD,
            AdvantageEstimator.GAE_MARFT,
        ]:
            self.use_critic = False
        else:
            # If the algorithm is not recognized, raise an error.
            raise NotImplementedError

        # --- Create resource manager ---
        # Define the specification for the global resource pool, typically GPUs per node.
        self.global_pool_id = "global_resource_pool"
        resource_pool_spec = {
            self.global_pool_id: [self.base_config.trainer.n_gpus_per_node] * self.base_config.trainer.nnodes,
        }
        # Instantiate the manager to oversee the allocation of these resources.
        self.resource_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec)
        # The actor manager is initialized later in the `init_workers` method.
        self.ray_actor_manager = None
        self.device_name = device_name

        # Perform a comprehensive validation of all configurations upon initialization.
        self._validate_config()

    def _check_mutually_exclusive(self, component_config: Dict[str, Any], param_name: str, param_per_gpu_name: str, component_id_str: str) -> None:
        """
        A helper function to validate that only one of two mutually exclusive batch size parameters is set.

        It enforces that users set the 'per_gpu' version of a parameter and not the deprecated total version.

        Args:
            component_config: The configuration dictionary for a specific component.
            param_name: The name of the deprecated total batch size parameter (e.g., "ppo_micro_batch_size").
            param_per_gpu_name: The name of the preferred per-GPU batch size parameter (e.g., "ppo_micro_batch_size_per_gpu").
            component_id_str: A string identifying the component for clear error messages (e.g., "Node 1 (Actor)").
        """
        # Get both parameter values from the config dict.
        mbs = component_config.get(param_name, None)
        mbs_per_gpu = component_config.get(param_per_gpu_name, None)

        # Fail if neither parameter is provided.
        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{component_id_str}] Please set at least one of '{param_name}' or '{param_per_gpu_name}'.")

        # Fail if both are provided, guiding the user to use the preferred 'per_gpu' parameter.
        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(f"[{component_id_str}] You have set both '{param_name}' AND '{param_per_gpu_name}'. Please remove '{param_name}' because only '{param_per_gpu_name}' is supported (the former is deprecated).")

    def validate_actor_config(self, node: Node, actor_conf: ActorArguments, use_remove_padding: bool = False) -> None:
        """Validates configuration parameters specific to an Actor training node."""
        logger.debug(f"Validating Actor specific configurations for Node: {node.node_id} using provided actor_conf")

        # Check if KL is used both as a reward penalty and a loss term, which is a valid but notable configuration.
        if self.base_config.algorithm.use_kl_in_reward and actor_conf.use_kl_loss:
            logger.info(f"Node {node.node_id} (Actor): Both in-reward KL and KL loss are enabled for this actor configuration.")

        # Extract relevant actor parameters for validation.
        ppo_mini_batch_size = actor_conf.ppo_mini_batch_size
        ppo_micro_batch_size_per_gpu = actor_conf.ppo_micro_batch_size_per_gpu
        ppo_micro_batch_size = actor_conf.ppo_micro_batch_size
        sp_size = actor_conf.ulysses_sequence_parallel_size
        loss_agg_mode = actor_conf.loss_agg_mode
        strategy = actor_conf.strategy
        use_dynamic_bsz = actor_conf.use_dynamic_bsz

        # Perform batch size validations if not using a dynamic batch size.
        if not use_dynamic_bsz:
            self._check_mutually_exclusive(actor_conf.to_dict(), "ppo_micro_batch_size", "ppo_micro_batch_size_per_gpu", f"Node {node.node_id} (Actor)")
            # Ensure the global training batch size is at least as large as the PPO mini-batch size.
            assert self.base_config.data.train_batch_size * self.base_config.actor_rollout_ref.rollout.n >= ppo_mini_batch_size, f"Node {node.node_id} (Actor): train_batch_size ({self.base_config.data.train_batch_size}) must be >= ppo_mini_batch_size ({ppo_mini_batch_size})"
            # If micro-batch size is set, perform further divisibility checks.
            if ppo_micro_batch_size is not None:
                component_total_mbs = ppo_micro_batch_size_per_gpu * self.total_gpu
                assert ppo_mini_batch_size % ppo_micro_batch_size == 0, f"Node {node.node_id} (Actor): ppo_mini_batch_size ({ppo_mini_batch_size}) must be divisible by component_total_mbs ({component_total_mbs})."
                # This assertion seems to have a typo, but keeping logic as-is. It compares total vs per_gpu * sp.
                assert ppo_micro_batch_size * sp_size >= self.total_gpu, f"Node {node.node_id} (Actor): ppo_micro_batch_size_per_gpu * SP size ({ppo_micro_batch_size_per_gpu} * {sp_size}) must be >= {self.total_gpu}"

        # Ensure the loss aggregation mode is one of the supported values.
        assert loss_agg_mode in ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"], f"Node {node.node_id} (Actor): Invalid loss_agg_mode: {loss_agg_mode}"

        # For FSDP with sequence parallelism, padding must be removed to avoid hangs.
        if strategy == "fsdp" and sp_size > 1:
            assert use_remove_padding, f"Node {node.node_id} (Actor): When using SP (>1) with FSDP, enable `use_remove_padding` in the relevant model config."

    def validate_reference_config(self, node: Node, reference_conf: RefArguments, use_remove_padding: bool = False) -> None:
        """Validates configuration parameters specific to a Reference Policy inference node."""
        logger.debug(f"Validating Reference Policy specific configurations for Node: {node.node_id}")
        log_prob_use_dynamic_bsz = reference_conf.log_prob_use_dynamic_bsz
        strategy = reference_conf.strategy
        ulysses_sequence_parallel_size = reference_conf.ulysses_sequence_parallel_size

        # Validate micro batch size settings if not using dynamic batching.
        if not log_prob_use_dynamic_bsz:
            self._check_mutually_exclusive(reference_conf.to_dict(), "log_prob_micro_batch_size", "log_prob_micro_batch_size_per_gpu", f"Node {node.node_id} (Reference)")

        # For FSDP with sequence parallelism, padding must be removed.
        if strategy == "fsdp" and ulysses_sequence_parallel_size > 1:
            assert use_remove_padding, f"Node {node.node_id} (Reference): When using SP (>1) with FSDP, enable `use_remove_padding` in relevant model config."

    def validate_rollout_config(self, node: Node, rollout_conf: RolloutArguments, use_remove_padding: bool = False):
        """Validates configuration parameters specific to a Rollout (generation) node."""
        logger.debug(f"Validating Rollout specific configurations for Node: {node.node_id}")

        # Validate micro batch size for log-probability calculations if not dynamic.
        log_prob_use_dynamic_bsz = rollout_conf.log_prob_use_dynamic_bsz
        if not log_prob_use_dynamic_bsz:
            self._check_mutually_exclusive(rollout_conf.to_dict(), "log_prob_micro_batch_size", "log_prob_micro_batch_size_per_gpu", f"Node {node.node_id} (Rollout)")

        # Extract generation parameters for validation.
        do_sample = rollout_conf.val_kwargs.do_sample
        multi_turn_enable = rollout_conf.multi_turn.enable
        temperature = rollout_conf.temperature
        tool_config_path = rollout_conf.multi_turn.tool_config_path

        # If sampling is enabled, temperature must be positive.
        if do_sample:
            assert temperature > 0, f"Node {node.node_id} (Rollout): validation gen temperature > 0 for do_sample."

        # If multi-turn rollouts (e.g., with tools) are enabled, a tool config must be provided.
        if multi_turn_enable:
            assert tool_config_path is not None, f"Node {node.node_id} (Rollout): tool_config_path required for multi_turn."
            # Check if the algorithm is compatible with multi-turn rollouts.
            assert self.base_config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], f"only GRPO is tested for multi-turn with tool"

    def validate_critic_config(self, node: Node, critic_conf: CriticArguments, use_remove_padding: bool = False):
        """Validates configuration parameters specific to a Critic training node."""
        logger.debug(f"Validating Critic specific configurations for Node: {node.node_id}")

        # Extract critic parameters.
        use_dynamic_bsz = critic_conf.use_dynamic_bsz
        ppo_mini_batch_size = critic_conf.ppo_mini_batch_size
        ppo_micro_batch_size_per_gpu = critic_conf.ppo_micro_batch_size_per_gpu
        ppo_micro_batch_size = critic_conf.ppo_micro_batch_size
        sp_size = critic_conf.ulysses_sequence_parallel_size
        strategy = critic_conf.strategy

        # Validate batch sizes if not dynamic.
        if not use_dynamic_bsz:
            self._check_mutually_exclusive(critic_conf.to_dict(), "ppo_micro_batch_size", "ppo_micro_batch_size_per_gpu", f"Node {node.node_id} (Critic)")
            assert self.base_config.data.train_batch_size >= ppo_mini_batch_size
            if ppo_micro_batch_size is not None:
                effective_mbs_per_gpu = ppo_micro_batch_size_per_gpu
                assert ppo_mini_batch_size % ppo_micro_batch_size == 0
                assert ppo_micro_batch_size * sp_size >= 1

        # For FSDP with sequence parallelism, padding must be removed.
        if strategy == "fsdp" and sp_size > 1:
            assert use_remove_padding, f"Node {node.node_id} (Critic): When using SP (>1) with FSDP, enable `use_remove_padding` in critic.model."

    def validate_reward_model_config(self, node: Node, reward_model_conf: RewardModelArguments, use_remove_padding: bool = False):
        """Validates configuration parameters specific to a Reward Model training node."""
        logger.debug(f"Validating Reward Model specific configurations for Node: {node.node_id}")
        use_dynamic_bsz = reward_model_conf.use_dynamic_bsz
        # Validate micro batch size settings if not using dynamic batching.
        if not use_dynamic_bsz:
            self._check_mutually_exclusive(reward_model_conf.to_dict(), "micro_batch_size", "micro_batch_size_per_gpu", f"Node {node.node_id} (RewardModel)")

    def validate_configurations_for_task_graph(self, task_graph: TaskGraph) -> None:
        """
        Iterates through all nodes in a task graph and dispatches to the appropriate validation function.

        Args:
            task_graph: The TaskGraph object to validate.
        """
        logger.info(f"Starting configuration validation for TaskGraph: {task_graph.graph_id}")

        # Loop over each node in the graph.
        for node_id, node in task_graph.nodes.items():
            logger.debug(f"Processing Node ID: {node.node_id}, Type: {node.node_type.value}, Role: {node.node_role.value}")

            # Initialize variables for the dispatcher.
            node_specific_config: Any = None
            validator_function = None
            component_name_for_logging = ""
            use_remove_padding = False
            intern_config = None
            # The actual component-specific config is stored in a special key.
            if INTERN_CONFIG in node.config:
                intern_config = node.config[INTERN_CONFIG]

            # Based on the node's type and role, select the correct config object and validator function.
            if node.node_type == NodeType.MODEL_TRAIN and node.node_role == NodeRole.ACTOR:
                assert isinstance(intern_config, ActorRolloutRefArguments), f"Node {node_id} intern config illegal"
                # Calculate the effective batch size considering the number of rollouts per sample.
                real_train_batch_size = self.base_config.data.train_batch_size * intern_config.rollout.n
                assert real_train_batch_size % self.total_gpu == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({self.total_gpu})."
                node_specific_config = intern_config.actor
                validator_function = self.validate_actor_config
                use_remove_padding = intern_config.model.use_remove_padding
                component_name_for_logging = "Actor"
            elif node.node_type == NodeType.MODEL_INFERENCE and node.node_role == NodeRole.ROLLOUT:
                assert isinstance(intern_config, ActorRolloutRefArguments), f"Node {node_id} intern config illegal"
                node_specific_config = intern_config.rollout
                validator_function = self.validate_rollout_config
                component_name_for_logging = "Rollout"
            elif node.node_type == NodeType.MODEL_TRAIN and node.node_role == NodeRole.CRITIC:
                assert isinstance(intern_config, CriticArguments), f"Node {node_id} intern config illegal"
                if self.base_config.critic:
                    node_specific_config = intern_config
                    validator_function = self.validate_critic_config
                    component_name_for_logging = "Critic"
                    use_remove_padding = intern_config.model.use_remove_padding
            elif node.node_type == NodeType.MODEL_TRAIN and node.node_role == NodeRole.REWARD:
                assert isinstance(intern_config, RewardModelArguments), f"Node {node_id} intern config illegal"
                if self.base_config.reward_model.enable:
                    node_specific_config = intern_config
                    validator_function = self.validate_reward_model_config
                    component_name_for_logging = "RewardModel"
                    use_remove_padding = intern_config.model.use_remove_padding
            elif node.node_type == NodeType.MODEL_TRAIN and node.node_role == NodeRole.REFERENCE:
                assert isinstance(intern_config, ActorRolloutRefArguments), f"Node {node_id} intern config illegal"
                node_specific_config = intern_config.ref
                validator_function = self.validate_reference_config
                component_name_for_logging = "ReferencePolicy"
                use_remove_padding = intern_config.ref.use_remove_padding

            # If a validator was found for the node, execute it.
            if validator_function and node_specific_config is not None:
                try:
                    logger.debug(f"Running {component_name_for_logging} validation for Node: {node.node_id}")
                    # Pass the node, its specific config, and padding flag to the validator.
                    validator_function(node, node_specific_config, use_remove_padding)
                except (AssertionError, ValueError) as e:
                    # If validation fails, log a fatal error and re-raise to halt execution.
                    logger.error(f"Configuration validation FAILED for Node {node.node_id} ({component_name_for_logging}): {e}")
                    raise
            elif validator_function and node_specific_config is None:
                # This case handles when a node should be validated (e.g., Critic) but is disabled in the main config.
                logger.warning(f"Node {node.node_id} ({node.node_type.value}, {node.node_role.value}) mapped to {component_name_for_logging} validator, but its config section was not found or component not enabled. Skipping specialized validation.")
            else:
                # For nodes that do not require specialized validation (e.g., data nodes).
                logger.trace(f"No specialized validator or config section for Node {node.node_id} ({node.node_type.value}, {node.node_role.value}).")

        logger.info(f"All configuration checks passed successfully for TaskGraph: {task_graph.graph_id}!")

    def _validate_config(self):
        """Entry point for configuration validation. Validates all unique task graphs."""
        for graph_id, task_graph in self.unique_graphs_map.items():
            self.validate_configurations_for_task_graph(task_graph)

    def init_workers(self):
        """Initializes the resources and Ray actors required for training."""
        # Step 1: Create the resource pool based on the spec defined in __init__.
        self.resource_manager.create_resource_pool()
        # Step 2: Create the RayActorManager, which will be responsible for creating and managing the actual DAGWorker actors.
        ray_actor_manager_kwargs = {"ray_wait_register_center_timeout": self.base_config.trainer.ray_wait_register_center_timeout}

        self.ray_actor_manager = RayActorManager(
            resource_pool=self.resource_manager.get_resource_pool(self.global_pool_id),
            base_config=self.base_config,
            process_manager=self.process_group_manager,
            rank_taskgraph_mapping=self.rank_taskgraph_mapping,
            data_buffer_handles=self.data_buffer_handles,
            device_name=self.device_name,
            **ray_actor_manager_kwargs,
        )

    def start_workers(self):
        """
        Starts all DAGWorkers and enters a monitoring loop that waits for them to complete.
        This method handles progress logging and robustly detects and reports actor failures.
        """
        logger.success("create workers finished, try start training")
        # 1. Asynchronously start the main task (`execute_task_graph`) on all workers.
        # This returns a list of "futures", which are placeholders for the eventual results.
        work_futures = self.ray_actor_manager.map_async(method_name="execute_task_graph")

        start_time = time.time()
        num_workers = len(self.ray_actor_manager.workers)

        # Create a mapping from a future to its worker's name for easier logging upon completion or failure.
        future_to_worker_name = {future: name for future, name in zip(work_futures, self.ray_actor_manager.worker_names)}

        # Create a copy of the futures list to track which workers are still running.
        remaining_futures = work_futures.copy()

        # Loop until all workers have completed their tasks.
        while remaining_futures:
            try:
                # 2. Wait for ANY of the remaining tasks to complete.
                # Use a timeout so the loop doesn't block indefinitely, allowing for periodic logging.
                ready_futures, remaining_futures = ray.wait(
                    remaining_futures,
                    num_returns=1,  # Return as soon as one worker finishes.
                    timeout=60.0,  # Wait for up to 60 seconds.
                )

                # If the wait timed out and no futures are ready, it means workers are still running.
                # Log progress and continue to the next iteration of the while loop.
                if not ready_futures:
                    elapsed_time = time.time() - start_time
                    finished_count = num_workers - len(remaining_futures)
                    logger.info(f"INFO: Training for {elapsed_time:.0f} seconds... {finished_count}/{num_workers} workers have finished.")
                    continue

                # 3. Process the futures that are now ready (i.e., workers that have finished).
                for future in ready_futures:
                    worker_name = future_to_worker_name[future]
                    try:
                        # `ray.get()` retrieves the result of the future.
                        # CRITICAL: If the remote actor task failed with an exception,
                        # `ray.get()` will re-raise that exception here, allowing us to catch it.
                        result = ray.get(future)
                        logger.success(f"Worker {worker_name} has finished its task graph. Result: {result}")

                    # Specifically catch the case where a Ray actor process has died.
                    except ray.exceptions.ActorDiedError:
                        logger.error(f"FATAL: Worker {worker_name} died unexpectedly during its task. Halting execution.")
                        # Re-raise the exception to stop the entire training job.
                        # A dead worker is a critical failure that cannot be recovered from.
                        raise
                    # Catch any other exception that the worker might have thrown.
                    except Exception as e:
                        logger.error(f"FATAL: Worker {worker_name} failed with an exception: {e}")
                        # Re-raise to halt the training job.
                        raise

            except Exception as e:
                # Catch unexpected errors in the monitoring loop itself.
                logger.error(f"An unexpected error occurred during worker monitoring: {e}")
                raise

        # This point is reached only if all workers complete successfully.
        elapsed_time = time.time() - start_time
        logger.success(f"All {num_workers} DAGWorkers have successfully finished their task graphs. Total cost: {elapsed_time:.2f}s")
        return
