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

import os
import time

import hydra
import ray
from omegaconf import DictConfig

from siirl.execution.scheduler.enums import AdvantageEstimator, AlgorithmType
from siirl.execution.scheduler.graph_updater import display_node_config, update_task_graph_node_configs
from siirl.execution.scheduler.launch import RayTrainer
from siirl.execution.scheduler.process_group_manager import ProcessGroupManager, log_process_group_manager_details
from siirl.execution.scheduler.task_scheduler import TaskScheduler, log_schedule_assignments
from siirl.utils.logger.logging_utils import set_basic_config
from siirl.global_config.params import SiiRLArguments, log_dict_formatted, parse_config
from siirl.execution.dag import TaskGraph
from siirl.execution.dag.builtin_pipelines import grpo_pipeline, ppo_pipeline, dapo_pipeline
from siirl.data_coordinator import init_data_buffer



# --- Constants ---
RAY_RUNTIME_ENV_VARS = {
    "TOKENIZERS_PARALLELISM": "true",
    "NCCL_DEBUG": "WARN",
    "VLLM_LOGGING_LEVEL": "WARN",
}

# The main runner is an orchestrator, not a heavy workload.
# Assigning it a full CPU is often wasteful. A fractional CPU is more efficient.
MAIN_RUNNER_CPU_RESERVATION = 5


def load_pipeline(siirl_args: SiiRLArguments) -> TaskGraph:
    """
    Load training pipeline using the Python-based Pipeline API.

    This function supports two modes (in priority order):
    1. Custom pipeline via dag.custom_pipeline_fn (user-specified Python function)
    2. Built-in Python pipelines (grpo_pipeline, ppo_pipeline, dapo_pipeline)

    Args:
        siirl_args: Configuration arguments

    Returns:
        TaskGraph: Loaded and validated task graph

    Raises:
        ImportError: If custom pipeline function cannot be loaded
        NotImplementedError: If no suitable pipeline is found
    """
    # Import logger locally to avoid Ray serialization issues
    from loguru import logger

    # Mode 1: User-specified custom pipeline function
    if hasattr(siirl_args.dag, 'custom_pipeline_fn') and siirl_args.dag.custom_pipeline_fn:
        logger.info(f"Loading custom pipeline: {siirl_args.dag.custom_pipeline_fn}")

        try:
            # Parse function path: "module.path:function_name"
            if ":" not in siirl_args.dag.custom_pipeline_fn:
                raise ValueError(
                    f"Invalid custom_pipeline_fn format: '{siirl_args.dag.custom_pipeline_fn}'. "
                    f"Expected format: 'module.path:function_name'"
                )

            module_path, func_name = siirl_args.dag.custom_pipeline_fn.rsplit(":", 1)

            # Dynamically import the module and function
            import importlib
            module = importlib.import_module(module_path)
            pipeline_fn = getattr(module, func_name)

            if not callable(pipeline_fn):
                raise ValueError(
                    f"'{siirl_args.dag.custom_pipeline_fn}' is not callable. "
                    f"It should be a function that returns TaskGraph."
                )

            # Call the function to get TaskGraph
            taskgraph = pipeline_fn()

            if not isinstance(taskgraph, TaskGraph):
                raise ValueError(
                    f"Custom pipeline function '{siirl_args.dag.custom_pipeline_fn}' "
                    f"must return a TaskGraph object, got {type(taskgraph)}"
                )

            logger.success(f"Custom pipeline loaded successfully: {taskgraph.graph_id}")
            return taskgraph

        except Exception as e:
            logger.error(f"Failed to load custom pipeline '{siirl_args.dag.custom_pipeline_fn}': {e}")
            raise

    # Mode 2: Built-in Python pipelines (default)
    logger.info(f"Using built-in Python pipeline for algorithm: {siirl_args.algorithm.adv_estimator}")

    # Set CPGD-specific config
    if siirl_args.algorithm.adv_estimator == AdvantageEstimator.CPGD:
        siirl_args.actor_rollout_ref.actor.use_cpgd_loss = True

    # Select appropriate built-in pipeline
    if siirl_args.algorithm.adv_estimator == AdvantageEstimator.GRPO:
        return grpo_pipeline()
    elif siirl_args.algorithm.adv_estimator == AdvantageEstimator.CPGD:
        return grpo_pipeline()  # CPGD uses GRPO structure
    elif siirl_args.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return ppo_pipeline()
    elif siirl_args.algorithm.algorithm_name == AlgorithmType.DAPO.value:
        return dapo_pipeline()
    else:
        raise NotImplementedError(
            f"No built-in pipeline for algorithm '{siirl_args.algorithm.adv_estimator}'. "
            f"Please specify dag.custom_pipeline_fn to use a custom pipeline."
        )


def get_databuffer_shard_number(siirl_args: SiiRLArguments) -> int:
    assert siirl_args.data.train_batch_size % siirl_args.trainer.nnodes == 0, f"Config Error: train_batch_size ({siirl_args.data.train_batch_size}) must be divisible by nnodes ({siirl_args.trainer.nnodes}). Please adjust your configuration."

    batch_size_per_node = siirl_args.data.train_batch_size // siirl_args.trainer.nnodes
    intermediate_value = batch_size_per_node * siirl_args.actor_rollout_ref.rollout.n
    SHARDING_FACTOR = 8
    assert intermediate_value % SHARDING_FACTOR == 0, f"Config Error: The result of '(train_batch_size / nnodes) * rollout_n' ({intermediate_value}) must be divisible by {SHARDING_FACTOR}. Please adjust your configuration."
    databuffer_number = ((siirl_args.data.train_batch_size // siirl_args.trainer.nnodes) * siirl_args.actor_rollout_ref.rollout.n) // 8
    databuffer_number = min(databuffer_number, siirl_args.trainer.nnodes)
    return databuffer_number


@ray.remote(num_cpus=MAIN_RUNNER_CPU_RESERVATION)
class MainRunner:
    """
    A Ray actor responsible for orchestrating the entire RL training workflow.

    This actor handles loading configurations, scheduling task graphs, initializing
    process groups, and launching the distributed Ray trainers. Isolating this
    orchestration logic in a dedicated actor ensures the main process remains clean
    and that the setup process is managed within the Ray cluster.
    """

    def run(self, siirl_args: SiiRLArguments) -> None:
        """
        Executes the main training workflow.

        Args:
            siirl_args: A SiiRLArguments object containing all parsed configurations.
        """
        set_basic_config()
        from loguru import logger

        logger.info("MainRunner started. Beginning workflow setup...")
        start_time = time.time()

        # 1. Init DataBuffer
        logger.info(f"Init DataBuffer with sharding number: {siirl_args.trainer.nnodes}")
        databuffer_number = get_databuffer_shard_number(siirl_args)
        data_buffer_handlers = init_data_buffer(databuffer_number)

        # 2. Load and configure the workflow task graph (DAG)
        logger.info("Loading training pipeline...")
        workerflow_taskgraph = load_pipeline(siirl_args)
        update_task_graph_node_configs(workerflow_taskgraph, siirl_args)
        display_node_config(workerflow_taskgraph)

        # 3. Schedule the task graph across available resources
        logger.info("Scheduling tasks across nodes and GPUs...")
        total_workers = siirl_args.trainer.nnodes * siirl_args.trainer.n_gpus_per_node
        task_scheduler = TaskScheduler(siirl_args.trainer.nnodes, siirl_args.trainer.n_gpus_per_node)
        rank_taskgraph_mapping = task_scheduler.schedule_and_assign_tasks([workerflow_taskgraph])
        log_schedule_assignments(rank_taskgraph_mapping, total_workers)
        unique_graphs_map = task_scheduler.get_unique_assigned_task_graphs()

        # 4. Create and configure process groups for communication
        logger.info("Initializing process groups for distributed communication...")
        process_group_manager = ProcessGroupManager(total_workers, rank_taskgraph_mapping)
        log_process_group_manager_details(process_group_manager, log_level="debug")
        # set process_group info into env for inference_actor
        inference_process_group = []
        inference_groups = process_group_manager.node_type_process_group_mapping["MODEL_INFERENCE"]
        for group_name in inference_groups:
            inference_process_group.append(process_group_manager.process_group_spec[group_name])
        os.environ["DGA_PROCESS_GROUP"] = str(inference_process_group)
        # 6. Initialize the main trainer
        logger.info("Initializing RayTrainer...")
        trainer = RayTrainer(
            config=siirl_args,
            process_group_manager=process_group_manager,
            rank_taskgraph_mapping=rank_taskgraph_mapping,
            unique_graphs_map=unique_graphs_map,
            data_buffer_handles=data_buffer_handlers,  # Placeholder for DataBuffer
            device_name=siirl_args.trainer.device,
        )

        # 7. Initialize and start DAGWorkers
        logger.info("Initializing and starting DAG workers...")
        trainer.init_workers()
        trainer.start_workers()

        setup_duration = time.time() - start_time
        logger.info(f"Workflow setup and worker launch complete. Time cost: {setup_duration:.2f}s")


def main() -> None:
    """
    Main entry point for launching the PPO DAG training job.

    This function initializes Ray, parses configurations using Hydra, and
    starts the MainRunner actor to orchestrate the distributed training workflow.

    Args:
        siirl_config: The configuration object provided by Hydra.
    """
    # Import logger locally to avoid Ray serialization issues
    from loguru import logger

    start_time = time.time()

    # Initialize Ray cluster if not already running
    if not ray.is_initialized():
        logger.info("Initializing local Ray cluster...")
        ray.init(runtime_env={"env_vars": RAY_RUNTIME_ENV_VARS}, num_cpus=None)
    logger.success(f"Ray is initialized. Time cost: {(time.time() - start_time) * 1000:.2f} ms")

    # Parse the complete configuration into a structured object
    siirl_args = parse_config()
    log_dict_formatted(siirl_args.to_dict(), "SiiRLArguments")

    # Launch the main orchestration actor and wait for it to complete.
    logger.info("Starting MainRunner actor to orchestrate the job.")
    runner = MainRunner.remote()
    # This is a blocking call that waits for the remote `run` method to finish.
    ray.get(runner.run.remote(siirl_args))

    logger.success("MainRunner has completed its execution. Shutting down.")


if __name__ == "__main__":
    main()
