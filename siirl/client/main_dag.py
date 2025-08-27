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
from loguru import logger
from omegaconf import DictConfig

from siirl.scheduler.enums import AdvantageEstimator, AlgorithmType
from siirl.scheduler.graph_updater import display_node_config, update_task_graph_node_configs
from siirl.scheduler.launch import RayTrainer
from siirl.scheduler.process_group_manager import ProcessGroupManager, log_process_group_manager_details
from siirl.scheduler.task_scheduler import TaskScheduler, log_schedule_assignments
from siirl.utils.logger.logging_utils import set_basic_config
from siirl.utils.params import SiiRLArguments, log_dict_formatted, parse_config
from siirl.workers.dag import DAGConfigLoader
from siirl.workers.databuffer import init_data_buffer
from siirl.workers.environment import EnvironmentConfigLoader

# --- Constants ---
RAY_RUNTIME_ENV_VARS = {
    "TOKENIZERS_PARALLELISM": "true",
    "NCCL_DEBUG": "WARN",
    "VLLM_LOGGING_LEVEL": "WARN",
}

# The main runner is an orchestrator, not a heavy workload.
# Assigning it a full CPU is often wasteful. A fractional CPU is more efficient.
MAIN_RUNNER_CPU_RESERVATION = 5


def determine_workflow_config(self, siirl_args: SiiRLArguments) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if siirl_args.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return os.path.join(current_dir, "config/workflow_ppo.yaml")
    elif siirl_args.algorithm.adv_estimator in [
        AdvantageEstimator.GRPO,
        AdvantageEstimator.GRPO_PASSK,
        AdvantageEstimator.REINFORCE_PLUS_PLUS,
        AdvantageEstimator.REMAX,
        AdvantageEstimator.RLOO,
        AdvantageEstimator.OPO,
        AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        AdvantageEstimator.CPGD,
    ]:
        if siirl_args.algorithm.algorithm_name == AlgorithmType.DAPO.value:
            return os.path.join(current_dir, "config/workflow_dapo.yaml")
        return os.path.join(current_dir, "config/workflow_grpo.yaml")
    else:
        raise NotImplementedError


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

        # 1. Load environment configurations
        environment_handlers = None
        if siirl_args.dag.env_enable:
            assert siirl_args.dag.environment_path is not None, "Environment path must be provided when env_enable is True."
            logger.info(f"Loading environments from: {siirl_args.dag.environment_path}")
            environment_handlers = EnvironmentConfigLoader().load_environments_from_file(siirl_args.dag.environment_path)

        # 2. Init DataBuffer
        logger.info(f"Init DataBuffer with sharding number: {siirl_args.trainer.nnodes}")
        databuffer_number = get_databuffer_shard_number(siirl_args)
        data_buffer_handlers = init_data_buffer(databuffer_number)

        # 3. Load and configure the workerflow task graph (DAG)
        if siirl_args.dag.workflow_path is None:
            # If no workerflow path is provided, determine the default workflow config
            workflow_path = determine_workflow_config(self, siirl_args)
            logger.info(f"No workerflow path provided. Using {workflow_path} determined by adv_estimator: {siirl_args.algorithm.adv_estimator}")
        else:
            workflow_path = siirl_args.dag.workflow_path
        logger.info(f"Loading workerflow from: {siirl_args.dag.workflow_path}")
        if siirl_args.algorithm.adv_estimator == AdvantageEstimator.CPGD:
            siirl_args.actor_rollout_ref.actor.use_cpgd_loss = True
        workerflow_taskgraph = DAGConfigLoader.load_from_file(workflow_path)
        update_task_graph_node_configs(workerflow_taskgraph, siirl_args)
        display_node_config(workerflow_taskgraph)

        # 4. Schedule the task graph across available resources
        logger.info("Scheduling tasks across nodes and GPUs...")
        total_workers = siirl_args.trainer.nnodes * siirl_args.trainer.n_gpus_per_node
        task_scheduler = TaskScheduler(siirl_args.trainer.nnodes, siirl_args.trainer.n_gpus_per_node)
        rank_taskgraph_mapping = task_scheduler.schedule_and_assign_tasks([workerflow_taskgraph])
        log_schedule_assignments(rank_taskgraph_mapping, total_workers)
        unique_graphs_map = task_scheduler.get_unique_assigned_task_graphs()

        # 5. Create and configure process groups for communication
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
            environments_handles=environment_handlers,
            device_name=siirl_args.trainer.device,
        )

        # 7. Initialize and start DAGWorkers
        logger.info("Initializing and starting DAG workers...")
        trainer.init_workers()
        trainer.start_workers()

        setup_duration = time.time() - start_time
        logger.info(f"Workflow setup and worker launch complete. Time cost: {setup_duration:.2f}s")


@hydra.main(config_path="config", config_name="ppo_dag_trainer", version_base=None)
def main(siirl_config: DictConfig) -> None:
    """
    Main entry point for launching the PPO DAG training job.

    This function initializes Ray, parses configurations using Hydra, and
    starts the MainRunner actor to orchestrate the distributed training workflow.

    Args:
        siirl_config: The configuration object provided by Hydra.
    """
    start_time = time.time()

    # Initialize Ray cluster if not already running
    if not ray.is_initialized():
        logger.info("Initializing local Ray cluster...")
        ray.init(runtime_env={"env_vars": RAY_RUNTIME_ENV_VARS}, num_cpus=siirl_config.ray_init.num_cpus)
    logger.success(f"Ray is initialized. Time cost: {(time.time() - start_time) * 1000:.2f} ms")

    # Parse the complete configuration into a structured object
    siirl_args = parse_config(siirl_config)
    log_dict_formatted(siirl_args.to_dict(), "SiiRLArguments")

    # Launch the main orchestration actor and wait for it to complete.
    logger.info("Starting MainRunner actor to orchestrate the job.")
    runner = MainRunner.remote()
    # This is a blocking call that waits for the remote `run` method to finish.
    ray.get(runner.run.remote(siirl_args))

    logger.success("MainRunner has completed its execution. Shutting down.")


if __name__ == "__main__":
    main()
