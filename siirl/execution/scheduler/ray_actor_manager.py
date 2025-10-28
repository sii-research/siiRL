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
Manages a group of Ray actors for distributed workloads, handling their
creation, lifecycle, and communication.
"""

import os
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import asyncio

import ray
from loguru import logger
from ray.actor import ActorHandle
from ray.experimental.state.api import get_actor
from ray.util import list_named_actors

from siirl.utils.extras.device import get_device_name
from siirl.execution.scheduler.process_group_manager import ProcessGroupManager
from siirl.params import SiiRLArguments
from siirl.engine.base_worker import RayClassWithInitArgs, RayResourcePool, WorkerGroup, get_random_string, sort_placement_group_by_node_ip
from siirl.execution.dag import TaskGraph
from siirl.dag_worker.dagworker import DAGWorker
from siirl.execution.rollout_flow.multiturn.agent_loop import AgentLoopManager


class DistributedEnv(Enum):
    """Enumeration for distributed environment variable keys."""

    MASTER_ADDR = "MASTER_ADDR"
    MASTER_PORT = "MASTER_PORT"
    WORLD_SIZE = "WORLD_SIZE"
    RANK = "RANK"
    WG_PREFIX = "WG_PREFIX"
    WG_BACKEND = "WG_BACKEND"
    RAY_LOCAL_WORLD_SIZE = "RAY_LOCAL_WORLD_SIZE"
    RAY_LOCAL_RANK = "RAY_LOCAL_RANK"
    DGA_PROCESS_GROUP = "DGA_PROCESS_GROUP"


# --- Constants ---
ACTOR_STATE_ALIVE = "ALIVE"
REGISTER_CENTER_POLL_INTERVAL_S = 1
REGISTER_CENTER_LOG_INTERVAL_S = 30
RAY_BACKEND = "ray"


class RayActorManager(WorkerGroup):
    """
    Manages the lifecycle of a group of distributed Ray actors (workers).

    This class handles the creation of actors based on resource availability,
    assigns ranks, and sets up the necessary environment variables for
    distributed communication. It provides a unified interface to execute
    methods synchronously or asynchronously across all or specific workers.

    Attributes:
        worker_names (List[str]): A list of the generated names for each actor.
        master_address (str): The network address of the rank 0 worker.
        master_port (str): The network port of the rank 0 worker.
        workers (List[ActorHandle]): The list of Ray actor handles managed by this group.
        world_size (int): The total number of workers in the group.
    """

    def __init__(
        self,
        resource_pool: RayResourcePool,
        base_config: SiiRLArguments,
        process_manager: ProcessGroupManager,
        rank_taskgraph_mapping: Dict[int, "TaskGraph"],
        data_coordinator_handle: ActorHandle,
        bin_pack: bool = True,
        name_prefix: Optional[str] = None,
        ray_wait_register_center_timeout: int = 300,
        device_name="cuda",
        **kwargs,
    ) -> None:
        """
        Initializes the RayActorManager.

        Args:
            resource_pool: The pool of resources available for placing actors.
            base_config: Base configuration arguments for the workers.
            process_manager: Manager for the distributed process group.
            rank_taskgraph_mapping: Mapping of worker ranks to their task graphs.
            data_coordinator_handle: Handle to the central DataCoordinator actor.
            bin_pack: If True, use strict packing strategy for placement groups.
            name_prefix: A custom prefix for actor names. A random one is
                         generated if None.
            ray_wait_register_center_timeout: Seconds to wait for the rank 0
                                              registration actor to appear.
            **kwargs: Additional arguments for the base WorkerGroup.
        """
        super().__init__(resource_pool=resource_pool, **kwargs)

        self.name_prefix: str = get_random_string(length=6) if name_prefix is None else name_prefix
        self._ray_wait_register_center_timeout = ray_wait_register_center_timeout

        self._worker_names: List[str] = []
        self._world_size: int = resource_pool.world_size
        self._master_addr: Optional[str] = None
        self._master_port: Optional[str] = None

        self.base_config = base_config
        self.process_manager = process_manager
        self.rank_taskgraph_mapping = rank_taskgraph_mapping
        self.data_coordinator_handle = data_coordinator_handle
        self.device_name = device_name

        # Prepare the Ray actor class with its initial arguments.
        self.ray_actor_class = RayClassWithInitArgs(
            ray.remote(DAGWorker),
            config=self.base_config,
            process_group_manager=self.process_manager,
            taskgraph_mapping=self.rank_taskgraph_mapping,
            data_coordinator=self.data_coordinator_handle,
            device_name=self.device_name,
        )

        self._initialize_workers(resource_pool=resource_pool, bin_pack=bin_pack)

    def _initialize_workers(self, resource_pool: RayResourcePool, bin_pack: bool) -> None:
        """
        Creates and configures all worker actors based on the resource pool.

        This method orchestrates the creation of placement groups, iterates through
        them to launch actors with the correct rank and environment variables,
        and establishes the master address for distributed coordination.
        """
        strategy = "STRICT_PACK" if bin_pack else "PACK"
        placement_groups = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
        sorted_pgs = sort_placement_group_by_node_ip(placement_groups)

        num_gpus_per_worker = 1 / resource_pool.max_colocate_count
        local_world_size = resource_pool.store[0]
        rank = -1

        for pg_index, placement_group in enumerate(sorted_pgs):
            if local_world_size > placement_group.bundle_count:
                raise ValueError(f"Placement group for '{self.name_prefix}' has too few bundles ({placement_group.bundle_count}) to support the required local world size ({local_world_size}).")

            for local_rank in range(local_world_size):
                rank += 1
                worker = self._create_worker_actor(
                    rank=rank,
                    local_rank=local_rank,
                    local_world_size=local_world_size,
                    pg_index=pg_index,
                    placement_group=placement_group,
                    num_gpus_per_worker=num_gpus_per_worker,
                    use_gpu=resource_pool.use_gpu,
                    device_name=self.device_name,
                )
                self._workers.append(worker)

                if rank == 0:
                    # Rank 0 worker is special: it establishes the master
                    # address and port for the entire worker group.
                    self._master_addr, self._master_port = self._get_register_center_and_master_info()
        work_futures = self.map_async(method_name="init_graph")
        ray.get(work_futures)
        # only support single agent
        
    def _get_register_center_and_master_info(self) -> Tuple[str, str]:
        """
        Waits for the registration actor to be available and fetches the
        master address and port from it.

        Returns:
            A tuple containing the master address and master port.

        Raises:
            TimeoutError: If the registration actor cannot be found within the
                          configured timeout.
        """
        register_center_name = f"{self.name_prefix}_register_center"
        logger.info(f"Waiting for registration center actor: '{register_center_name}'...")
        start_time = time.time()

        while time.time() - start_time < self._ray_wait_register_center_timeout:
            try:
                # Use list_named_actors for a more robust check.
                if register_center_name in list_named_actors():
                    register_center_actor = ray.get_actor(register_center_name)
                    logger.success(f"Successfully connected to '{register_center_name}'.")
                    rank_zero_info = ray.get(register_center_actor.get_rank_zero_info.remote())

                    master_addr = rank_zero_info[DistributedEnv.MASTER_ADDR.value]
                    master_port = rank_zero_info[DistributedEnv.MASTER_PORT.value]
                    return master_addr, master_port
            except Exception as e:
                logger.warning(f"Attempt to get register center failed, will retry. Error: {e}")

            elapsed = int(time.time() - start_time)
            if elapsed > 0 and elapsed % REGISTER_CENTER_LOG_INTERVAL_S == 0:
                logger.warning(f"Still waiting for '{register_center_name}'. Elapsed: {elapsed}s / {self._ray_wait_register_center_timeout}s.")
            time.sleep(REGISTER_CENTER_POLL_INTERVAL_S)

        raise TimeoutError(
            f"Failed to get register_center_actor '{register_center_name}' "
            f"within {self._ray_wait_register_center_timeout} seconds. "
            f"Existing actors: {list_named_actors(all_namespaces=True)}. "
            "Ensure Ray resources from previous runs are cleaned up or "
            "increase 'trainer.ray_wait_register_center_timeout'."
        )

    def _create_worker_actor(
        self,
        rank: int,
        local_rank: int,
        local_world_size: int,
        pg_index: int,
        placement_group: "ray.util.placement_group.PlacementGroup",
        num_gpus_per_worker: float,
        use_gpu: bool,
        device_name: str,
    ) -> ActorHandle:
        """
        Configures and creates a single worker actor.

        Args:
            rank: The global rank of the worker.
            local_rank: The rank of the worker on its local node.
            local_world_size: The number of workers on the local node.
            pg_index: The index of the placement group being used.
            placement_group: The Ray placement group for this worker.
            num_gpus_per_worker: The number of GPUs to assign to the worker.
            use_gpu: A boolean indicating if the worker should use a GPU.
            device_name: The name of the device to use ("cuda" or "npu").

        Returns:
            The handle to the newly created Ray actor.
        """
        # --- 1. Prepare Environment Variables ---
        env_vars = {
            DistributedEnv.WORLD_SIZE.value: str(self.world_size),
            DistributedEnv.RANK.value: str(rank),
            DistributedEnv.WG_PREFIX.value: self.name_prefix,
            DistributedEnv.WG_BACKEND.value: RAY_BACKEND,
            DistributedEnv.RAY_LOCAL_WORLD_SIZE.value: str(local_world_size),
            DistributedEnv.RAY_LOCAL_RANK.value: str(local_rank),
            DistributedEnv.DGA_PROCESS_GROUP.value: os.environ.get("DGA_PROCESS_GROUP", ""),
        }
        if rank != 0:
            if not self._master_addr or not self._master_port:
                raise ConnectionError("Master address and port not set before creating non-zero rank workers.")
            env_vars[DistributedEnv.MASTER_ADDR.value] = self._master_addr
            env_vars[DistributedEnv.MASTER_PORT.value] = self._master_port

        # --- 2. Generate a unique and descriptive actor name ---
        base_class_repr = type(self.ray_actor_class.cls).__name__  # e.g., "ActorClass(DAGWorker)"
        match = re.search(r"ActorClass\(([^)]+)\)", base_class_repr)
        actor_class_name = match.group(1) if match else base_class_repr
        actor_name = f"{self.name_prefix}_{actor_class_name}_{pg_index}:{local_rank}"
        self._worker_names.append(actor_name)

        # --- 3. Set actor-specific options ---
        self.ray_actor_class.update_options(
            {
                "runtime_env": {"env_vars": env_vars},
                "name": actor_name,
            }
        )

        # --- 4. Create the actor ---
        logger.debug(f"Creating actor '{actor_name}' with rank {rank}.")
        worker = self.ray_actor_class(placement_group=placement_group, placement_group_bundle_idx=local_rank, use_gpu=use_gpu, num_gpus=num_gpus_per_worker, device_name=device_name)
        return worker

    def _is_worker_alive(self, worker: ActorHandle) -> bool:
        """
        Checks if a given worker actor is in the 'ALIVE' state.

        Note: This uses a Ray experimental API, which may change in the future.

        Args:
            worker: The actor handle to check.

        Returns:
            True if the worker is alive, False otherwise.
        """
        try:
            worker_state_dict = get_actor(worker._actor_id.hex())
            return worker_state_dict.get("state", "undefined") == ACTOR_STATE_ALIVE if worker_state_dict else False
        except Exception:
            return False

    def _invoke_on_worker(self, worker: ActorHandle, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invokes a method on a single remote worker actor."""
        remote_method = getattr(worker, method_name)
        return remote_method.remote(*args, **kwargs)

    def map_sync(self, method_name: str, *args: Any, **kwargs: Any) -> List[Any]:
        """
        Executes a method on all workers and waits for all results to complete.

        This is the synchronous (blocking) version of `map_async`.

        Args:
            method_name: The name of the method to execute.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            A list containing the results from each worker.
        """
        return ray.get(self.map_async(method_name, *args, **kwargs))

    def map_async(self, method_name: str, *args: Any, **kwargs: Any) -> List[ray.ObjectRef]:
        """
        Executes a method on all workers asynchronously.

        Special Behavior:
        If all positional and keyword arguments are lists of the same length
        as the number of workers, the arguments are "unzipped" and distributed.
        The i-th worker receives the i-th element from each argument list.
        Otherwise, all workers receive the same arguments.

        Args:
            method_name: The name of the method to execute.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            A list of ray.ObjectRef handles to the results.
        """
        num_workers = len(self._workers)

        # Check for the special argument-splitting case
        all_args_are_distributable = all(isinstance(arg, list) and len(arg) == num_workers for arg in args) and all(isinstance(val, list) and len(val) == num_workers for val in kwargs.values())

        if all_args_are_distributable and (args or kwargs):
            futures = []
            for i in range(num_workers):
                # Slice arguments for the i-th worker
                sliced_args = tuple(arg[i] for arg in args)
                sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                futures.append(self._invoke_on_worker(self._workers[i], method_name, *sliced_args, **sliced_kwargs))
            return futures

        # Default case: all workers get the same arguments
        return [self._invoke_on_worker(w, method_name, *args, **kwargs) for w in self._workers]

    def map(self, method_name: str, *args: Any, **kwargs: Any) -> List[ray.ObjectRef]:
        """
        Alias for `map_async`. Executes a method on all workers asynchronously.
        """
        return self.map_async(method_name, *args, **kwargs)

    @property
    def worker_names(self) -> List[str]:
        """Returns the names of all managed workers."""
        return self._worker_names

    @property
    def master_address(self) -> Optional[str]:
        """Returns the address of the master (rank 0) worker."""
        return self._master_addr

    @property
    def master_port(self) -> Optional[str]:
        """Returns the port of the master (rank 0) worker."""
        return self._master_port

    @property
    def workers(self) -> List[ActorHandle]:
        """Returns the list of Ray actor handles."""
        return self._workers

    @property
    def world_size(self) -> int:
        """Returns the total number of workers."""
        return self._world_size
