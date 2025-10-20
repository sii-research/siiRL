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

import asyncio
from typing import Dict, List, Optional, Tuple, Callable, Any

import ray
import loguru
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import time
from collections import deque
from siirl.data_coordinator.sample import SampleInfo


@ray.remote
class DataBuffer:
    """
    A lightweight, distributed Actor, with each instance typically located on a physical node.
    Its sole responsibility is to hold object references (ObjectRef) of samples created by
    RolloutWorkers via `ray.put` on the local node, to prevent them from being prematurely
    garbage collected by Ray's GC.
    """
    def __init__(self, buffer_id: int):
        self.buffer_id = buffer_id
        # Stores only ObjectRefs, not the actual data
        self._ref_store: List[ray.ObjectRef] = []
        loguru.logger.info(f"DataBuffer (ID: {self.buffer_id}) initialized on node {ray.get_runtime_context().get_node_id()}.")

    def put_ref(self, sample_ref: ray.ObjectRef):
        """Receives and holds the ObjectRef of a sample."""
        self._ref_store.append(sample_ref)

    def get_ref_count(self) -> int:
        """Returns the current number of held references."""
        return len(self._ref_store)
    
    def clear(self):
        """Clears all stored references."""
        self._ref_store.clear()
        # Note: After clearing this list, if the corresponding references are also
        # removed from the DataCoordinator, Ray GC will be able to reclaim the
        # memory occupied by these objects.

    def __repr__(self) -> str:
        return f"<DataBuffer(id={self.buffer_id}, stored_ref_count={len(self._ref_store)})>"

    def get_node_id(self) -> str:
        """Returns the node ID where the current actor is located."""
        return ray.get_runtime_context().get_node_id()


@ray.remote
class DataCoordinator:
    """
    A globally unique central Actor responsible for coordinating data producers (RolloutWorkers)
    and consumers (Trainers). It does not store the actual sample data, only the sample
    metadata (SampleInfo) and object references (ObjectRef). This allows it to implement
    complex global sampling strategies at a very low cost.
    """
    def __init__(self):
        # Use a deque to store tuples of metadata and references for efficient FIFO operations
        self._sample_queue: deque[Tuple[SampleInfo, ray.ObjectRef]] = deque()
        # A map from node_id to a list of DataBuffer actor handles
        self._buffer_map: Dict[str, List[ray.actor.ActorHandle]] = {}
        self._put_counter = 0  # Used for round-robin buffer selection
        self.lock = asyncio.Lock()
        loguru.logger.info("Global DataCoordinator initialized.")

    def register_buffers(self, buffer_info: Dict[str, List[ray.actor.ActorHandle]]):
        """
        Called by the initialization logic to register all DataBuffers and their
        corresponding node IDs.
        """
        self._buffer_map = buffer_info
        for node_id, buffers in self._buffer_map.items():
            loguru.logger.info(f"Registered {len(buffers)} DataBuffers for node {node_id}.")

    async def put(self, sample_info: SampleInfo, sample_ref: Any):
        """
        Called by a RolloutWorker to register a new sample reference and its metadata.
        This method automatically routes the ObjectRef to a DataBuffer on its local
        node to be held.
        """
        # Due to Ray's small object optimization, an ObjectRef passed by the client
        # might be automatically resolved to its actual value. Here, we ensure that
        # we are always handling an ObjectRef.
        if not isinstance(sample_ref, ray.ObjectRef):
            #loguru.logger.warning(
            #    f"Coordinator.put received a value of type {type(sample_ref)} "
            #    "instead of an ObjectRef. This might be due to Ray's small object "
            #    "optimization. Automatically calling ray.put() to store a reference."
            #)
            sample_ref = ray.put(sample_ref)

        # 1. Get the node ID of the caller
        caller_node_id = ray.get_runtime_context().get_node_id()

        # 2. Inject the node ID into SampleInfo for subsequent filtering
        #    Only inject if node_id has not been manually set, to facilitate testing.
        if sample_info.node_id is None:
            sample_info.node_id = caller_node_id
        
        # 3. Find the list of DataBuffers on the corresponding node and delegate the ref holding
        local_buffers = self._buffer_map.get(caller_node_id)
        if local_buffers:
            # Use round-robin to distribute references evenly among the buffers on the node
            buffer_to_use = local_buffers[self._put_counter % len(local_buffers)]
            buffer_to_use.put_ref.remote(sample_ref)
            self._put_counter += 1
        else:
            loguru.logger.warning(f"No DataBuffer found for node {caller_node_id}. The sample reference will not be held, which may lead to premature garbage collection.")

        # 4. Register the metadata and reference to the global queue
        async with self.lock:
            # More complex logic can be implemented here, such as inserting into a
            # priority queue based on priority
            self._sample_queue.append((sample_info, sample_ref))

    async def get_batch(self, batch_size: int, filter_plugin: Optional[Callable[[SampleInfo], bool]] = None) -> List[ray.ObjectRef]:
        """
        Called by a Trainer to get a batch of sample ObjectRefs.
        Supports an optional filter plugin to implement custom sampling logic.
        """
        async with self.lock:
            # No filter plugin, use efficient FIFO
            if not filter_plugin:
                if len(self._sample_queue) < batch_size:
                    loguru.logger.warning(f"Coordinator queue size ({len(self._sample_queue)}) is less than requested batch size ({batch_size}). Returning empty list.")
                    return []
                
                # Efficient O(batch_size) implementation using deque's O(1) popleft
                batch_refs = []
                for _ in range(batch_size):
                    _, sample_ref = self._sample_queue.popleft()
                    batch_refs.append(sample_ref)
                return batch_refs

            # With filter plugin, use O(N) filtering and reconstruction
            else:
                # 1. The filtering process does not consume elements from the queue
                potential_items = [item for item in self._sample_queue if filter_plugin(item[0])]
                
                # 2. Check if there are enough samples
                if len(potential_items) < batch_size:
                    loguru.logger.warning(f"After filtering, coordinator has {len(potential_items)} samples, which is less than requested batch size ({batch_size}). Returning empty list.")
                    return []
                
                # 3. Extract a batch from the filtered list (FIFO)
                batch_items_to_return = potential_items[:batch_size]
                batch_refs = [item[1] for item in batch_items_to_return]

                # 4. Efficiently remove the selected items from the original queue
                # Use ObjectRef (guaranteed unique and hashable) to identify items for removal
                refs_to_remove = {item[1] for item in batch_items_to_return}
                self._sample_queue = deque(item for item in self._sample_queue if item[1] not in refs_to_remove)
                
                return batch_refs

    async def get_valid_size(self) -> int:
        """Returns the number of samples in the current queue."""
        async with self.lock:
            return len(self._sample_queue)

    def __repr__(self) -> str:
        return f"<DataCoordinator(total_samples={len(self._sample_queue)})>"


# ====================================================================
# Initialization Logic
# ====================================================================

def init_data_coordinator(num_buffers: int, force_local: bool = False) -> ray.actor.ActorHandle:
    """
    Initializes the data coordination system, which includes a global DataCoordinator
    and multiple distributed DataBuffers. Returns a single, unified DataCoordinator
    handle to the user.

    Args:
        num_buffers: The number of distributed DataBuffer instances to create,
                     usually equal to the number of nodes or total GPUs.
        force_local: If True, forces all Buffers to be created on the local node,
                     for single-machine testing.

    Returns:
        The Actor handle for the DataCoordinator.
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before calling init_data_coordinator.")

    # 1. Create or get the globally unique DataCoordinator
    # Use a global name to ensure the coordinator's uniqueness
    coordinator_name = "global_data_coordinator"
    try:
        coordinator = ray.get_actor(coordinator_name)
        loguru.logger.info(f"Connected to existing DataCoordinator actor '{coordinator_name}'.")
    except ValueError:
        loguru.logger.info(f"Creating new DataCoordinator actor with global name '{coordinator_name}'.")
        coordinator = DataCoordinator.options(name=coordinator_name, lifetime="detached").remote()

    if force_local:
        # --- Local testing mode ---
        loguru.logger.warning("force_local=True. Creating all DataBuffers on the local node for testing purposes.")
        data_buffers = [DataBuffer.remote(buffer_id=i) for i in range(num_buffers)]
        # In local mode, all buffers are on the same node, so put them all in one list
        local_node_id = ray.get_runtime_context().get_node_id()
        buffer_info = {local_node_id: data_buffers}
    else:
        # --- Distributed deployment mode ---
        # 2. Wait for and create the distributed DataBuffers
        wait_timeout = 300  # seconds
        poll_interval = 2  # seconds
        start_time = time.time()
        
        loguru.logger.debug(f"Waiting for at least {num_buffers} nodes to be available for DataBuffers (timeout: {wait_timeout}s).")
        while time.time() - start_time < wait_timeout:
            num_available_nodes = len([node for node in ray.nodes() if node.get("Alive", False)])
            if num_available_nodes >= num_buffers:
                loguru.logger.success(f"Found {num_available_nodes} nodes. Proceeding to create placement group for {num_buffers} DataBuffers.")
                break
            loguru.logger.warning(f"Waiting for more nodes... Available: {num_available_nodes}/{num_buffers}. Retrying in {poll_interval}s.")
            time.sleep(poll_interval)
        else: # This else belongs to the while loop, it executes if the loop finishes without break
            num_available_nodes = len([node for node in ray.nodes() if node.get("Alive", False)])
            raise TimeoutError(f"Timed out after {wait_timeout}s. Cannot create {num_buffers} buffers with 'STRICT_SPREAD' strategy on {num_available_nodes} available nodes.")

        # 3. Use a Placement Group to ensure DataBuffers are spread across different nodes
        bundles = [{"CPU": 1} for _ in range(num_buffers)]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        loguru.logger.debug(f"Waiting for placement group for {num_buffers} DataBuffers to be ready...")
        ray.get(pg.ready())
        loguru.logger.debug("Placement group is ready.")

        scheduling_strategy = PlacementGroupSchedulingStrategy(placement_group=pg)
        data_buffers = [DataBuffer.options(scheduling_strategy=scheduling_strategy).remote(buffer_id=i) for i in range(num_buffers)]
        
        # Get the node ID for each buffer
        buffer_nodes = ray.get([b.get_node_id.remote() for b in data_buffers])
        # Group buffers on the same node
        buffer_info: Dict[str, List[ray.actor.ActorHandle]] = {}
        for node_id, buffer in zip(buffer_nodes, data_buffers):
            if node_id not in buffer_info:
                buffer_info[node_id] = []
            buffer_info[node_id].append(buffer)
    
    # 4. Register all DataBuffers with the Coordinator
    # Use ray.get to wait for registration to complete, ensuring the map is
    # populated for subsequent operations
    ray.get(coordinator.register_buffers.remote(buffer_info))

    loguru.logger.success(f"Successfully created and registered {len(data_buffers)} DataBuffer actors.")
    
    return coordinator
