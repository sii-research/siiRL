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
import heapq

import ray
import loguru
import time
from collections import deque
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from siirl.data_coordinator.sample import SampleInfo


def get_seqlen_balanced_partitions_constrained_lpt(seqlen_list: List[int], k_partitions: int) -> List[List[int]]:
    """Partitions items into k subsets of equal item count with balanced sums.

    This function implements a constrained version of the LPT (Longest
    Processing Time) heuristic. It strictly adheres to the constraint that each
    partition must have a nearly equal number of items, and then uses the LPT
    principle to balance the sum of sequence lengths within that constraint.
    This is the recommended approach when a fixed number of items per worker
    is a hard requirement.

    Args:
        seqlen_list: A list of integers representing the "size" of each item.
        k_partitions: The desired number of partitions.

    Returns:
        A list of lists, where each inner list contains the original indices
        of the items assigned to that partition. Each list will have a size of
        len(seqlen_list) // k or len(seqlen_list) // k + 1.
    """
    if k_partitions <= 0:
        raise ValueError("Number of partitions (k_partitions) must be positive.")
    num_items = len(seqlen_list)
    
    # Ensure the data size is perfectly divisible.
    # If this fails, it indicates an unexpected issue in the data pipeline.
    if num_items % k_partitions != 0:
        loguru.logger.warning(
            f"Data size ({num_items}) is not evenly divisible by the number of partitions ({k_partitions}). "
            f"This may lead to uneven partition sizes."
        )
    
    # 1. Sort items by length in descending order, preserving original indices.
    indexed_lengths = sorted(enumerate(seqlen_list), key=lambda x: x[1], reverse=True)
    
    # 2. Determine the target number of items for each partition.
    base_size = num_items // k_partitions
    rem = num_items % k_partitions
    partition_target_sizes = [base_size + 1] * rem + [base_size] * (k_partitions - rem)
    
    # 3. Initialize partitions and a min-heap to track partition sums.
    #    The heap stores tuples of (current_sum, partition_index).
    partitions = [[] for _ in range(k_partitions)]
    partition_heap = [(0, i) for i in range(k_partitions)]
    heapq.heapify(partition_heap)
    
    # 4. Iterate through sorted items and assign each to a non-full partition
    #    with the smallest current sum.
    for original_idx, length in indexed_lengths:
        # Find the smallest, non-full partition.
        # Pop from the heap until we find a partition that is not yet full.
        while True:
            smallest_sum, smallest_idx = heapq.heappop(partition_heap)
            
            # Check if the selected partition is already full.
            if len(partitions[smallest_idx]) < partition_target_sizes[smallest_idx]:
                # This partition is not full, so we can assign the item.
                partitions[smallest_idx].append(original_idx)
                new_sum = smallest_sum + length
                
                # If the partition is still not full after adding, push it back.
                if len(partitions[smallest_idx]) < partition_target_sizes[smallest_idx]:
                    heapq.heappush(partition_heap, (new_sum, smallest_idx))
                break
    
    return partitions


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

    def put_refs(self, sample_refs: List[ray.ObjectRef]):
        """Receives and holds a list of ObjectRefs."""
        self._ref_store.extend(sample_refs)

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

    async def put(self, sample_info: SampleInfo, sample_ref: Any, caller_node_id: Optional[str] = None):
        """
        Called by a RolloutWorker to register a new sample reference and its metadata.
        This method automatically routes the ObjectRef to a DataBuffer on its local
        node to be held.
        
        Args:
            sample_info: Metadata about the sample
            sample_ref: Ray ObjectRef or the actual sample data
            caller_node_id: The node ID of the caller. If None, will try to get it from
                          the runtime context (but this won't work correctly for remote calls)
        """
        # Due to Ray's small object optimization, an ObjectRef passed by the client
        # might be automatically resolved to its actual value. Here, we ensure that
        # we are always handling an ObjectRef.
        if not isinstance(sample_ref, ray.ObjectRef):
            sample_ref = ray.put(sample_ref)

        # 1. Get the node ID of the caller
        # Note: When called remotely, ray.get_runtime_context().get_node_id() returns
        # the node ID of the DataCoordinator actor, not the caller. So we require the
        # caller to pass their node_id explicitly.
        if caller_node_id is None:
            caller_node_id = ray.get_runtime_context().get_node_id()
            loguru.logger.warning(
                "DataCoordinator.put() called without caller_node_id. "
                f"Using DataCoordinator's node_id {caller_node_id[:16]}... which may be incorrect."
            )

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

    async def put_batch(self, sample_infos: List[SampleInfo], sample_refs: List[ray.ObjectRef], caller_node_id: Optional[str] = None):
        """
        Called by a worker to register a batch of new sample references and their metadata.
        This method routes the ObjectRefs to DataBuffers on their local nodes.
        
        Args:
            sample_infos: List of metadata for each sample
            sample_refs: List of Ray ObjectRefs
            caller_node_id: The node ID of the caller. If None, will try to get it from
                          the runtime context (but this won't work correctly for remote calls)
        """
        if not sample_refs:
            return

        # Get the node ID of the caller
        # Note: When called remotely, ray.get_runtime_context().get_node_id() returns
        # the node ID of the DataCoordinator actor, not the caller. So we require the
        # caller to pass their node_id explicitly.
        if caller_node_id is None:
            caller_node_id = ray.get_runtime_context().get_node_id()
            loguru.logger.warning(
                "DataCoordinator.put_batch() called without caller_node_id. "
                f"Using DataCoordinator's node_id {caller_node_id[:16]}... which may be incorrect."
            )

        for i in range(len(sample_infos)):
            if sample_infos[i].node_id is None:
                sample_infos[i].node_id = caller_node_id
        
        local_buffers = self._buffer_map.get(caller_node_id)
        if local_buffers:
            buffer_to_use = local_buffers[self._put_counter % len(local_buffers)]
            buffer_to_use.put_refs.remote(sample_refs)
            self._put_counter += 1
        else:
            loguru.logger.warning(f"No DataBuffer found for node {caller_node_id}. The sample reference will not be held, which may lead to premature garbage collection.")

        async with self.lock:
            self._sample_queue.extend(zip(sample_infos, sample_refs))

    async def get_batch(
        self, 
        batch_size: int, 
        filter_plugin: Optional[Callable[[SampleInfo], bool]] = None,
        balance_partitions: Optional[int] = None
    ) -> List[ray.ObjectRef]:
        """Called by a Trainer to get a batch of sample ObjectRefs.
        
        Supports an optional filter plugin to implement custom sampling logic, and an
        optional length balancing feature.
        
        Args:
            batch_size: The requested batch size.
            filter_plugin: An optional filter function for custom sampling logic.
            balance_partitions: If specified, the returned samples will be optimized
                              for even distribution among the given number of workers,
                              balancing the sum of sequence lengths for each worker.
                              Defaults to None (no length balancing).
        
        Returns:
            A list of sample ObjectRefs. If length balancing is enabled, the order
            of samples will be optimized.
        """
        async with self.lock:
            # No filter plugin, use efficient FIFO
            if not filter_plugin:
                if len(self._sample_queue) < batch_size:
                    loguru.logger.warning(f"Coordinator queue size ({len(self._sample_queue)}) is less than requested batch size ({batch_size}). Returning empty list.")
                    return []
                
                # Efficient O(batch_size) implementation using deque's O(1) popleft
                batch_items = []
                for _ in range(batch_size):
                    item = self._sample_queue.popleft()
                    batch_items.append(item)
                
                # Apply length balancing if requested
                if balance_partitions and balance_partitions > 1:
                    batch_refs = self._apply_length_balancing(batch_items, balance_partitions)
                else:
                    batch_refs = [item[1] for item in batch_items]
                
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

                # 4. Efficiently remove the selected items from the original queue
                # Use ObjectRef (guaranteed unique and hashable) to identify items for removal
                refs_to_remove = {item[1] for item in batch_items_to_return}
                self._sample_queue = deque(item for item in self._sample_queue if item[1] not in refs_to_remove)
                
                # Apply length balancing if requested
                if balance_partitions and balance_partitions > 1:
                    batch_refs = self._apply_length_balancing(batch_items_to_return, balance_partitions)
                else:
                    batch_refs = [item[1] for item in batch_items_to_return]
                
                return batch_refs
    
    def _apply_length_balancing(
        self, 
        batch_items: List[Tuple[SampleInfo, ray.ObjectRef]], 
        k_partitions: int
    ) -> List[ray.ObjectRef]:
        """Applies the length balancing algorithm to reorder samples.
        
        Uses the LPT (Longest Processing Time) algorithm to reorder samples so that
        if they are evenly distributed among k_partitions workers, the sum of
        sample lengths for each worker is as balanced as possible.
        
        Args:
            batch_items: A list of (SampleInfo, ObjectRef) tuples.
            k_partitions: The number of partitions (typically the DP size).
            
        Returns:
            A reordered list of ObjectRefs.
        """
        # Extract the length of each sample.
        # Use sum_tokens as the length metric (includes prompt + response).
        seqlen_list = [item[0].sum_tokens for item in batch_items]
        
        try:
            # Use the LPT algorithm to calculate the optimal partitions.
            partitions = get_seqlen_balanced_partitions_constrained_lpt(seqlen_list, k_partitions)
            
            # Reorder the samples based on the partitioning result.
            # Concatenate the partitions in order: [all samples from partition_0, all from partition_1, ...]
            reordered_refs = []
            for partition in partitions:
                for original_idx in partition:
                    reordered_refs.append(batch_items[original_idx][1])
            
            loguru.logger.debug(
                f"Applied length balancing: {len(batch_items)} samples reordered into {k_partitions} partitions"
            )
            
            return reordered_refs
            
        except Exception as e:
            loguru.logger.warning(
                f"Failed to apply length balancing: {e}. Falling back to original order."
            )
            # If length balancing fails, return the original order.
            return [item[1] for item in batch_items]

    async def get_all_by_filter(self, filter_plugin: Callable[[SampleInfo], bool]) -> List[ray.ObjectRef]:
        """
        Gets ALL sample ObjectRefs that match the filter plugin, consuming them from the queue.
        This is useful for pipeline-based data passing where a downstream stage needs the
        entire output of an upstream stage.
        """
        async with self.lock:
            # 1. Find all items that match the filter.
            items_to_return = [item for item in self._sample_queue if filter_plugin(item[0])]
            
            if not items_to_return:
                return []

            # 2. Extract their ObjectRefs.
            batch_refs = [item[1] for item in items_to_return]

            # 3. Efficiently remove the selected items from the original queue.
            refs_to_remove = {ref for ref in batch_refs}
            self._sample_queue = deque(item for item in self._sample_queue if item[1] not in refs_to_remove)
            
            return batch_refs

    async def get_valid_size(self) -> int:
        """Returns the number of samples in the current queue."""
        async with self.lock:
            return len(self._sample_queue)
    
    async def peek_source_dp_size(self, filter_plugin: Callable[[SampleInfo], bool]) -> Optional[int]:
        """
        Peek at the source_dp_size of matching samples without consuming them.
        
        Args:
            filter_plugin: Filter function to find matching samples
            
        Returns:
            The source_dp_size if found, None otherwise
        """
        async with self.lock:
            for sample_info, _ in self._sample_queue:
                if filter_plugin(sample_info):
                    source_dp_size = sample_info.dict_info.get('source_dp_size')
                    if source_dp_size is not None:
                        return source_dp_size
            return None

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
        alive_nodes = []
        while time.time() - start_time < wait_timeout:
            alive_nodes = [node for node in ray.nodes() if node.get("Alive", False)]
            if len(alive_nodes) >= num_buffers:
                loguru.logger.success(f"Found {len(alive_nodes)} nodes. Proceeding to create {num_buffers} DataBuffers.")
                break
            loguru.logger.warning(f"Waiting for more nodes... Available: {len(alive_nodes)}/{num_buffers}. Retrying in {poll_interval}s.")
            time.sleep(poll_interval)
        else: # This else belongs to the while loop, it executes if the loop finishes without break
            alive_nodes = [node for node in ray.nodes() if node.get("Alive", False)]
            raise TimeoutError(f"Timed out after {wait_timeout}s. Cannot create {num_buffers} buffers on {len(alive_nodes)} available nodes.")

        # 3. Explicitly create one DataBuffer per node to ensure proper distribution
        # Use NodeAffinitySchedulingStrategy to pin each buffer to a specific node
        data_buffers = []
        buffer_info: Dict[str, List[ray.actor.ActorHandle]] = {}
        
        for i in range(num_buffers):
            # Select a node in round-robin fashion
            target_node = alive_nodes[i % len(alive_nodes)]
            node_id = target_node.get("NodeID")
            
            # Create buffer with explicit node affinity scheduling
            scheduling_strategy = NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False  # Hard constraint: must be on this node
            )
            buffer = DataBuffer.options(
                scheduling_strategy=scheduling_strategy
            ).remote(buffer_id=i)
            data_buffers.append(buffer)
            
            loguru.logger.debug(f"Created DataBuffer {i} targeting node {node_id[:8]}...")
        
        # Get the actual node ID for each buffer to verify placement
        buffer_nodes = ray.get([b.get_node_id.remote() for b in data_buffers])
        
        # Group buffers by their actual node
        for node_id, buffer in zip(buffer_nodes, data_buffers):
            if node_id not in buffer_info:
                buffer_info[node_id] = []
            buffer_info[node_id].append(buffer)
        
        # Log the distribution
        loguru.logger.info(f"DataBuffer distribution across {len(buffer_info)} nodes:")
        for node_id, buffers in buffer_info.items():
            loguru.logger.info(f"  Node {node_id[:16]}...: {len(buffers)} buffer(s)")
    
    # 4. Register all DataBuffers with the Coordinator
    # Use ray.get to wait for registration to complete, ensuring the map is
    # populated for subsequent operations
    ray.get(coordinator.register_buffers.remote(buffer_info))

    loguru.logger.success(f"Successfully created and registered {len(data_buffers)} DataBuffer actors.")
    
    return coordinator
