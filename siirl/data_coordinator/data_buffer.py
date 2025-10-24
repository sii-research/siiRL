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
import heapq
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import ray
from loguru import logger
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import time
from asyncio import Queue
from siirl.data_coordinator.protocol import DataProto
from siirl.execution.rollout_flow.multi_agent.utils import AgentOutput
# ====================================================================
# Sequence Balancing Utility Functions
# ====================================================================


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

    # Enforce the guarantee that the data size is perfectly divisible.
    # If this ever fails, it indicates an unexpected issue in the data pipeline.
    assert num_items % k_partitions == 0, f"Data size ({num_items}) is not evenly divisible by the number of partitions ({k_partitions}). The system expects data to be perfectly divisible."

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

    # A temporary list to hold partitions that become full.
    full_partitions = []

    # 4. Iterate through sorted items and assign each to a non-full partition
    #    with the smallest current sum.
    for original_idx, length in indexed_lengths:
        # Find the smallest, non-full partition.
        # Pop from the heap until we find a partition that is not yet full.

        # This loop is guaranteed to find a non-full partition because the total
        # number of items equals the sum of all target sizes.
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
            else:
                # This partition is full. Add it to a temporary list to be
                # re-added to the heap later if necessary (though not strictly needed
                # with this logic, it's good practice for other variations).
                full_partitions.append((smallest_sum, smallest_idx))

    return partitions


# ====================================================================
# DataBuffer Class Definition
# ====================================================================


class DataBuffer:
    """A distributed, thread-safe data buffer for high-concurrency workloads.

    DataBuffer acts as a simple key-value store, designed to be controlled by an
    external orchestrator (e.g., a DAG worker). It features several key
    optimizations for performance in distributed machine learning pipelines.

    Key Optimizations:
    - Deferred Concatenation: `put` operations have minimal overhead. The expensive
      `concat` operation is deferred until the first `get` call.
    - Advanced Caching: Concatenated data and computed balancing plans are cached
      to eliminate redundant computations on subsequent `get` calls.
    - Fine-Grained Locking: Locks are scoped to the minimum critical section
      to maximize concurrency.
    - Double-Buffer Reset: Provides a zero-overhead `reset` from the caller's
      perspective by cleaning up the old buffer in a background task.
    """

    def __init__(self, buffer_id: int, max_workers: int = 2):
        """Initializes the DataBuffer instance.

        Args:
            buffer_id: A unique identifier for this buffer instance.
            max_workers: The maximum number of worker threads for background tasks.
        """
        self.buffer_id = buffer_id
        # The stored value can be one of:
        # 1. List[DataProto]: Uncached state, a list of data shards.
        # 2. Tuple[DataProto, Dict]: Cached state, a tuple containing the
        #    concatenated proto and a dictionary of balancing plans.
        # 3. ray.ObjectRef: A direct reference to a Ray object.
        self.storage: Dict[str, Union[ray.ObjectRef, List[DataProto], Tuple[DataProto, Dict]]] = {}

        self.active_storage = self.storage
        self.old_storage: Optional[Dict[str, Union[ray.ObjectRef, List[DataProto], Tuple[DataProto, Dict], str]]] = None
        
        self.storage_queue: Dict[str, Queue]
        self.active_storage_queue = self.storage
        self.old_storage_queue: Optional[Dict[str,  Queue]] = None
        
        self.cleanup_task: Optional[asyncio.Task] = None

        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"DataBuffer-{self.buffer_id}")

        logger.debug(f"DataBuffer actor (ID: {self.buffer_id}) initialized on node {ray.get_runtime_context().get_node_id()}.")

    async def put(self, key: str, value: Union[List[ray.ObjectRef], DataProto, str]) -> bool:
        """Atomically places a data item into the buffer.

        If a `put` occurs for a key that is already cached (from a previous `get`),
        the cache for that key is invalidated to ensure data consistency for
        future reads.

        Args:
            key: The globally unique key for the data.
            value: The data to store, either as a `DataProto` or a list
                   containing a single `ray.ObjectRef`.

        Returns:
            True if the operation was successful, False otherwise.
        """
        async with self.lock:
            is_wrapped_ref = isinstance(value, list) and len(value) == 1 and isinstance(value[0], ray.ObjectRef)
            if is_wrapped_ref:
                self.active_storage[key] = value[0]
                return True
            if isinstance(value, AgentOutput):
                if key not in self.active_storage_queue:
                    self.active_storage_queue[key] = Queue()
                await self.active_storage_queue[key].put(value)
                return True
            if isinstance(value, str):
                # used for env
                self.active_storage[key] = value
                return True
            if isinstance(value, DataProto):
                current_item = self.active_storage.get(key)
                if current_item is None:
                    # First item for this key.
                    self.active_storage[key] = [value]
                elif isinstance(current_item, list):
                    # Common case: append to an existing list of shards.
                    current_item.append(value)
                else:  # It's a cached item (tuple), so we must invalidate.
                    logger.warning(f"DataBuffer (ID: {self.buffer_id}): Received a 'put' for key '{key}' which has already been read and cached. Invalidating cache for this key.")
                    # Revert to a list to include the new data.
                    cached_proto, _ = current_item
                    self.active_storage[key] = [cached_proto, value]
                return True

            logger.error(f"DataBuffer (ID: {self.buffer_id}): Invalid type for 'value' in put for key '{key}': {type(value)}")
            return False
    
    async def get_queue(self, key: str) -> Union[List[AgentOutput], None]:
        '''
            get AgentOuput from queue
        '''
        item_container = None
        async with self.lock:
            if key in self.active_storage_queue:
                queue:Queue = self.active_storage_queue.get(key)
                while queue.qsize() > 0:
                    if item_container:
                        item_container.append(await queue.get())
                    else:
                        item_container = [await queue.get()]
        return item_container
        
    async def get(self, key: str, requesting_dag_worker_dp_rank: int = 0, requesting_dag_worker_world_size: int = 1) -> Union[DataProto, ray.ObjectRef, None]:
        """Retrieves a balanced data shard for a specific worker.

        This method implements a "compute-once, cache-result" strategy for both
        data concatenation and sequence-length balancing. It robustly finds the
        correct keys for sequence length calculation, even if they are prefixed.

        Args:
            key: The globally unique key for the data.
            requesting_dag_worker_dp_rank: The rank of the requesting worker
                within its data-parallel group.
            requesting_dag_worker_world_size: The total size of the worker's
                data-parallel group.

        Returns:
            A `DataProto` object containing the balanced data shard, a direct
            `ray.ObjectRef` if one was stored, or `None` if the key is not
            found or an error occurs.
        """
        final_proto: DataProto

        def _get_sequence_lengths(proto: DataProto) -> Optional[List[int]]:
            """Robustly finds sequence lengths with a priority-based fallback."""
            batch = proto.batch
            batch_keys = list(batch.keys())

            # Priority 1: Look for the standard 'attention_mask' key.
            if "attention_mask" in batch:
                logger.debug(f"DataBuffer (ID: {self.buffer_id}): Using 'attention_mask' for length calculation.")
                return batch["attention_mask"].sum(dim=-1).tolist()

            # Priority 2: Look for a prefixed attention mask.
            for k in batch_keys:
                if k.endswith("_attention_mask"):
                    logger.debug(f"DataBuffer (ID: {self.buffer_id}): Using prefixed key '{k}' for length calculation.")
                    return batch[k].sum(dim=-1).tolist()

            # Fallback 1: Use 'input_ids' shape.
            if "input_ids" in batch:
                logger.debug(f"DataBuffer (ID: {self.buffer_id}): 'attention_mask' not found. Falling back to 'input_ids' shape.")
                num_items = batch["input_ids"].shape[0]
                seq_len = batch["input_ids"].shape[1]
                return [seq_len] * num_items

            # Fallback 2: Use prefixed 'input_ids' shape.
            for k in batch_keys:
                if k.endswith("_input_ids"):
                    logger.debug(f"DataBuffer (ID: {self.buffer_id}): 'attention_mask' not found. Falling back to prefixed key '{k}' shape.")
                    num_items = batch[k].shape[0]
                    seq_len = batch[k].shape[1]
                    return [seq_len] * num_items

            # If all attempts fail, report a comprehensive error.
            logger.error(f"DataBuffer (ID: {self.buffer_id}): Cannot determine sequence length for key '{key}'. Neither 'attention_mask' nor 'input_ids' (or prefixed versions) were found. Available keys: {batch_keys}")
            return None

        def _log_balance_distribution(partitions: List[List[int]], lengths: List[int]):
            """Formats and logs the distribution of sequence lengths after balancing."""
            distribution_details = [f"Rank {i}: {len(p_indices)} items, total_len={sum(lengths[j] for j in p_indices)}" for i, p_indices in enumerate(partitions)]
            logger.debug(f"DataBuffer (ID: {self.buffer_id}): Balanced key '{key}' for world_size={requesting_dag_worker_world_size}. Distribution: {'; '.join(distribution_details)}")

        async with self.lock:
            item_container = self.active_storage.get(key)

        if item_container is None:
            return None
        if isinstance(item_container, ray.ObjectRef) :
            return item_container

        balanced_partitions = None

        if isinstance(item_container, list):  # First-time access: Uncached.
            final_proto = DataProto.concat(item_container)
            seqlen_list = _get_sequence_lengths(final_proto)
            if seqlen_list is None:
                return None  # Error already logged.

            balanced_partitions = get_seqlen_balanced_partitions_constrained_lpt(seqlen_list, k_partitions=requesting_dag_worker_world_size)
            _log_balance_distribution(balanced_partitions, seqlen_list)

            async with self.lock:
                cached_partitions_dict = {requesting_dag_worker_world_size: balanced_partitions}
                self.active_storage[key] = (final_proto, cached_partitions_dict)

        elif isinstance(item_container, tuple):  # Cached access.
            final_proto, cached_partitions_dict = item_container

            if requesting_dag_worker_world_size in cached_partitions_dict:  # Cache hit on plan.
                balanced_partitions = cached_partitions_dict[requesting_dag_worker_world_size]
            else:  # Cache miss on plan, must compute a new one.
                seqlen_list = _get_sequence_lengths(final_proto)
                if seqlen_list is None:
                    return None  # Error already logged.

                balanced_partitions = get_seqlen_balanced_partitions_constrained_lpt(seqlen_list, k_partitions=requesting_dag_worker_world_size, equal_size=True)
                _log_balance_distribution(balanced_partitions, seqlen_list)

                async with self.lock:
                    # Re-fetch and update atomically to avoid race conditions.
                    current_item = self.active_storage.get(key)
                    if isinstance(current_item, tuple):
                        _, current_dict = current_item
                        current_dict.update({requesting_dag_worker_world_size: balanced_partitions})
        else:
            logger.error(f"DataBuffer (ID: {self.buffer_id}): Stored item for key '{key}' has an unexpected type: {type(item_container)}")
            return None

        # --- Final step: Return the correct data subset using the plan.
        if final_proto is None or balanced_partitions is None:
            logger.error(f"DataBuffer (ID: {self.buffer_id}): Failed to obtain proto or balancing plan for key '{key}'.")
            return None

        if not (0 <= requesting_dag_worker_dp_rank < len(balanced_partitions)):
            logger.error(f"DataBuffer (ID: {self.buffer_id}): Rank {requesting_dag_worker_dp_rank} out of bounds for key '{key}'.")
            return None

        my_indices = balanced_partitions[requesting_dag_worker_dp_rank]
        return final_proto.subset(my_indices)

    async def pop(self, key: str) -> Union[List[DataProto], Tuple[DataProto, Dict], ray.ObjectRef, None]:
        """Atomically removes and returns the value for a given key."""
        async with self.lock:
            return self.active_storage.pop(key, None)

    async def batch_pop(self, keys: List[str]) -> Dict[str, Union[List[DataProto], Tuple[DataProto, Dict], ray.ObjectRef, None]]:
        """Atomically removes and returns values for multiple keys in a batch."""
        async with self.lock:
            if not keys:
                return {}
            results = {}
            for key in keys:
                results[key] = self.active_storage.pop(key, None)
            return results

    async def reset(self) -> None:
        """Atomically resets the buffer using a double-buffer pattern for zero-overhead cleanup."""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                logger.warning(f"DataBuffer (ID: {self.buffer_id}): Cancelled a pending cleanup task.")
        async with self.lock:
            self.old_storage = self.active_storage
            self.active_storage = {}
            self.old_storage_queue = self.active_storage_queue
            self.active_storage_queue = {}
        logger.debug(f"DataBuffer (ID: {self.buffer_id}): Switched to new empty storage. Old storage with {len(self.old_storage)} items is scheduled for cleanup.")
        self.cleanup_task = asyncio.create_task(self._cleanup_old_storage())

    async def _cleanup_old_storage(self) -> None:
        """Asynchronously clears the old storage buffer in the background."""
        if self.old_storage is None:
            return
        try:
            await asyncio.sleep(0)  # Yield control to the event loop.
            old_count = len(self.old_storage)
            self.old_storage.clear()
            self.old_storage_queue.clear()
            self.old_storage = None
            logger.debug(f"DataBuffer (ID: {self.buffer_id}): Background cleanup finished. Freed {old_count} items.")
        except asyncio.CancelledError:
            logger.warning(f"DataBuffer (ID: {self.buffer_id}): Cleanup task was cancelled during background execution.")
        except Exception as e:
            logger.error(f"DataBuffer (ID: {self.buffer_id}): Error during old storage cleanup: {e}", exc_info=True)

    async def get_all_keys(self) -> List[str]:
        """Returns a list of all keys currently in the active storage."""
        async with self.lock:
            return list(self.active_storage.keys())

    async def get_storage_size(self) -> int:
        """Returns the number of items currently in the active storage."""
        async with self.lock:
            return len(self.active_storage)

    def __repr__(self) -> str:
        return f"<DataBuffer(id={self.buffer_id}, stored_keys_count={len(self.active_storage)})>"

    def __del__(self):
        """Shuts down the thread pool when the DataBuffer is garbage collected."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)


def init_data_buffer(sharding_number: int, max_workers: int = 2) -> List["ray.actor.ActorHandle"]:
    """
    Initializes and returns a list of DataBuffer actors with a guaranteed
    one-actor-per-node distribution.

    This function uses a Ray Placement Group with the 'STRICT_SPREAD' strategy,
    ensuring each actor is scheduled on a unique physical node. The function
    will wait up to 60 seconds for enough nodes to become available before
    raising an error if the number of requested actors exceeds the number of nodes.

    Args:
        sharding_number: The number of DataBuffer actors to create.
        max_workers: The maximum number of workers for each DataBuffer actor.

    Returns:
        A list of actor handles for the created DataBuffer actors.

    Raises:
        RuntimeError: If Ray is not initialized.
        TimeoutError: If a sufficient number of nodes is not available within 60 seconds.
        Exception: For other errors during placement group or actor creation.
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before calling init_data_buffer.")

    # 1. Precondition Check with a 60-second wait loop to handle node startup delays
    wait_timeout = 300  # seconds
    poll_interval = 2  # seconds
    start_time = time.time()
    nodes_ready = False
    num_available_nodes = 0

    logger.debug(f"Waiting for at least {sharding_number} nodes to be available (timeout: {wait_timeout}s).")

    while time.time() - start_time < wait_timeout:
        alive_nodes = [node for node in ray.nodes() if node.get("Alive", False)]
        num_available_nodes = len(alive_nodes)

        if num_available_nodes >= sharding_number:
            logger.success(f"Sufficient nodes are available. Found {num_available_nodes} nodes. Proceeding to create placement group for {sharding_number} actors.")
            nodes_ready = True
            break
        else:
            logger.warning(f"Waiting for more nodes... Available: {num_available_nodes}/{sharding_number}. Retrying in {poll_interval} seconds.")
            time.sleep(poll_interval)

    if not nodes_ready:
        raise TimeoutError(f"Timed out after {wait_timeout}s waiting for nodes. Cannot satisfy 'STRICT_SPREAD' strategy: The requested number of buffers ({sharding_number}) is greater than the final number of available nodes ({num_available_nodes}).")

    # 2. Placement Group Definition with STRICT_SPREAD
    bundles = [{"CPU": 1} for _ in range(sharding_number)]
    pg = None

    try:
        logger.debug(f"Creating a 'STRICT_SPREAD' placement group for {sharding_number} DataBuffer actors.")
        pg = placement_group(bundles, strategy="STRICT_SPREAD")

        logger.debug("Waiting for placement group resources to be allocated...")
        ray.get(pg.ready())
        logger.debug("Placement group is ready. Initializing actors within the group...")

        scheduling_strategy = PlacementGroupSchedulingStrategy(placement_group=pg)

        # 3. Actor Creation within the Guaranteed Placement Group
        data_buffer_cls = ray.remote(DataBuffer)
        # max_concurrency may need to defined by agent num
        data_buffer_actors = [data_buffer_cls.options(scheduling_strategy=scheduling_strategy, num_cpus=1, max_concurrency=8).remote(buffer_id=i, max_workers=max_workers) for i in range(sharding_number)]
        logger.success("Successfully submitted creation tasks for all DataBuffer actors, each guaranteed to be on a unique node.")

    except Exception as e:
        logger.error(f"Failed to initialize data buffers with placement group: {e}", exc_info=True)
        if pg:
            logger.warning("Removing placement group due to an error during actor initialization.")
            ray.util.remove_placement_group(pg)
        raise e

    return data_buffer_actors
