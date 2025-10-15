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
from typing import Dict, List, Optional, Tuple, Callable

import ray
from loguru import logger
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import time
from collections import deque
from siirl.data_coordinator.sample import SampleInfo


@ray.remote
class DataBuffer:
    """
    一个轻量级的、分布式的Actor，每个实例通常位于一个物理节点上。
    它的唯一职责是持有本地节点上由RolloutWorkers通过`ray.put`创建的样本对象引用(ObjectRef)，
    以防止它们被Ray的垃圾回收机制过早地清理掉。
    """
    def __init__(self, buffer_id: int):
        self.buffer_id = buffer_id
        # 只存储ObjectRef，不存储实际数据
        self._ref_store: List[ray.ObjectRef] = []
        logger.info(f"DataBuffer (ID: {self.buffer_id}) initialized on node {ray.get_runtime_context().get_node_id()}.")

    def put_ref(self, sample_ref: ray.ObjectRef):
        """接收并持有一个样本的ObjectRef。"""
        self._ref_store.append(sample_ref)

    def get_ref_count(self) -> int:
        """返回当前持有的引用数量。"""
        return len(self._ref_store)
    
    def clear(self):
        """清空所有已存储的引用。"""
        self._ref_store.clear()
        # 注意：这里清空列表后，如果DataCoordinator中也删除了对应的引用，
        # Ray GC将能够回收这些对象占用的内存。

    def __repr__(self) -> str:
        return f"<DataBuffer(id={self.buffer_id}, stored_ref_count={len(self._ref_store)})>"

    def get_node_id(self) -> str:
        """返回当前actor所在的节点ID。"""
        return ray.get_runtime_context().get_node_id()


@ray.remote
class DataCoordinator:
    """
    一个全局唯一的中央Actor，负责协调数据的生产者(RolloutWorkers)和消费者(Trainers)。
    它不存储实际的样本数据，只存储样本的元信息(SampleInfo)和对象引用(ObjectRef)。
    这使得它能够以极低的成本实现复杂的全局采样策略。
    """
    def __init__(self):
        # 使用双端队列存储元信息和引用的元组，实现高效的先进先出操作
        self._sample_queue: deque[Tuple[SampleInfo, ray.ObjectRef]] = deque()
        # 建立从 node_id 到 DataBuffer actor 句柄的映射
        self._buffer_map: Dict[str, ray.actor.ActorHandle] = {}
        self.lock = asyncio.Lock()
        logger.info("Global DataCoordinator initialized.")

    def register_buffers(self, buffer_info: Dict[str, ray.actor.ActorHandle]):
        """
        由初始化逻辑调用，用于注册所有DataBuffer及其所在的节点ID。
        """
        self._buffer_map = buffer_info
        logger.info(f"Registered {len(self._buffer_map)} DataBuffers.")

    async def put(self, sample_info: SampleInfo, sample_ref: ray.ObjectRef):
        """
        由RolloutWorker调用，用于注册一个新的样本引用及其元数据。
        该方法会自动将ObjectRef路由到其所在节点的DataBuffer进行持有。
        """
        # 1. 获取调用方所在的节点ID
        caller_node_id = ray.get_runtime_context().get_node_id()
        
        # 2. 找到对应节点的DataBuffer并委托其持有ref
        local_buffer = self._buffer_map.get(caller_node_id)
        if local_buffer:
            # 这是一个“发后即忘”的调用，无需等待
            local_buffer.put_ref.remote(sample_ref)
        else:
            logger.warning(f"No DataBuffer found for node {caller_node_id}. The sample reference will not be held, which may lead to premature garbage collection.")

        # 3. 将元数据和引用注册到全局队列
        async with self.lock:
            # 在这里可以实现更复杂的逻辑，例如根据priority插入到优先队列中
            self._sample_queue.append((sample_info, sample_ref))

    async def get_batch(self, batch_size: int, filter_plugin: Optional[Callable[[SampleInfo], bool]] = None) -> List[ray.ObjectRef]:
        """
        由Trainer调用，用于获取一批样本的ObjectRef。
        支持传入一个可选的过滤插件，用于实现自定义采样逻辑。
        """
        async with self.lock:
            # 无过滤插件，采用高效的FIFO
            if not filter_plugin:
                if len(self._sample_queue) < batch_size:
                    logger.warning(f"Coordinator queue size ({len(self._sample_queue)}) is less than requested batch size ({batch_size}). Returning empty list.")
                    return []
                
                # 高效的 O(batch_size) 实现，利用 deque 的 O(1) popleft
                batch_refs = []
                for _ in range(batch_size):
                    _, sample_ref = self._sample_queue.popleft()
                    batch_refs.append(sample_ref)
                return batch_refs

            # 有过滤插件，采用 O(N) 的筛选和重构
            else:
                # 1. 过滤过程不会消耗队列中的元素
                potential_items = [item for item in self._sample_queue if filter_plugin(item[0])]
                
                # 2. 检查是否有足够的样本
                if len(potential_items) < batch_size:
                    logger.warning(f"After filtering, coordinator has {len(potential_items)} samples, which is less than requested batch size ({batch_size}). Returning empty list.")
                    return []
                
                # 3. 从筛选后的列表中提取一个批次 (FIFO)
                batch_items_to_return = potential_items[:batch_size]
                batch_refs = [item[1] for item in batch_items_to_return]

                # 4. 从原始队列中高效地移除已选中的样本
                items_to_remove = set(batch_items_to_return)
                self._sample_queue = deque(item for item in self._sample_queue if item not in items_to_remove)
                
                return batch_refs

    async def get_valid_size(self) -> int:
        """返回当前队列中的样本数量。"""
        async with self.lock:
            return len(self._sample_queue)

    def __repr__(self) -> str:
        return f"<DataCoordinator(total_samples={len(self._sample_queue)})>"


# ====================================================================
# Initialization Logic
# ====================================================================

def init_data_system(num_buffers: int) -> ray.actor.ActorHandle:
    """
    初始化数据协调系统，包含一个全局DataCoordinator和多个分布式的DataBuffer。
    对用户只返回一个统一的DataCoordinator句柄。

    Args:
        num_buffers: 要创建的分布式DataBuffer实例的数量，通常等于节点数或GPU总数。

    Returns:
        DataCoordinator的Actor句柄。
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before calling init_data_system.")

    # 1. 创建或获取全局唯一的DataCoordinator
    # 使用全局命名空间来确保coordinator的唯一性
    coordinator_name = "global_data_coordinator"
    try:
        coordinator = ray.get_actor(coordinator_name)
        logger.info(f"Connected to existing DataCoordinator actor '{coordinator_name}'.")
    except ValueError:
        logger.info(f"Creating new DataCoordinator actor with global name '{coordinator_name}'.")
        coordinator = DataCoordinator.options(name=coordinator_name, lifetime="detached").remote()

    # 2. 等待并创建分布式的DataBuffer
    wait_timeout = 300  # seconds
    poll_interval = 2  # seconds
    start_time = time.time()
    
    logger.debug(f"Waiting for at least {num_buffers} nodes to be available for DataBuffers (timeout: {wait_timeout}s).")
    while time.time() - start_time < wait_timeout:
        num_available_nodes = len([node for node in ray.nodes() if node.get("Alive", False)])
        if num_available_nodes >= num_buffers:
            logger.success(f"Found {num_available_nodes} nodes. Proceeding to create placement group for {num_buffers} DataBuffers.")
            break
        logger.warning(f"Waiting for more nodes... Available: {num_available_nodes}/{num_buffers}. Retrying in {poll_interval}s.")
        time.sleep(poll_interval)
    else: # This else belongs to the while loop, it executes if the loop finishes without break
        num_available_nodes = len([node for node in ray.nodes() if node.get("Alive", False)])
        raise TimeoutError(f"Timed out after {wait_timeout}s. Cannot create {num_buffers} buffers with 'STRICT_SPREAD' strategy on {num_available_nodes} available nodes.")

    # 3. 使用Placement Group确保DataBuffer分布在不同节点
    bundles = [{"CPU": 1} for _ in range(num_buffers)]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    logger.debug(f"Waiting for placement group for {num_buffers} DataBuffers to be ready...")
    ray.get(pg.ready())
    logger.debug("Placement group is ready.")

    scheduling_strategy = PlacementGroupSchedulingStrategy(placement_group=pg)
    data_buffers = [DataBuffer.options(scheduling_strategy=scheduling_strategy).remote(buffer_id=i) for i in range(num_buffers)]
    
    # 获取每个buffer所在的节点ID
    buffer_nodes = ray.get([b.get_node_id.remote() for b in data_buffers])
    buffer_info = {node_id: buffer for node_id, buffer in zip(buffer_nodes, data_buffers)}

    # 4. 在Coordinator中注册所有DataBuffer
    coordinator.register_buffers.remote(buffer_info)

    logger.success(f"Successfully created and registered {len(data_buffers)} DataBuffer actors.")
    
    return coordinator
