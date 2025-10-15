import asyncio
import time
import ray
import torch
import numpy as np
from tensordict import TensorDict
from typing import List
import datetime

# 根据你的项目结构，确保这里的导入路径是正确的
# 假设脚本从项目根目录运行
from siirl.data_coordinator.data_buffer import DataBuffer
from siirl.data_coordinator.protocol import DataProto

# ====================================================================
# Performance Test Configuration
# ====================================================================
# --- Data Generation Parameters ---
# 要放入buffer的并发数据项数量
# 原始参数 (1000, 16, 2048, 4096) 会产生约1TB的内存占用，导致测试失败。
# 当前参数产生的总数据量约为 6.4GB，这是一个更合理的基准测试负载。
NUM_ITEMS = 200
# 每个数据项内部的batch size
BATCH_SIZE_PER_ITEM = 8
# 序列长度
SEQ_LEN = 1024
# 特征维度
EMBED_DIM = 1024

# --- Workload Parameters ---
# 模拟并发读取数据的下游worker数量
NUM_GET_WORKERS = 8
# Buffer Actor的ID
BUFFER_ID = 99


def log_with_time(message: str):
    """Prints a message with a timestamp and flushes the output."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{now}] {message}", flush=True)


def create_mock_dataprote(item_idx: int) -> DataProto:
    """创建一个模拟的DataProto对象用于测试"""
    tensor_data = {
        "input_ids": torch.randint(0, 32000, (BATCH_SIZE_PER_ITEM, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH_SIZE_PER_ITEM, SEQ_LEN, dtype=torch.long),
        "hidden_states": torch.randn(BATCH_SIZE_PER_ITEM, SEQ_LEN, EMBED_DIM),
    }
    non_tensor_data = {
        "metadata_field": np.array([f"item_{item_idx}_{i}" for i in range(BATCH_SIZE_PER_ITEM)])
    }
    meta_info = {"source_id": f"generator_{item_idx % 10}"}

    td = TensorDict(tensor_data, batch_size=[BATCH_SIZE_PER_ITEM])
    return DataProto(batch=td, non_tensor_batch=non_tensor_data, meta_info=meta_info)


async def main():
    """主性能测试函数"""
    if not ray.is_initialized():
        ray.init(num_cpus=4, ignore_reinit_error=True, logging_level="error")

    log_with_time("=" * 80)
    log_with_time("      DataBuffer Performance Benchmark Script")
    log_with_time("=" * 80)
    log_with_time(f"Configuration:")
    log_with_time(f"  - Concurrent Items (put): {NUM_ITEMS}")
    log_with_time(f"  - Batch Size per Item:    {BATCH_SIZE_PER_ITEM}")
    log_with_time(f"  - Sequence Length:        {SEQ_LEN}")
    log_with_time(f"  - Embedding Dimension:    {EMBED_DIM}")
    log_with_time(f"  - Concurrent Workers (get): {NUM_GET_WORKERS}")
    log_with_time("-" * 80)

    # 1. 初始化DataBuffer Actor
    actor_name = f"PerfTestBuffer_{time.time()}"
    RemoteDataBuffer = ray.remote(DataBuffer)
    buffer = RemoteDataBuffer.options(name=actor_name, lifetime="detached").remote(buffer_id=BUFFER_ID)  # type: ignore
    # 确保actor已经启动
    await buffer.get_storage_size.remote()
    log_with_time(f"✅ DataBuffer Actor '{actor_name}' initialized.")

    # 2. 生成测试数据
    log_with_time("\n🔄 Generating mock data...")
    data_protos: List[DataProto] = [create_mock_dataprote(i) for i in range(NUM_ITEMS)]
    total_samples = NUM_ITEMS * BATCH_SIZE_PER_ITEM
    log_with_time(f"✅ Generated {len(data_protos)} DataProto items, with a total of {total_samples} samples.")

    # 3. 测试并发 put 性能
    log_with_time("\n--- Testing Concurrent `put` Performance ---")
    put_key = "perf_test_key"

    start_time = time.perf_counter()
    put_tasks = [buffer.put.remote(put_key, dp) for dp in data_protos]
    await asyncio.gather(*put_tasks)
    end_time = time.perf_counter()

    total_put_time = end_time - start_time
    put_throughput = NUM_ITEMS / total_put_time
    avg_put_latency = (total_put_time / NUM_ITEMS) * 1000  # ms

    log_with_time(f"  - Total time to put {NUM_ITEMS} items: {total_put_time:.4f} seconds")
    log_with_time(f"  - Throughput: {put_throughput:.2f} puts/sec")
    log_with_time(f"  - Average Latency: {avg_put_latency:.4f} ms/put")

    # 验证数据是否已全部存入
    stored_item = await buffer.pop.remote(put_key)
    # pop出来的是一个list
    assert isinstance(stored_item, list) and len(stored_item) == NUM_ITEMS
    # 再把它put回去，用于get测试
    for item in stored_item:
        await buffer.put.remote(put_key, item)

    # 4. 测试首次 get 性能 (冷缓存)
    log_with_time("\n--- Testing First `get` Performance (Cold Cache) ---")
    log_with_time("This measures concatenation and initial balancing plan computation.")

    start_time = time.perf_counter()
    # 模拟第一个worker发起请求
    _ = await buffer.get.remote(put_key, requesting_dag_worker_dp_rank=0, requesting_dag_worker_world_size=NUM_GET_WORKERS)
    end_time = time.perf_counter()

    first_get_time = end_time - start_time
    log_with_time(f"  - Time for the first `get` call: {first_get_time:.4f} seconds")

    # 5. 测试并发 get 性能 (热缓存)
    log_with_time("\n--- Testing Concurrent `get` Performance (Warm Cache) ---")
    log_with_time(f"Simulating {NUM_GET_WORKERS} workers fetching data simultaneously.")

    start_time = time.perf_counter()
    get_tasks = [
        buffer.get.remote(put_key, requesting_dag_worker_dp_rank=i, requesting_dag_worker_world_size=NUM_GET_WORKERS)
        for i in range(NUM_GET_WORKERS)
    ]
    results = await asyncio.gather(*get_tasks)
    end_time = time.perf_counter()

    total_get_time = end_time - start_time
    get_throughput = NUM_GET_WORKERS / total_get_time
    avg_get_latency = (total_get_time / NUM_GET_WORKERS) * 1000  # ms

    log_with_time(f"  - Total time for {NUM_GET_WORKERS} concurrent gets: {total_get_time:.4f} seconds")
    log_with_time(f"  - Throughput: {get_throughput:.2f} gets/sec")
    log_with_time(f"  - Average Latency: {avg_get_latency:.4f} ms/get")

    # 验证获取到的数据分片是否正确
    total_retrieved_samples = sum(res.batch.batch_size[0] for res in results)
    assert total_retrieved_samples == total_samples, \
        f"Mismatch in sample count! Expected {total_samples}, got {total_retrieved_samples}"
    log_with_time("✅ Data integrity check passed for `get` results.")

    # 6. 清理资源
    log_with_time("\n\n🧹 Cleaning up resources...")
    ray.kill(buffer, no_restart=True)
    ray.shutdown()
    log_with_time("=" * 80)
    log_with_time("               Benchmark Finished")
    log_with_time("=" * 80)


if __name__ == "__main__":
    # 使用 python -m tests.data_buffer.performance_test_data_buffer 从项目根目录运行
    asyncio.run(main())
