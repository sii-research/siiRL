import asyncio
import time
import ray
import torch
import numpy as np
from tensordict import TensorDict
from typing import List, Tuple
import datetime
import uuid
import statistics

# Make sure the import path is correct based on your project structure
from siirl.data_coordinator.data_buffer import init_data_coordinator
from siirl.data_coordinator.sample import SampleInfo


# ====================================================================
# Performance Test Configuration
# ====================================================================
# --- Data Generation Parameters ---
TOTAL_SAMPLES = 256  # Total number of samples to generate for the detailed test
BATCH_SIZE_PER_SAMPLE = 1
SEQ_LEN = 1024
EMBED_DIM = 1024

# --- Workload Parameters ---
NUM_PRODUCERS = 4  # Number of concurrent producers to simulate
NUM_BUFFERS = NUM_PRODUCERS

# --- Batching Simulation Parameters ---
BATCH_SIZE_FOR_SIM = 64  # How many samples to group in our simulated batch


def log_with_time(message: str):
    """Prints a message with a timestamp and flushes the output."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{now}] {message}", flush=True)


def create_mock_sample() -> TensorDict:
    """Creates a mock sample (TensorDict) for testing."""
    tensor_data = {
        "input_ids": torch.randint(0, 32000, (BATCH_SIZE_PER_SAMPLE, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH_SIZE_PER_SAMPLE, SEQ_LEN, dtype=torch.long),
        "hidden_states": torch.randn(BATCH_SIZE_PER_SAMPLE, SEQ_LEN, EMBED_DIM),
    }
    return TensorDict(tensor_data, batch_size=[BATCH_SIZE_PER_SAMPLE])


async def producer_task_detailed_profile(
    producer_id: int, 
    coordinator: ray.actor.ActorHandle, 
    num_samples_to_produce: int
) -> Tuple[List[float], List[float]]:
    """
    Simulates a producer and records the detailed timings for 
    `ray.put` and `coordinator.put.remote`.
    """
    ray_put_timings = []
    coord_put_timings = []

    for _ in range(num_samples_to_produce):
        sample_data = create_mock_sample()
        
        # 1. Profile `ray.put` (Serialization + Local Object Store Write)
        start_put = time.perf_counter()
        sample_ref = ray.put(sample_data)
        end_put = time.perf_counter()
        ray_put_timings.append(end_put - start_put)
        
        sample_info = SampleInfo(uid=uuid.uuid4().int)
        
        # 2. Profile `coordinator.put.remote` (RPC Overhead + Remote Execution)
        start_coord = time.perf_counter()
        await coordinator.put.remote(sample_info, sample_ref)
        end_coord = time.perf_counter()
        coord_put_timings.append(end_coord - start_coord)
        
    return ray_put_timings, coord_put_timings


def analyze_timings(operation_name: str, timings: List[float]):
    """Analyzes and prints statistics for a list of timings."""
    if not timings:
        log_with_time(f"  - No data for {operation_name}.")
        return

    total_time = sum(timings)
    mean_time = statistics.mean(timings) * 1000  # ms
    std_dev = statistics.stdev(timings) * 1000 if len(timings) > 1 else 0.0  # ms
    
    log_with_time(f"  - Analysis for '{operation_name}':")
    log_with_time(f"    - Total Time: {total_time:.4f} seconds for {len(timings)} calls")
    log_with_time(f"    - Average Latency: {mean_time:.4f} ms/call")
    log_with_time(f"    - Standard Deviation: {std_dev:.4f} ms")


async def main():
    """Main performance test function."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="error")

    log_with_time("=" * 80)
    log_with_time("  Detailed Performance Profile: ray.put vs coordinator.put.remote")
    log_with_time("=" * 80)

    coordinator = init_data_coordinator(NUM_BUFFERS, force_local=True)
    
    # --- Part 1: Detailed Profile of the Current "Sample-by-Sample" Method ---
    log_with_time("\n--- Part 1: Profiling Current Sample-by-Sample Approach ---")
    samples_per_producer = TOTAL_SAMPLES // NUM_PRODUCERS
    
    producer_tasks = []
    for i in range(NUM_PRODUCERS):
        task = producer_task_detailed_profile(i, coordinator, samples_per_producer)
        producer_tasks.append(task)
            
    results = await asyncio.gather(*producer_tasks)
    
    all_ray_put_timings = [t for res in results for t in res[0]]
    all_coord_put_timings = [t for res in results for t in res[1]]

    analyze_timings("ray.put (Serialization)", all_ray_put_timings)
    print() # Spacer
    analyze_timings("coordinator.put.remote (RPC)", all_coord_put_timings)

    total_ray_put_time = sum(all_ray_put_timings)
    total_coord_put_time = sum(all_coord_put_timings)
    
    log_with_time("\n  - Conclusion for Part 1:")
    if total_ray_put_time > total_coord_put_time:
        ratio = total_ray_put_time / total_coord_put_time if total_coord_put_time > 0 else float('inf')
        log_with_time(f"    - Serialization (`ray.put`) is the dominant cost, taking {total_ray_put_time:.4f}s.")
        log_with_time(f"    - `ray.put` is {ratio:.2f}x slower than the coordinator RPC.")
    else:
        ratio = total_coord_put_time / total_ray_put_time if total_ray_put_time > 0 else float('inf')
        log_with_time(f"    - RPC (`coordinator.put.remote`) is the dominant cost, taking {total_coord_put_time:.4f}s.")
        log_with_time(f"    - The RPC is {ratio:.2f}x slower than serialization.")

    # --- Part 2: Local Benchmark of a Potential "Batched" Optimization ---
    log_with_time("\n--- Part 2: Simulating Batched `ray.put` Optimization Potential ---")
    
    # Create a batch of samples locally
    mock_batch = [create_mock_sample() for _ in range(BATCH_SIZE_FOR_SIM)]
    
    # Time a single, batched ray.put
    start_batch_put = time.perf_counter()
    ray.put(mock_batch)
    end_batch_put = time.perf_counter()
    batched_put_time = end_batch_put - start_batch_put
    
    # Get the average time for individual puts from our earlier test
    avg_single_put_time = statistics.mean(all_ray_put_timings)
    equivalent_individual_time = avg_single_put_time * BATCH_SIZE_FOR_SIM
    
    log_with_time(f"  - Time to `ray.put` {BATCH_SIZE_FOR_SIM} samples INDIVIDUALLY: {equivalent_individual_time:.4f} seconds (estimated from Part 1)")
    log_with_time(f"  - Time to `ray.put` {BATCH_SIZE_FOR_SIM} samples as a single BATCH: {batched_put_time:.4f} seconds")
    
    if batched_put_time > 0 :
        speedup_factor = equivalent_individual_time / batched_put_time
        log_with_time(f"  - Potential Speedup Factor: {speedup_factor:.2f}x")
    log_with_time("    - This demonstrates the potential performance gain from reducing serialization overhead.")

    # --- Cleanup ---
    log_with_time("\n\n🧹 Cleaning up resources...")
    ray.kill(coordinator, no_restart=True)
    ray.shutdown()
    log_with_time("=" * 80)
    log_with_time("               Benchmark Finished")
    log_with_time("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
