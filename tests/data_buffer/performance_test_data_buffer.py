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
import time
import ray
import torch
import numpy as np
from tensordict import TensorDict
from typing import List
import datetime
import random
import uuid

# Make sure the import path is correct based on your project structure
from siirl.data_coordinator.data_buffer import init_data_coordinator
from siirl.data_coordinator.sample import SampleInfo

# ====================================================================
# Performance Test Configuration for New Architecture
# ====================================================================
# --- Data Generation Parameters ---
# Total number of samples to generate (aligned with old test: 200 items * 8 batch/item = 1600 samples)
TOTAL_SAMPLES = 1600
# Batch size within each sample (usually 1, simulating a single trajectory)
BATCH_SIZE_PER_SAMPLE = 1
# Sequence length 
SEQ_LEN = 1024
# Embedding dimension 
EMBED_DIM = 1024

# --- Workload Parameters ---
# Number of concurrent RolloutWorkers to simulate data production
NUM_PRODUCERS = 8
# Batch size for a simulated Trainer to get at once
TRAINER_BATCH_SIZE = 128
# Number of distributed DataBuffers (should equal NUM_PRODUCERS to simulate each worker having its own local buffer)
NUM_BUFFERS = NUM_PRODUCERS


def log_with_time(message: str):
    """Prints a message with a timestamp and flushes the output."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{now}] {message}", flush=True)


def create_mock_sample(item_idx: int) -> TensorDict:
    """Creates a mock sample (TensorDict) for testing."""
    tensor_data = {
        "input_ids": torch.randint(0, 32000, (BATCH_SIZE_PER_SAMPLE, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH_SIZE_PER_SAMPLE, SEQ_LEN, dtype=torch.long),
        "hidden_states": torch.randn(BATCH_SIZE_PER_SAMPLE, SEQ_LEN, EMBED_DIM),
    }
    return TensorDict(tensor_data, batch_size=[BATCH_SIZE_PER_SAMPLE])


async def producer_task(
    producer_id: int, 
    coordinator: ray.actor.ActorHandle, 
    num_samples_to_produce: int
):
    """Simulates the behavior of a RolloutWorker: produce data, store it locally, and register it globally."""
    for i in range(num_samples_to_produce):
        sample_data = create_mock_sample(producer_id * num_samples_to_produce + i)
        
        # 1. Store the sample data in the Ray object store of the current node
        sample_ref = ray.put(sample_data)
        
        # 2. Create metadata
        sample_info = SampleInfo(
            agent_group=producer_id,
            sum_tokens=SEQ_LEN,
            prompt_length=SEQ_LEN,
            response_length=0, # Assume response is empty in the producer phase
            uid=uuid.uuid4().int # Generate a unique integer ID
        )
        
        # 3. Register the metadata and reference with the global DataCoordinator
        #    The DataCoordinator will automatically handle holding the reference locally
        await coordinator.put.remote(sample_info, sample_ref)


async def main():
    """Main performance test function."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="error")

    log_with_time("=" * 80)
    log_with_time("      New DataCoordinator Architecture - Performance Benchmark")
    log_with_time("=" * 80)
    log_with_time(f"Configuration:")
    log_with_time(f"  - Total Samples to Generate: {TOTAL_SAMPLES}")
    log_with_time(f"  - Concurrent Producers (RolloutWorkers): {NUM_PRODUCERS}")
    log_with_time(f"  - Trainer Batch Size: {TRAINER_BATCH_SIZE}")
    log_with_time(f"  - Distributed Buffers: {NUM_BUFFERS}")
    log_with_time("-" * 80)

    # 1. Initialize the data coordination system
    # For single-machine testing, force_local=True must be set to avoid waiting for multiple nodes
    coordinator = init_data_coordinator(NUM_BUFFERS, force_local=True)
    log_with_time(f"✅ Data system initialized with 1 Coordinator.")

    # 2. Test concurrent Producer (RolloutWorker) performance
    log_with_time("\n--- Testing Concurrent Producer (put) Performance ---")
    
    samples_per_producer = TOTAL_SAMPLES // NUM_PRODUCERS
    if TOTAL_SAMPLES % NUM_PRODUCERS != 0:
        log_with_time(f"Warning: Total samples not evenly divisible by producers. Some producers will generate more samples.")
    
    start_time = time.perf_counter()
    
    producer_tasks = []
    for i in range(NUM_PRODUCERS):
        # The RolloutWorker now only needs to interact with the Coordinator
        num_to_produce = samples_per_producer + (1 if i < TOTAL_SAMPLES % NUM_PRODUCERS else 0)
        if num_to_produce > 0:
            task = producer_task(i, coordinator, num_to_produce)
            producer_tasks.append(task)
            
    await asyncio.gather(*producer_tasks)
    end_time = time.perf_counter()

    total_put_time = end_time - start_time
    put_throughput = TOTAL_SAMPLES / total_put_time
    avg_put_latency = (total_put_time / TOTAL_SAMPLES) * 1000  # ms per sample

    log_with_time(f"  - Total time to produce and register {TOTAL_SAMPLES} samples: {total_put_time:.4f} seconds")
    log_with_time(f"  - Producer Throughput: {put_throughput:.2f} samples/sec")
    log_with_time(f"  - Average Producer Latency: {avg_put_latency:.4f} ms/sample")
    
    # Verify that all data has been stored in the Coordinator
    queue_size = await coordinator.get_valid_size.remote()
    assert queue_size == TOTAL_SAMPLES, f"Coordinator size mismatch! Expected {TOTAL_SAMPLES}, got {queue_size}"
    log_with_time("✅ All samples registered in Coordinator.")

    # 3. Test Consumer (Trainer) performance
    log_with_time("\n--- Testing Consumer (get_batch) Performance ---")
    num_batches_to_get = TOTAL_SAMPLES // TRAINER_BATCH_SIZE
    log_with_time(f"Simulating a trainer fetching {num_batches_to_get} batches of size {TRAINER_BATCH_SIZE}.")
    
    consumer_start_time = time.perf_counter()
    
    total_retrieved_samples = 0
    for _ in range(num_batches_to_get):
        # 1. Get a batch of sample references from the Coordinator (or the values directly due to Ray optimization)
        batch_refs_or_values = await coordinator.get_batch.remote(TRAINER_BATCH_SIZE)
        if not batch_refs_or_values:
            log_with_time("  - Coordinator returned empty batch, stopping consumer test.")
            break
            
        # 2. Differentiate between returned ObjectRefs and resolved values, and handle them accordingly
        resolved_batch = []
        refs_to_get = []
        for item in batch_refs_or_values:
            if isinstance(item, ray.ObjectRef):
                refs_to_get.append(item)
            else:
                # Ray might return the value directly because the caller is the owner
                resolved_batch.append(item)
        
        # Batch get all ObjectRefs that need to be resolved
        if refs_to_get:
            loop = asyncio.get_running_loop()
            resolved_from_refs = await loop.run_in_executor(None, ray.get, refs_to_get)
            resolved_batch.extend(resolved_from_refs)

        actual_data_batch = resolved_batch
        total_retrieved_samples += len(actual_data_batch)

    consumer_end_time = time.perf_counter()

    total_get_time = consumer_end_time - consumer_start_time
    get_throughput = total_retrieved_samples / total_get_time
    avg_batch_latency = (total_get_time / num_batches_to_get) * 1000 # ms per batch

    log_with_time(f"  - Total time for consumer to fetch {total_retrieved_samples} samples: {total_get_time:.4f} seconds")
    log_with_time(f"  - Consumer Throughput: {get_throughput:.2f} samples/sec")
    log_with_time(f"  - Average Batch Latency: {avg_batch_latency:.4f} ms/batch")
    
    assert total_retrieved_samples == num_batches_to_get * TRAINER_BATCH_SIZE, "Mismatch in retrieved sample count!"
    log_with_time("✅ Data integrity check passed for consumer results.")

    # 4. Clean up resources
    log_with_time("\n\n🧹 Cleaning up resources...")
    # We cannot access the data_buffers directly, but we can manage them indirectly
    # through the coordinator or directly via ray.kill.
    # For simplicity, we only kill the coordinator as it's the only handle we have.
    ray.kill(coordinator, no_restart=True)
    # Note: Placement group and detached actors might need manual cleanup if not managed properly.
    # For this test, shutting down ray is sufficient.
    ray.shutdown()
    log_with_time("=" * 80)
    log_with_time("               Benchmark Finished")
    log_with_time("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
