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


import ray
import asyncio

from ray.actor import ActorHandle

from typing import Optional, Any
from loguru import logger
from tensordict import TensorDict

from siirl.execution.metric_worker.utils import *
from siirl.utils.metrics.metric_utils import *

# Special metric configurations where specific aggregation logic is needed
# e.g., "graph_output_handling" uses MaxMetric since only rollout_tp 0 needs data buffer handling
Special_Metric = {
    "graph_output_handling": MaxMetric,
}

class MetricClient():
    """Client class for interacting with the MetricWorker actor
    
    Provides methods to submit metrics, wait for submissions to complete,
    and retrieve final aggregated metrics from the worker.
    """
    def __init__(self, metric_worker: ActorHandle):
        """Initialize MetricClient with a reference to the MetricWorker actor
        
        Args:
            metric_worker: Ray actor handle for the MetricWorker instance
        """
        self.metric_worker = metric_worker
        self.fut = []  # List to track pending metric submission futures
        
    def stop(self):
        """Stop the metric worker and terminate its processing loop"""
        self.is_running = False
        ray.get(self.metric_worker.stop.remote())
    
    def submit_metric(self, metrics: dict, world_size):
        """Submit a dictionary of metrics to the worker for aggregation
        
        Args:
            metrics: Dictionary containing metric names and values
            world_size: Total number of processes in the distributed system
        """
        self.fut.append(self.metric_worker.submit_metric.remote(metrics, world_size))

    
    def wait_submit(self):
        """Wait for all pending metric submissions to complete"""
        ray.get(self.fut)
        self.fut = []  # Clear the list after all futures are resolved
    
    def wait_final_res(self):
        """Retrieve the final aggregated metrics from the worker
        
        Returns:
            Dictionary of aggregated metrics
        """
        metrics = ray.get(self.metric_worker.wait_final_res.remote())
        return metrics
    
    def compute_local_data_metric(self, data: TensorDict, world_size: int):
        """Compute and submit metrics from local data
        
        Extracts necessary fields from the TensorDict, computes metrics,
        and submits them to the worker.
        
        Args:
            data: TensorDict containing the data to process
            world_size: Total number of processes in the distributed system
        """
        need_key = ["responses", "attention_mask", "token_level_scores",
            "token_level_rewards", "advantages", "returns"]
        # Add optional keys if present in data
        if "values" in data:
            need_key.append("values")
        if "response_mask" in data:
            need_key.append("response_mask")
        if "__num_turns__" in data:
            need_key.append("__num_turns__")
        
        need_data = data.select(*need_key)
        self.fut.append(self.metric_worker.submit_metric.remote(
            compute_data_metric(need_data), world_size)
        )

    def compute_local_throughout_metrics(self, data: TensorDict, timing_raw: dict, n_gpu: int, world_size: int):
        """Compute and submit throughput metrics
        
        Args:
            data: TensorDict containing relevant data (e.g., token counts)
            timing_raw: Dictionary containing raw timing data
            n_gpu: Number of GPUs used (should be 1)
            world_size: Total number of processes in the distributed system
        """
        need_key = ["global_token_num"]
        need_data = data.select(*need_key)
        self.fut.append(self.metric_worker.submit_metric.remote(
            compute_throughout_metrics(need_data, timing_raw, n_gpu), world_size)
        )
        
    def compute_local_timing_metrics(self, data: TensorDict, timing_raw: dict, world_size: int):
        """Compute and submit timing metrics
        
        Args:
            data: TensorDict containing relevant data
            timing_raw: Dictionary containing raw timing data
            world_size: Total number of processes in the distributed system
        """
        self.fut.append(self.metric_worker.submit_metric.remote(
            compute_timing_metrics(data, timing_raw), world_size)
        )
    
    def process_local_validation_metrics(self, data_sources: list[str], sample_inputs: list[str], 
                                        infos_dict: dict[str, list[Any]], sample_turns: list[int], world_size: int):
        """Process and submit validation metrics
        
        Args:
            data_sources: List of data source names
            sample_inputs: List of sample input strings
            infos_dict: Dictionary containing information about validation samples
            sample_turns: List of turn counts for each sample
            world_size: Total number of processes in the distributed system
        """
        self.fut.append(self.metric_worker.submit_metric.remote(
            process_validation_metrics(data_sources, sample_inputs, infos_dict, sample_turns), world_size)
        )
    

@ray.remote(num_cpus=1)
class MetricWorker:
    """Ray actor responsible for aggregating metrics from distributed processes
    
    Runs an asynchronous loop to process incoming metrics, aggregate them
    across all processes, and provide final results when requested.
    """
    def __init__(self) -> None:
        self.metric_queue = asyncio.Queue()  # Queue for incoming metric submissions
        self.is_running = False  # Flag to control the processing loop
        self.process_task: Optional[asyncio.Task] = None  # Task for the main processing loop
        self.step = 0  # Current step counter (not actively used in shown code)
        self.final_metrics = {}  # Aggregated final metrics
        self.working_metrics = {}  # Metrics currently being collected/aggregated
    
    async def start(self):
        """Start the metrics processing loop
        
        Initializes and starts the asynchronous loop that processes metrics
        from the queue.
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.process_task = asyncio.create_task(self._process_metrics_loop())

    async def submit_metric(self, metric: dict, world_size: int):
        """Submit a metric dictionary to the worker's processing queue
        
        Args:
            metric: Dictionary of metric names and values
            world_size: Total number of processes in the distributed system
        """
        await self.metric_queue.put((metric, world_size))

        
    
    async def stop(self):
        """Stop the metrics processing loop and clean up resources"""
        self.is_running = False
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass  # Expected when task is cancelled
    
    async def compute_metric(self, metric_name, metrics):
        """Compute the final aggregated value for a metric
        
        Uses the appropriate metric function to aggregate values from all processes.
        
        Args:
            metric_name: Name of the metric to compute
            metrics: List of Metric objects containing values from each process
        """
        metric_func = MetricFunc(metric_name)
        res = metric_func(metrics)
        self.working_metrics.pop(metric_name)  # Remove from working set after computation
        
        # Rename timing metrics for consistency in output
        if metric_name.startswith("timing_s/"):
            metric_name = metric_name.replace("timing_s/", "perf/delta_time/")
            
        self.final_metrics[metric_name] = res
        
           
    async def parse_metric(self, metric_data: tuple):
        """Process incoming metric data and aggregate when all processes have submitted
        
        Collects metric values from each process and triggers computation when
        all values (one per process) have been received.
        
        Args:
            metric_data: Tuple containing (metric_dict, world_size)
        """
        metric_dict, world_size = metric_data
        
        for key, value in metric_dict.items():
            metric = Metric(name=key, value=value, world_size=world_size)
            
            # Add to working metrics or compute if all values are collected
            if key in self.working_metrics:
                metrics = self.working_metrics[key]
                metrics.append(metric)
                # Check if we have received values from all processes
                if len(metrics) == world_size:
                    await self.compute_metric(key, metrics)
            else:
                self.working_metrics[key] = [metric]
            
    
    async def _process_metrics_loop(self):
        """Main loop for processing metrics from the queue
        
        Continuously retrieves and processes metric data while the worker is running.
        """
        while self.is_running:
            metric_data = await self.metric_queue.get()            
            await self.parse_metric(metric_data)
            

    async def wait_final_res(self):
        """Wait for all metrics to be processed and return the final results
        
        Ensures all remaining metrics in the queue are processed, computes any
        remaining aggregated values, and returns the final metrics.
        
        Returns:
            Dictionary of final aggregated metrics
        """
        await self.stop()
        
        # Process any remaining metrics in the queue
        while self.metric_queue.qsize():      
            metric_data = await self.metric_queue.get()
            await self.parse_metric(metric_data)
        
        # Compute any metrics still in working set
        items = list(self.working_metrics.items())
        for key, value in items:
            await self.compute_metric(key, value)
        
        # Restart the worker for potential future use
        await self.start()
        
        # Capture and reset metrics before returning
        final_metrics = self.final_metrics
        self.final_metrics = {}
        self.working_metrics = {}
        return final_metrics