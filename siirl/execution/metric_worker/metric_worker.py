import ray
import asyncio

from ray.actor import ActorHandle

from typing import Optional, Any
from loguru import logger
from tensordict import TensorDict

from siirl.execution.metric_worker.utils import *
from siirl.utils.metrics.metric_utils import *

Special_Metric = {
    "graph_output_handling": MaxMetric, # because only rollout_tp 0 need to put_data_buffer
}

class MetricClient():
    def __init__(self, metric_worker:ActorHandle):
        self.metric_worker = metric_worker
        self.fut = []
        
    def stop(self):
        self.is_running = False
        ray.get(self.metric_worker.stop.remote())
    
    def submit_metirc(self, metrics:dict, world_size):
        self.fut.append(self.metric_worker.submit_metric.remote(metrics, world_size))

    
    def wait_submit(self):
        ray.get(self.fut)
        self.fut = []
    
    def wait_final_res(self):
        metrics = ray.get(self.metric_worker.wait_final_res.remote())
        return metrics
    
    def compute_local_data_metric(self, data:TensorDict, world_size: int):
        need_key = ["responses", "attention_mask", "attention_mask", "token_level_scores", \
            "token_level_rewards", "advantages", "returns"]
        if "values" in data:
            need_key.append("values")
        if  "response_mask" in data:
            need_key.append("response_mask")
        if "__num_turns__" in data:
            need_key.append("__num_turns__")
        need_data = data.select(*need_key)
        self.fut.append(self.metric_worker.submit_metric.remote(compute_data_metric(need_data,), world_size))

    def compute_local_throughout_metrics(self, data:TensorDict, timing_raw:dict, n_gpu: int, world_size: int):
        # n_gpu should be 1
        need_key = ["global_token_num"]
        need_data = data.select(*need_key)
        self.fut.append(self.metric_worker.submit_metric.remote(compute_throughout_metrics(need_data, timing_raw, n_gpu), world_size))
        
    def compute_local_timing_metrics(self, data: TensorDict, timing_raw: dict, world_size:int):
        self.fut.append(self.metric_worker.submit_metric.remote(compute_timing_metrics(data, timing_raw), world_size))
    
    def process_local_validation_metrics(self, data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], sample_turns: list[int], world_size: int):
        self.fut.append(self.metric_worker.submit_metric.remote(process_validation_metrics(data_sources, sample_inputs,infos_dict,sample_turns), world_size))
    

@ray.remote(num_cpus=1)
class MetricWorker:
    def __init__(self) -> None:
        self.metric_queue = asyncio.Queue()
        self.is_running = False
        self.process_task: Optional[asyncio.Task] = None
        self.step = 0
        self.final_metrics = {}
        self.working_metrics = {}
    
    async def start(self):
        """start metrics process loop"""
        if self.is_running:
            return
        
        self.is_running = True
        self.process_task = asyncio.create_task(self._process_metrics_loop())

    async def submit_metric(self, metric:dict, world_size: int):
        await self.metric_queue.put((metric, world_size))

        
    
    
    async def stop(self):
        self.is_running = False
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass
    
    async def compute_metric(self, metric_name, metrics):
        metric_func = MetricFunc(metric_name)
        res = metric_func(metrics)
        self.working_metrics.pop(metric_name)
        if metric_name.startswith("timing_s/"):
            metric_name = metric_name.replace("timing_s/", "perf/delta_time/")
        self.final_metrics[metric_name] = res
        
           
    async def parse_metric(self, metric_data: tuple):
        metric_dict, world_size = metric_data
        
        for key, value in metric_dict.items():
            metric = Metric(name=key, value=value, world_size=world_size)
            if key in self.working_metrics:
                metrics = self.working_metrics[key]
                metrics.append(metric)
                waiting_size = len(metrics)
                if waiting_size == world_size:
                    await self.compute_metric(key, metrics)
            else:
                self.working_metrics[key] = [metric]
            
    
    async def _process_metrics_loop(self):
        while self.is_running:
            metric_data = await self.metric_queue.get()            
            await self.parse_metric(metric_data)
            

    async def wait_final_res(self):
        # wait metric_qeueue until empty
        await self.stop()
        while self.metric_queue.qsize():      
            metric_data = await self.metric_queue.get()
            await self.parse_metric(metric_data)
        items = list(self.working_metrics.items())
        for key,value in items:
            await self.compute_metric(key, value)
        await self.start()
        final_metrics = self.final_metrics
        self.final_metrics = {}
        self.working_metrics = {}
        return final_metrics
        
        
    
    