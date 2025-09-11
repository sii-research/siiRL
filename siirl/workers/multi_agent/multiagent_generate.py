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
import torch
import asyncio
import random
import copy
import numpy as np
import torch.distributed  as dist

from asyncio import Queue
from siirl.models.loader import load_tokenizer
from tensordict import TensorDict
from uuid import uuid4
from .utils import AgentOutput, AgentOutputStatus
from typing import Dict, List, Any, Tuple, Optional, Union
from codetiming import Timer

from loguru import logger
from siirl.workers.rollout.sglang_rollout.async_sglang_server import AsyncSglangServer
from siirl.workers.fsdp_workers import ActorRolloutRefWorker
from siirl.workers.dag import TaskGraph, Node, NodeRole, NodeType
from siirl.workers.databuffer.protocol import DataProto
from siirl.utils.params import RolloutArguments, ActorRolloutRefArguments
from siirl.workers.dag_worker.mixins.utilities_mixin import UtilitiesMixin
from siirl.workers.dag_worker.dag_utils import remove_prefix_from_dataproto

class MultiAgentLoop(UtilitiesMixin):
    def __init__(self, dag, config: ActorRolloutRefArguments, node_workers:Dict, local_dag:TaskGraph, databuffer:List["ray.actor.ActorHandle"], placement_mode: str = 'colocate'):
        # dely import Dag after dagworker finish init
        from siirl.workers.dag_worker.dagworker import DAGWorker
        assert config.rollout.name == 'sglang', "MultiAgent only support sglang because vllm can't sleep in multi times"
        self.dag:DAGWorker = dag
        self.graph = local_dag
        self.placement_mode = placement_mode
        self.rollout_config = config.rollout
        self.internal_data_cache: Dict[str, Queue] = {}
        self.data_buffers = databuffer
        self.workers = node_workers
        self.max_model_len = int(self.rollout_config.max_model_len or self.rollout_config.prompt_length + self.rollout_config.response_length)
        self.max_model_len = min(self.max_model_len, self.rollout_config.prompt_length + self.rollout_config.response_length)
        self._parse_graph(local_dag)
        self.finish_generate = False 
        assert placement_mode == 'colocate' #in ['colocate', 'spread']
        if self.rollout_config.multi_turn.max_assistant_turns is None:
            self.rollout_config.multi_turn.max_assistant_turns = 1
    
        
    def _parse_graph(self, graph:TaskGraph):
        node_queue = graph.get_entry_nodes()
        visited_nodes = set()
        self.node_queue = []
        while node_queue:
            cur_node = node_queue.pop(0)
            if cur_node.node_role != NodeRole.ROLLOUT:
                break
            self.node_queue.append(cur_node)
            next_nodes = graph.get_downstream_nodes(cur_node.node_id)
            for n in next_nodes:
                if n.node_id not in visited_nodes:
                    node_queue.append(n)
        tail_node = self.node_queue[-1]
        tail_worker :ActorRolloutRefWorker = self.workers[self._generate_node_worker_key(tail_node)]
        self.tail_device_mesh = tail_worker.rollout.get_device_mesh()


    def _generate_node_worker_key(self, node: Node) -> str:
        """Generates a unique string key for a node's worker instance."""
        return f"{node.agent_group}_{node.node_type.value}_{node.node_role.value}"
    def node_if_local(self, node):
    # used in spread mode to judge agent node if in current gpu_worker
        pass

    def _preprocess(self, batch:DataProto) -> List[str]:
        '''Preprocess data from dataloader and return prompt to generate task'''
        n = 1 if batch.meta_info.get("validate", False) else self.rollout_config.n
        batch = batch.repeat(n, interleave=True)
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        reward_model = batch.non_tensor_batch['reward_model'] if self.rollout_config.agent.rewards_with_env else None
        raw_prompts = [p.tolist() for p in raw_prompts]
        if reward_model is None:
            ground_truth = [-1] * len(raw_prompts)
        else:
            ground_truth = [reward['ground_truth'] for reward in reward_model]
        return raw_prompts, ground_truth

    def _generate_key(self, cur_node: Node, next_node: Node, batch_id: int, global_bs: int = 0) :
        """
        Generates a unique key for routing data between nodes in the DAG, considering their Data Parallel (DP) configurations.
        The key ensures data is correctly routed from the current node to the next node, accounting for differences
        in DP sizes (e.g., when the current node's DP size is larger or smaller than the next node's).
        Args:
            cur_node: Current node in the DAG (source of the data).
            next_node: Next node in the DAG (destination for the data).
            batch_id: Index of the sample within the local batch (for fine-grained routing).
            global_bs: Global batch size (optional, used when DP sizes differ).
        Returns:
            str: Unique key formatted as "{next_node_id}_{destination_dp_rank}" for data routing.
        """
        cur_dp_size, cur_dp_rank, *_ = self.dag._get_node_dp_info(cur_node)
        next_dp_size, next_dp_rank, *_ = self.dag._get_node_dp_info(next_node)
        if cur_dp_size == cur_dp_size:
            return f"{next_node.node_id}_{next_dp_rank}"
        elif cur_dp_size > next_dp_size:
            assert cur_dp_size % next_dp_size == 0, f"dp size of {cur_node.node_id} should div by {next_node.node_id}"
            return f"{next_node.node_id}_{cur_dp_rank / next_dp_size}"
        else:
            assert next_dp_size % cur_dp_size == 0, f"dp size of {next_node.node_id} should div by {cur_node.node_id}"
            # todo(hujr): may not suitable in GRPO
            next_rank_range = next_dp_size / cur_dp_size
            next_bs_range = global_bs / next_dp_size
            next_rank = batch_id / next_bs_range + cur_dp_rank * next_rank_range
            return f"{next_node.node_id}_{next_rank}"
          
    async def async_put_data(self, key: str, value: Tuple[AgentOutput, str], source_dp_size: int, dest_dp_size: int, timing_raw: Dict[str, float]):
        """
        Asynchronously puts data into local cache or distributed data buffers, based on source and destination DP sizes.
        - Uses local cache when source and destination DP sizes match (no cross-DP communication needed).
        - Uses distributed buffers when DP sizes differ (requires cross-DP data sharing).

        Args:
            key: Unique key to identify the data (generated by `_generate_key`).
            value: Data to store (either an `AgentOutput` object or a string).
            source_dp_size: DP size of the source node.
            dest_dp_size: DP size of the destination node.
            timing_raw: Dictionary to track timing metrics (not yet implemented).
        """
        if  source_dp_size == dest_dp_size:
            if isinstance(value, AgentOutput):
                if key not in self.internal_data_cache:
                    self.internal_data_cache[key] = Queue()
                await self.internal_data_cache[key].put(value)
            elif isinstance(value, str):
                self.internal_data_cache[key] = value
            else:
                raise NotImplementedError("This Should not Happen")
        else:   
            # random save to databuffers
            buffer = random.choice(self.data_buffers)
            await buffer.put.remote(key, value)
            
    async def async_get_envdata(self, key: str, timing_raw: Dict[str, float]):
        """
        Asynchronously retrieves environment-related data (e.g., observations) from local cache or distributed buffers.
        Checks local cache first for efficiency; falls back to distributed buffers if not found.

        Args:
            key: Unique key identifying the environment data.
            timing_raw: Dictionary to track timing metrics (not yet implemented).

        Returns:
            The requested data (if found) or None (if not found).
        """
        data = None  
        if key in self.internal_data_cache:
            data = self.internal_data_cache.pop(key, None)
        else:
            tasks = [buffer.pop.remote(key) for buffer in self.data_buffers]
           
            temp_data = await asyncio.gather(*tasks)
            # temp_data = self.data_buffers.get(key) 
            data = [t for t in temp_data if t is not None]
        return data[0] if data else None
    
    async def async_get_data(self, key: str, timing_raw: Dict[str, float]):
        """
        Asynchronously retrieves generated data (e.g., AgentOutputs) from local cache or distributed buffers.
        
        Handles queue-based data in local cache (for multiple entries) and aggregates results from distributed buffers.

        Args:
            key: Unique key identifying the data (generated by `_generate_key`).
            timing_raw: Dictionary to track timing metrics (not yet implemented).

        Returns:
            List of retrieved data entries (or None if no data found).
        """    
        data = None  
        if key in self.internal_data_cache:
            queue:Queue = self.internal_data_cache.get(key)
            while queue.qsize() > 0:
                if data:
                    data.append(await queue.get())
                else:
                    data = [await queue.get()]
        else:
            tasks = [buffer.get_queue.remote(key) for buffer in self.data_buffers]
            temp_data = await asyncio.gather(*tasks)
            # temp_data = self.data_buffers.get(key) 
            data = [item for t in temp_data if t is not None for item in t]
            
        return data
    

    async def spread_task(self, cur_node, node_idx, batch_idx):
        ''' Not support now'''
        while True:
            key = self._generate_key(batch_idx, cur_node.dp_rank, cur_node.node_id)
            prompt,should_stop = self.databuffer.get(key)
            if multiturn > max_multiturn:
                return response, response_mask
            if should_stop:
                if node_idx != len(self.node_queue) - 1:
                    next_node = self.node_queue[node_idx + 1] if node_idx < len(self.node_queue) - 1  else self.node_queue[0]
                    next_key = self._generate_key(batch_idx, next_node.dp_rank, next_node.node_id)
                    self.databuffer.put(next_key, [[], should_stop]) 
                return response, response_mask
            response = await node_worker.rollout.generate(XXXX)
            prompt = prompt + response
            if cur_node.node_id == self.node_queue[-1]:
                # last agent need to interaction with env
                tool_response, rewards, should_stop = self.env.execute(prompt)
                prompt = prompt + tool_response
            next_node = self.node_queue[node_idx + 1] if node_idx < len(self.node_queue) - 1  else self.node_queue[0]
            next_key = self._generate_key(batch_idx, next_node.dp_rank, next_node.node_id)
            self.databuffer.put(next_key, [prompt, should_stop]) 
            multiturn = multiturn + 1
                
    async def generate_spread(self):
        ''' Not support now'''
        for i in range(len(self.node_queue)):
            cur_node = self.node_queue[i]
            if node_if_local(cur_node) is False:
                continue
            dp_rank,dp_size,tp_rank,tp_size = get_node_info(cur_node)
            node_worker = self.workers[self._generate_node_worker_key(cur_node)]
            # wakeup before generate
            node_worker.wake_up()
            if tp_rank == 0:
                bs = get_batch_size(cur_node)
                tasks = []
                for bs_idx in range(bs):
                    tasks.append(spread_task(cur_node, i, bs_idx))
                res = await asyncio.gather(*tasks)
            barrier()
            node_worker.sleep()
            return res
    async def check_colocate_running(self, finished_res: Dict, visited_agentoutputs: Dict):
        """
        Asynchronously checks whether the current worker should continue running (local_running status)
        by verifying if all tracked samples have been fully processed across all DAG nodes.
        Args:
            finished_res: Dictionary tracking finished samples. Key = node ID, Value = set of 
                        request IDs that have been fully processed for that node.
            visited_agentoutputs: Set (or dict-like) of request IDs representing samples the 
                                current worker has fetched/processed in the current cycle.

        Returns:
            bool: True if the worker should continue running (unprocessed samples remain), 
                False if all visited samples are finished (worker can stop).
        """
        finish = True
        for node in self.node_queue:
            if node.node_id in finished_res:
                if len(finished_res[node.node_id]) == len(visited_agentoutputs):
                    finish = False
                elif len(finished_res[node.node_id]) > len(visited_agentoutputs):
                    assert False, "This should not happen"
                else:
                    finish = True
            else:
                finish = True
        return finish
    
    async def colocate_task(self, agent_output:AgentOutput, agent_res:Dict,  finished_res: Dict, cur_node: Node, node_idx: int, sampling_params: Dict[str, Any], global_bs: int,  timing_raw: Dict[str, float]):
        """
        Asynchronous task for generate with a single `AgentOutput` in colocated mode.
        
        Handles end-to-end processing for one sample: environment observation fetching, prompt preprocessing,
        model generation, response postprocessing, environment step execution (if enabled), and data propagation
        to the next node in the DAG. Also tracks finished samples and updates result dictionaries.
        
        Args:
            agent_output: `AgentOutput` object containing the sample's prompt, metadata, and status.
            agent_res: Global dictionary to store processed `AgentOutput` results (key: node ID, value: request ID → AgentOutput).
            finished_res: Global dictionary to track fully processed samples (key: node ID, value: set of request IDs).
            cur_node: Current DAG node being processed (defines agent logic and environment config).
            node_idx: Index of `cur_node` in the DAG node queue (for next node lookup).
            sampling_params: Model sampling hyperparameters (e.g., temperature, top_p) for sequence generation.
            global_bs: Global batch size (DP size × local batch size) for consistent data routing.
            timing_raw: Dictionary to record raw timing metrics (e.g., preprocessing, generation, environment step latency).
        """
        cur_dp_size, cur_dp_rank, *_ = self.dag._get_node_dp_info(cur_node)
        node_worker:ActorRolloutRefWorker = self.workers[self._generate_node_worker_key(cur_node)]
        next_node = self.node_queue[node_idx + 1] if node_idx < len(self.node_queue) - 1  else self.node_queue[0]
        next_key = self._generate_key(cur_node, next_node, agent_output.batch_id, global_bs)
        next_dp_size, *_ = self.dag._get_node_dp_info(next_node)
        obs = None
        if agent_output.status !=  AgentOutputStatus.RUNNING:
            if cur_node.node_id not in finished_res:
                finished_res[cur_node.node_id] = set()
            
            finished_res[cur_node.node_id].add(agent_output.request_id)
            # pre agent use same rewards with last agent
            await self.async_put_data(next_key, agent_output, cur_dp_size, next_dp_size, timing_raw)
            return
        if cur_node.agent_options and cur_node.agent_options.obs_with_env:
            obs = await self.async_get_envdata(agent_output.request_id + f'_{cur_node.agent_group}', timing_raw)
        agent_output.original_prompt, agent_output.templated_prompt = cur_node.agent_process.apply_pre_process(prompt=agent_output.original_prompt, obs = obs)
        agent_output.templated_prompt = agent_output.templated_prompt[:self.rollout_config.prompt_length]

        response = await node_worker.rollout.generate(
            request_id=agent_output.request_id, prompt_ids=agent_output.templated_prompt, sampling_params=sampling_params
            )
        if len(response) == 0:
            # if response is None, padding response some prompt for training
            response = "<|im_end|>"
        agent_output.original_prompt, agent_output.templated_prompt, agent_output.response_mask \
            = cur_node.agent_process.apply_post_process(oridinal_prompt = agent_output.original_prompt, templated_prompt = agent_output.templated_prompt, response = response)     
                   
        # if have env
        if cur_node.agent_options and cur_node.agent_options.obs_with_env:
            if cur_node.agent_process.env:    
                pre_agent_actions = {}
                for i in range(cur_node.agent_group):
                    pre_agent_actions[i] = await self.async_get_envdata(agent_output.request_id + f'_{i}', timing_raw)
            
                for env_id, env_manager in enumerate(cur_node.agent_process.env_managers):
                    # only support one env now
                    if agent_output.request_id not in env_manager:
                        env_class = cur_node.agent_process.env[env_id]
                        env_manager[agent_output.request_id + f'{cur_node.agent_group}'] = env_class()
                    env_instance = env_manager[agent_output.request_id + f'{cur_node.agent_group}']
                    
                    
                    pre_agent_actions = [data for data in list(pre_agent_actions.values()) if data is not None]
                    next_obs, rewards, should_stop = env_instance.step(actions = pre_agent_actions + [agent_output.original_prompt], ground_truth = agent_output.ground_truth)
                    
                    agent_output.rewards = rewards
                    agent_output.original_prompt = next_obs[-1]
                    
                    if should_stop:
                        agent_output.status = AgentOutputStatus.ENV_FINISH
                    # todo: add multienv process
                if isinstance(next_obs, list) and (isinstance(next_obs[0], list) or isinstance(next_obs[0], str)):
                    # have multi-agent obs
                    assert len(next_obs) == len(self.node_queue), f"env return {len(next_node)} obs, should equal agent num {len(self.node_queue)}"
                    # this data may be get in last node ,force put to databuffer temporarily
                    for i in range(cur_node.agent_group):
                        if next_obs[i] is None:
                            assert False
                        await self.async_put_data(agent_output.request_id + f'_{i}', next_obs[i], 2, 4, timing_raw)
            else:
                # this data will be get in last node ,force put to databuffer temporarily
                assert isinstance(agent_output.original_prompt, str)
                if agent_output.original_prompt is None:
                    assert False
                await self.async_put_data(agent_output.request_id + f'_{cur_node.agent_group}', agent_output.original_prompt, 2, 4, timing_raw)
                
        input_and_response = agent_output.templated_prompt
        agent_output.templated_prompt = input_and_response[: len(agent_output.templated_prompt) - len(agent_output.response_mask)]
        agent_output.response_mask = agent_output.response_mask[:self.rollout_config.response_length]
        if len(agent_output.response_mask) == 0:
            # multi-agent may response none
            agent_output.response_id = []
        else:
            agent_output.response_id = input_and_response[-len(agent_output.response_mask) :]
        if cur_node.node_id not in agent_res:
            agent_res[cur_node.node_id] = {}
            agent_res[cur_node.node_id][agent_output.request_id] = []
        if agent_output.request_id not in agent_res[cur_node.node_id]:
            agent_res[cur_node.node_id][agent_output.request_id] = []
        agent_res[cur_node.node_id][agent_output.request_id].append(copy.deepcopy(agent_output))
        # agent_res[cur_node.node_id][agent_output.request_id]=[copy.deepcopy(agent_output)]
        # last node need to add turn
        if node_idx == len(self.node_queue) - 1:
            agent_output.turn = agent_output.turn + 1
            if agent_output.turn >= self.rollout_config.multi_turn.max_assistant_turns:
                agent_output.status = AgentOutputStatus.Turn_FINISH
            if len(agent_output.templated_prompt) >= self.max_model_len:
                agent_output.status = AgentOutputStatus.LENGTH_FINISH

        if agent_output.status !=  AgentOutputStatus.RUNNING:
            if cur_node.node_id not in finished_res:
                finished_res[cur_node.node_id] = set()
            finished_res[cur_node.node_id].add(agent_output.request_id)
       
        await self.async_put_data(next_key, agent_output, cur_dp_size, next_dp_size, timing_raw)
        
        return
    
    async def generate_colocate(self, bs, sampling_params: Dict[str, Any], timing_raw: Dict[str, float]):
        """
        Asynchronously generates sequences in **colocated mode** (model and data reside on the same worker),
        handling distributed coordination (Data Parallelism/DP + Tensor Parallelism/TP) across all nodes in the DAG.
        
        This function manages a loop to fetch input data, dispatch generation tasks, and synchronize across
        distributed ranks until all samples in the batch are fully processed.
        
        Args:
            bs: Local batch size (number of samples to process per individual worker).
            sampling_params: Dictionary of model sampling hyperparameters (e.g., temperature, top_p, repetition_penalty).
            timing_raw: Dictionary to record raw timing metrics (e.g., data fetch latency, task execution time) for debugging/benchmarking.
        
        Returns:
            agent_res: Dictionary mapping node IDs (str) to lists of `AgentOutput` objects. Each `AgentOutput` contains
                    the generated sequence, prompt metadata, and other task-related results for a single sample.
        """
        agent_res: Dict[str, List[AgentOutput]] = {}
        finished_res: Dict[str, set] = {}
        agent_num = len(self.node_queue)  
        global_running = True
        local_running = True  
        while global_running:
            for i in range(agent_num):
                visited_agentoutputs = set()
                cur_node:Node = self.node_queue[i]  
                cur_dp_size, cur_dp_rank, cur_tp_rank, *_ = self.dag._get_node_dp_info(cur_node)
                if i == 0:
                    global_bs = cur_dp_size * bs
                node_worker :ActorRolloutRefWorker = self.workers[self._generate_node_worker_key(cur_node)]
                key = f"{cur_node.node_id}_{cur_dp_rank}"
                workers = []
                await node_worker.rollout.wake_up()
                
                # assert self.tran_bs * self.rollout_config.n % cur_dp_size == 0, f"global batch size is f{self.tran_bs * self.rollout_config.n} can't div by f{cur_dp_size} in node {cur_node.node_id}"
                
                if cur_tp_rank == 0 and local_running:
                    while True:
                        agent_outputs:List[AgentOutput] = await self.async_get_data(key, timing_raw)               
                        if agent_outputs is not None:
                            for agent_output in agent_outputs:
                                visited_agentoutputs.add(agent_output.request_id)
                                worker_task = asyncio.create_task(
                                        self.colocate_task(agent_output = agent_output, 
                                                        agent_res = agent_res, 
                                                        finished_res = finished_res,
                                                        cur_node = cur_node, 
                                                        node_idx = i, 
                                                        sampling_params = sampling_params, 
                                                        global_bs = global_bs,
                                                        timing_raw = timing_raw)
                                    )
                                workers.append(worker_task)
                        
                        if len(visited_agentoutputs) == bs:
                            await asyncio.gather(*workers)
                            break   

                torch.distributed.barrier(node_worker.rollout.get_device_mesh()["tp"].get_group())  


                # Note: in async mode, can't global barrier
                await node_worker.rollout.sleep()        
                local_running = await self.check_colocate_running(finished_res, visited_agentoutputs)
                # tp 0 broadcast to other tp
                tp_group = self.tail_device_mesh["tp"].get_group()
                tp_local_rank = self.tail_device_mesh["tp"].get_local_rank()
                src_local_rank = 0
                src_global_rank = self.tail_device_mesh["tp"].mesh.tolist()[src_local_rank]
                broadcast_list = [None]
                if tp_local_rank == src_local_rank:
                    broadcast_list[0] = local_running

                dist.broadcast_object_list(
                    object_list=broadcast_list,  
                    src=src_global_rank,    
                    group=tp_group          
                )
                local_running = broadcast_list[0]
                        
                        
                finish_flag_tensor = torch.tensor(0 if local_running else 1, device="cuda" if torch.cuda.is_available() else "cpu")
                dist.all_reduce(finish_flag_tensor, op=dist.ReduceOp.SUM)
                total_finish = finish_flag_tensor.item()
                if total_finish == dist.get_world_size():
                    global_running = False
                else:
                    global_running = True
                
        return agent_res   
    def _postprocess(self, agent_outputs: Dict[str, List[AgentOutput]]) -> DataProto:
        """
        Postprocesses generated agent outputs into a structured DataProto object.
        
        Combines prompts and responses into formatted tensors (input_ids, attention_mask, etc.)
        with proper padding and metadata, handling multiple nodes in the DAG.
        
        Args:
            agent_outputs: Dictionary mapping node IDs to lists of AgentOutput objects containing
                        generated responses, prompts, and metadata for each sample.
        
        Returns:
            DataProto object containing concatenated batch data (tensors) and metadata from all nodes.
        """
        # NOTE: Consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # - prompts: Left-padded to fixed length
        # - responses: Right-padded to fixed length
        # - input_ids: Concatenation of prompt and response token IDs
        # - attention_mask: Combines prompt mask (left) and response mask (right)
        #   Format: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # - position_ids: Sequential numbering for valid tokens (masked tokens get 0)
        #   Format: [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        async def _single_postprocess(agent_outputs: List[AgentOutput], node: Node):
            """
            Helper function to postprocess outputs for a single node in the DAG.
            
            Args:
                agent_outputs: List of AgentOutput objects for the current node.
                node: Current DAG node (contains agent processor and tokenizer).
            
            Returns:
                DataProto with processed tensors for the current node.
            """
            # Sort agent outputs by batch ID to maintain original order
            cur_agent_outputs = list(agent_outputs.values())
            cur_agent_outputs = [
                list(agent_output)  
                for agent_output in sorted(cur_agent_outputs, key=lambda x: x[0].batch_id)
            ]
            prompt_texts = [step_output.templated_prompt for agent_output in cur_agent_outputs for step_output in agent_output]
            prompt_texts = node.agent_process.tokenizer.batch_decode(prompt_texts, skip_special_tokens=True)
            node.agent_process.tokenizer.padding_side = "left"
            input_ids = [{"input_ids": step_output.templated_prompt} for agent_output in cur_agent_outputs for step_output in agent_output]
            batch_size = len(input_ids)
            world_size = dist.get_world_size()
            pad_batch_size = 0
            if remainder := batch_size % world_size:
                # need pad    
                pad_batch_size = world_size - remainder
                for _ in range(pad_batch_size):
                    input_ids.append(input_ids[0].copy())
            outputs = node.agent_process.tokenizer.pad(
                input_ids,
                padding="max_length",
                max_length=self.rollout_config.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]
            # responses
            node.agent_process.tokenizer.padding_side = "right"
            response_ids = [{"input_ids": step_output.response_id} for agent_output in cur_agent_outputs for step_output in agent_output]
            if pad_batch_size:
                for _ in range(pad_batch_size):
                    response_ids.append(response_ids[0].copy())
            outputs = node.agent_process.tokenizer.pad(
                response_ids,
                padding="max_length",
                max_length=self.rollout_config.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

            # response_mask
            response_masks = [{"input_ids": step_output.response_mask} for agent_output in cur_agent_outputs for step_output in agent_output]
            
            if pad_batch_size:
                for _ in range(pad_batch_size):
                    response_masks.append({"input_ids":[0] * len(response_masks[0]["input_ids"])})
            outputs = node.agent_process.tokenizer.pad(
                response_masks,
                padding="max_length",
                max_length=self.rollout_config.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            response_mask = outputs["input_ids"]
            assert response_ids.shape == response_mask.shape, (
                f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
            )
            response_mask = response_mask * response_attention_mask
            request_ids = []
            traj_len = []
            traj_step = []
            for agent_output in cur_agent_outputs:
                traj = len(agent_output)
                step = 0
                for step_output in agent_output:
                    request_ids.append(step_output.request_id)
                    traj_len.append(traj)
                    traj_step.append(step)
                    step = step + 1
            if pad_batch_size:
                for _ in range(pad_batch_size):
                    request_ids.append("pad_request")
                    traj_len.append(1)
                    traj_step.append(0)
                    prompt_texts.append(prompt_texts[0])
            input_ids = torch.cat([prompt_ids, response_ids], dim=1)
            attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
            position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask
            prefix = f"agent_group_{node.agent_group}_"
            batch = TensorDict(
                { 
                    prefix + "prompts": prompt_ids,  # [bsz, prompt_length]
                    prefix + "responses": response_ids,  # [bsz, response_length]
                    prefix + "response_mask": response_mask,  # [bsz, response_length]
                    prefix + "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                    prefix + "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                    prefix + "position_ids": position_ids,  # [bsz, prompt_length + response_length]
                    
                },
                batch_size=len(input_ids),
            )
            non_tensor_batch = {
                prefix + "request_id": np.array(request_ids),
                prefix + "traj_len": np.array(traj_len),
                prefix + "traj_step": np.array(traj_step),
                prefix + "prompt_texts": np.array(prompt_texts)
            }
            if node.node_id == self.node_queue[-1].node_id and self.rollout_config.agent.rewards_with_env:
                reward_tensor = torch.zeros_like(batch[prefix + "responses"], dtype=torch.float32)
                idx = 0
                for agent_outputs in cur_agent_outputs:
                    for step_output in agent_outputs:
                        prompt_ids = batch[prefix + "prompts"][idx]
                        prompt_length = prompt_ids.shape[-1]
                        valid_response_length = batch[prefix + "attention_mask"][idx][prompt_length:].sum()
                        reward_tensor[idx, valid_response_length - 1] = step_output.rewards
                        idx = idx + 1
                batch[prefix + "token_level_rewards"] = reward_tensor
                batch[prefix + "token_level_scores"] = copy.deepcopy(reward_tensor)
                
            if node.agent_process.env:
                node.agent_process.env_managers.clear()
                node.agent_process.env_managers = [{}]
            return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"metrics": {}})
        
        tasks = []
        for i in range(len(self.node_queue)):
            cur_node = self.node_queue[i]
            _, cur_dp_rank, cur_tp_rank, *_ = self.dag._get_node_dp_info(cur_node)
            if cur_tp_rank == 0:
                tasks.append(_single_postprocess(agent_outputs = agent_outputs[cur_node.node_id], node = cur_node))
        loop = asyncio.get_event_loop()
        datas = loop.run_until_complete(asyncio.gather(*tasks))    
        dataproto = None
        for data in datas:
            if dataproto:
                dataproto.union(data)
            else:
                dataproto = data
        return dataproto

    def generate_sequence(self, batch:DataProto, timing_raw: Dict[str, float] = {}):
        """
        Generate model output sequences based on the input DataProto batch, handling both colocated and spread placement modes.
        Args:
            batch: Input DataProto object containing raw data (e.g., prompts, ground truth) for sequence generation.
            timing_raw: Dictionary to record raw timing metrics (e.g., data transfer, generation latency). Defaults to empty dict.
        Returns:
            Processed DataProto object with generated sequences, metadata (including metrics), and prefix removed for downstream DAG worker.
        """
        prompts = None
        metrics = {}
        loop = asyncio.get_event_loop()
        entry_node = self.node_queue[0]
        sampling_params = dict(
            temperature=self.rollout_config.temperature,
            top_p=self.rollout_config.top_p,
            repetition_penalty=1.0,
        )
        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = self.rollout_config.val_kwargs.top_p
            sampling_params["temperature"] = self.rollout_config.val_kwargs.temperature
            
        if self.placement_mode == 'colocate' or self.node_if_local(entry_node):
            prompts, ground_truth = self._preprocess(batch)
            prompts_ids = entry_node.agent_process.tokenizer.apply_chat_template(
                        prompts,
                        add_generation_prompt=True,
                        tokenize=True,
                    )
            tasks = []
            dp_size, dp_rank, tp_rank, *_ = self.dag._get_node_dp_info(entry_node)
            if tp_rank == 0:
                for i in range(len(prompts_ids)):
                    key = self._generate_key(entry_node, entry_node, i)
                    tasks.append(self.async_put_data(key, 
                        AgentOutput(batch_id = i, 
                                    original_prompt = prompts_ids[i], 
                                    templated_prompt = '', 
                                    should_stop = False, 
                                    response_mask = [0] * len(prompts_ids[i]), 
                                response_id = [], 
                                    request_id = uuid4().hex,
                                    ground_truth = ground_truth[i]), 
                        dp_size, dp_size, timing_raw))
                loop.run_until_complete(asyncio.gather(*tasks))      
        with Timer(name="generate_sequences", logger=None) as timer:
            if self.placement_mode == 'spread':
                # if in different GPUWorker
                response,response_mask = loop.run_until_complete(self.generate_spread(timing_raw))
            elif self.placement_mode == 'colocate':
                # if in same GPUWorker
                agent_outputs = loop.run_until_complete(self.generate_colocate(len(prompts_ids), sampling_params, timing_raw))
        delta_time = timer.last
        metrics["perf/delta_time/actor"] = delta_time
        generated_proto = self._postprocess(agent_outputs) 
          
        # remove last node prefix, because it will be add in dagworker
        if generated_proto:
            generated_proto.meta_info.update({"metrics": metrics})
            generated_proto = remove_prefix_from_dataproto(generated_proto, self.node_queue[-1])
        # databuffer will reset in dagworker, so only reset internal_dict
        # but in validate step, databuffer will not clean
        if batch.meta_info.get("validate", False):
            dist.barrier()
            if dist.get_rank() == 0:
                tasks = [databuffer.reset.remote() for databuffer in self.data_buffers]
                ray.get(tasks)
        self.internal_data_cache.clear()
        # logger.info(f"batch keys :{batch.batch.keys()} ||| {batch.non_tensor_batch.keys()} ||| {batch.meta_info.keys()}")
        return generated_proto
    

 
