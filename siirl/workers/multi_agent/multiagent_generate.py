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

import numpy as np
import torch.distributed  as dist

from asyncio import Queue
from siirl.models.loader import load_tokenizer
from tensordict import TensorDict
from uuid import uuid4
from .utils import AgentOutput, AgentOutputStatus
from typing import Dict, List, Any, Tuple, Optional, Union
from codetiming import Timer


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
            self.rollout_config.multi_turn.max_assistant_turns = 2
    
        
    def _parse_graph(self, graph:TaskGraph):
        node_queue = graph.get_entry_nodes()
        visited_nodes = set()
        self.node_queue = []
        while node_queue:
            cur_node = node_queue.pop(0)
            if cur_node.node_role != NodeRole.ROLLOUT:
                break
            # 先不带 env
            # if cur_node.agent_options.get('env', None):
            #     cur_node.env = 
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
        n = 1 if batch.meta_info.get("validate", False) else self.rollout_config.n
        raw_prompts = batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)
        raw_prompts = [p.tolist() for p in raw_prompts]
        return raw_prompts

    def _generate_key(self, cur_node: Node, next_node: Node, batch_id: int, global_bs: int = 0) :
        cur_dp_size, cur_dp_rank, _, _, _, _ = self.dag._get_node_dp_info(cur_node)
        next_dp_size, next_dp_rank, _, _, _, _ = self.dag._get_node_dp_info(next_node)
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
          
    async def async_put_data(self, key: str, value: AgentOutput, source_dp_size: int, dest_dp_size: int, timing_raw: Dict[str, float]):
        # todo(hujr): timing_raw support in multi asyncio task
        # if  source_dp_size == dest_dp_size:
        #     if key not in self.internal_data_cache:
        #         self.internal_data_cache[key] = Queue()
        #     await self.internal_data_cache[key].put(value)
        # else:   
            # random save to databuffers
        buffer = random.choice(self.data_buffers)
        await buffer.put.remote(key, value)

    
    async def async_get_data(self, key: str, timing_raw: Dict[str, float]):
        # todo(hujr): timing_raw support in multi asyncio task    
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
        while True:
            key = self._generate_key(batch_idx, cur_node.dp_rank, cur_node.node_id)
            prompt,should_stop = self.databuffer.get(key)
            if multiturn > max_multiturn:
                return response, response_mask
            if should_stop:
                #链式传递 should_stop
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
                # response mask 计算参考verl tool_agent_loop
            next_node = self.node_queue[node_idx + 1] if node_idx < len(self.node_queue) - 1  else self.node_queue[0]
            next_key = self._generate_key(batch_idx, next_node.dp_rank, next_node.node_id)
            self.databuffer.put(next_key, [prompt, should_stop]) 
            multiturn = multiturn + 1
                
    async def generate_spread(self):
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
    async def colocate_task(self, agent_output:AgentOutput, agent_res:Dict,  cur_node: Node, node_idx: int, sampling_params: Dict[str, Any], global_bs: int,  timing_raw: Dict[str, float]):
        cur_dp_size, cur_dp_rank, _, _, _, _ = self.dag._get_node_dp_info(cur_node)
        node_worker:ActorRolloutRefWorker = self.workers[self._generate_node_worker_key(cur_node)]
        # if agent_output.should_stop:
        #     return agent_output

        agent_output.original_prompt, agent_output.templated_prompt = cur_node.agent_process.apply_pre_process(prompt=agent_output.original_prompt)
        agent_output.templated_prompt = agent_output.templated_prompt[:self.rollout_config.prompt_length]
        if agent_output.status != AgentOutputStatus.LENGTH_FINISH:
            response = await node_worker.rollout.generate(
                request_id=agent_output.request_id, prompt_ids=agent_output.templated_prompt, sampling_params=sampling_params
                )
        else:
            response = []
        agent_output.original_prompt, agent_output.templated_prompt, agent_output.response_mask \
            = cur_node.agent_process.apply_post_process(oridinal_prompt = agent_output.original_prompt, templated_prompt = agent_output.templated_prompt, response = response)
        # if cur_node.env
        #     tool_response, rewards, should_stop = self.env.execute(prompt)
        #     prompt = prompt + tool_response
        #     # response mask 计算参考verl tool_agent_loop
        if len(agent_output.templated_prompt) >= self.max_model_len:
            agent_output.status = AgentOutputStatus.LENGTH_FINISH
        next_node = self.node_queue[node_idx + 1] if node_idx < len(self.node_queue) - 1  else self.node_queue[0]
        next_key = self._generate_key(cur_node, next_node, agent_output.batch_id, global_bs)
        next_dp_size, _, _, _, _, _ = self.dag._get_node_dp_info(next_node)
        if agent_output.turn >= self.rollout_config.multi_turn.max_assistant_turns - 1:
            agent_output.templated_prompt = agent_output.templated_prompt[: len(agent_output.templated_prompt) - len(agent_output.response_mask)]
            agent_output.response_mask = agent_output.response_mask[:self.rollout_config.response_length]
            agent_output.response_id = agent_output.templated_prompt[-len(agent_output.response_mask) :]  
            if cur_node.node_id not in agent_res:
                agent_res[cur_node.node_id] = []
            agent_res[cur_node.node_id].append(agent_output)
        # last node need to add turn
        if node_idx == len(self.node_queue) - 1:
            agent_output.turn = agent_output.turn + 1
        await self.async_put_data(next_key, agent_output, cur_dp_size, next_dp_size, timing_raw)
        return
    
    async def generate_colocate(self, bs, sampling_params: Dict[str, Any], timing_raw: Dict[str, float]):
        agent_res: Dict[str, List[AgentOutput]] = {}
        agent_num = len(self.node_queue)
        loop = 0
        while self.finish_generate == False:
            loop = loop + 1
            for i in range(agent_num):
                visited_agentoutputs = set()
                cur_node:Node = self.node_queue[i]  
                cur_dp_size, cur_dp_rank, cur_tp_rank, cur_tp_size, cur_pp_rank, cur_pp_size = self.dag._get_node_dp_info(cur_node)
                if i == 0:
                    global_bs = cur_dp_size * bs
                node_worker :ActorRolloutRefWorker = self.workers[self._generate_node_worker_key(cur_node)]
                key = f"{cur_node.node_id}_{cur_dp_rank}"
                workers = []
                await node_worker.rollout.wake_up()
                # assert self.tran_bs * self.rollout_config.n % cur_dp_size == 0, f"global batch size is f{self.tran_bs * self.rollout_config.n} can't div by f{cur_dp_size} in node {cur_node.node_id}"
                
                if cur_tp_rank == 0:
                    while True:
                        agent_outputs:List[AgentOutput] = await self.async_get_data(key, timing_raw)               
                        if agent_outputs is not None:
                            for agent_output in agent_outputs:
                                visited_agentoutputs.add(agent_output.request_id)
                                worker_task = asyncio.create_task(
                                        self.colocate_task(agent_output = agent_output, 
                                                        agent_res = agent_res, 
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
                
                if i == agent_num - 1 :
                    if cur_node.node_id in agent_res:
                        if len(agent_res[self.node_queue[agent_num - 1].node_id]) == len(visited_agentoutputs):
                            self.finish_generate = True
                        elif len(agent_res[self.node_queue[agent_num - 1].node_id]) > len(visited_agentoutputs):
                            assert False, "This should not happen"
                    # tp 0 broadcast to other tp
                    tp_group = self.tail_device_mesh["tp"].get_group()
                    tp_local_rank = self.tail_device_mesh["tp"].get_local_rank()
                    src_local_rank = 0
                    src_global_rank = self.tail_device_mesh["tp"].mesh.tolist()[src_local_rank]
                    broadcast_list = [None]
                    if tp_local_rank == src_local_rank:
                        broadcast_list[0] = self.finish_generate

                    dist.broadcast_object_list(
                        object_list=broadcast_list,  
                        src=src_global_rank,    
                        group=tp_group          
                    )
                    self.finish_generate = broadcast_list[0]
        return agent_res   
    def _postprocess(self, agent_outputs: Dict[str, List[AgentOutput]]) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts
        async def _single_postprocess(agent_outputs: List[AgentOutput], node: Node):
            agent_outputs = sorted(agent_outputs, key=lambda x: x.batch_id)
            node.agent_process.tokenizer.padding_side = "left"
            input_ids = [{"input_ids": agent_output.templated_prompt} for agent_output in agent_outputs]
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
            outputs = node.agent_process.tokenizer.pad(
                [{"input_ids": agent_output.response_id} for agent_output in agent_outputs],
                padding="max_length",
                max_length=self.rollout_config.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

            # response_mask
            outputs = node.agent_process.tokenizer.pad(
                [{"input_ids": agent_output.response_mask} for agent_output in agent_outputs],
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
            return DataProto(batch=batch, non_tensor_batch={}, meta_info={"metrics": {}})
        
        tasks = []
        for i in range(len(self.node_queue)):
            cur_node = self.node_queue[i]
            _, _, cur_tp_rank, _, _, _ = self.dag._get_node_dp_info(cur_node)
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
        # return DataProto.concat(datas)

    def generate_sequence(self, batch:DataProto, timing_raw: Dict[str, float] = {}):
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
            prompts = self._preprocess(batch)
            prompts_ids = entry_node.agent_process.apply_chat_template(
                        prompts,
                        add_generation_prompt=True,
                        tokenize=True,
                    )
            tasks = []
            dp_size, dp_rank, tp_rank, _, _, _ = self.dag._get_node_dp_info(entry_node)
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
                                    request_id = uuid4().hex), 
                        dp_size, dp_size, timing_raw))
                loop.run_until_complete(asyncio.gather(*tasks))     
        self.finish_generate = False  
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
            tasks = [databuffer.reset.remote() for databuffer in self.data_buffers]
            ray.get(tasks)
        self.internal_data_cache.clear()
        return generated_proto
    

 
