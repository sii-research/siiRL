import torch
import numpy as np
import asyncio
import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Union, Set
from siirl.data_coordinator.protocol import DataProto
from tensordict import TensorDict

@dataclass
class MetaInfo:
    # single instance
    _instance = None
    
    # data
    validate: bool=field(default=False)
    metrics: dict=field(default_factory=dict)
    temperature: float=field(default=0)
    total_input_tokens: int=field(default=0)
    total_output_tokens: int=field(default=0)
    global_token_num: List=field(default_factory=list)
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    

@dataclass
class SampleInfo:
    agent_group: int = field(default=0)
    sum_tokens: int = field(default=0)
    prompt_length: int = field(default=0)
    response_length: int = field(default=0)
    dict_info: dict = field(default_factory=dict)
    uid: int = field(default=None)
    node_id: Optional[str] = field(default=None)

@dataclass
class Sample:
    # from tensodict of Dataproto
    prompts: List[int] = field(default_factory=list)
    responses: List[int] = field(default_factory=list)
    response_mask: List[int] = field(default_factory=list)
    input_ids: List[int] = field(default_factory=list, metadata={"help":"initial: prompts with pad, after rollout: prompts + response"})
    attention_mask: List[int] = field(default_factory=list)
    position_ids: List[int] = field(default_factory=list)
    acc: float = field(default=0.0)
    token_level_rewards: List[float] = field(default_factory=list)
    token_level_scores: List[float] = field(default_factory=list)
    values: List[float] = field(default=list)
    advantages: List[float] = field(default_factory=list)
    old_log_probs: List[float] = field(default_factory=list)
    ref_log_prob: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    # from non_tensor_batch of Dataproto
    raw_prompt: str = field(default="")
    raw_prompt_ids: List[int] = field(default_factory=list)
    prompt_texts: str = field(default="", metadata={"help":"used in validate, decode form 'input_ids', but may is same with raw_prompt"})
    reward_model: dict = field(default_factory=dict)
    data_source: str = field(default="", metadata={"help":"name of datasets"})
    extra_info: dict = field(default_factory=dict)
    tools_kwargs: dict = field(default_factory=dict, metadata={"help":"used in multi-turn, may not need to be sample dimension"})
    interaction_kwargs: dict = field(default_factory=dict, metadata={"help":"used in multi-turn, may not need to be sample dimension"})
    request_id: str = field(default="", metadata={"help":"used in multi-agent"})
    traj_len: int = field(default=0, metadata={"help":"used in multi-agent"})
    traj_step: int = field(default=0, metadata={"help":"used in multi-agent"})
    seq_final_reward: float = field(default=0.0, metadata={"help":"used in dapo"})
    seq_reward: float = field(default=0.0, metadata={"help":"used in dapo"})
    
@dataclass
class GroupSample:
    group_sample: List[Sample] = field(default_factory=list)
    
    
def Dict2Samples(data:Dict, n:int):
    batch_size = len(data['input_ids'])
    async def calc_sample(index, n):
        local_sample = Sample()
        local_sample.input_ids = data['input_ids'][index]
        local_sample.attention_mask = data['attention_mask'][index]
        local_sample.position_ids = data['position_ids'][index]
        local_sample.data_source = data['data_source'][index]
        local_sample.reward_model = data['reward_model'][index]
        local_sample.raw_prompt_ids = data['raw_prompt_ids'][index]
        local_sample.extra_info = data['extra_info'][index]
        if 'raw_prompt' in data:
            local_sample.raw_prompt = data['raw_prompt'][index]
        local_sample.uid = str(uuid.uuid4())
        group_sample = GroupSample(group_sample = [local_sample] * n)
        return group_sample
    loop = asyncio.get_event_loop()
    futures = []
    for index in range(batch_size):
        futures.append(calc_sample(index, n))
    group_samples = loop.run_until_complete(asyncio.gather(*futures))   
    del data 
    return group_samples
            
        

# used for develop
def get_sample_fields() -> Set[str]:
    """获取Sample类所有字段名的集合"""
    # 使用dataclasses.fields获取所有字段，提取字段名
    return {field.name for field in fields(Sample)}

def is_key_in_sample(key: str) -> bool:
    """判断单个key是否是Sample类的字段"""
    return key in get_sample_fields()

def Sample2DataProto(samples: List[Sample]):
    """将Sample对象转换为Dataproto，拆分到tensordict和nontensorbatch"""
    # 1. 构建tensordict（适合张量操作的列表/数值字段）
    tensordict = TensorDict ({
        "prompts": torch.tensor([sample.prompts for sample in samples]),
        "responses": torch.tensor([sample.responses for sample in samples]),
        "response_mask": torch.tensor([sample.response_mask for sample in samples]),
        "input_ids": torch.tensor([sample.input_ids for sample in samples]),
        "attention_mask": torch.tensor([sample.attention_mask for sample in samples]),
        "position_ids": torch.tensor([sample.position_ids for sample in samples]),
        "acc": torch.tensor([sample.acc for sample in samples]),
        "token_level_rewards": torch.tensor([sample.token_level_rewards for sample in samples]),
        "token_level_scores": torch.tensor([sample.token_level_scores for sample in samples]),
        "values": torch.tensor([sample.values for sample in samples]),
        "advantages": torch.tensor([sample.advantages for sample in samples]),
        "old_log_probs": torch.tensor([sample.old_log_probs for sample in samples]),
        "ref_log_prob": torch.tensor([sample.ref_log_prob for sample in samples]),
        "returns": torch.tensor([sample.returns for sample in samples]),
        },
        batch_size=len(samples)
    )
    
    # 2. 构建nontensorbatch（元数据/非张量字段）
    non_tensor_batch = {
        "uid": np.array([sample.uid for sample in samples]),
        "raw_prompt": np.array([sample.raw_prompt for sample in samples]),
        "raw_prompt_ids": np.array([sample.raw_prompt_ids for sample in samples]),
        "prompt_texts": np.array([sample.prompt_texts for sample in samples]),
        "reward_model": np.array([sample.reward_model for sample in samples]),
        "data_source": np.array([sample.data_source for sample in samples]),
        "extra_info": np.array([sample.extra_info for sample in samples]),
        "tools_kwargs": np.array([sample.tools_kwargs for sample in samples]),
        "interaction_kwargs": np.array([sample.interaction_kwargs for sample in samples]),
        "request_id": np.array([sample.request_id for sample in samples]),
        "traj_len": np.array([sample.traj_len for sample in samples]),
        "traj_step": np.array([sample.traj_step for sample in samples]),
        "seq_final_reward": np.array([sample.seq_final_reward for sample in samples]),
        "seq_reward": np.array([sample.seq_reward for sample in samples]),
    }
    meta = MetaInfo()
    meta_info = {
        "validate": meta.validate,
        "metrics": meta.metrics,
        "temperature": meta.temperature,
        "total_input_tokens": meta.total_input_tokens,
        "total_output_tokens": meta.total_output_tokens,
        "global_token_num": meta.global_token_num
    }
    # 3. 实例化Dataproto并返回（假设Dataproto接受这两个参数）
    print(f"[hujr] DataProto2Sample token_level_rewards {tensordict.get('token_level_rewards', [])}")
    return DataProto(
        batch=tensordict,
        non_tensor_batch=non_tensor_batch,
        meta_info=meta_info
    )

def DataProto2Sample(data:DataProto):
    """将DataProto对象转换为Sample列表"""
    samples = []
    batch_size = None
    # 确定批次大小（从tensordict中任意一个张量获取）
    for key, tensor in data.batch.items():
        if batch_size is None:
            batch_size = tensor.shape[0]
        break
    
    if batch_size is None:
        return []  # 空批次处理
    
    # 判断key是否存在
    for key in data.batch.keys():
        if not is_key_in_sample(key):
            print(f"[hujr] tensordict key {key} not  exist in Sample")
    for key in data.non_tensor_batch.keys():
        if not is_key_in_sample(key):
            print(f"[hujr] non_tensor_batch key {key} not  exist in Sample")
    
    # 遍历每个样本索引，构建Sample对象
    for i in range(batch_size):
        # 从tensordict提取数据（转换为列表）
        sample_kwargs = {
            "prompts": data.batch["prompts"][i].tolist() if "prompts" in data.batch else [],
            "responses": data.batch["responses"][i].tolist() if "responses" in data.batch else [],
            "response_mask": data.batch["response_mask"][i].tolist() if "response_mask" in data.batch else [],
            "input_ids": data.batch["input_ids"][i].tolist() if "input_ids" in data.batch else [],
            "attention_mask": data.batch["attention_mask"][i].tolist() if "attention_mask" in data.batch else [],
            "position_ids": data.batch["position_ids"][i].tolist() if "position_ids" in data.batch else [],
            "acc": float(data.batch["acc"][i].item()) if "acc" in data.batch else 0.0,
            "token_level_rewards": data.batch["token_level_rewards"][i].tolist() if "token_level_rewards" in data.batch else [],
            "token_level_scores": data.batch["token_level_scores"][i].tolist() if "token_level_scores" in data.batch else [],
            "values": data.batch["values"][i].tolist() if "values" in data.batch else [],
            "advantages": data.batch["advantages"][i].tolist() if "advantages" in data.batch else [],
            "old_log_probs": data.batch["old_log_probs"][i].tolist() if "old_log_probs" in data.batch else [],
            "ref_log_prob": data.batch["ref_log_prob"][i].tolist() if "ref_log_prob" in data.batch else [],
            "returns": data.batch["returns"][i].tolist() if "returns" in data.batch else [],
            
        }
        
        # 从non_tensor_batch提取数据（处理numpy数组）
        non_tensor = data.non_tensor_batch
        sample_kwargs.update({
            "uid": non_tensor["uid"][i].item() if "uid" in non_tensor else None,
            "raw_prompt": str(non_tensor["raw_prompt"][i]) if "raw_prompt" in non_tensor else "",
            "raw_prompt_ids": non_tensor["raw_prompt_ids"][i].tolist() if "raw_prompt_ids" in non_tensor else [],
            "prompt_texts": str(non_tensor["prompt_texts"][i]) if "prompt_texts" in non_tensor else "",
            "reward_model": dict(non_tensor["reward_model"][i]) if "reward_model" in non_tensor else {},
            "data_source": str(non_tensor["data_source"][i]) if "data_source" in non_tensor else "",
            "extra_info": dict(non_tensor["extra_info"][i]) if "extra_info" in non_tensor else {},
            "tools_kwargs": dict(non_tensor["tools_kwargs"][i]) if "tools_kwargs" in non_tensor else {},
            "interaction_kwargs": dict(non_tensor["interaction_kwargs"][i]) if "interaction_kwargs" in non_tensor else {},
            "request_id": str(non_tensor["request_id"][i]) if "request_id" in non_tensor else "",
            "traj_len": int(non_tensor["traj_len"][i].item()) if "traj_len" in non_tensor else 0,
            "traj_step": int(non_tensor["traj_step"][i].item()) if "traj_step" in non_tensor else 0,
            "seq_final_reward": float(non_tensor["seq_final_reward"][i].item()) if "seq_final_reward" in non_tensor else 0.0,
            "seq_reward": float(non_tensor["seq_reward"][i].item()) if "seq_reward" in non_tensor else 0.0,
        })
        
        # 创建Sample对象并添加到列表
        samples.append(Sample(** sample_kwargs))
    meta = MetaInfo()
    meta.global_token_num = (torch.sum(data.batch["attention_mask"], dim=-1)).tolist()
    meta.validate = data.meta_info.get("validate", False)
    meta.metrics = data.meta_info.get("metrics", {})
    meta.temperature = data.meta_info.get("temperature", 0)
    meta.total_input_tokens = data.meta_info.get("total_input_tokens", 0)
    meta.total_output_tokens = data.meta_info.get("total_output_tokens", 0)
    print(f"[hujr] DataProto2Sample token_level_rewards {data.batch.get('token_level_rewards', [])}")
    return samples
    