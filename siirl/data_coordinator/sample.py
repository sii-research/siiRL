import torch
import numpy as np
import asyncio
import ray
import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Union, Set
from siirl.data_coordinator.protocol import DataProto
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

@dataclass
class MetaInfo:
    # single instance
    _instance = None
    
    # data
    eos_token_id: int=field(default=0)
    pad_token_id: int=field(default=0)
    validate: bool=field(default=False)
    do_sample: bool=field(default=True)
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
    uid: str = field(default=None)

@dataclass
class Sample:
    # from tensodict of Dataproto
    prompts: torch.tensor = field(default=None)
    responses: torch.tensor = field(default=None)
    response_mask: torch.tensor = field(default=None)
    input_ids: torch.tensor = field(default=None, metadata={"help":"initial: prompts with pad, after rollout: prompts + response"})
    attention_mask: torch.tensor = field(default=None)
    position_ids: torch.tensor = field(default=None)
    acc: float = field(default=0.0)
    token_level_rewards: torch.tensor = field(default=None)
    token_level_scores: torch.tensor = field(default=None)
    values: torch.tensor = field(default=None)
    advantages: torch.tensor = field(default=None)
    returns: torch.tensor = field(default=None)
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
    multi_modal_inputs: dict = field(default=None)

@dataclass   
class SampleManager:
    # 使用default而不是default_factory来设置None默认值
    sample_info: Optional[SampleInfo] = field(default=None)
    sample: Optional[Sample | ray.ObjectRef] = field(default=None)
    
    
def preprocess_dataloader(data:Dict, n:int = 1):
    batch_size = len(data['input_ids'])
    uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)])
    data['uid'] = uid
    for key,value in data.items():
        if isinstance(value, np.ndarray): 
            value = np.repeat(value, n, axis=0)
            data[key] = value

    tensor_dict = TensorDict(data, batch_size=batch_size)
    tensor_dict = tensor_dict.repeat_interleave(n)
    return tensor_dict
   
def Dict2Samples(data:TensorDict):
    batch_size = data.batch_size[0]
    async def calc_sample(index):
        local_sample = Sample()
        sample_info = SampleInfo()
        local_sample.input_ids = data['input_ids'][index]
        local_sample.attention_mask = data['attention_mask'][index]
        local_sample.position_ids = data['position_ids'][index]
        local_sample.data_source = data['data_source'][index]
        local_sample.reward_model = data['reward_model'][index]
        local_sample.prompts = data['prompts'][index]
        local_sample.responses = data['responses'][index]
        local_sample.response_mask = data['response_mask'][index]
        local_sample.raw_prompt_ids = data['raw_prompt_ids'][index] if 'raw_prompt_ids' in data else None
        local_sample.advantages = data['advantages'][index] if 'advantages' in data else None
        local_sample.raw_prompt = data['raw_prompt'][index] if 'raw_prompt' in data else None
        local_sample.advantages = data['advantages'][index] if 'advantages' in data else None
        local_sample.returns = data['returns'][index] if 'returns' in data else None
        local_sample.token_level_rewards = data['token_level_rewards'][index] if 'token_level_rewards' in data else None
        local_sample.token_level_scores = data['token_level_scores'][index] if 'token_level_scores' in data else None
        
        if 'multi_modal_inputs' in data:
            local_sample.multi_modal_inputs = data["multi_modal_inputs"][index]
        sample_info.uid = str(uuid.uuid4())
        # local_sample = ray.put(local_sample)
        sample_manager = SampleManager(sample_info=sample_info, sample=local_sample)
        return sample_manager
    loop = asyncio.get_event_loop()
    futures = []
    for index in range(batch_size):
        futures.append(calc_sample(index))
    sample_managers = loop.run_until_complete(asyncio.gather(*futures))   
    del data 
    return sample_managers

def Samples2Dict(samples: List[SampleManager]) -> TensorDict:
    async def get_sample(samples, index):
        # sample = ray.get(samples[index].sample)
        # samples[index].sample = sample
        return samples[index]
    futures = []
    for i in range(len(samples)):
        futures.append(get_sample(samples, i))
    loop = asyncio.get_event_loop()
    samples = loop.run_until_complete(asyncio.gather(*futures))
    # convert to tensordict
    sample_fields = [f.name for f in fields(Sample)]
    
    aggregated: Dict[str, List[Any]] = {}
    aggregated["uid"] = []
    for sm in samples:
        sample = sm.sample
        sample_info = sm.sample_info
        if sample is None:
            raise ValueError("SampleManager 中的 sample 不能为 None")
        for field in sample_fields:
            val = getattr(sample, field)
            if val is not None:
                if isinstance(val, (torch.Tensor, list, np.ndarray, dict, str)):
                    if field not in aggregated:
                        aggregated[field] = []
                    aggregated[field].append(val)
                elif isinstance(val, (int, float, bool)):
                    aggregated[field] = val
                else:
                    print(f"key {field} type{type(val)} not support")
        aggregated["uid"].append(sample_info.uid)        
    tensordict_data: Dict[str, Any] = {}
    batch_size = (len(samples),)  
    
    for key, values in aggregated.items():
        if isinstance(values, list):
            first_val = values[0]
            if isinstance(first_val, torch.Tensor):
                tensordict_data[key] = torch.stack(values, dim=0) if first_val.ndim >= 1 else torch.cat(values, dim=0)
            elif isinstance(first_val, np.ndarray):
                tensordict_data[key] = np.stack(values, axis=0) if first_val.ndim >= 1 else np.concatenate(values, axis=0)
            elif isinstance(first_val, str):
                tensordict_data[key] = values
            else:
                tensordict_data[key] = NonTensorData(
                    data=values,
                    batch_size=batch_size
                )
            
        else:
            tensordict_data[key] = NonTensorData(
                data=values,
                batch_size=batch_size
            )
    tensordict_data["global_token_num"] = NonTensorData((torch.sum(tensordict_data["attention_mask"], dim=-1)).tolist())
    return TensorDict(tensordict_data, batch_size=batch_size)
    
    