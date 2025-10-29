import torch
import numpy as np
import asyncio
import ray
import uuid
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union, Set
from siirl.data_coordinator.protocol import DataProto
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from typing import get_args, get_origin


class SampleInfo(BaseModel):
    agent_group: int = Field(default=0)
    sum_tokens: int = Field(default=0)
    prompt_length: int = Field(default=0)
    response_length: int = Field(default=0)
    dict_info: Dict[str, Any] = Field(default_factory=dict)
    uid: Optional[str] = Field(default=None)
    node_id: Optional[str] = Field(default=None)


class Sample(BaseModel):
    # from tensordict of Dataproto
    prompts: Optional[np.ndarray] = Field(default=None)
    responses: Optional[np.ndarray] = Field(default=None)
    response_mask: Optional[np.ndarray] = Field(default=None)
    input_ids: Optional[np.ndarray] = Field(
        default=None,
        metadata={"help": "initial: prompts with pad, after rollout: prompts + response"}
    )
    attention_mask: Optional[np.ndarray] = Field(default=None)
    position_ids: Optional[np.ndarray] = Field(default=None)
    acc: float = Field(default=None)
    token_level_rewards: Optional[np.ndarray] = Field(default=None)
    token_level_scores: Optional[np.ndarray] = Field(default=None)
    values: Optional[np.ndarray] = Field(default=None)
    advantages: Optional[np.ndarray] = Field(default=None)
    returns: Optional[np.ndarray] = Field(default=None)
    old_log_probs: Optional[np.ndarray] = Field(default=None)
    ref_log_prob: Optional[np.ndarray] = Field(default=None)

    # from  non_tensor_batch of Dataproto
    raw_prompt: str = Field(default="")
    raw_prompt_ids: List[int] = Field(default_factory=list)
    prompt_texts: str = Field(
        default="",
        metadata={"help": "used in validate, decode form 'input_ids', but may is same with raw_prompt"}
    )
    reward_model: Dict[str, Any] = Field(default_factory=dict)
    data_source: str = Field(
        default="",
        metadata={"help": "name of datasets"}
    )
    extra_info: Dict[str, Any] = Field(default_factory=dict)
    tools_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        metadata={"help": "used in multi-turn, may not need to be sample dimension"}
    )
    interaction_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        metadata={"help": "used in multi-turn, may not need to be sample dimension"}
    )
    request_id: str = Field(
        default="",
        metadata={"help": "used in multi-agent"}
    )
    traj_len: int = Field(
        default=None,
        metadata={"help": "used in multi-agent"}
    )
    traj_step: int = Field(
        default=None,
        metadata={"help": "used in multi-agent"}
    )
    seq_final_reward: float = Field(
        default=None,
        metadata={"help": "used in dapo"}
    )
    seq_reward: float = Field(
        default=None,
        metadata={"help": "used in dapo"}
    )
    multi_modal_inputs: Optional[Dict[str, Any]] = Field(default=None)
    uid: Optional[str] = Field(default=None)

    temperature: float = Field(
        default=None,
        metadata={"help": "temperature"}
    )

    class Config:
        arbitrary_types_allowed = True


class SampleManager(BaseModel):
    sample_info: Optional[SampleInfo] = Field(default=None)
    sample: Optional[Union[Sample, ray.ObjectRef]] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True



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

def Dict2Samples(data:TensorDict)-> List[SampleManager]:
    batch_size = data.batch_size[0]
    async def calc_sample(index):
        local_sample = Sample()
        local_sample.input_ids = data['input_ids'][index].numpy()
        local_sample.attention_mask = data['attention_mask'][index].numpy()
        local_sample.position_ids = data['position_ids'][index].numpy()
        local_sample.data_source = data['data_source'][index]
        local_sample.reward_model = data['reward_model'][index]
        local_sample.prompts = data['prompts'][index].numpy() if 'prompts' in data else None
        local_sample.responses = data['responses'][index].numpy() if 'responses' in data else None
        local_sample.response_mask = data['response_mask'][index].numpy() if 'response_mask' in data else None
        local_sample.values = data['values'][index].numpy() if 'values' in data else None
        local_sample.raw_prompt_ids = data['raw_prompt_ids'][index] if 'raw_prompt_ids' in data else None
        local_sample.advantages = data['advantages'][index].numpy() if 'advantages' in data else None
        local_sample.raw_prompt = data['raw_prompt'][index] if 'raw_prompt' in data else None
        local_sample.returns = data['returns'][index].numpy() if 'returns' in data else None
        local_sample.token_level_rewards = data['token_level_rewards'][index].numpy() if 'token_level_rewards' in data else None
        local_sample.token_level_scores = data['token_level_scores'][index].numpy() if 'token_level_scores' in data else None
        local_sample.old_log_probs = data['old_log_probs'][index].numpy() if 'old_log_probs' in data else None
        local_sample.ref_log_prob = data['ref_log_prob'][index].numpy() if 'ref_log_prob' in data else None
        local_sample.extra_info = data['extra_info'][index] if 'extra_info' in data else None

        if 'multi_modal_inputs' in data:
            local_sample.multi_modal_inputs = data["multi_modal_inputs"][index]
        local_sample.uid = data['uid'][index]
        # local_sample = ray.put(local_sample)
        return local_sample
    loop = asyncio.get_event_loop()
    futures = []
    for index in range(batch_size):
        futures.append(calc_sample(index))
    samples = loop.run_until_complete(asyncio.gather(*futures))   
    del data 
    return samples

def Samples2Dict(samples: List[Sample]) -> TensorDict:
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
    fields = Sample.model_fields 
    sample_fields = [name for name in fields.keys()]

    aggregated: Dict[str, List[Any]] = {}
    for sample in samples:
        if sample is None:
            raise ValueError("Sample Should not be None")
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
    tensordict_data: Dict[str, Any] = {}
    batch_size = (len(samples),)  

    for key, values in aggregated.items():
        if isinstance(values, list):
            first_val = values[0]
            
            # if internal val is not ""/ {} ...
            if isinstance(first_val, np.ndarray):
                tensordict_data[key] = np.stack(values, axis=0) if first_val.ndim >= 1 else np.concatenate(values, axis=0)
                default_type = fields[key].annotation
                if get_origin(default_type) is Union:
                    args = get_args(default_type)       
                    actual_type = next((arg for arg in args if arg is not type(None)), None)
                    if actual_type is np.ndarray:
                        tensordict_data[key] = torch.tensor(tensordict_data[key]) 
                elif default_type is np.ndarray:
                    tensordict_data[key] = torch.tensor(tensordict_data[key])
            elif isinstance(first_val, str):
                if first_val:
                    tensordict_data[key] = values
            else:
                if first_val:
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