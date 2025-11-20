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
    
    # used for vla
    pixel_values: Optional[np.ndarray] = Field(default=None)
    finish_step: Optional[np.ndarray] = Field(default=None)
    
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
    # Manually repeat all numpy arrays and torch tensors
    # This ensures consistent handling of all fields
    batch_size = None
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # Repeat numpy arrays along axis 0
            data[key] = np.repeat(value, n, axis=0)
        elif isinstance(value, torch.Tensor):
            # Repeat torch tensors along dim 0
            if batch_size is None:
                batch_size = value.shape[0]
            data[key] = value.repeat_interleave(n, dim=0)
            
        elif isinstance(value, list):
            # Convert list to numpy array and repeat
            data[key] = np.repeat(np.array(value), n, axis=0)
            
    # Create integer indices for GRPO grouping
    # Each prompt gets a unique index (0, 1, 2, ..., batch_size-1)
    # This will be repeated to [0,0,0,...,1,1,1,...,2,2,2,...] after repeat
    uid = np.arange(batch_size, dtype=np.int64)
    data['uid'] = np.repeat(uid, n, axis=0)
    # Now all fields have batch_size * n
    # Create TensorDict with the expanded batch size
    tensor_dict = TensorDict(data, batch_size=batch_size * n)
    
    return tensor_dict

def Dict2Samples(data:TensorDict)-> List[SampleManager]:
    batch_size = data.batch_size[0]
    async def calc_sample(index):
        local_sample = Sample()
        local_sample.input_ids = data['input_ids'][index].numpy() if 'input_ids' in data else None
        local_sample.attention_mask = data['attention_mask'][index].numpy() if 'attention_mask' in data else None
        local_sample.position_ids = data['position_ids'][index].numpy() if 'position_ids' in data else None
        local_sample.data_source = data['data_source'][index] if 'data_source' in data else None
        local_sample.reward_model = data['reward_model'][index] if 'reward_model' in data else None
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
        local_sample.pixel_values = data['pixel_values'][index].numpy() if 'pixel_values' in data else None
        local_sample.finish_step = data['finish_step'][index].numpy() if 'finish_step' in data else None
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
                tensordict_data[key] = np.stack(values, axis=0) if first_val.ndim >= 1 else np.array(values)
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

    tensordict_data["global_token_num"] = NonTensorData(torch.sum(tensordict_data["attention_mask"], dim=-1).flatten().tolist())

    return TensorDict(tensordict_data, batch_size=batch_size)

def filter_tensordict(batch: TensorDict, indices: List[int]) -> TensorDict:
    """
    Filter a TensorDict by selecting only the samples at the specified indices.
    
    This function is used by DAPO to filter out trajectory groups with zero variance.
    It properly handles both regular tensor fields and NonTensorData fields.
    
    Args:
        batch: The input TensorDict to filter
        indices: List of indices to keep
        
    Returns:
        A new TensorDict containing only the selected samples
    """
    if not indices:
        # Return an empty TensorDict with the same structure but batch_size=0
        return TensorDict({}, batch_size=(0,))
    
    # Convert indices to both tensor and numpy for different field types
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    indices_np = np.array(indices, dtype=np.int64)
    target_batch_size = len(indices)
    original_batch_size = batch.batch_size[0] if isinstance(batch.batch_size, tuple) else batch.batch_size
    
    from loguru import logger
    
    # Manually filter each field to ensure NonTensorData is handled correctly
    filtered_dict = {}
    for key, value in batch.items():
        try:
            if isinstance(value, NonTensorData):
                # NonTensorData needs special handling
                if isinstance(value.data, np.ndarray):
                    # numpy array wrapped in NonTensorData
                    data_len = len(value.data)
                    if data_len == original_batch_size:
                        # This is batched data - filter it
                        filtered_data = value.data[indices_np]
                        filtered_dict[key] = NonTensorData(data=filtered_data, batch_size=[target_batch_size])
                    else:
                        # This is metadata (length doesn't match batch_size) - keep as is
                        filtered_dict[key] = value
                elif isinstance(value.data, (list, tuple)):
                    # list/tuple wrapped in NonTensorData
                    data_len = len(value.data)
                    if data_len == original_batch_size:
                        # This is batched data - filter it
                        filtered_data = [value.data[i] for i in indices]
                        filtered_dict[key] = NonTensorData(data=filtered_data, batch_size=[target_batch_size])
                    else:
                        # This is metadata (length doesn't match batch_size) - keep as is
                        filtered_dict[key] = value
                else:
                    # scalar or other type - keep as is (it's metadata)
                    filtered_dict[key] = value
            elif isinstance(value, torch.Tensor):
                # Regular tensor - use tensor indexing
                filtered_dict[key] = value[indices_tensor]
            else:
                # Other types - try tensor indexing or keep as is
                try:
                    filtered_dict[key] = value[indices_tensor]
                except:
                    filtered_dict[key] = value
        except Exception as e:
            # Provide detailed error message for debugging
            raise RuntimeError(
                f"Error filtering field '{key}' in filter_tensordict: {e}\n"
                f"  Field type: {type(value)}\n"
                f"  Value type: {type(value.data) if isinstance(value, NonTensorData) else 'N/A'}\n"
                f"  Data length: {len(value.data) if isinstance(value, NonTensorData) and hasattr(value.data, '__len__') else 'N/A'}\n"
                f"  Original batch size: {original_batch_size}\n"
                f"  Target batch size: {target_batch_size}\n"
                f"  Indices: {indices[:10]}... (showing first 10)\n"
                f"  Max index: {max(indices) if indices else 'N/A'}"
            ) from e
    
    # Create new TensorDict with filtered data
    filtered_batch = TensorDict(filtered_dict, batch_size=target_batch_size)
    
    return filtered_batch