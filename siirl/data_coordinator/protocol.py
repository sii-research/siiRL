# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

import contextlib
import copy
import logging
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import ray
import tensordict
import torch
import torch.distributed
import torch.nn.functional as F
from packaging import version
from tensordict import TensorDict
from torch.utils.data import DataLoader

from siirl.utils.extras.device import get_device_id, get_torch_device
from siirl.utils.extras.py_functional import union_two_dict
from siirl.utils.model_utils.torch_functional import allgather_dict_tensors

__all__ = ["union_tensor_dict"]

with contextlib.suppress(Exception):
    tensordict.set_lazy_legacy(False).set()


class _TensorDictConfigMeta(type):
    _config = {}

    auto_padding_key = "_siirl_auto_padding"

    @property
    def auto_padding(cls):
        enabled_by_env = os.getenv("SIIRL_AUTO_PADDING", "FALSE").upper() in ["TRUE", "1"]
        return enabled_by_env or cls._config.get(cls.auto_padding_key, False)

    @auto_padding.setter
    def auto_padding(cls, enabled: bool):
        assert isinstance(enabled, bool), f"enabled must be a boolean, got {enabled} as {type(enabled)}"
        cls._config[cls.auto_padding_key] = enabled


class TensorDictConfig(metaclass=_TensorDictConfigMeta):
    pass


_padding_size_key = "_padding_size_key_x123d"


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, (
        f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    )
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert tensor_dict1[key].equal(tensor_dict2[key]), (
                f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
            )

    return tensor_dict1


def union_numpy_dict(tensor_dict1: dict[str, np.ndarray], tensor_dict2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            # to properly deal with nan and object type
            assert pd.DataFrame(tensor_dict2[key]).equals(pd.DataFrame(tensor_dict1[key])), (
                f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
            )
        tensor_dict1[key] = val

    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output


def all_gather_data_proto(data: TensorDict, process_group):
    # Note that this is an inplace operator just like torch.distributed.all_gather
    group_size = torch.distributed.get_world_size(group=process_group)
    print(f"all gather dataproto process_group size:{group_size}")
    assert isinstance(data, TensorDict)
    prev_device = data.device
    data = data.to(get_device_id())
    data = allgather_dict_tensors(data.contiguous(), size=group_size, group=process_group, dim=0)
    data = data.to(prev_device)


def select_idxs(batch: TensorDict, idxs):
    """
    Select specific indices from the TensorDict.

    Args:
        batch (TensorDict): data to be select
        idxs (torch.Tensor or numpy.ndarray or list): Indices to select

    Returns:
        TensorDict: A new TensorDict containing only the selected indices
    """
    if isinstance(idxs, list):
        idxs = torch.tensor(idxs)
        if idxs.dtype != torch.bool:
            idxs = idxs.type(torch.int32)

    if isinstance(idxs, np.ndarray):
        idxs_np = idxs
        idxs_torch = torch.from_numpy(idxs)
    else:  # torch.Tensor
        idxs_torch = idxs
        idxs_np = idxs.detach().cpu().numpy()

    batch_size = int(idxs_np.sum()) if idxs_np.dtype == bool else idxs_np.shape[0]
    filtered_data = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            filtered_data[key] = value[idxs_torch]
        elif isinstance(value, np.ndarray):
            filtered_data[key] = value[idxs_np]
        elif isinstance(value, list):
            filtered_data[key] = np.ndarray(value)[idxs_np]
        else:
            filtered_data[key] = value
    return TensorDict(filtered_data, batch_size=batch_size)