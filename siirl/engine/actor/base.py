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
The base class for Actor
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch

from tensordict import TensorDict

__all__ = ["BasePPOActor"]


class BasePPOActor(ABC):
    def __init__(self, config):
        """The base class for PPO actor

        Args:
            config (DictConfig): a config passed to the PPOActor. We expect the type to be
                DictConfig (https://omegaconf.readthedocs.io/), but it can be any namedtuple in general.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: TensorDict) -> torch.Tensor:
        """Compute logits given a batch of data.

        Args:
            data (TensorDict): a batch of data represented by TensorDict. It must contain key ```input_ids```,
                ```attention_mask``` and ```position_ids```.

        Returns:
            TensorDict: a TensorDict containing the key ```log_probs```


        """
        pass

    @abstractmethod
    def update_policy(self, data: TensorDict) -> Dict:
        """Update the policy with an iterator of TensorDict

        Args:
            data (TensorDict): an iterator over the TensorDict that returns by
                ```make_minibatch_iterator```

        Returns:
            Dict: a dictionary contains anything. Typically, it contains the statistics during updating the model
            such as ```loss```, ```grad_norm```, etc,.

        """
        pass
