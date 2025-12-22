# Copyright (c) 2025, Shanghai Innovation Institute.  All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     <url id="d0rb3ebacc47j8if2om0" type="url" status="parsed" title="Apache License, Version 2.0" wc="10467">http://www.apache.org/licenses/LICENSE-2.0</url>
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, Dict, Optional, Tuple
import yaml
import json
import os
import importlib
import ray
import sys
from loguru import logger


# --- 1. Abstract Base Class for Environment ---


async def get_ddp_world_size_rank(local_world_size, local_rank, local_parallel_size):
    ddp_world_size = local_world_size // local_parallel_size
    ddp_rank = local_rank // local_parallel_size
    return ddp_world_size, ddp_rank


class BaseEnvironment:
    """
    BaseEnvironment defines functions for users to implement
    """
    def __init__(self):
        logger.info(f"[BaseEnvironment] Environment  initialized, configuration") 
    
    @abstractmethod
    async def reset(self) -> Any:
        """
        Resets the environment to its initial state.
        Args:
            seed: The seed used for environment randomness.
            options: Environment-specific reset options.
        Returns:
            Initial observation (raw observation).
        """
        logger.info("Reset not implemented")
        raise NotImplementedError

    @abstractmethod
    async def step(self, actions: Any, ground_truth: Optional[Any] = None) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Executes an action in the environment. Subclasses can override this method. Alternatively, you can override do_action, get_rewards, and get_obs, but with limited flexibility.
        Args:
            action: Input parameters for executing the action.
        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
                   - observation: The environment's raw observation.
                   - reward: The reward obtained after executing the action.
                   - info: A dictionary containing additional diagnostic information as needed.
        """
        logger.info("Step not implemented")
        raise NotImplementedError
        return next_obs, rewards, info

    async def get_rewards(self, actions: Any) -> Any:
        """Returns the rewards of action"""
        return None
