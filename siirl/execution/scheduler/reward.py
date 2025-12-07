# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

"""
This module provides functionalities for dynamically loading and computing rewards
"""

import importlib.util
import multiprocessing
import os
import sys
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union
from tensordict import TensorDict
import ray
from loguru import logger

from siirl.params import SiiRLArguments
from siirl.utils.reward_score import default_compute_score
from siirl.engine.reward_manager import (
    DAPORewardManager,
    NaiveRewardManager,
    ParallelRewardManager,
    EmbodiedRewardManager
)

Tokenizer = Any
RewardTensor = Any
AnyRewardManager = Union[NaiveRewardManager, DAPORewardManager, ParallelRewardManager, EmbodiedRewardManager]


def load_custom_reward_function(config: SiiRLArguments) -> Optional[Callable]:
    """
    Dynamically loads a custom reward function from a user-specified file.

    This function reads the path and function name from the configuration,
    imports the module, and returns the specified function.

    Args:
        config: The main SiiRLArguments configuration object which contains
                the `custom_reward_function` settings.

    Returns:
        The loaded custom reward function wrapped with its configured keyword
        arguments, or None if no custom function is specified.

    Raises:
        FileNotFoundError: If the specified Python file does not exist.
        AttributeError: If the function is not found within the specified file.
        RuntimeError: If the module cannot be loaded for other reasons.
    """
    reward_fn_config = config.custom_reward_function
    file_path = reward_fn_config.path

    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Custom reward function file not found: '{file_path}'")

    # Dynamically import the module from the given file path.
    module_name = "custom_module"  # A placeholder name for the module.
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create module spec from '{file_path}'")

    module = importlib.util.module_from_spec(spec)
    # This allows the module to be discoverable by other parts of the system
    # if necessary, for instance during deserialization (unpickling).
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Failed to execute module from '{file_path}': {e}") from e

    function_name = reward_fn_config.name
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in custom reward file '{file_path}'.")

    logger.info(f"Using custom reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)
    reward_kwargs = dict(reward_fn_config.reward_kwargs)

    # Wrap the function to pre-fill the custom keyword arguments.
    return partial(raw_fn, **reward_kwargs)


def create_reward_manager(
    config: SiiRLArguments,
    tokenizer: Tokenizer,
    num_examine: int,
    **reward_kwargs,
) -> AnyRewardManager:
    """
    Factory function to instantiate and return the appropriate reward manager.

    It selects the reward manager class based on the configuration and wires it
    up with the correct scoring function, which can be a default, a sandbox-
    based, or a custom function.

    Args:
        config: The SiiRLArguments configuration object.
        tokenizer: The tokenizer instance to be used by the reward manager.
        num_examine: The number of candidates to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instantiated reward manager object.

    Raises:
        NotImplementedError: If the specified `reward_manager_name` is unknown.
    """
    
    # Map manager names to their respective classes for clean, extensible selection.
    manager_map = {
        "naive": NaiveRewardManager,
        "dapo": DAPORewardManager,
        "parallel": ParallelRewardManager,
        "embodied": EmbodiedRewardManager
    }

    # Map each manager to its default compute_score function
    # Note: compute_embodied_reward is imported lazily to avoid loading sklearn for LLM/VLM tasks
    default_compute_score_map = {
        "naive": default_compute_score,
        "prime": default_compute_score,
        "batch": default_compute_score,
        "dapo": default_compute_score,
        "parallel": default_compute_score,
        "embodied": None,  # Will be loaded lazily if needed
    }
    reward_manager_name = config.reward_model.reward_manager
    reward_manager_cls = manager_map.get(reward_manager_name)

    if reward_manager_cls is None:
        raise NotImplementedError(f"Reward manager '{reward_manager_name}' is not implemented.")

    # Determine the final scoring function.
    # Priority: Custom function > Sandbox function > Default function
    compute_score_fn = load_custom_reward_function(config)

    if compute_score_fn is None:
        sandbox_config = config.reward_model.sandbox_fusion
        sandbox_url = sandbox_config.get("url") if sandbox_config else None

        if sandbox_url:
            logger.info(f"Using sandbox-based reward scoring at URL: {sandbox_url}")
            # This semaphore should be managed carefully. Creating it here assumes
            # this function is called once per worker/process.
            manager = multiprocessing.Manager()
            semaphore = manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            compute_score_fn = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=semaphore,
            )
        else:
            # Fallback to the default scoring function.
            compute_score_fn = default_compute_score_map.get(
                reward_manager_name,
                default_compute_score  # Fallback for any unmapped managers
            )
            
            # Lazy import for embodied reward to avoid loading sklearn for LLM/VLM tasks
            if compute_score_fn is None and reward_manager_name == "embodied":
                from siirl.utils.reward_score.embodied import compute_embodied_reward
                compute_score_fn = compute_embodied_reward
                logger.info("Loaded embodied reward function (with sklearn dependencies)")

    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score_fn,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: TensorDict, reward_fn: Callable) -> Tuple[RewardTensor, Dict[str, Any]]:
    """
    Computes the reward for a given batch of data using the provided function.

    This function includes robust error handling. If the reward function fails,
    it logs a warning and returns a placeholder reward (e.g., zero) instead of
    crashing.

    Args:
        data: A TensorDict object containing the batch of input data.
        reward_fn: The reward function or manager method to call.

    Returns:
        A tuple containing:
        - The reward tensor for the batch.
        - A dictionary with any extra metadata returned by the reward function.
    """
    try:
        # Assumes reward_fn can return a dictionary with specific keys.
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        extra_info = reward_result.get("reward_extra_info", {})
    except Exception:
        # If the structured return fails, try a simpler call.
        try:
            reward_tensor = reward_fn(data)
            extra_info = {}
        except Exception as e:
            logger.warning(f"Error computing reward: {e}. Returning a zero tensor.")
            # Create a zero tensor of the expected shape as a fallback.
            # This requires knowing the expected tensor type and shape.
            # Assuming a shape of (batch_size,) and using a generic placeholder.
            # This part may need adjustment based on the actual tensor library (torch/tf).
            batch_size = len(data.prompts)  # Example of getting batch size
            reward_tensor = [0.0] * batch_size  # Placeholder for actual tensor
            extra_info = {}

    return reward_tensor, extra_info
