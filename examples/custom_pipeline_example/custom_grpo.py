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
Custom Pipeline Examples

This file demonstrates how users can define custom training pipelines
using the new Pipeline API. All functions are explicitly visible in the code.
"""

import numpy as np
from siirl.execution.dag.pipeline import Pipeline, NodeConfig
from siirl.execution.dag.task_graph import TaskGraph
from siirl.data_coordinator import DataProto
from siirl.dag_worker.data_structures import NodeOutput


# ============================================================================
# Example 1: Use Built-in Pipeline (Simplest)
# ============================================================================

def example_builtin_grpo() -> TaskGraph:
    """
    Simplest way: Use built-in GRPO pipeline directly.

    This is recommended for users who want to use standard algorithms
    without customization.
    """
    from siirl.execution.dag.builtin_pipelines import grpo_pipeline
    return grpo_pipeline()


# ============================================================================
# Example 2: GRPO with Custom Reward Function
# ============================================================================

def grpo_with_custom_reward() -> TaskGraph:
    """
    Customize the reward computation while keeping other parts standard.

    This example shows how to replace the reward node with a custom function
    while keeping the rest of the pipeline standard.
    """
    pipeline = Pipeline(
        "grpo_custom_reward",
        "GRPO pipeline with custom reward function"
    )

    # Standard rollout
    pipeline.add_node(
        "rollout_actor",
        func="siirl.dag_worker.dagworker:DAGWorker.generate",
        deps=[]
    )

    # Custom reward function (user's own implementation)
    pipeline.add_node(
        "custom_reward",
        func="examples.custom_pipeline_example.custom_grpo:my_custom_reward_fn",
        deps=["rollout_actor"]
    )

    # Standard advantage calculation and training
    pipeline.add_node(
        "calculate_advantages",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
        deps=["custom_reward"]
    ).add_node(
        "actor_old_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_old_log_prob",
        deps=["calculate_advantages"],
        only_forward_compute=True
    ).add_node(
        "reference_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_ref_log_prob",
        deps=["actor_old_log_prob"]
    ).add_node(
        "actor_train",
        func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
        deps=["reference_log_prob"]
    )

    return pipeline.build()


def my_custom_reward_fn(batch: DataProto, **kwargs) -> NodeOutput:
    """
    User's custom reward function.

    This function can implement any custom reward logic.
    Here we show a simple example, but users can implement
    arbitrarily complex reward computations.

    Args:
        batch: DataProto containing prompts and responses
        **kwargs: Additional arguments (config, etc.)

    Returns:
        NodeOutput: Batch with computed rewards
    """
    # Option 1: Use built-in reward computation as base
    from siirl.execution.scheduler.reward import compute_reward
    reward_output = compute_reward(batch, kwargs.get("config"))

    # Option 2: Fully custom reward logic
    # responses = batch.non_tensor_batch.get("responses", [])
    # custom_rewards = np.array([score_response(r) for r in responses])
    # batch.batch["rewards"] = custom_rewards
    # reward_output = NodeOutput(batch=batch, metrics={"avg_reward": custom_rewards.mean()})

    return reward_output

