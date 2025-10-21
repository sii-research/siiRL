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
Built-in pipeline definitions for standard RL algorithms.

All function paths are explicitly visible, making it easy to understand
what each node in the pipeline executes.
"""

from siirl.execution.dag.pipeline import Pipeline
from siirl.execution.dag.task_graph import TaskGraph
from siirl.execution.dag.node import NodeType, NodeRole


def grpo_pipeline() -> TaskGraph:
    """
    Standard GRPO (Group Relative Policy Optimization) pipeline.

    Workflow:
        1. rollout_actor: Generate sequences using the policy model
        2. function_reward: Compute rewards for generated sequences
        3. calculate_advantages: Calculate advantage estimates
        4. actor_old_log_prob: Compute log probabilities with old policy (forward only)
        5. reference_log_prob: Compute log probabilities with reference model
        6. actor_train: Train the actor model

    Returns:
        TaskGraph: A validated task graph ready for execution
    """
    pipeline = Pipeline("grpo_training_pipeline", "Standard GRPO workflow")

    # All function paths are explicitly visible!
    pipeline.add_node(
        "rollout_actor",
        func="siirl.dag_worker.dagworker:DAGWorker.generate",
        deps=[],
        node_type=NodeType.MODEL_INFERENCE,
        node_role=NodeRole.ROLLOUT
    ).add_node(
        "function_reward",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
        deps=["rollout_actor"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.REWARD
    ).add_node(
        "calculate_advantages",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
        deps=["function_reward"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.ADVANTAGE
    ).add_node(
        "actor_old_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_old_log_prob",
        deps=["calculate_advantages"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR,
        only_forward_compute=True
    ).add_node(
        "reference_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_ref_log_prob",
        deps=["actor_old_log_prob"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.REFERENCE
    ).add_node(
        "actor_train",
        func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
        deps=["reference_log_prob"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR
    )

    return pipeline.build()


def ppo_pipeline() -> TaskGraph:
    """
    Standard PPO (Proximal Policy Optimization) pipeline.

    Workflow:
        1. rollout_actor: Generate sequences using the policy model
        2. function_reward: Compute rewards for generated sequences
        3. compute_value: Compute value function estimates (forward only)
        4. calculate_advantages: Calculate GAE (Generalized Advantage Estimation)
        5. actor_old_log_prob: Compute log probabilities with old policy (forward only)
        6. reference_log_prob: Compute log probabilities with reference model
        7. actor_train: Train the actor model
        8. critic_train: Train the critic (value) model

    Returns:
        TaskGraph: A validated task graph ready for execution
    """
    pipeline = Pipeline("ppo_training_pipeline", "Standard PPO workflow")

    pipeline.add_node(
        "rollout_actor",
        func="siirl.dag_worker.dagworker:DAGWorker.generate",
        deps=[],
        node_type=NodeType.MODEL_INFERENCE,
        node_role=NodeRole.ROLLOUT
    ).add_node(
        "function_reward",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
        deps=["rollout_actor"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.REWARD
    ).add_node(
        "compute_value",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_value",
        deps=["function_reward"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.CRITIC,
        only_forward_compute=True
    ).add_node(
        "calculate_advantages",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
        deps=["compute_value"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.ADVANTAGE
    ).add_node(
        "actor_old_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_old_log_prob",
        deps=["calculate_advantages"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR,
        only_forward_compute=True
    ).add_node(
        "reference_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_ref_log_prob",
        deps=["actor_old_log_prob"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.REFERENCE
    ).add_node(
        "actor_train",
        func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
        deps=["reference_log_prob"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR
    ).add_node(
        "critic_train",
        func="siirl.dag_worker.dagworker:DAGWorker.train_critic",
        deps=["reference_log_prob"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.CRITIC
    )

    return pipeline.build()


def dapo_pipeline() -> TaskGraph:
    """
    DAPO (Data-Augmented Policy Optimization) pipeline.

    DAPO is a variant of GRPO with an additional post-processing step
    after rollout for sampling augmentation.

    Workflow:
        1. rollout_actor: Generate sequences using the policy model
        2. postprocess_sampling: DAPO-specific sampling post-processing
        3. function_reward: Compute rewards for augmented sequences
        4. calculate_advantages: Calculate advantage estimates
        5. actor_old_log_prob: Compute log probabilities with old policy (forward only)
        6. reference_log_prob: Compute log probabilities with reference model
        7. actor_train: Train the actor model

    Returns:
        TaskGraph: A validated task graph ready for execution
    """
    pipeline = Pipeline("dapo_training_pipeline", "DAPO workflow")

    pipeline.add_node(
        "rollout_actor",
        func="siirl.dag_worker.dagworker:DAGWorker.generate",
        deps=[],
        node_type=NodeType.MODEL_INFERENCE,
        node_role=NodeRole.ROLLOUT
    ).add_node(
        "postprocess_sampling",
        func="siirl.dag_worker.dagworker:DAGWorker.postprocess_sampling",
        deps=["rollout_actor"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.POSTPROCESS_SAMPLING
    ).add_node(
        "function_reward",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
        deps=["postprocess_sampling"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.REWARD
    ).add_node(
        "calculate_advantages",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
        deps=["function_reward"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.ADVANTAGE
    ).add_node(
        "actor_old_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_old_log_prob",
        deps=["calculate_advantages"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR,
        only_forward_compute=True
    ).add_node(
        "reference_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_ref_log_prob",
        deps=["actor_old_log_prob"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.REFERENCE
    ).add_node(
        "actor_train",
        func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
        deps=["reference_log_prob"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR
    )

    return pipeline.build()


__all__ = ["grpo_pipeline", "ppo_pipeline", "dapo_pipeline"]
