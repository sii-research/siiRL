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

import uuid

import numpy as np
import torch

from siirl.scheduler.reward import compute_reward
from siirl.workers.dag.node import NodeRole
from siirl.workers.dag_worker.algorithms import apply_kl_penalty, compute_advantage, compute_response_mask
from siirl.workers.dag_worker.core_algos import agg_loss
from siirl.workers.dag_worker.data_structures import NodeOutput
from siirl.workers.databuffer import DataProto
from siirl.utils.debug import DistProfiler


class NodeExecutorsMixin:
    """Contains the specific execution methods for different node roles in the DAG."""

    from typing import Any, Dict
    from siirl.utils.params import SiiRLArguments
    from siirl.workers.dag.node import NodeRole
    from siirl.workers.base_worker import Worker

    agent_group_worker: Dict[int, Dict[NodeRole, Worker]]
    config: SiiRLArguments
    reward_fn: Any
    kl_ctrl_in_reward: Any
    _rank: int
    global_steps: int

    _prepare_generation_batch: Any
    _get_node_process_group: Any
    _get_node: Any
    _reduce_and_broadcast_metrics: Any

    @DistProfiler.annotate(role="generate")
    def generate(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """Generates sequences for a training batch using the rollout model."""
        if self.rollout_mode == 'sync':
            gen_batch = self._prepare_generation_batch(batch)
            if self.config.actor_rollout_ref.rollout.name == 'sglang':
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            gen_output = self.agent_group_worker[worker_group_index][NodeRole.ROLLOUT].generate_sequences(gen_batch)
            metrics = gen_output.meta_info.get("metrics", {})
            gen_output.meta_info = {}
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])
            batch = batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True).union(gen_output)
            if "response_mask" not in batch.batch:
                batch.batch["response_mask"] = compute_response_mask(batch)
            return NodeOutput(batch=batch, metrics=metrics)
        elif self._async_rollout_manager is not None:
            gen_batch = self._prepare_generation_batch(batch)
            gen_output = self._async_rollout_manager.generate_sequences(gen_batch)
            metrics = gen_output.meta_info.get("metrics", {})
            gen_output.meta_info = {}
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])
            batch = batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True).union(gen_output)
            if "response_mask" not in batch.batch:
                batch.batch["response_mask"] = compute_response_mask(batch)
            return NodeOutput(batch=batch, metrics=metrics)
        return NodeOutput(batch=batch, metrics={})

    @DistProfiler.annotate(role="compute_reward")
    def compute_reward(self, batch: DataProto, tp_size: int, **kwargs) -> NodeOutput:
        """Calculates rewards for a batch of generated sequences."""
        batch.meta_info["global_token_num"] = (torch.sum(batch.batch["attention_mask"], dim=-1) // tp_size).tolist()
        reward_tensor, extra_infos = compute_reward(batch, self.reward_fn)
        batch.batch["token_level_scores"] = reward_tensor

        if extra_infos:
            batch.non_tensor_batch.update({k: np.array(v) for k, v in extra_infos.items()})

        metrics = {}
        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl_in_reward, self.config.algorithm.kl_penalty)
            metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        return NodeOutput(batch=batch, metrics=metrics)

    @DistProfiler.annotate(role="compute_old_log_prob")
    def compute_old_log_prob(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Computes log probabilities from the actor model before the policy update."""
        if 'global_token_num' not in batch.meta_info:
            # in multi-agent, agentA may don't have reward node
            # insert some info needed
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.ACTOR].compute_log_prob(batch)
        process_group = self._get_node_process_group(self._get_node(NodeRole.ACTOR, worker_group_index))

        local_metrics = processed_data.meta_info.get("metrics", {})
        if "entropys" in processed_data.batch:
            entropy = agg_loss(processed_data.batch["entropys"], processed_data.batch["response_mask"].to("cpu"), self.config.actor_rollout_ref.actor.loss_agg_mode)
            local_metrics["actor/entropy_loss"] = entropy.item()
        metrics = self._reduce_and_broadcast_metrics(local_metrics, process_group)

        processed_data.meta_info.pop("metrics", None)
        processed_data.batch.pop("entropys", None)

        if "rollout_log_probs" in processed_data.batch and self._rank == 0:
            rollout_probs, actor_probs = torch.exp(processed_data.batch["rollout_log_probs"]), torch.exp(processed_data.batch["old_log_probs"])
            rollout_probs_diff = torch.masked_select(torch.abs(rollout_probs.cpu() - actor_probs), processed_data.batch["response_mask"].bool().cpu())
            if rollout_probs_diff.numel() > 0:
                metrics.update({"training/rollout_probs_diff_max": torch.max(rollout_probs_diff).item(), "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).item(), "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).item()})
        return NodeOutput(batch=processed_data, metrics=metrics)

    @DistProfiler.annotate(role="compute_ref_log_prob")
    def compute_ref_log_prob(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Computes log probabilities from the frozen reference model."""
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.REFERENCE].compute_ref_log_prob(batch)
        metrics = processed_data.meta_info.get("metrics", {})
        return NodeOutput(batch=processed_data, metrics=metrics)

    @DistProfiler.annotate(role="compute_value")
    def compute_value(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Computes value estimates from the critic model."""
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.CRITIC].compute_values(batch)
        return NodeOutput(batch=processed_data)

    @DistProfiler.annotate(role="compute_advantage")
    def compute_advantage(self, batch: DataProto, **kwargs) -> NodeOutput:
        """Computes advantages and returns for PPO using GAE."""
        adv_config = self.config.algorithm
        rollout_config = self.config.actor_rollout_ref.rollout
        if 'token_level_rewards' not in batch.batch:
            # make sure rewards of angentB has been compute
            cur_node = kwargs['cur_node']
            if depend_nodes := self.taskgraph.get_dependencies(cur_node.node_id):
                depend_node = depend_nodes[0]
                batch.batch['token_level_rewards'] = torch.zeros_like(
                    batch.batch[f"agent_group_{depend_node.agent_group}_token_level_rewards"]
                )
                node_output = self.compute_value(batch, cur_node.agent_group)
                node_output.batch.batch['pre_values'] = batch.batch[f"agent_group_{depend_node.agent_group}_values"]
                node_output.batch.batch['pre_advantages'] = batch.batch[f"agent_group_{depend_node.agent_group}_advantages"]
                
                batch = node_output.batch    
            else:
                raise RuntimeError(f"cur_node {cur_node.node_id} have no rewards with can't find it's dependencies reward")
        return NodeOutput(
            batch=compute_advantage(
                batch, adv_estimator=adv_config.adv_estimator, gamma=adv_config.gamma, lam=adv_config.lam, num_repeat=rollout_config.n, norm_adv_by_std_in_grpo=adv_config.norm_adv_by_std_in_grpo, weight_factor_in_cpgd=adv_config.weight_factor_in_cpgd, multi_turn=rollout_config.multi_turn.enable
            )
        )

    @DistProfiler.annotate(role="train_critic")
    def train_critic(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Performs a single training step on the critic model."""
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.CRITIC].update_critic(batch)
        process_group = self._get_node_process_group(self._get_node(NodeRole.CRITIC, worker_group_index))
        metrics = self._reduce_and_broadcast_metrics(processed_data.meta_info.get("metrics"), process_group)
        return NodeOutput(batch=processed_data, metrics=metrics)

    @DistProfiler.annotate(role="train_actor")
    def train_actor(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Performs a single training step on the actor (policy) model."""
        if self.config.trainer.critic_warmup > self.global_steps:
            return NodeOutput(batch=batch)  # Skip actor update during critic warmup

        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.ACTOR].update_actor(batch)
        process_group = self._get_node_process_group(self._get_node(NodeRole.ACTOR, worker_group_index))
        metrics = self._reduce_and_broadcast_metrics(processed_data.meta_info.get("metrics"), process_group)
        return NodeOutput(batch=processed_data, metrics=metrics)
