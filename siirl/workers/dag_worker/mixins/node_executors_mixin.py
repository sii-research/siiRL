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

from siirl.scheduler.enums import AdvantageEstimator
from siirl.scheduler.reward import compute_reward
from siirl.utils.debug import DistProfiler
from siirl.workers.dag.node import NodeRole
from siirl.workers.dag_worker.algorithms import apply_kl_penalty, compute_advantage, compute_response_mask
from siirl.workers.dag_worker.core_algos import agg_loss
from siirl.workers.dag_worker.data_structures import NodeOutput
from siirl.workers.databuffer import DataProto


class NodeExecutorsMixin:
    """Contains the specific execution methods for different node roles in the DAG."""

    from typing import Any, Dict

    from siirl.utils.params import SiiRLArguments
    from siirl.workers.base_worker import Worker
    from siirl.workers.dag.node import NodeRole

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
    _find_first_non_compute_ancestor: Any
    rebalance_sampled_data: Any

    @DistProfiler.annotate(role="generate")
    def generate_sync_mode(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """
        Sync mode generate
        """
        gen_batch: DataProto = self._prepare_generation_batch(batch)
        if self.config.actor_rollout_ref.rollout.name == "sglang":
            gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        gen_output = self.agent_group_worker[worker_group_index][NodeRole.ROLLOUT].generate_sequences(gen_batch)
        metrics = gen_output.meta_info.get("metrics", {})
        gen_output.meta_info = {}
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])
        batch = batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True).union(gen_output)
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)
        return NodeOutput(batch=batch, metrics=metrics)

    @DistProfiler.annotate(role="generate")
    def generate_async_mode(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """Generates sequences for a training batch using the async rollout model."""
        if self._async_rollout_manager is not None:
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

    @DistProfiler.annotate(role="generate")
    def generate_multi_agent_mode(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """Generates sequences for a training batch using the multi-agent rollout model."""
        gen_batch = self._prepare_generation_batch(batch)
        if self.config.actor_rollout_ref.rollout.agent.rewards_with_env and "reward_model" in batch.non_tensor_batch:
            gen_batch.non_tensor_batch["reward_model"] = batch.non_tensor_batch["reward_model"]
        assert self.config.actor_rollout_ref.rollout.name == "sglang"
        gen_output = self.multi_agent_loop.generate_sequence(gen_batch)
        if gen_output:
            metrics = gen_output.meta_info.get("metrics", {})
            # gen_output.meta_info = {}
            # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])
            # batch = batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True).union(gen_output)
            # if "response_mask" not in batch.batch:
            #     batch.batch["response_mask"] = compute_response_mask(batch)
            return NodeOutput(batch=gen_output, metrics=metrics)
        return NodeOutput(batch=batch, metrics={})

    @DistProfiler.annotate(role="generate")
    def generate_embodied_mode(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """
        Generates embodied episodes for training.
        
        This method follows the same pattern as _generate_for_embodied_validation in validation_mixin,
        but configured for training mode (do_sample=True, validate=False).
        
        For embodied tasks, the batch contains task metadata (task_id, trial_id, etc.) from the dataloader.
        The rollout worker interacts with the environment and generates all required data
        (input_ids, pixel_values, responses, etc.) during environment rollout.
        
        Unlike text generation, we do NOT call _prepare_generation_batch because:
        1. The input batch doesn't have text-generation keys (input_ids, attention_mask, etc.)
        2. These keys will be generated by the embodied rollout worker during env interaction
        """
        from loguru import logger
        
        rollout_worker = self.agent_group_worker[worker_group_index][NodeRole.ROLLOUT]
        
        # Set meta_info for embodied training (aligned with verl implementation)
        # Reference: verl/trainer/ppo/ray_trainer.py:526-530
        batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') else None,
            "n_samples": self.config.actor_rollout_ref.rollout.n,
            "pad_token_id": self.tokenizer.pad_token_id if hasattr(self, 'tokenizer') else None,
        }
        
        logger.info(f"[Embodied Training] Batch size: {batch.batch.batch_size[0]}, meta_info: {batch.meta_info}")
        
        # Generate embodied episodes
        gen_output = rollout_worker.generate_sequences(batch)
        metrics = gen_output.meta_info.get("metrics", {})
        gen_output.meta_info = {}
        
        # Add unique IDs for tracking
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])
        
        # Union the generated data with the original batch
        # Note: For embodied, we don't need to repeat the batch since rollout already handles n_samples
        batch = batch.union(gen_output)
        
        # Compute response mask if not already present
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)
        
        return NodeOutput(batch=batch, metrics=metrics)

    @DistProfiler.annotate(role="generate")
    def generate(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """Generates sequences for a training batch using the rollout model."""
        # Check if this is embodied mode
        is_embodied = self.config.actor_rollout_ref.model.model_type == "embodied"
        
        if is_embodied:
            # Use dedicated embodied generation path (mirrors validation logic)
            return self.generate_embodied_mode(worker_group_index, batch, **kwargs)
        elif self._multi_agent is False:
            if self.rollout_mode == "sync":
                return self.generate_sync_mode(worker_group_index, batch, **kwargs)
            else:
                return self.generate_async_mode(worker_group_index, batch, **kwargs)
        else:
            return self.generate_multi_agent_mode(worker_group_index, batch, **kwargs)

    @DistProfiler.annotate(role="compute_reward")
    def compute_reward(self, batch: DataProto, tp_size: int, **kwargs) -> NodeOutput:
        """Calculates rewards for a batch of generated sequences."""
        if "token_level_rewards" in batch.batch:
            return NodeOutput(batch=batch, metrics={})
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
        if "global_token_num" not in batch.meta_info:
            # in multi-agent, agentA may don't have reward node
            # insert some info needed
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.ACTOR].compute_log_prob(batch)
        process_group = self._get_node_process_group(self._get_node(NodeRole.ACTOR, worker_group_index))

        local_metrics = processed_data.meta_info.get("metrics", {})
        if "entropys" in processed_data.batch:
            entropy = agg_loss(
                processed_data.batch["entropys"],
                processed_data.batch["response_mask"].to("cpu"),
                self.config.actor_rollout_ref.actor.loss_agg_mode,
            )
            local_metrics["actor/entropy_loss"] = entropy.item()
        metrics = self._reduce_and_broadcast_metrics(local_metrics, process_group)

        processed_data.meta_info.pop("metrics", None)
        processed_data.batch.pop("entropys", None)

        if "rollout_log_probs" in processed_data.batch and self._rank == 0:
            rollout_probs, actor_probs = (
                torch.exp(processed_data.batch["rollout_log_probs"]),
                torch.exp(processed_data.batch["old_log_probs"]),
            )
            rollout_probs_diff = torch.masked_select(
                torch.abs(rollout_probs.cpu() - actor_probs), processed_data.batch["response_mask"].bool().cpu()
            )
            if rollout_probs_diff.numel() > 0:
                metrics.update(
                    {
                        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).item(),
                        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).item(),
                        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).item(),
                    }
                )
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
    def compute_multi_agent_advantage(self, batch: DataProto, **kwargs) -> NodeOutput:
        adv_config = self.config.algorithm
        rollout_config = self.config.actor_rollout_ref.rollout
        cur_node = kwargs["cur_node"]
        if "token_level_rewards" not in batch.batch:
            # make sure rewards of angentB has been compute
            # GAE_MARFT adv need make sure only last agent has adv node
            if depend_nodes := self.taskgraph.get_dependencies(cur_node.node_id):
                depend_node = depend_nodes[0]
                if adv_config.share_reward_in_agent:
                    batch.batch["token_level_rewards"] = batch.batch[
                        f"agent_group_{depend_node.agent_group}_token_level_rewards"
                    ].clone()
                else:
                    batch.batch["token_level_rewards"] = torch.zeros_like(
                        batch.batch[f"agent_group_{depend_node.agent_group}_token_level_rewards"]
                    )
                batch.batch["token_level_scores"] = batch.batch[
                    f"agent_group_{depend_node.agent_group}_token_level_scores"
                ].clone()
            else:
                raise RuntimeError(
                    f"cur_node {cur_node.node_id} have no rewards with can't find it's dependencies reward"
                )
        if adv_config.adv_estimator == AdvantageEstimator.GAE_MARFT:
            # make sure adv node define in last agent node
            cur_agent_id = len(self.agent_group_worker) - 1
            agent_groups_ids = list(range(cur_agent_id))
            kwargs["agent_group_ids"] = agent_groups_ids
            # pre_agent may have no reward token
            for agent_id in reversed(agent_groups_ids):
                key_prefix = f"agent_group_{agent_id}_token_level_rewards"
                if key_prefix not in batch.batch:
                    pre_key_prefix = (
                        f"agent_group_{agent_id + 1}_token_level_rewards"
                        if agent_id != cur_agent_id - 1
                        else "token_level_rewards"
                    )
                    if adv_config.share_reward_in_agent:
                        batch.batch[key_prefix] = batch.batch[pre_key_prefix].clone()
                    else:
                        batch.batch[key_prefix] = torch.zeros_like(batch.batch[pre_key_prefix])
                batch.batch[f"agent_group_{agent_id}_token_level_scores"] = batch.batch[key_prefix].clone()

        return NodeOutput(
            batch=compute_advantage(
                batch,
                adv_estimator=adv_config.adv_estimator,
                gamma=adv_config.gamma,
                lam=adv_config.lam,
                num_repeat=rollout_config.n,
                norm_adv_by_std_in_grpo=adv_config.norm_adv_by_std_in_grpo,
                weight_factor_in_cpgd=adv_config.weight_factor_in_cpgd,
                multi_turn=rollout_config.multi_turn.enable,
                **kwargs,
            )
        )

    @DistProfiler.annotate(role="compute_advantage")
    def compute_advantage(self, batch: DataProto, **kwargs) -> NodeOutput:
        """Computes advantages and returns for PPO using GAE."""
        if self._multi_agent:
            return self.compute_multi_agent_advantage(batch, **kwargs)
        adv_config = self.config.algorithm
        rollout_config = self.config.actor_rollout_ref.rollout
        return NodeOutput(
            batch=compute_advantage(
                batch,
                adv_estimator=adv_config.adv_estimator,
                gamma=adv_config.gamma,
                lam=adv_config.lam,
                num_repeat=rollout_config.n,
                norm_adv_by_std_in_grpo=adv_config.norm_adv_by_std_in_grpo,
                weight_factor_in_cpgd=adv_config.weight_factor_in_cpgd,
                multi_turn=rollout_config.multi_turn.enable,
                **kwargs,
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

    @DistProfiler.annotate(role="data_rebalance")
    def data_rebalance(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """
        A stateful, distributed-aware node that rebalances data across ranks.
        """
        node_id = kwargs.get("node_config", {}).get("_node_id_", "Unknown")

        # Call the generic rebalancing logic from the mixin
        final_batch = self.rebalance_sampled_data(batch, node_id)

        # Determine the status based on the outcome
        # An empty batch indicates that the data was cached.
        if not final_batch:
            status = "CACHED_INSUFFICIENT_DATA"
        else:
            status = "OK_REBALANCED"

        return NodeOutput(batch=final_batch, metrics={"postprocess_status": status})
