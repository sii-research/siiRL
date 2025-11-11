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

from typing import Any, Dict, Optional

from loguru import logger

from siirl.scheduler.enums import AdvantageEstimator
from siirl.scheduler.reward import compute_reward
from siirl.utils.params import SiiRLArguments
from siirl.workers.dag import Node, NodeRole, NodeStatus, NodeType
from siirl.workers.dag_worker import core_algos
from siirl.workers.databuffer import DataProto


class ComputeNode(Node):
    """
    A specialized node for computation tasks like advantage or reward calculation.
    The `node_type` is fixed to `COMPUTE`.
    The `node_role` is restricted to `ADVANTAGE` or `REWARD`.
    The `executable_ref` is automatically set based on the `node_role`.
    """

    def __init__(
        self,
        node_id: str,
        node_role: NodeRole,  # node_role is now mandatory
        global_config: SiiRLArguments,
        config: Optional[Dict[str, Any]] = None,
        retry_limit: int = 0,
    ):
        """
        Initialize a ComputeNode.

        Args:
            node_id (str): The unique identifier of the node.
            node_role (NodeRole): The role of the node, must be ADVANTAGE or REWARD.
            global_config (SiiRLArguments): The arguments from config file.
            config (Optional[Dict[str, Any]]): Configuration for this node.
            retry_limit (int): Maximum number of retries on failure.
        """
        if node_role not in [NodeRole.ADVANTAGE, NodeRole.REWARD]:
            raise ValueError(f"ComputeNode role must be NodeRole.ADVANTAGE or NodeRole.REWARD, got {node_role}")

        node_type = NodeType.COMPUTE

        super().__init__(node_id=node_id, node_type=node_type, node_role=node_role, config=config, executable_ref=None, retry_limit=retry_limit)

        if node_role == NodeRole.ADVANTAGE:
            self._executable = self.compute_advantage

            self.adv_estimator = global_config.algorithm.adv_estimator
            self.gamma = global_config.algorithm.gamma
            self.lam = global_config.algorithm.lam
            self.num_repeat = global_config.actor_rollout_ref.rollout.n
            self.norm_adv_by_std_in_grpo = global_config.algorithm.norm_adv_by_std_in_grpo
            self.weight_factor_in_cpgd = global_config.algorithm.weight_factor_in_cpgd
            self.multi_turn = global_config.actor_rollout_ref.rollout.multi_turn.enable
        elif node_role == NodeRole.REWARD:
            self._executable = compute_reward
        else:
            self._executable = None

    @staticmethod
    def compute_response_mask(data: DataProto):
        """Compute the attention mask for the response part of the sequence.
        
        Handles both 2D responses (NLP) and 3D responses (Embodied AI).
        
        Returns:
            torch.Tensor: The attention mask for the response tokens (always 2D).
        """
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        batch_size = responses.size(0)
        
        # Handle 3D responses (Embodied AI): (batch_size, traj_len, action_token_len)
        if responses.ndim == 3:
            traj_len = responses.size(1)
            action_token_len = responses.size(2)
            
            # Check if attention_mask is also 3D
            if attention_mask.ndim == 3:
                # attention_mask: (batch_size, traj_len, tot_pad_len)
                # Extract response part from last dimension: (batch_size, traj_len, action_token_len)
                response_mask = attention_mask[:, :, -action_token_len:]
                # Flatten to 2D: (batch_size, traj_len * action_token_len)
                response_mask = response_mask.reshape(batch_size, -1)
            else:
                # attention_mask is 2D: (batch_size, total_length)
                # Calculate flattened response_length and slice
                response_length = traj_len * action_token_len
                response_mask = attention_mask[:, -response_length:]
        # Handle 2D responses (NLP): (batch_size, response_length)
        elif responses.ndim == 2:
            response_length = responses.size(1)
            response_mask = attention_mask[:, -response_length:]
        else:
            raise ValueError(f"Unexpected responses shape: {responses.shape}, ndim={responses.ndim}")
        
        return response_mask

    def compute_advantage(self, data: DataProto):
        # Back-compatible with trainers that do not compute response mask in fit
        if "response_mask" not in data.batch:
            data.batch["response_mask"] = self.compute_response_mask(data)
        # prepare response group
        # TODO: add other ways to estimate advantages
        if self.adv_estimator == AdvantageEstimator.GAE:
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                response_mask=data.batch["response_mask"],
                gamma=self.gamma,
                lam=self.lam,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
        elif self.adv_estimator == AdvantageEstimator.GRPO:
            # TODO: test on more adv estimator type
            grpo_calculation_mask = data.batch["response_mask"]
            # Call compute_grpo_outcome_advantage with parameters matching its definition
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=self.norm_adv_by_std_in_grpo,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
        elif self.adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
            advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                index=data.non_tensor_batch["uid"],
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
        elif self.adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
            advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                gamma=self.gamma,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
        elif self.adv_estimator == AdvantageEstimator.REMAX:
            advantages, returns = core_algos.compute_remax_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                reward_baselines=data.batch["reward_baselines"],
                response_mask=data.batch["response_mask"],
            )

            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
        elif self.adv_estimator == AdvantageEstimator.RLOO:
            advantages, returns = core_algos.compute_rloo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                index=data.non_tensor_batch["uid"],
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
        elif self.adv_estimator == AdvantageEstimator.CPGD:
            # TODO: test on more adv estimator type
            cpgd_calculation_mask = data.batch["response_mask"]
            # Call compute_cpgd_outcome_advantage with parameters matching its definition
            advantages, returns = core_algos.compute_cpgd_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=cpgd_calculation_mask,
                index=data.non_tensor_batch["uid"],
                weight_factor_in_cpgd=self.weight_factor_in_cpgd,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
        else:
            raise NotImplementedError
        return data

    def run(self, **kwargs: Any) -> Any:
        """
        Execute the task of the node.
        Args:
            **kwargs: Parameters passed to the executable function, usually the outputs of its dependent nodes.
        Returns:
            Any: The result of the node execution.
        """
        logger.info(f"Starting to execute node: {self.node_id} (Type: {self.node_type.value}, Role: {self.node_role.value})")
        self.update_status(NodeStatus.RUNNING)

        try:
            if self.node_role == NodeRole.ADVANTAGE:
                if self._executable is not None:
                    self.output = self._executable(kwargs["data"])
            elif self.node_role == NodeRole.REWARD:
                if self._executable is not None:
                    self.output = self._executable(kwargs["data"], kwargs["reward_fn"])
            self.update_status(NodeStatus.COMPLETED)
            logger.info(f"Node {self.node_id} execution completed.")
            return self.output
        except Exception as e:
            error_message = f"An error occurred while executing node {self.node_id}: {e}"
            self.update_status(NodeStatus.FAILED, error_message)
            # An exception can be raised here, or the scheduler can handle the FAILED status
            raise RuntimeError(error_message) from e
