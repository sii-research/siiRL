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

import time
import asyncio
import numpy as np
import torch.distributed as dist
from collections import defaultdict
from typing import Dict, List, Callable, Any, Optional
from loguru import logger
from tensordict import TensorDict, NonTensorData
from torch.distributed import ProcessGroup

from siirl.data_coordinator import DataProto, preprocess_dataloader
from siirl.data_coordinator.dataloader import DataLoaderNode
from siirl.execution.dag.node import NodeRole, Node
from siirl.dag_worker.data_structures import ValidationPayload, ValidationResult
from siirl.dag_worker.dag_utils import prepare_generation_batch, dump_validation_generations, timer
from siirl.utils.metrics.metric_utils import aggregate_validation_metrics
from siirl.global_config.params import SiiRLArguments


class Validator:
    """
    Handles the complete validation workflow for distributed RL training.

    This class orchestrates the validation process including:
    - Batch preparation and generation
    - Reward scoring and result packaging
    - Metrics aggregation across all ranks
    - Performance logging

    The validator operates in a distributed manner, coordinating across multiple
    ranks and aggregating results on rank 0.
    """

    def __init__(
        self,
        config: SiiRLArguments,
        dataloader: DataLoaderNode,
        val_reward_fn: Callable,
        validate_tokenizer: Any,
        agent_group_worker: Dict[int, Dict[NodeRole, Any]],
        rollout_mode: str,
        async_rollout_manager: Optional[Any],
        multi_agent_loop: Optional[Any],
        multi_agent: bool,
        rank: int,
        world_size: int,
        gather_group: ProcessGroup,
        first_rollout_node: Node,
        get_node_dp_info_fn: Callable,
        enable_perf: bool = False,
    ):
        """
        Initialize the Validator with explicit dependencies.

        Args:
            config: Training configuration
            dataloader: Data loading utilities
            val_reward_fn: Validation reward function
            validate_tokenizer: Tokenizer for decoding sequences
            agent_group_worker: Worker groups for generation (indexed by agent_group -> role)
            rollout_mode: Generation mode ('sync' or 'async')
            async_rollout_manager: Manager for async rollout (None if not using async)
            multi_agent_loop: Manager for multi-agent generation (None if not multi-agent)
            multi_agent: Whether in multi-agent mode
            rank: Current process rank
            world_size: Total number of processes
            gather_group: Process group for distributed gathering
            first_rollout_node: First rollout node for getting DP/TP/PP info
            get_node_dp_info_fn: Function to get node parallelism info
            enable_perf: Whether to enable performance profiling
        """
        self.config = config
        self.dataloader = dataloader
        self.val_reward_fn = val_reward_fn
        self.validate_tokenizer = validate_tokenizer
        self.agent_group_worker = agent_group_worker
        self.rollout_mode = rollout_mode
        self.async_rollout_manager = async_rollout_manager
        self.multi_agent_loop = multi_agent_loop
        self.multi_agent = multi_agent
        self.rank = rank
        self.world_size = world_size
        self.gather_group = gather_group
        self.first_rollout_node = first_rollout_node
        self.get_node_dp_info_fn = get_node_dp_info_fn
        self.enable_perf = enable_perf

        # Validation timing tracking
        self.val_timedict = defaultdict(float)

    def validate(self, global_step: int) -> Dict[str, float]:
        """
        Executes the complete validation workflow.

        This is the main entry point for validation. It:
        1. Prepares validation batches from the dataloader
        2. Generates sequences using the rollout model
        3. Scores the generated sequences
        4. Aggregates metrics across all ranks
        5. Logs performance breakdown (on rank 0)

        Args:
            global_step: Current training step (for logging and checkpointing)

        Returns:
            Dict[str, float]: Validation metrics (only on rank 0, empty dict on other ranks)
        """
        self.val_timedict = defaultdict(float)
        if self.rank == 0:
            logger.info("=" * 60)
            logger.info(f"Starting Validation @ Global Step {global_step}...")
            logger.info("=" * 60)
            self.val_timedict["overall_start_time"] = time.perf_counter()

        all_scored_results: List[ValidationResult] = []

        # Check if num_val_batches > 0 to avoid unnecessary loops.
        if self.dataloader.num_val_batches <= 0:
            if self.rank == 0:
                logger.warning("num_val_batches is 0. Skipping validation.")
            return {}

        for i in range(self.dataloader.num_val_batches):
            if self.rank == 0:
                logger.debug(f"Processing validation batch {i + 1}/{self.dataloader.num_val_batches}")

            with timer(self.enable_perf, "prep_and_generate", self.val_timedict):
                test_batch = self.dataloader.run(is_validation_step=True)
                val_batch = preprocess_dataloader(test_batch, self.config.actor_rollout_ref.rollout.val_kwargs.n)
                generated_proto = self._generate_for_validation(val_batch)
                dist.barrier(self.gather_group)

            with timer(self.enable_perf, "score_and_package", self.val_timedict):
                scored_results = self._score_and_package_results(generated_proto)
                all_scored_results.extend(scored_results)

        dump_validation_generations(self.config, global_step, self.rank, all_scored_results)
        dist.barrier(self.gather_group)

        _, _, tp_rank, _, pp_rank, _ = self.get_node_dp_info_fn(self.first_rollout_node)
        # Gather all payloads to rank 0
        with timer(self.enable_perf, "gather_payloads", self.val_timedict):
            payloads_for_metrics = []
            if tp_rank == 0 and pp_rank == 0:
                # Only the master rank of the TP group (tp_rank=0) and first PP stage (pp_rank=0) prepares the payload.
                payloads_for_metrics = [
                    ValidationPayload(r.input_text, r.score, r.data_source, r.extra_rewards) for r in all_scored_results
                ]
            gathered_payloads_on_rank0 = [None] * self.world_size if self.rank == 0 else None
            dist.gather_object(payloads_for_metrics, gathered_payloads_on_rank0, dst=0, group=self.gather_group)

        # Rank 0 performs the final aggregation and logging
        if self.rank == 0:
            flat_payload_list = [p for sublist in gathered_payloads_on_rank0 if sublist for p in sublist]
            final_metrics = self._aggregate_and_log_validation_metrics(flat_payload_list)
        dist.barrier(self.gather_group)

        return final_metrics if self.rank == 0 else {}

    def _generate_for_validation(self, batch: TensorDict) -> TensorDict:
        """
        Generates sequences using the rollout worker for a validation batch.

        Supports three generation modes:
        - Sync mode: Direct generation via rollout worker
        - Async mode: Generation via async rollout manager
        - Multi-agent mode: Generation via multi-agent loop

        Args:
            batch_proto: Input batch containing prompts

        Returns:
            DataProto: Batch with generated sequences added
        """
        rollout_worker = self.agent_group_worker[0][NodeRole.ROLLOUT]
        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs

        prompt_texts = self.validate_tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        batch["prompt_texts"] = prompt_texts
        batch["meta_info"] = NonTensorData({
            "eos_token_id": self.validate_tokenizer.eos_token_id,
            "pad_token_id": self.validate_tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": val_kwargs.do_sample,
            "validate": True,
        })
        logger.info(f"_generate_for_validation gen batch meta_info: {batch['meta_info']}")

        output = None
        if self.multi_agent is False:
            if self.rollout_mode == 'sync':
                output = rollout_worker.generate_sequences(batch)
            elif self.async_rollout_manager:
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(self.async_rollout_manager.generate_sequences(batch))
        else:
            output = self.multi_agent_loop.generate_sequence(batch)

        if output is not None:
            return output
        return batch

    def _score_and_package_results(self, generated_proto: TensorDict) -> List[ValidationResult]:
        """
        Scores generated sequences and packages them into ValidationResult objects.

        This method:
        1. Computes rewards for generated sequences (or uses pre-computed rewards)
        2. Decodes input prompts and output responses
        3. Packages everything into ValidationResult objects
        4. Filters out padded duplicates for trailing ranks

        Args:
            generated_proto: Batch containing generated sequences

        Returns:
            List[ValidationResult]: Scored and packaged validation results
        """
        if self.rollout_mode == 'async' and self.async_rollout_manager is None:
            return []
        if self.multi_agent and 'responses' not in generated_proto:
            return []
        if "token_level_rewards" in generated_proto:
            reward_result = {"reward_tensor": generated_proto["token_level_rewards"],
                             "reward_extra_info": {}}
        else:
            reward_result = self.val_reward_fn(generated_proto, return_dict=True)
        scores = reward_result["reward_tensor"].sum(-1).cpu()

        input_texts = generated_proto["prompt_texts"] if "prompt_texts" in generated_proto else None
        if input_texts is None:
            logger.error(
                "FATAL: `prompt_texts` not found in `non_tensor_batch`. "
                "The prompt data was lost during the process. Falling back to decoding the full sequence, "
                "but please be aware the resulting `input_text` will be INCORRECT (it will contain prompt + response)."
            )
            # Fallback to prevent a crash, but the output is known to be wrong.
            input_texts = self.validate_tokenizer.batch_decode(
                generated_proto["input_ids"], skip_special_tokens=True
            )

        output_texts = self.validate_tokenizer.batch_decode(generated_proto["responses"], skip_special_tokens=True)
        data_sources = generated_proto["data_source"] if "data_source" in generated_proto else ["unknown"] * len(scores)
        extra_info = generated_proto["extra_info"] if "data_source" in generated_proto else [None] * len(scores)

        packaged_results = []
        for i in range(len(scores)):
            if self.dataloader.is_val_trailing_rank and isinstance(extra_info[i], dict) and extra_info[i].get("padded_duplicate", None):
                logger.debug(f"Rank {self.rank} skip append padded duplicate item {i}: score={scores[i].item()}")
                continue
            extra_rewards = {k: v[i] for k, v in reward_result.get("reward_extra_info", {}).items()}
            packaged_results.append(ValidationResult(input_texts[i], output_texts[i], scores[i].item(), data_sources[i], reward_result["reward_tensor"][i], extra_rewards))
        return packaged_results

    def _aggregate_and_log_validation_metrics(self, all_payloads: List[ValidationPayload]) -> Dict[str, float]:
        """
        Aggregates all validation results and logs performance (rank 0 only).

        This method:
        1. Calls _aggregate_validation_results to compute final metrics
        2. Logs a detailed performance breakdown of the validation process
        3. Reports total validation time

        Args:
            all_payloads: All validation payloads gathered from all ranks

        Returns:
            Dict[str, float]: Final aggregated validation metrics
        """
        if not all_payloads:
            logger.warning("Validation finished with no results gathered on Rank 0 to aggregate.")
            return {}

        logger.info(f"Rank 0: Aggregating {len(all_payloads)} validation results...")
        with timer(self.enable_perf, "final_aggregation", self.val_timedict):
            final_metrics = self._aggregate_validation_results(all_payloads)

        # Log performance breakdown
        total_time = time.perf_counter() - self.val_timedict.pop("overall_start_time", time.perf_counter())
        logger.info("--- Validation Performance Breakdown (Rank 0) ---")
        for name, duration in self.val_timedict.items():
            logger.info(f"  Total {name.replace('_', ' ').title():<25}: {duration:.4f}s")
        known_time = sum(self.val_timedict.values())
        logger.info(f"  {'Other/Overhead':<25}: {max(0, total_time - known_time):.4f}s")
        logger.info(f"  {'TOTAL VALIDATION TIME':<25}: {total_time:.4f}s")
        logger.info("=" * 51)

        return final_metrics

    def _aggregate_validation_results(self, all_payloads: List[ValidationPayload]) -> Dict[str, float]:
        """
        Computes the final metric dictionary from all gathered validation payloads.

        This method processes validation results to compute:
        - Mean/majority/best metrics for different data sources
        - Pass@N accuracy metrics
        - Per-data-source test scores

        Args:
            all_payloads: All validation payloads from all ranks

        Returns:
            Dict[str, float]: Final validation metrics organized by data source and metric type
        """
        data_sources = [p.data_source for p in all_payloads]
        sample_inputs = [p.input_text for p in all_payloads]

        infos_dict = defaultdict(list)
        for p in all_payloads:
            infos_dict["reward"].append(p.score)
            for key, value in p.extra_rewards.items():
                infos_dict[key].append(value)

        data_src2var2metric2val = aggregate_validation_metrics(data_sources=data_sources, sample_inputs=sample_inputs, infos_dict=infos_dict)

        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                if not metric2val:
                    continue

                # Robustly parse '@N' to prevent crashes from malformed metric names.
                n_max_values = []
                for name in metric2val.keys():
                    if "@" in name and "/mean" in name:
                        try:
                            n_val = int(name.split("@")[-1].split("/")[0])
                            n_max_values.append(n_val)
                        except (ValueError, IndexError):
                            continue  # Ignore malformed metric names

                n_max = max(n_max_values) if n_max_values else 1

                for metric_name, metric_val in metric2val.items():
                    is_core_metric = (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name)

                    metric_sec = "val-core" if is_core_metric else "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Re-calculate test_score per data source
        data_source_rewards = defaultdict(list)
        for p in all_payloads:
            data_source_rewards[p.data_source].append(p.score)

        for source, rewards in data_source_rewards.items():
            if rewards:
                metric_dict[f"val/test_score/{source}"] = np.mean(rewards)

        return metric_dict
