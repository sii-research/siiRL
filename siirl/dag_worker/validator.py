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
import torch
import numpy as np
import torch.distributed as dist
from ray.actor import ActorHandle
from collections import defaultdict
from typing import Dict, List, Callable, Any, Optional, Tuple
from loguru import logger
from tensordict import TensorDict, NonTensorData
from torch.distributed import ProcessGroup

from siirl.data_coordinator import preprocess_dataloader
from siirl.data_coordinator.dataloader import DataLoaderNode
from siirl.execution.dag.node import NodeRole, Node
from siirl.execution.scheduler.reward import create_reward_manager
from siirl.dag_worker.data_structures import ValidationPayload, ValidationResult
from siirl.dag_worker.dag_utils import dump_validation_generations, timer
from siirl.utils.metrics.metric_utils import aggregate_validation_metrics
from siirl.params import SiiRLArguments



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
        validate_tokenizer: Any,
        multi_agent_group: Dict[int, Dict[NodeRole, Any]],
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
        metric_worker: ActorHandle = None
    ):
        """
        Initialize the Validator with explicit dependencies.

        Args:
            config: Training configuration
            dataloader: Data loading utilities
            val_reward_fn: Validation reward function
            validate_tokenizer: Tokenizer for decoding sequences
            multi_agent_group: Worker groups for generation (indexed by agent_group -> role)
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
        self.validate_tokenizer = validate_tokenizer
        self.multi_agent_group = multi_agent_group
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
        
        self.val_reward_fn = create_reward_manager(
            self.config,
            self.validate_tokenizer,
            num_examine=1,
            max_resp_len=self.config.data.max_response_length,
            overlong_buffer_cfg=self.config.reward_model.overlong_buffer,
        )

        # Validation timing tracking
        self.val_timedict = defaultdict(float)
        self.metric_worker = metric_worker

    def validate(self, global_step: int) -> Dict[str, float]:
        """Performs validation based on dataset type."""
        # Correctly use the existing dataset_type parameter from the data config.
        dataset_type = getattr(self.config.data, "dataset_type", "llm")
        if dataset_type == "embodied":
            return self._validate_embodied(global_step)
        else:
            return self._validate_text_generation(global_step)
    
    def _validate_embodied(self, global_step) -> Dict[str, float]:
        """
        Performs embodied validation by running interactive episodes.
        
        This is the main entry point that orchestrates the entire validation flow:
        1. Initialize timers and check prerequisites
        2. Iterate through validation batches (each rank processes a shard)
        3. Generate embodied episodes via rollout worker
        4. Score results using val_reward_fn
        5. Gather payloads from all ranks to rank 0
        6. Aggregate and return final metrics
        
        Returns:
            Dict[str, float]: Validation metrics (only on rank 0, empty dict on other ranks)
        """
        # 1. Initialize timers
        self.timers = defaultdict(float)
        if self.rank == 0:
            logger.info("=" * 60)
            logger.info(f"Starting Embodied Validation @ Global Step {global_step}...")
            logger.info("=" * 60)
            self.timers["overall_start_time"] = time.perf_counter()
        
        # 2. Check if num_val_batches > 0 to avoid unnecessary loops
        if self.dataloader.num_val_batches <= 0:
            if self.rank == 0:
                logger.warning("num_val_batches is 0. Skipping embodied validation.")
            return {}
        
        # 3. Collect payloads from all batches
        all_payloads = []
        
        for i in range(self.dataloader.num_val_batches):
            if self.rank == 0:
                logger.debug(f"Processing embodied validation batch {i + 1}/{self.dataloader.num_val_batches}")
            
            # 3.1 Prepare and generate
            with timer(self.enable_perf, "prep_and_generate", self.val_timedict):
                batch_proto = self._prepare_embodied_validation_batch()
                generated_proto = self._generate_for_embodied_validation(batch_proto, global_step)
                dist.barrier(self.gather_group)
            
            # 3.2 Score
            with timer(self.enable_perf, "score", self.val_timedict):
                batch_payloads = self._score_embodied_results(generated_proto)
                all_payloads.extend(batch_payloads)
        # 4. Gather payloads to rank 0 (only TP/PP master ranks prepare payload)
        dp_size, _, tp_rank, _, pp_rank, _ = self.get_node_dp_info_fn(self.first_rollout_node)
        with timer(self.enable_perf, "gather_payloads", self.val_timedict):
            if tp_rank == 0 and pp_rank == 0:
                self.metric_worker.submit_metric(self._aggregate_and_log_embodied_metrics(all_payloads, global_step), dp_size)
        # with timer(self.enable_perf, "gather_payloads", self.val_timedict):
        #     payloads_for_metrics = []
        #     if tp_rank == 0 and pp_rank == 0:
        #         payloads_for_metrics = all_payloads
            
        #     gathered_payloads_on_rank0 = [None] * self.world_size if self._rank == 0 else None
        #     dist.gather_object(payloads_for_metrics, gathered_payloads_on_rank0, dst=0, group=self._gather_group)
        
        # # 5. Rank 0 aggregates and logs metrics
        # if self.rank == 0:
        #     flat_payload_list = [p for sublist in gathered_payloads_on_rank0 if sublist for p in sublist]
        #     final_metrics = self._aggregate_and_log_embodied_metrics(flat_payload_list)
        
        dist.barrier(self.gather_group)
        
        return
    
    
    def _validate_text_generation(self, global_step: int) -> Dict[str, float]:
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
        sample_turns = []
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

        dp_size, _, tp_rank, _, pp_rank, _ = self.get_node_dp_info_fn(self.first_rollout_node)
        
        # Gather all payloads to rank 0
        with timer(self.enable_perf, "gather_payloads", self.val_timedict):
            if tp_rank == 0 and pp_rank == 0:
                # Only the master rank of the TP group (tp_rank=0) and first PP stage (pp_rank=0) prepares the payload.
                payloads_for_metrics = [
                    ValidationPayload(r.input_text, r.score, r.data_source, r.extra_rewards) for r in all_scored_results
                ]
                self.metric_worker.submit_metric(self._aggregate_and_log_validation_metrics(payloads_for_metrics), dp_size)

        dist.barrier(self.gather_group)
        return 

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
            TensorDict: Batch with generated sequences added
        """
        rollout_worker = self.multi_agent_group[0][NodeRole.ROLLOUT]
        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs

        prompt_texts = self.validate_tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        batch["prompt_texts"] = prompt_texts
        batch["eos_token_id"] = NonTensorData(self.validate_tokenizer.eos_token_id)
        batch["pad_token_id"] = NonTensorData(self.validate_tokenizer.eos_token_id)
        batch["recompute_log_prob"] = NonTensorData(self.validate_tokenizer.eos_token_id)
        batch["validate"] = NonTensorData(True)
        batch["do_sample"] = NonTensorData(val_kwargs.do_sample)

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
            Dict: extra_rewards 
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

        
        with timer(self.enable_perf, "final_aggregation", self.val_timedict):
            final_metrics = self._aggregate_validation_results(all_payloads)

        # Log performance breakdown
        total_time = time.perf_counter() - self.val_timedict.pop("overall_start_time", time.perf_counter())
        if self.rank == 0:
            logger.info(f"Rank 0: Aggregating {len(all_payloads)} validation results...")
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

    
    def _prepare_embodied_validation_batch(self) -> TensorDict:
        """
        Fetches and prepares a single embodied validation batch.
        
        Unlike text-generation validation, embodied validation does NOT repeat batches
        because running embodied episodes is expensive.
        
        Returns:
            TensorDict: The validation batch
        """
        test_batch = self.dataloader.run(is_validation_step=True)
        test_batch_proto = preprocess_dataloader(test_batch)

        # No repeat for embodied validation (confirmed by user)
        return test_batch_proto
    
    def _generate_for_embodied_validation(self, batch: TensorDict, global_step:int) -> TensorDict:
        """
        Generates embodied episodes using the rollout worker.
        
        Sets up meta_info for validation mode (validate=True, do_sample=False) and
        calls the appropriate generation method based on rollout configuration.
        
        Args:
            batch: The input batch containing task information
            
        Returns:
            TensorDict: The batch with generated episode data (actions, observations, rewards, etc.)
        """
        rollout_worker = self.multi_agent_group[0][NodeRole.ROLLOUT]
        
        # Set meta_info for embodied validation
    
        batch["eos_token_id"] = NonTensorData(self.validate_tokenizer.eos_token_id)
        batch["pad_token_id"] = NonTensorData(self.validate_tokenizer.eos_token_id)
        batch["recompute_log_prob"] = NonTensorData(self.validate_tokenizer.eos_token_id)
        batch["validate"] = NonTensorData(True)
        batch["do_sample"] = NonTensorData(False)
        batch["global_steps"] = NonTensorData(global_step)
        
        logger.info(
            f"[Embodied Validation] Batch variables: "
            f"eos_token_id={batch['eos_token_id']}, "
            f"pad_token_id={batch['pad_token_id']}, "
            f"recompute_log_prob={batch['recompute_log_prob']}, "
            f"validate={batch['validate']}, "
            f"do_sample={batch['do_sample']}, "
            f"global_steps={batch['global_steps']}"
        )
        
        # Generate episodes based on rollout mode
        output = None
        output = rollout_worker.generate_sequences(batch)
        
        # Union the output with the original batch
        if output is not None:
            return output
        
        return batch
    
    def _score_embodied_results(self, generated_proto: TensorDict) -> List[ValidationPayload]:
        """
        Scores generated embodied episodes using val_reward_fn and packages lightweight payloads.
        
        Unlike text-generation, embodied validation:
        - Uses val_reward_fn.verify() instead of val_reward_fn()
        - Returns (verifier_score, reward_metrics, format_metrics, reward_format_metrics)
        - Doesn't need to decode text prompts/responses
        
        Args:
            generated_proto: The batch with generated episode data
            
        Returns:
            List[ValidationPayload]: Lightweight payloads for gathering
        """
        if self.val_reward_fn:
            verifier_score, reward_metrics, format_metrics, reward_format_metrics = self.val_reward_fn.verify(generated_proto)
            reward_tensor = torch.tensor(verifier_score, dtype=torch.float32).unsqueeze(-1)
            
            # Store batch-level metrics (without prefix, will be added during aggregation)
            batch_metrics = {
                'reward_metrics': reward_metrics,
                'format_metrics': format_metrics,
                'reward_format_metrics': reward_format_metrics,
            }
        
        # 3. Get data sources (task suite name)
        task_suite_name = getattr(self.config.actor_rollout_ref.embodied.env, 'env_name', 'unknown_task')
        data_sources = generated_proto.get("data_source", [task_suite_name] * reward_tensor.shape[0])
        
        # 4. Get input identifiers (for debugging/logging)
        # For embodied tasks, we use task_file_name if available
        if "task_file_name" in generated_proto:
            task_file_names_bytes = generated_proto["task_file_name"].cpu().numpy()
            input_texts = []
            for tfn_bytes in task_file_names_bytes:
                # Decode bytes to string
                tfn_str = bytes(tfn_bytes).decode('utf-8').rstrip('\x00')
                input_texts.append(f"Task: {tfn_str}")
        else:
            # Fallback: use generic episode identifiers
            input_texts = [f"Episode_{i}" for i in range(reward_tensor.shape[0])]
        
        # 5. Compute scores
        scores = reward_tensor.sum(-1).cpu()
        
        # 6. Package payloads (lightweight, no full reward tensor)
        # Only the first sample in each batch carries the batch_metrics to avoid duplication
        packaged_payloads = []
        for i in range(len(scores)):
            payload = ValidationPayload(
                input_text=input_texts[i],
                score=scores[i].item(),
                data_source=data_sources[i],
                extra_rewards=batch_metrics if i == 0 else {}  # Only first sample carries batch metrics
            )
            packaged_payloads.append(payload)
        
        return packaged_payloads
    
    
    def _aggregate_and_log_embodied_metrics(self, all_payloads: List[ValidationPayload], global_step: int) -> Dict[str, float]:
        """
        On Rank 0, aggregates all embodied validation results and logs performance.
        
        This function runs only on rank 0 after gathering payloads from all ranks.
        
        Args:
            all_payloads: All ValidationPayload objects gathered from all ranks
            
        Returns:
            Dict[str, float]: Final validation metrics
        """
        if not all_payloads:
            logger.warning("Embodied validation finished with no results gathered on Rank 0.")
            return {}
        
        logger.info(f"Rank 0: Aggregating {len(all_payloads)} embodied validation results...")
        
        # 1. Aggregate results
        with timer(self.enable_perf, "final_aggregation", self.val_timedict):
            final_metrics = self._aggregate_embodied_results(all_payloads)
        
        # 2. Log performance breakdown
        if self.rank == 0:
            total_time = time.perf_counter() - self.timers.pop("overall_start_time", time.perf_counter())
            logger.info("=" * 60)
            logger.info("--- Embodied Validation Performance Breakdown (Rank 0) ---")
            for name, duration in self.timers.items():
                logger.info(f"  Total {name.replace('_', ' ').title():<30}: {duration:.4f}s")
            known_time = sum(self.timers.values())
            logger.info(f"  {'Other/Overhead':<30}: {max(0, total_time - known_time):.4f}s")
            logger.info(f"  {'TOTAL EMBODIED VALIDATION TIME':<30}: {total_time:.4f}s")
            logger.info("=" * 60)
            
            # 3. Log final metrics
            logger.info(f"Embodied Validation Metrics (Global Step {global_step}):")
            for metric_name, metric_value in sorted(final_metrics.items()):
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            logger.info("=" * 60)
        
        return final_metrics
    
    def _aggregate_embodied_results(self, all_payloads: List[ValidationPayload]) -> Dict[str, float]:
        """
        Computes the final metric dictionary from all gathered embodied validation payloads.
        
        Aggregation strategy:
        1. Group rewards by data_source
        2. Compute mean reward per data_source and overall
        3. Collect and average batch-level metrics from all batches
        
        Args:
            all_payloads: All ValidationPayload objects from all ranks
            
        Returns:
            Dict[str, float]: Final metrics with structure:
                - embodied/test_score/{data_source}: Mean reward per data source
                - embodied/test_score/all: Overall mean reward
                - embodied/reward/{metric_name}: Averaged reward metrics across all batches
                - embodied/format/{metric_name}: Averaged format metrics across all batches
                - embodied/reward_format/{metric_name}: Averaged combined metrics across all batches
        """
        # 1. Group rewards by data_source
        data_source_rewards = {}
        for payload in all_payloads:
            data_source = payload.data_source
            if data_source not in data_source_rewards:
                data_source_rewards[data_source] = []
            data_source_rewards[data_source].append(payload.score)
        
        # 2. Compute per-data-source metrics
        metric_dict = {}
        for data_source, rewards in data_source_rewards.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        
        # 3. Compute overall mean reward
        all_rewards = [p.score for p in all_payloads]
        metric_dict['val/test_score/all'] = np.mean(all_rewards)
        
        # 4. Collect and average batch-level metrics from all batches
        batch_metrics_list = []
        for payload in all_payloads:
            if payload.extra_rewards and 'reward_metrics' in payload.extra_rewards:
                batch_metrics_list.append(payload.extra_rewards)
        
        if batch_metrics_list:
            num_batches = len(batch_metrics_list)
            logger.info(f"[_aggregate_embodied_results] Collected metrics from {num_batches} batches")
            
            # 4.1 Collect all reward_metrics (including 'all' and per-task metrics)
            aggregated_reward_metrics = {}
            for batch_metrics in batch_metrics_list:
                for key, value in batch_metrics['reward_metrics'].items():
                    if key not in aggregated_reward_metrics:
                        aggregated_reward_metrics[key] = []
                    aggregated_reward_metrics[key].append(value)
            
            # 4.2 Collect all format_metrics
            aggregated_format_metrics = {}
            for batch_metrics in batch_metrics_list:
                for key, value in batch_metrics['format_metrics'].items():
                    if key not in aggregated_format_metrics:
                        aggregated_format_metrics[key] = []
                    aggregated_format_metrics[key].append(value)
            
            # 4.3 Collect all reward_format_metrics
            aggregated_reward_format_metrics = {}
            for batch_metrics in batch_metrics_list:
                for key, value in batch_metrics['reward_format_metrics'].items():
                    if key not in aggregated_reward_format_metrics:
                        aggregated_reward_format_metrics[key] = []
                    aggregated_reward_format_metrics[key].append(value)
            
            # 5. Compute average values and add to metric_dict
            for key, values in aggregated_reward_metrics.items():
                metric_dict[f'embodied/reward/{key}'] = np.mean(values)
            
            for key, values in aggregated_format_metrics.items():
                metric_dict[f'embodied/format/{key}'] = np.mean(values)
            
            for key, values in aggregated_reward_format_metrics.items():
                metric_dict[f'embodied/reward_format/{key}'] = np.mean(values)
        
        return metric_dict
