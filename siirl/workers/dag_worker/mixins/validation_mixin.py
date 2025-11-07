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

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch.distributed as dist
from loguru import logger
import torch

from siirl.utils.metrics.metric_utils import aggregate_validation_metrics
from siirl.workers.dag.node import NodeRole
from siirl.workers.dag_worker.data_structures import ValidationPayload, ValidationResult
from siirl.workers.databuffer import DataProto, pad_dataproto_to_divisor, unpad_dataproto


class ValidationMixin:
    """Handles the validation process, including generation, scoring, and aggregation."""

    from typing import Any, Dict, List, Optional
    from collections import defaultdict
    import torch.distributed as dist
    from siirl.utils.params import SiiRLArguments
    from siirl.dataloader import DataLoaderNode
    from siirl.workers.dag.node import Node, NodeRole
    from siirl.workers.base_worker import Worker

    timers: defaultdict
    _rank: int
    global_steps: int
    logger: Any  # Can be more specific if Tracking class is imported
    dataloader: DataLoaderNode
    _gather_group: Optional[dist.ProcessGroup]
    world_size: int  # Assuming this is available, though it's not a self attribute in DAGWorker
    config: SiiRLArguments
    agent_group_worker: Dict[int, Dict[NodeRole, Worker]]
    validate_tokenizer: Any
    first_rollout_node: Node
    val_reward_fn: Any

    _timer: Any
    _get_node_dp_info: Any

    def _validate(self) -> Dict[str, float]:
        """Performs validation based on dataset type."""
        # Correctly use the existing dataset_type parameter from the data config.
        dataset_type = getattr(self.config.data, "dataset_type", "llm")

        if dataset_type == "embodied":
            return self._validate_embodied()
        else:
            return self._validate_text_generation()

    def _validate_text_generation(self) -> Dict[str, float]:
        """Performs validation by generating, scoring, and aggregating metrics across all ranks."""
        self.timers = defaultdict(float)
        if self._rank == 0:
            logger.info("=" * 60)
            logger.info(f"Starting Validation @ Global Step {self.global_steps}...")
            logger.info("=" * 60)
            self.timers["overall_start_time"] = time.perf_counter()

        all_scored_results: List[ValidationResult] = []

        # Check if num_val_batches > 0 to avoid unnecessary loops.
        if self.dataloader.num_val_batches <= 0:
            if self._rank == 0:
                logger.warning("num_val_batches is 0. Skipping validation.")
            return {}

        for i in range(self.dataloader.num_val_batches):
            if self._rank == 0:
                logger.debug(f"Processing validation batch {i + 1}/{self.dataloader.num_val_batches}")

            # Each rank performs generation and scoring on its slice of data
            with self._timer("prep_and_generate", self.timers):
                batch_proto = self._prepare_validation_batch()
                generated_proto = self._generate_for_validation(batch_proto)
                dist.barrier(self._gather_group)  # Ensure generation is complete on all ranks

            with self._timer("score_and_package", self.timers):
                scored_results = self._score_and_package_results(generated_proto)
                all_scored_results.extend(scored_results)
                # payloads = [ValidationPayload(r.input_text, r.score, r.data_source, r.extra_rewards) for r in scored_results]

        self._dump_validation_generations(all_scored_results)
        dist.barrier(self._gather_group)
        
        _, _, tp_rank, _, pp_rank, _ = self._get_node_dp_info(self.first_rollout_node)
        # Gather all lightweight payloads to rank 0
        with self._timer("gather_payloads", self.timers):
            # Create payloads from the local results before gathering
            payloads_for_metrics = []
            if tp_rank == 0 and pp_rank == 0:
                # Only the master rank of the TP group (tp_rank=0) and first PP stage (pp_rank=0) prepares the payload.
                payloads_for_metrics = [
                    ValidationPayload(r.input_text, r.score, r.data_source, r.extra_rewards) for r in all_scored_results
                ]
            gathered_payloads_on_rank0 = [None] * self.world_size if self._rank == 0 else None
            dist.gather_object(payloads_for_metrics, gathered_payloads_on_rank0, dst=0, group=self._gather_group)

        # Rank 0 performs the final aggregation and logging
        if self._rank == 0:
            flat_payload_list = [p for sublist in gathered_payloads_on_rank0 if sublist for p in sublist]
            final_metrics = self._aggregate_and_log_validation_metrics(flat_payload_list)
        dist.barrier(self._gather_group)
        
        return final_metrics if self._rank == 0 else {}   # Non-zero ranks return an empty dict

    def _prepare_validation_batch(self) -> DataProto:
        """Fetches and prepares a single batch for validation."""
        test_batch = self.dataloader.run(is_validation_step=True)
        test_batch_proto = DataProto.from_single_dict(test_batch)
        n_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
        return test_batch_proto.repeat(n_samples, interleave=True)

    def _prepare_generation_batch(self, batch: DataProto) -> DataProto:
        """Pops keys from a batch to isolate data needed for sequence generation."""
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_inputs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        return batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

    def _generate_for_validation(self, batch_proto: DataProto) -> DataProto:
        """Generates sequences using the rollout worker for a validation batch."""
        rollout_worker = self.agent_group_worker[0][NodeRole.ROLLOUT]
        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs

        # Decode the prompts into strings at the very beginning, when input_ids are clean.
        # This is conceptually simple and robust, prioritizing correctness over performance/memory.
        prompt_texts = self.validate_tokenizer.batch_decode(
            batch_proto.batch["input_ids"], skip_special_tokens=True
        )
        # Store the list of decoded strings to be passed along through the process.
        batch_proto.non_tensor_batch["prompt_texts"] = prompt_texts

        gen_batch = self._prepare_generation_batch(batch_proto)

        if self.config.actor_rollout_ref.rollout.agent.rewards_with_env and "reward_model" in batch_proto.non_tensor_batch:
            gen_batch.non_tensor_batch["reward_model"] = batch_proto.non_tensor_batch["reward_model"] 
        gen_batch.meta_info = {
            "eos_token_id": self.validate_tokenizer.eos_token_id,
            "pad_token_id": self.validate_tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": val_kwargs.do_sample,
            "validate": True,
        }
        logger.info(f"_generate_for_validation gen batch meta_info: {gen_batch.meta_info}")
        
        output = None
        if self._multi_agent is False:
            if self.rollout_mode == 'sync':
                output = rollout_worker.generate_sequences(gen_batch)
            elif self._async_rollout_manager:
                output = self._async_rollout_manager.generate_sequences(gen_batch)
        else:
            output = self.multi_agent_loop.generate_sequence(gen_batch)
            if output:
                return output
            return batch_proto
        if output:
            batch_proto.union(output)
        return batch_proto

    def _score_and_package_results(self, generated_proto: DataProto) -> List[ValidationResult]:
        """Scores generated sequences and packages them into ValidationResult objects."""
        if self.rollout_mode == 'async' and self._async_rollout_manager is None:
            return []
        if self._multi_agent and 'responses' not in generated_proto.batch:
            return []
        if "token_level_rewards" in generated_proto.batch:
            reward_result = {"reward_tensor": generated_proto.batch["token_level_rewards"],
                             "reward_extra_info": {}}
        else:    
            reward_result = self.val_reward_fn(generated_proto, return_dict=True)
        scores = reward_result["reward_tensor"].sum(-1).cpu()

        input_texts = generated_proto.non_tensor_batch.get("prompt_texts")
        if input_texts is None:
            logger.error(
                "FATAL: `prompt_texts` not found in `non_tensor_batch`. "
                "The prompt data was lost during the process. Falling back to decoding the full sequence, "
                "but please be aware the resulting `input_text` will be INCORRECT (it will contain prompt + response)."
            )
            # Fallback to prevent a crash, but the output is known to be wrong.
            input_texts = self.validate_tokenizer.batch_decode(
                generated_proto.batch["input_ids"], skip_special_tokens=True
            )

        output_texts = self.validate_tokenizer.batch_decode(generated_proto.batch["responses"], skip_special_tokens=True)
        data_sources = generated_proto.non_tensor_batch.get("data_source", ["unknown"] * len(scores))
        extra_info = generated_proto.non_tensor_batch.get("extra_info", [None] * len(scores))

        packaged_results = []
        for i in range(len(scores)):
            if self.dataloader.is_val_trailing_rank and isinstance(extra_info[i], dict) and extra_info[i].get("padded_duplicate", None):
                logger.debug(f"Rank {self._rank} skip append padded duplicate item {i}: score={scores[i].item()}")
                continue
            extra_rewards = {k: v[i] for k, v in reward_result.get("reward_extra_info", {}).items()}
            packaged_results.append(ValidationResult(input_texts[i], output_texts[i], scores[i].item(), data_sources[i], reward_result["reward_tensor"][i], extra_rewards))
        return packaged_results

    def _aggregate_and_log_validation_metrics(self, all_payloads: List[ValidationPayload]) -> Dict[str, float]:
        """On Rank 0, aggregates all validation results and logs performance."""
        if not all_payloads:
            logger.warning("Validation finished with no results gathered on Rank 0 to aggregate.")
            return {}

        logger.info(f"Rank 0: Aggregating {len(all_payloads)} validation results...")
        with self._timer("final_aggregation", self.timers):
            final_metrics = self._aggregate_validation_results(all_payloads)

        # Log performance breakdown
        total_time = time.perf_counter() - self.timers.pop("overall_start_time", time.perf_counter())
        logger.info("--- Validation Performance Breakdown (Rank 0) ---")
        for name, duration in self.timers.items():
            logger.info(f"  Total {name.replace('_', ' ').title():<25}: {duration:.4f}s")
        known_time = sum(self.timers.values())
        logger.info(f"  {'Other/Overhead':<25}: {max(0, total_time - known_time):.4f}s")
        logger.info(f"  {'TOTAL VALIDATION TIME':<25}: {total_time:.4f}s")
        logger.info("=" * 51)

        return final_metrics

    def _aggregate_validation_results(self, all_payloads: List[ValidationPayload]) -> Dict[str, float]:
        """Computes the final metric dictionary from all gathered validation payloads."""
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

    def _dump_validation_generations(self, results: List[ValidationResult]):
        """
        Dumps local validation generation results to a rank-specific JSONL file.

        This method is called by each rank to dump its own portion of the
        validation data into a shared directory.

        Args:
            results: A list of ValidationResult objects containing the local
                     data for the current rank.
        """
        dump_path_str = self.config.trainer.rollout_data_dir
        if not dump_path_str:
            return
        dump_path = Path(dump_path_str)

        try:
            dump_path.mkdir(parents=True, exist_ok=True)

            # Use .json extension for pretty-printed, multi-line JSON format.
            filename = dump_path / f"step_{self.global_steps}_rank_{self._rank}.json"

            # Collect all entries into a list of dictionaries
            entries = []
            for res in results:
                entry = {
                    "rank": self._rank,
                    "global_step": self.global_steps,
                    "data_source": res.data_source,
                    "input": res.input_text,
                    "output": res.output_text,
                    "score": res.score,
                }
                if res.extra_rewards:
                    entry.update(res.extra_rewards) #

                entries.append(entry) #

            # Write the entire list to a file with indentation for readability
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=4)

            if self._rank == 0:
                logger.info(f"Validation generations are being dumped by all ranks to: {dump_path.resolve()}")
            logger.debug(f"Rank {self._rank}: Dumped {len(results)} validation generations to {filename}")

        except (OSError, IOError) as e:
            logger.error(f"Rank {self._rank}: Failed to write validation dump file to {dump_path}: {e}")
        except Exception as e:
            logger.error(f"Rank {self._rank}: An unexpected error occurred during validation dumping: {e}", exc_info=True)


    def _validate_embodied(self) -> Dict[str, float]:
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
        if self._rank == 0:
            logger.info("=" * 60)
            logger.info(f"Starting Embodied Validation @ Global Step {self.global_steps}...")
            logger.info("=" * 60)
            self.timers["overall_start_time"] = time.perf_counter()
        
        # 2. Check if num_val_batches > 0 to avoid unnecessary loops
        if self.dataloader.num_val_batches <= 0:
            if self._rank == 0:
                logger.warning("num_val_batches is 0. Skipping embodied validation.")
            return {}
        
        # 3. Collect payloads from all batches
        all_payloads = []
        
        for i in range(self.dataloader.num_val_batches):
            if self._rank == 0:
                logger.debug(f"Processing embodied validation batch {i + 1}/{self.dataloader.num_val_batches}")
            
            # 3.1 Prepare and generate
            with self._timer("prep_and_generate", self.timers):
                batch_proto = self._prepare_embodied_validation_batch()
                generated_proto = self._generate_for_embodied_validation(batch_proto)
                dist.barrier(self._gather_group)
            
            # 3.2 Score
            with self._timer("score", self.timers):
                batch_payloads = self._score_embodied_results(generated_proto)
                all_payloads.extend(batch_payloads)
        
        # 4. Gather payloads to rank 0 (only TP/PP master ranks prepare payload)
        _, _, tp_rank, _, pp_rank, _ = self._get_node_dp_info(self.first_rollout_node)
        with self._timer("gather_payloads", self.timers):
            payloads_for_metrics = []
            if tp_rank == 0 and pp_rank == 0:
                payloads_for_metrics = all_payloads
            
            gathered_payloads_on_rank0 = [None] * self.world_size if self._rank == 0 else None
            dist.gather_object(payloads_for_metrics, gathered_payloads_on_rank0, dst=0, group=self._gather_group)
        
        # 5. Rank 0 aggregates and logs metrics
        if self._rank == 0:
            flat_payload_list = [p for sublist in gathered_payloads_on_rank0 if sublist for p in sublist]
            final_metrics = self._aggregate_and_log_embodied_metrics(flat_payload_list)
        
        dist.barrier(self._gather_group)
        
        return final_metrics if self._rank == 0 else {}
    
    def _prepare_embodied_validation_batch(self) -> DataProto:
        """
        Fetches and prepares a single embodied validation batch.
        
        Unlike text-generation validation, embodied validation does NOT repeat batches
        because running embodied episodes is expensive.
        
        Returns:
            DataProto: The validation batch
        """
        test_batch = self.dataloader.run(is_validation_step=True)
        test_batch_proto = DataProto.from_single_dict(test_batch)

        # No repeat for embodied validation (confirmed by user)
        return test_batch_proto
    
    def _generate_for_embodied_validation(self, batch_proto: DataProto) -> DataProto:
        """
        Generates embodied episodes using the rollout worker.
        
        Sets up meta_info for validation mode (validate=True, do_sample=False) and
        calls the appropriate generation method based on rollout configuration.
        
        Args:
            batch_proto: The input batch containing task information
            
        Returns:
            DataProto: The batch with generated episode data (actions, observations, rewards, etc.)
        """
        rollout_worker = self.agent_group_worker[0][NodeRole.ROLLOUT]
        
        # Set meta_info for embodied validation
        batch_proto.meta_info = {
            "eos_token_id": self.validate_tokenizer.eos_token_id,
            "pad_token_id": self.validate_tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": False,  # Greedy/deterministic for validation
            "validate": True,
            "global_steps": self.global_steps,
        }
        
        logger.info(f"[Embodied Validation] Batch meta_info: {batch_proto.meta_info}")
        
        # Generate episodes based on rollout mode
        output = None
        output = rollout_worker.generate_sequences(batch_proto)
        
        # Union the output with the original batch
        if output:
            batch_proto = batch_proto.union(output)
        
        return batch_proto
    
    def _score_embodied_results(self, generated_proto: DataProto) -> List[ValidationPayload]:
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
            # verl pattern: verify returns (scores, reward_metrics, format_metrics, reward_format_metrics)
            verifier_score, reward_metrics, format_metrics, reward_format_metrics = self.val_reward_fn.verify(generated_proto)
            reward_tensor = torch.tensor(verifier_score, dtype=torch.float32).unsqueeze(-1)
            
            # Store batch-level metrics (without prefix, will be added during aggregation)
            batch_metrics = {
                'reward_metrics': reward_metrics,
                'format_metrics': format_metrics,
                'reward_format_metrics': reward_format_metrics,
            }
        
        # 3. Get data sources (task suite name)
        task_suite_name = getattr(self.config.data, 'task_suite_name', 'unknown')
        data_sources = generated_proto.non_tensor_batch.get("data_source", [task_suite_name] * reward_tensor.shape[0])
        
        # 4. Get input identifiers (for debugging/logging)
        # For embodied tasks, we use task_file_name if available
        if "task_file_name" in generated_proto.batch:
            task_file_names_bytes = generated_proto.batch["task_file_name"].cpu().numpy()
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
    
    def _aggregate_and_log_embodied_metrics(self, all_payloads: List[ValidationPayload]) -> Dict[str, float]:
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
        with self._timer("final_aggregation", self.timers):
            final_metrics = self._aggregate_embodied_results(all_payloads)
        
        # 2. Log performance breakdown
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
        logger.info(f"Embodied Validation Metrics (Global Step {self.global_steps}):")
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
            metric_dict[f'embodied/test_score/{data_source}'] = np.mean(rewards)
        
        # 3. Compute overall mean reward
        all_rewards = [p.score for p in all_payloads]
        metric_dict['embodied/test_score/all'] = np.mean(all_rewards)
        
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
