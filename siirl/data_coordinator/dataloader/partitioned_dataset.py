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

import os
import re

from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Optional, Sequence

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import siirl.utils.model_utils.torch_functional as F
from siirl.utils.model_utils.model import compute_position_id_with_mask
from siirl.params import SiiRLArguments


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class PartitionedRLHFDataset(Dataset):
    """
    An efficient Dataset class for distributed training. It only load and process
    the data partition (slice) of the RLHF dataset corresponding to the current DDP rank.

    Args:
        config (SiiRLArguments): Configuration object containing data and preprocessing arguments.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text prompts.
        processor (Optional[ProcessorMixin]): Optional processor for multi-modal data (e.g., images, videos).
        ddp_rank (int): The rank of the current process in DDP.
        ddp_world_size (int): Total number of DDP processes (world size).
        is_eval (bool): Whether the dataset is for evaluation (True) or training (False).
        drop_last (Optional[bool]): Whether to drop the last remainder
            if total rows is not divisible by world size.
            Defaults to True for training, False for evaluation.

    Notes:
        - This class is optimized for distributed training scenarios, ensuring each DDP process only
            loads and processes its own data partition.
        - Supports multi-modal data (text, images, videos) if a processor is provided.
        - Handles prompt filtering, truncation, and tokenization according to configuration.
        - Uses multiprocessing for efficient data preprocessing.
    """

    def __init__(
        self,
        config: SiiRLArguments,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        is_eval: bool = False,
        drop_last: Optional[bool] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_args = config.data
        self._rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.is_eval = is_eval
        self.drop_last = drop_last if drop_last is not None else (not is_eval)

        self.prompt_key = self.data_args.prompt_key
        self.image_key = self.data_args.image_key
        self.video_key = self.data_args.video_key
        self.max_prompt_length = self.data_args.max_prompt_length
        self.truncation = self.data_args.truncation
        self.return_raw_chat = self.data_args.return_raw_chat
        self.return_full_prompt = self.data_args.return_full_prompt
        self.filter_overlong_prompts = self.data_args.filter_overlong_prompts
        self.num_workers = self.data_args.preprocessing_num_workers if self.data_args.preprocessing_num_workers else max(1, os.cpu_count() // 8)
        self.force_on_the_fly = config.data.force_on_the_fly
        self.image_max_pixels = self.data_args.processor.image_max_pixels
        self.image_min_pixels = self.data_args.processor.image_min_pixels
        self.video_max_pixels = self.data_args.processor.video_max_pixels
        self.video_min_pixels = self.data_args.processor.video_min_pixels
        self.video_fps = self.data_args.processor.video_fps
        self.video_maxlen = self.data_args.processor.video_maxlen
        self.multi_turn = config.actor_rollout_ref.rollout.multi_turn.enable

        self.is_trailing_rank = False  # Indicates trailing ranks that received one less data item in round-robin partitioning.

        if self._rank == 0:
            logger.debug(f"Initializing PartitionedRLHFDataset with DDP rank {self.ddp_rank}, world size {self.ddp_world_size}, is_eval={self.is_eval}, drop_last={self.drop_last}")

            if self.processor is not None:
                logger.info(f"Set image_max_pixels={self.image_max_pixels}, image_min_pixels={self.image_min_pixels}, video_max_pixels={self.video_max_pixels}, video_min_pixels={self.video_min_pixels}, you can change these values via data.processor.image_max_pixels, etc.")

        # 1. Load the raw data partition for the current DDP rank
        dataset_files = self.data_args.val_files if is_eval else self.data_args.train_files
        raw_dataframe = self._load_partitioned_raw_data(dataset_files)

        if raw_dataframe is None or len(raw_dataframe) == 0:
            logger.warning(f"DDP rank {self.ddp_rank} received no data.")
            self.processed_dataframe = None
            return

        # 2. Filter out prompts that are too long from the loaded partition
        raw_dataframe = self._filter_overlong_prompts(raw_dataframe)
        
        # If this rank received fewer samples due to uneven partitioning, pad by duplicating the last row.
        if self.is_trailing_rank and raw_dataframe is not None and len(raw_dataframe) > 0:
            try:
                # Duplicate the last row to pad the partition
                last_row = raw_dataframe[-1].copy()
                if "extra_info" in last_row and isinstance(last_row["extra_info"], dict):
                    last_row["extra_info"]["padded_duplicate"] = True
                else:
                    last_row["extra_info"] = {"padded_duplicate": True}
                last_row_ds = datasets.Dataset.from_list([last_row])
                raw_dataframe = datasets.concatenate_datasets([raw_dataframe, last_row_ds])
                logger.debug(f"DDP rank {self.ddp_rank} is a trailing rank, duplicating last row to pad partition. New length: {len(raw_dataframe)}")
            except Exception:
                # We can safely ignore this exception because we mainly rely on the 'padded_duplicate' flag to identify padded elements.
                pass
            
        # 3. Preprocess the entire partition using multiple processes
        # By only removing the specific prompt_key, we ensure that other columns,
        # including complex types like dicts and strings from the original dataset,
        # are preserved. The .map() function will then add the new columns
        # returned by _preprocess_function. This is safer than removing all
        # columns and rebuilding the dataset from scratch.
        self.load_on_the_fly = self.force_on_the_fly
        if self.load_on_the_fly:
            self.processed_dataframe = raw_dataframe
        else:
            if self._rank == 0:
                logger.warning("Currently preloading and preprocessing the entire dataset. If you encounter Out-Of-Memory issues, please set data.force_on_the_fly=True to enable on-the-fly loading mode.")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                self.processed_dataframe = list(tqdm(
                    executor.map(self._preprocess_function, raw_dataframe),
                    total=len(raw_dataframe),
                    desc="Processing"
                ))

    def _load_partitioned_raw_data(self, dataset_files: Sequence[str]) -> Optional[datasets.Dataset]:
        """
        Loads a partition of Parquet data for the current DDP rank.
        """
        if not dataset_files:
            raise RuntimeError("No dataset files configured, aborting...")

        try:
            pq_files = [pq.ParquetFile(f) for f in dataset_files]

            # Gather (file_idx, row_group_idx, num_rows, start_row_idx_global)
            row_group_infos = []
            total_rows = 0
            for file_idx, pq_file in enumerate(pq_files):
                for rg_idx in range(pq_file.num_row_groups):
                    num_rows = pq_file.metadata.row_group(rg_idx).num_rows
                    row_group_infos.append({"file_idx": file_idx, "row_group_idx": rg_idx, "num_rows": num_rows, "start_row_idx_global": total_rows})
                    total_rows += num_rows

            if self._rank == 0:
                logger.debug(f"DDP rank={self.ddp_rank}, row group infos: {row_group_infos}")

            if total_rows < self.ddp_world_size:
                raise RuntimeError(f"Total rows ({total_rows}) is less than DDP world size ({self.ddp_world_size}),  cannot partition data across ranks. Please ensure enough data is available.")

            # Compute partition indices
            if self.drop_last:
                rows_per_rank = total_rows // self.ddp_world_size
                total_used_rows = rows_per_rank * self.ddp_world_size
                start = self.ddp_rank * rows_per_rank
                end = start + rows_per_rank
                if self._rank == 0:
                    logger.warning(
                        f"DDP Rank {self.ddp_rank} using drop_last=True, partitioning rows into {self.ddp_world_size} ranks with {rows_per_rank} rows each. Total used rows: {total_used_rows}, start={start}, end={end}. Total rows: {total_rows}, total dropped rows: {total_rows - total_used_rows}."
                    )
            else:
                # Distribute the remainder to the first (total_rows % ddp_world_size) ranks
                rows_per_rank = total_rows // self.ddp_world_size
                remainder = total_rows % self.ddp_world_size
                if self.ddp_rank < remainder:
                    start = self.ddp_rank * (rows_per_rank + 1)
                    end = start + rows_per_rank + 1
                else:
                    start = remainder * (rows_per_rank + 1) + (self.ddp_rank - remainder) * rows_per_rank
                    end = start + rows_per_rank
                    self.is_trailing_rank = True # There is one less sample compared to the previous ranks.

            if start >= end:
                raise RuntimeError(f"Rank {self.ddp_rank} assigned empty partition: start={start}, end={end}, total_rows={total_rows}")

            # Find which row groups overlap with [start, end)
            selected_chunks = []
            for info in row_group_infos:
                rg_start = info["start_row_idx_global"]
                rg_end = rg_start + info["num_rows"]
                # If overlap
                if rg_end > start and rg_start < end:
                    # Compute local slice within this row group
                    local_start = max(0, start - rg_start)
                    local_end = min(info["num_rows"], end - rg_start)
                    selected_chunks.append({"file_idx": info["file_idx"], "row_group_idx": info["row_group_idx"], "local_start": local_start, "local_end": local_end})

            # Read and slice the necessary row groups
            tables = []
            for chunk in selected_chunks:
                pq_file = pq_files[chunk["file_idx"]]
                table = pq_file.read_row_group(chunk["row_group_idx"])
                if chunk["local_start"] > 0 or chunk["local_end"] < table.num_rows:
                    table = table.slice(chunk["local_start"], chunk["local_end"] - chunk["local_start"])
                tables.append(table)

            if not tables:
                raise RuntimeError(f"DDP Rank {self.ddp_rank} assigned rows [{start}, {end}) but failed to read any data.")

            final_table = pa.concat_tables(tables)
            logger.debug(f"DDP rank={self.ddp_rank} loaded {len(final_table)} rows from {len(tables)} row groups. start={start}, end={end}, total_rows={total_rows}.")
            return datasets.Dataset(final_table)

        except Exception as e:
            logger.error(f"Failed during partitioned data loading for DDP rank {self.ddp_rank}: {dataset_files}. Error: {e}")
            raise

    def _filter_overlong_prompts(self, raw_dataframe: datasets.Dataset) -> datasets.Dataset:
        if self.filter_overlong_prompts:
            original_len = len(raw_dataframe)
            raw_dataframe = raw_dataframe.filter(
                lambda doc: len(self.tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Rank {self.ddp_rank} filtering prompts longer than {self.max_prompt_length} tokens",
            )
            filtered_len = len(raw_dataframe)
            if self._rank == 0:
                logger.info(f"Filtered prompts from {original_len} to {filtered_len} on each rank.")
        return raw_dataframe

    def __len__(self) -> int:
        return len(self.processed_dataframe) if self.processed_dataframe is not None else 0

    def _build_messages(self, example: dict) -> list:
        """Helper function to structure messages, adopted from RLHFDataset."""
        # The pop operation is safe here because map creates a copy for each process
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                # Simple split logic to handle interleaved text and images/videos
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    elif segment:  # Avoid adding empty strings
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
        return messages

    def _preprocess_function(self, row_dict: Dict) -> Dict:
        """
        The core preprocessing logic applied to each sample via `datasets.map()`.
        """
        processed_row = row_dict.copy()
        messages = self._build_messages(processed_row)
        model_inputs = {}

        # The output dict of this function becomes a row in the new dataset

        if self.processor is not None:
            # Note: Vision processing is kept here for simplicity.
            # For extreme performance, you might consider pre-serializing images/videos.
            from siirl.data_coordinator.dataloader.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            multi_modal_data = {}
            images = None
            if self.image_key in processed_row:
                images = [process_image(image, self.image_max_pixels, self.image_min_pixels) for image in processed_row.pop(self.image_key)]
                multi_modal_data["image"] = images
            videos = None
            if self.video_key in processed_row:
                videos = [process_video(video, fps=self.video_fps, fps_max_frames=self.video_maxlen, max_pixels=self.video_max_pixels, min_pixels=self.video_min_pixels) for video in processed_row.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            processed_row["multi_modal_data"] = multi_modal_data
            processed_row["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            processed_row["multi_modal_inputs"].pop("second_per_grid_ts", None)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from siirl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        processed_row["input_ids"] = input_ids[0]
        processed_row["attention_mask"] = attention_mask[0]
        processed_row["position_ids"] = position_ids[0]

        # Handle raw_prompt_ids for potential combination with other templates
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        processed_row["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            processed_row["raw_prompt"] = messages
        if self.return_full_prompt:
            processed_row["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if self.multi_turn:
            index = processed_row.get("extra_info", {}).get("index", 0)
            tools_kwargs = processed_row.get("extra_info", {}).get("tools_kwargs", {})
            interaction_kwargs = processed_row.get("extra_info", {}).get("interaction_kwargs", {})
            # need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
            # if need_tools_kwargs and not tools_kwargs:
            #     logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
            processed_row["index"] = index
            processed_row["tools_kwargs"] = tools_kwargs
            processed_row["interaction_kwargs"] = interaction_kwargs
        return processed_row

    def __getitem__(self, item: int) -> Dict:
        """
        Returns a preprocessed item from the dataset.
        """
        if self.processed_dataframe is None:
            raise IndexError("Dataset is empty or not initialized properly.")
        return self.processed_dataframe[item] if not self.load_on_the_fly else self._preprocess_function(self.processed_dataframe[item])
