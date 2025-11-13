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


from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

from siirl.params.model_args import ProcessorArguments


@dataclass
class DataArguments:
    dataset_type: str = field(
        default="llm",
        metadata={"help": "Type of dataset, e.g., 'llm' for traditional prompt-based datasets, "
                          "or 'vla' for Vision-Language-Action task manifests."}
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer configuration (null for auto-detection)"},
    )
    train_files: List[str] = field(
        default_factory=lambda: ["~/data/rlhf/gsm8k/train.parquet"],
        metadata={"help": "Training dataset path"},
    )
    val_files: List[str] = field(
        default_factory=lambda: ["~/data/rlhf/gsm8k/test.parquet"],
        metadata={"help": "Validation dataset path"},
    )
    prompt_key: str = field(default="prompt", metadata={"help": "Dataset column name for prompts"})
    max_prompt_length: int = field(default=512, metadata={"help": "Max token length for prompts"})
    max_response_length: int = field(default=512, metadata={"help": "Max token length for responses"})
    train_batch_size: int = field(default=1024, metadata={"help": "Training batch size"})
    gen_batch_size: Optional[int] = field(default=None, metadata={"help": "Generation batch size for DAPO (typically 3x train_batch_size)"})
    val_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Validation batch handling"})
    return_raw_input_ids: bool = field(default=False, metadata={"help": "Return raw token IDs"})
    return_raw_chat: bool = field(default=False, metadata={"help": "Return unprocessed chat data"})
    return_full_prompt: bool = field(default=False, metadata={"help": "Whether to return the full prompt with chat template"})
    filter_overlong_prompts: bool = field(default=False, metadata={"help": "For large-scale dataset, filtering overlong prompts could be timeconsuming."})
    shuffle: bool = field(default=True, metadata={"help": "Shuffle training data"})
    image_key: str = field(default="images", metadata={"help": "Dataset column name for images"})
    video_key: str = field(default="videos", metadata={"help": "Dataset column name for videos"})
    truncation: str = field(
        default="error",
        metadata={"help": "Truncate the input_ids or prompt length if they exceed max_prompt_length. Default is 'error', not allow exceed the max_prompt_length. The users should increase the max_prompt_length if throwing the error. You can also set ``left`` ``middle`` and ``right``"},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    )
    tool_format: Optional[str] = field(
        default=None,
        metadata={"help": "Tool format to use for constructing function calling examples."},
    )
    tokenized_path: Optional[str] = field(
        default=None,
        metadata={"help": ("Path to save or load the tokenized datasets. If tokenized_path not exists, it will save the tokenized datasets. If tokenized_path exists, it will load the tokenized datasets.")},
    )
    dataset_cache_dir: str = field(
        default="/tmp/.cache/siirl/rlhf",
        metadata={"help": "Local cache directory for rlhf dataset."},
    )
    filter_overlong_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to filter prompt which length > max_prompt_length for dataset."},
    )
    serialize_dataset: bool = field(
        default=False,
        metadata={"help": "Whether or not to store serialize dataset in state_dict."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    reward_fn_key: str = field(default="data_source", metadata={"help": "reward data source key"})
    multi_agent: bool = field(default=False, metadata={"help": "The DAG pipeline is multi agent or not"})
    auto_repeat: bool = field(default=False, metadata={"help": "Automatically repeats the training dataset. Recommended when the number of samples is smaller than the total training steps to prevent premature termination."})
    num_loader_workers: int = field(default=8, metadata={"help": "DataLoader worker number"})
    force_on_the_fly: bool = field(default=False, metadata={"help": "If True, the data will be loaded on-the-fly, which is useful for large datasets that cannot fit into memory."})
    processor: ProcessorArguments = field(
        default_factory=ProcessorArguments,
        metadata={"help": "Arguments for the processor."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.train_files = split_arg(self.train_files)
        self.val_files = split_arg(self.val_files)
        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")
        if self.interleave_probs is not None:
            if self.mix_strategy == "concat":
                raise ValueError("`interleave_probs` is only valid for interleaved mixing.")

            self.interleave_probs = list(map(float, split_arg(self.interleave_probs)))
            if self.train_files is not None and len(self.train_files) != len(self.interleave_probs):
                raise ValueError("The length of dataset and interleave probs should be identical.")

            if self.val_files is not None and len(self.val_files) != len(self.interleave_probs):
                raise ValueError("The length of eval dataset and interleave probs should be identical.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
