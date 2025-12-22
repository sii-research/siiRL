# Copyright 2025 the LlamaFactory team.
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

from types import MethodType
from typing import TYPE_CHECKING, Any, Dict

import torch
from peft import PeftModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    is_torch_npu_available,
)

from loguru import logger
from siirl.models.model_utils.visual import (
    get_image_seqlen,
    get_patch_size,
    get_vision_feature_select_strategy,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin

    from siirl.utils.params import ModelArguments


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments", config: "PretrainedConfig") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length != model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    if "InternVL" in config.architectures[0]:

        def eos_token_id_patch(self):
            return self.super().eos_token_id

        tokenizer.__class__.eos_token_id = property(eos_token_id_patch)
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")


def patch_processor(
    processor: "ProcessorMixin",
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)
    if getattr(config, "vision_config", None) is not None:  # visual models
        setattr(processor, "image_seqlen", get_image_seqlen(config))
        setattr(processor, "patch_size", get_patch_size(config, processor))
        setattr(processor, "image_max_pixels", model_args.image_max_pixels)
        setattr(processor, "image_min_pixels", model_args.image_min_pixels)
        setattr(processor, "video_max_pixels", model_args.video_max_pixels)
        setattr(processor, "video_min_pixels", model_args.video_min_pixels)
        setattr(processor, "video_fps", model_args.video_fps)
        setattr(processor, "video_maxlen", model_args.video_maxlen)
        setattr(
            processor,
            "vision_feature_select_strategy",
            get_vision_feature_select_strategy(config, processor),
        )
