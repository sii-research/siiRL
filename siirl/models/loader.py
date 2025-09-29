from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
)

from siirl.utils.extras.misc import skip_check_imports, try_download_model_from_other_hub
from siirl.models.patcher import patch_processor, patch_tokenizer

from loguru import logger


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from siirl.global_config.params import ModelArguments


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.
    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}")


def load_tokenizer(
    path: str = "",
    model_args: "ModelArguments" = None,
    correct_pad_token: bool = True,
    correct_gemma2=True,
) -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer and optionally loads processor.
    Note: including inplace operation of model_args.
    """
    init_kwargs = {}
    if model_args is not None:
        path = model_args.path
        init_kwargs = _get_init_kwargs(model_args)
    config = AutoConfig.from_pretrained(path, **init_kwargs)
    if correct_gemma2 and isinstance(path, str) and "gemma-2-2b-it" in path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        logger.warning("Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1)
        init_kwargs["eos_token"] = "<end_of_turn>"
        init_kwargs["eos_token_id"] = 107
    try:
        if "InternVL" in config.architectures[0] and "internlm2" in config.llm_config.model_type:
            from siirl.models.transformers.internvl_chat.tokenization_internlm2_fast import InternLM2TokenizerFast

            tokenizer = InternLM2TokenizerFast.from_pretrained(
                path,
                use_fast=True,
                split_special_tokens=model_args.split_special_tokens if model_args else False,
                padding_side="right",
                **init_kwargs,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                use_fast=model_args.use_fast_tokenizer if model_args else True,
                split_special_tokens=model_args.split_special_tokens if model_args else False,
                padding_side="right",
                **init_kwargs,
            )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    if model_args:
        patch_tokenizer(tokenizer, model_args, config)
    try:
        processor = AutoProcessor.from_pretrained(path, **init_kwargs)
        if model_args:
            patch_processor(processor, config, tokenizer, model_args)
    except Exception as e:
        logger.debug(f"Processor was not found: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    if processor is None and "InternVL" in config.architectures[0]:
        import torch
        from siirl.models.transformers.internvl import build_transform, dynamic_preprocess

        class InternVLProcessor:
            def __init__(self, proc, config):
                self.proc = proc
                self.image_size = config.force_image_size or config.vision_config.image_size
                self.patch_size = config.vision_config.patch_size
                self.dynamic_image_size = False  # config.dynamic_image_size
                self.min_dynamic_patch = config.min_dynamic_patch
                self.max_dynamic_patch = config.max_dynamic_patch
                self.use_thumbnail = config.use_thumbnail
                self.num_image_token = int((self.image_size // self.patch_size) ** 2 * (config.downsample_ratio**2))

            def process(self, images):
                transform = self.proc(True, input_size=self.image_size)
                pixel_values, image_flags = [], []
                for image in images:
                    pixel_values.append(transform(image))
                    image_flags.append(torch.tensor([1] * 1, dtype=torch.long))
                pixel_values = torch.stack(pixel_values)
                image_flags = torch.stack(image_flags)
                return {"pixel_values": pixel_values, "image_flags": image_flags}

        processor = InternVLProcessor(build_transform, config)
        logger.info("Build processor for model type ", config.model_type)

    if correct_pad_token:
        set_pad_token_id(tokenizer)

    return {"tokenizer": tokenizer, "processor": processor}
