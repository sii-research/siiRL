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


@dataclass
class MixedPrecisionArguments:
    param_dtype: Literal["float16", "bfloat16", "float32"] = field(
        default="bfloat16",
        metadata={"help": "Param precision to use for fsdp MixedPrecision model"},
    )
    reduce_dtype: Literal["float16", "bfloat16", "float32"] = field(
        default="float32",
        metadata={"help": "Reduce precision to use for fsdp MixedPrecision model"},
    )
    buffer_dtype: Literal["float16", "bfloat16", "float32"] = field(
        default="float32",
        metadata={"help": "Buffer precision to use for fsdp MixedPrecision model"},
    )
    keep_low_precision_grads: bool = field(default=False, metadata={"help": "Whether or not to use low precision grad"})
    cast_forward_inputs: bool = field(default=False, metadata={"help": "Whether or not to cast forward inputs"})
    cast_root_forward_inputs: bool = field(default=True, metadata={"help": "Whether or not to cast root forward inputs"})


@dataclass
class FSDPArguments:
    wrap_policy: Dict[str, Any] = field(
        default_factory=lambda: {"min_num_params": 0},
        metadata={"help": "Wrapping policy configuration"},
    )
    param_offload: bool = field(default=False, metadata={"help": "Parameter offloading"})
    optimizer_offload: bool = field(default=False, metadata={"help": "Optimizer state offloading"})
    fsdp_size: int = field(default=-1, metadata={"help": "FSDP group size"})
    model_dtype: Literal["float16", "bfloat16", "float32"] = field(
        default="float32",
        metadata={"help": "PrecisionType to use for model"},
    )
    mixed_precision: MixedPrecisionArguments = field(
        default_factory=MixedPrecisionArguments,
        metadata={"help": "fsdp mixed precision settings"},
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MegatronArguments:
    tensor_model_parallel_size: int = field(default=1, metadata={"help": "Tensor parallelism size"})
    pipeline_model_parallel_size: int = field(default=1, metadata={"help": "Pipeline parallelism size"})
    virtual_pipeline_model_parallel_size: Optional[int] = field(default=None, metadata={"help": "Virtual pipeline model parallel size"})
    sequence_parallel: bool = field(default=False, metadata={"help": "Whether the sequence parallel is enabled."})
    use_distributed_optimizer: bool = field(
        default=False,
        metadata={"help": "Whether the distributed optimizer is enabled."},
    )
    param_dtype: str = field(default="bfloat16", metadata={"help": "parameter data dtype"})
    seed: int = field(default=1, metadata={"help": "The random seed"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerArguments:
    lr: float = field(default=1e-6, metadata={"help": "Learning rate"})
    lr_warmup_steps_ratio: float = field(default=0.0, metadata={"help": "Warmup steps ratio"})
    min_lr_ratio: Optional[float] = field(default=None, metadata={"help": "Min learning rate ratio"})
    warmup_style: str = field(default="constant", metadata={"help": "Warmup strategy"})
    total_training_steps: int = field(default=0, metadata={"help": "Total training steps"})
    betas: tuple[float, float] = field(default=(0.9, 0.999), metadata={"help": "Beta params Of Optimizer"})
    weight_decay: float = field(default=1e-2, metadata={"help": "Weight decay params of Optimizer"})
    lr_warmup_steps: int = field(
        default=-1,
        metadata={"help": "Prioritized. Negative values mean delegating to lr_warmup_steps_ratio."},
    )
    clip_grad: float = field(default=1.0, metadata={"help": "gradient clip"})
    num_cycles: float = field(default=0.5, metadata={"help": "num cycles"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessorArguments:
    r"""
    Arguments pertaining to the image processor.
    """

    image_max_pixels: int = field(
        default=768 * 768,
        metadata={"help": "The maximum number of pixels of image inputs."},
    )
    image_min_pixels: int = field(
        default=32 * 32,
        metadata={"help": "The minimum number of pixels of image inputs."},
    )
    video_max_pixels: int = field(
        default=256 * 256,
        metadata={"help": "The maximum number of pixels of video inputs."},
    )
    video_min_pixels: int = field(
        default=16 * 16,
        metadata={"help": "The minimum number of pixels of video inputs."},
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "The frames to sample per second for video inputs."},
    )
    video_maxlen: int = field(
        default=128,
        metadata={"help": "The maximum number of sampled frames for video inputs."},
    )


@dataclass
class ModelArguments(ProcessorArguments):
    path: str = field(
        default="~/models/deepseek-llm-7b-chat",
        metadata={"help": "Model path or identifier"},
    )
    external_lib: Optional[str] = field(default=None, metadata={"help": "External model library"})
    override_config: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Model config overrides"})
    enable_gradient_checkpointing: bool = field(default=True, metadata={"help": "Gradient checkpointing"})
    use_remove_padding: bool = field(default=False, metadata={"help": "Padding removal optimization"})
    use_fused_kernels: bool = field(default=False, metadata={"help": "Kernels fuse optimization"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Download from hugging face, modelscope, openmind local cache dir"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    model_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    use_liger: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply Liger kernel to the model"},
    )
    fsdp_config: FSDPArguments = field(default_factory=FSDPArguments, metadata={"help": "FSDP settings"})
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    input_tokenizer: Optional[str] = field(default=None, metadata={"help": "input tokenizer path"})
    rm_tokenizer: Optional[str] = field(default=None, metadata={"help": "rmokenizer path"})
    lora_rank: int = field(default=0, metadata={"help": "set to positive value to enable LoRA (e.g., 32)"})
    lora_alpha: float = field(default=16, metadata={"help": "LoRA scaling factor"})
    target_modules: str = field(default="all-linear", metadata={"help": "all-linear or [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"})
    use_shm: bool = field(default=False)
    enable_activation_offload: bool = field(default=False, metadata={"help": "enable activation offload"})

    def __post_init__(self):
        if self.path is None:
            raise ValueError("Please provide `path`.")

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.new_special_tokens is not None:  # support multiple special tokens
            self.new_special_tokens = [token.strip() for token in self.new_special_tokens.split(",")]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointArguments:
    contents: List[str] = field(default_factory=["model", "hf_model", "optimizer", "extra"], metadata={"help": "The contents to save in the checkpoint."})


@dataclass
class ActorArguments:
    strategy: str = field(default="fsdp", metadata={"help": "Parallel strategy"})
    ppo_mini_batch_size: int = field(default=256, metadata={"help": "PPO mini-batch size"})
    ppo_micro_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Micro-batch size"})
    ppo_micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU micro-batch size"})
    use_dynamic_bsz: bool = field(default=False, metadata={"help": "Dynamic batch sizing"})
    ppo_max_token_len_per_gpu: int = field(default=16384, metadata={"help": "Max tokens per GPU"})
    grad_clip: float = field(default=1.0, metadata={"help": "Gradient clipping"})
    clip_ratio: float = field(default=0.2, metadata={"help": "Clipping ratio"})
    clip_ratio_low: float = field(default=0.2, metadata={"help": "Min value for clip ratio"})
    clip_ratio_high: float = field(default=0.2, metadata={"help": "Max value for clip ratio"})
    clip_ratio_c: float = field(default=3.0, metadata={"help": "lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729"})
    entropy_coeff: float = field(default=0.001, metadata={"help": "Entropy coefficient"})
    use_kl_loss: bool = field(default=False, metadata={"help": "Enable KL loss"})
    kl_loss_coef: float = field(default=0.001, metadata={"help": "KL loss coefficient"})
    kl_loss_type: str = field(default="low_var_kl", metadata={"help": "KL loss type"})
    ppo_epochs: int = field(default=1, metadata={"help": "PPO epochs"})
    shuffle: bool = field(default=False, metadata={"help": "Data shuffling"})
    ulysses_sequence_parallel_size: int = field(default=1, metadata={"help": "Sequence parallel size"})
    optim: OptimizerArguments = field(default_factory=OptimizerArguments, metadata={"help": "Optimizer settings"})
    fsdp_config: FSDPArguments = field(default_factory=FSDPArguments, metadata={"help": "FSDP settings"})
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    use_remove_padding: bool = field(default=False, metadata={"help": "Padding removal optimization"})
    use_fused_kernels: bool = field(default=False, metadata={"help": "Kernels fuse optimization"})
    use_torch_compile: bool = field(default=True, metadata={"help": "Whether or not use torch compile"})
    checkpoint: CheckpointArguments = field(default_factory=CheckpointArguments, metadata={"help": "Checkpoint configuration"})
    param_offload: bool = field(default=False, metadata={"help": "Enable param offload or not"})
    grad_offload: bool = field(default=False, metadata={"help": "Enable grad offload or not"})
    optimizer_offload: bool = field(default=False, metadata={"help": "Enable optimizer offload or not"})
    load_weight: bool = field(default=True)
    loss_agg_mode: str = field(default="token-mean", metadata={"help": "seq-mean-token-sum, seq-mean-token-mean"})
    recompute_old_log_prob: bool = field(default=True, metadata={"help": "recompute old log prob"})
    use_cpgd_loss: bool = field(default=False, metadata={"help": "use cpgd loss"})
    policy_drift_coeff: float = field(default=0.0, metadata={"help": "policy drift coeff for CPGD"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalSamplingArguments:
    top_k: int = field(default=-1, metadata={"help": "0 for hf rollout, -1 for vllm rollout"})
    top_p: float = field(default=1.0)
    temperature: int = field(default=0)
    n: int = field(default=1)
    do_sample: bool = field(default=False)


@dataclass
class LayerNameMapArguments:
    qkv_layer_name: str = field(default="qkv", metadata={"help": "QKV layer name map"})
    gate_proj_layer_name: str = field(default="linear_fc1.weight", metadata={"help": "Gate projection layer name map"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiTurnArguments:
    enable: bool = field(default=False, metadata={"help": "should set rollout.name to sglang_async if True"})
    max_assistant_turns: Optional[int] = field(default=None, metadata={"help": "null for no limit (default max_length // 3)"})
    tool_config_path: Optional[str] = field(default=None, metadata={"help": "null for no tool"})
    format: str = field(default="hermes", metadata={"help": "Format of the multi-turn interaction. Options: hermes, llama3_json, ..."})
    tool_config_path: Optional[str] = field(default=None, metadata={"help": " null for no tool"})
    max_user_turns: Optional[int] = field(default=None, metadata={"help": "null for no limit (default max_length // 3)"})
    max_parallel_calls: int = field(default=1, metadata={"help": "max parallel call for tools in single turn"})
    max_tool_response_length: int = field(default=256, metadata={"help": "max length of tool response"})
    tool_response_truncate_side: str = field(default="middle", metadata={"help": "truncate side of tool response: left, middle, right"})
    interaction_config_path: Optional[str] = field(default=None, metadata={"help": "null for no interaction"})
    completion_callback: Optional[str] = field(default=None, metadata={"help": "null for default callback"})
    use_inference_chat_template: bool = field(
        default=False,
        metadata={
            "help": "- When set to True, the model's default chat template is used for multi-turn rollout, which typically matches production behavior. \n \
    - When set to False, the token ids recorded for training are used instead; unlike the default chat template, these always include the model's full output, \n \
      which may contain additional content such as reasoning content. This maintains the consistency between training and rollout, but it will lead to longer prompts."
        },
    )
    tokenization_sanity_check_mode: str = field(
        default="strict",
        metadata={
            "help": "- disable: disable tokenization sanity check \n \
    - strict: enable strict tokenization sanity check (default) \n \
    - ignore_strippable: ignore strippable tokens when checking tokenization sanity"
        },
    )


@dataclass
class CustomAsyncServer:
    path: None
    # Path to the custom async server implementation
    name: None
    # Class name of the custom async server class (e.g. AsyncvLLMServer)


@dataclass
class AgentArguments:
    agent_name: str = field(default="single_turn_agent", metadata={"help": "choose which agent tool"})
    num_workers: int = field(default=1, metadata={"help": "custom async server configs"})
    # custom async server configs
    custom_async_server: CustomAsyncServer = field(default=None, metadata={"help": "custom async server configs"})
    # Path to the custom async server implementation

    train_cycle: int = field(default=None, metadata={"help": "Train cycle"})
    process_path: str = field(default=None, metadata={"help": "Path to the pre-process function"})
    pre_process_kwargs: dict = field(default_factory=dict, metadata={"help": "Pre-process function arguments"})
    post_process_kwargs: dict = field(default_factory=dict, metadata={"help": "Post-process function arguments"})
    env: list = field(default_factory=list, metadata={"help": "Env list"})

@dataclass
class EngineArguments:
    vllm: Dict[str, Any] = field(default_factory=lambda: {})
    sglang: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class RolloutArguments:
    name: str = field(default="vllm", metadata={"help": "Rollout engine"})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature"})
    top_k: int = field(default=-1, metadata={"help": "Top-k sampling"})
    top_p: float = field(default=1.0, metadata={"help": "Top-p sampling"})
    use_fire_sampling: bool = field(default=False, metadata={"help": "Fire sampling optimization"})
    prompt_length: int = field(default=512, metadata={"help": "Prompt length"})
    response_length: int = field(default=512, metadata={"help": "Response length"})
    dtype: str = field(default="bfloat16", metadata={"help": "Compute dtype"})
    gpu_memory_utilization: float = field(default=0.5, metadata={"help": "GPU memory usage"})
    ignore_eos: bool = field(default=False, metadata={"help": "Ignore EOS tokens"})
    enforce_eager: bool = field(default=True, metadata={"help": "Eager execution"})
    free_cache_engine: bool = field(default=True, metadata={"help": "Free GPU cache"})
    load_format: str = field(default="dummy_dtensor", metadata={"help": "Weight loading format"})
    tensor_model_parallel_size: int = field(default=1, metadata={"help": "Tensor parallelism"})
    max_num_batched_tokens: int = field(default=8192, metadata={"help": "Max batched tokens"})
    max_model_len: Optional[int] = field(default=None, metadata={"help": "Max model length"})
    max_num_seqs: int = field(default=1024, metadata={"help": "Max concurrent sequences"})
    limit_images: Optional[int] = field(default=None, metadata={"help": "support for multi-image data"})
    do_sample: bool = field(default=True, metadata={"help": "Enable sampling"})
    n: int = field(default=1, metadata={"help": "Number of responses"})
    log_prob_micro_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Log prob batch size"})
    log_prob_micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU log prob batch size"})
    log_prob_max_token_len_per_gpu: int = field(default=16384, metadata={"help": "Max tokens per GPU"})
    log_prob_use_dynamic_bsz: bool = field(default=False, metadata={"help": "Dynamic log prob batch size"})
    disable_log_stats: bool = field(default=True, metadata={"help": "Whether or not disable log stats"})
    enable_chunked_prefill: bool = field(default=True, metadata={"help": "Whether or not enable chunked prefill"})
    trust_remote_code: bool = field(default=False, metadata={"help": "trust the code or not."})
    val_kwargs: EvalSamplingArguments = field(default_factory=EvalSamplingArguments)
    layer_name_map: LayerNameMapArguments = field(default_factory=LayerNameMapArguments)
    seed: int = field(default=0, metadata={"help": "The random seed"})
    mode: str = field(default="sync", metadata={"help": "sync: LLM, async: AsyncLLM"})
    multi_turn: MultiTurnArguments = field(default_factory=MultiTurnArguments)
    micro_batch_size: Optional[int] = field(default=None, metadata={"help": "Inference micro-batch size"})
    engine_kwargs: EngineArguments = field(default_factory=EngineArguments)
    calculate_log_probs: bool = field(default=False, metadata={"help": "support logging rollout prob for debugging purpose"})
    agent: AgentArguments = field(default_factory=AgentArguments)
    multi_stage_wake_up: bool = field(default=False, metadata={"help": "# Whether to wake up inference engine in multi-stage. (Wake up model weights first, then resume kv cache)"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefArguments:
    strategy: str = field(default="fsdp", metadata={"help": "Parallel strategy"})
    fsdp_config: FSDPArguments = field(default_factory=FSDPArguments, metadata={"help": "Reference FSDP settings"})
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    log_prob_micro_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Log prob batch size"})
    log_prob_micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU log prob batch size"})
    log_prob_use_dynamic_bsz: bool = field(default=False, metadata={"help": "Dynamic log prob batch size"})
    log_prob_max_token_len_per_gpu: int = field(default=16384, metadata={"help": "Max tokens per GPU"})
    ulysses_sequence_parallel_size: int = field(default=1, metadata={"help": "Sequence parallel size"})
    use_remove_padding: bool = field(default=False, metadata={"help": "Padding removal optimization"})
    use_fused_kernels: bool = field(default=False, metadata={"help": "Kernels fuse optimization"})
    use_torch_compile: bool = field(default=True, metadata={"help": "Whether or not use torch compile"})
    ppo_micro_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Micro-batch size"})
    ppo_micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU micro-batch size"})
    param_offload: bool = field(default=False, metadata={"help": "Enable param offload or not"})
    grad_offload: bool = field(default=False, metadata={"help": "Enable grad offload or not"})
    optimizer_offload: bool = field(default=False, metadata={"help": "Enable optimizer offload or not"})
    load_weight: bool = field(default=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActorRolloutRefArguments:
    hybrid_engine: bool = field(default=True, metadata={"help": "Hybrid engine mode"})
    model: ModelArguments = field(default_factory=ModelArguments, metadata={"help": "Base model settings"})
    actor: ActorArguments = field(default_factory=ActorArguments, metadata={"help": "Actor configuration"})
    ref: RefArguments = field(default_factory=RefArguments, metadata={"help": "Reference model settings"})
    rollout: RolloutArguments = field(default_factory=RolloutArguments, metadata={"help": "Rollout parameters"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CriticArguments:
    strategy: str = field(default="fsdp", metadata={"help": "Parallel strategy"})
    optim: OptimizerArguments = field(
        default_factory=lambda: OptimizerArguments(lr=1e-5),
        metadata={"help": "Optimizer settings"},
    )
    model: ModelArguments = field(
        default_factory=lambda: ModelArguments(path="~/models/deepseek-llm-7b-chat", enable_gradient_checkpointing=True),
        metadata={"help": "Critic model"},
    )
    fsdp_config: FSDPArguments = field(default_factory=FSDPArguments, metadata={"help": "FSDP settings"})
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    ppo_mini_batch_size: int = field(default=256, metadata={"help": "PPO mini-batch size"})
    ppo_micro_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Micro-batch size"})
    ppo_micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU micro-batch size"})
    use_dynamic_bsz: bool = field(default=False, metadata={"help": "Dynamic batch size"})
    ppo_epochs: int = field(default=1, metadata={"help": "PPO epochs"})
    shuffle: bool = field(default=False, metadata={"help": "Data shuffling"})
    grad_clip: float = field(default=1.0, metadata={"help": "Gradient clipping"})
    cliprange_value: float = field(default=0.5, metadata={"help": "Value clipping range"})
    ulysses_sequence_parallel_size: int = field(default=1, metadata={"help": "Sequence parallel size"})
    forward_micro_batch_size_per_gpu: int = field(default=None, metadata={"help": "Forward micro batch size per gpu"})
    forward_micro_batch_size: int = field(default=None, metadata={"help": "Forward micro batch size"})
    forward_max_token_len_per_gpu: int = field(default=32768, metadata={"help": "Forward max token length in per gpu"})
    load_weight: bool = field(default=True)
    rollout_n: int = field(default=5, metadata={"help": "rollout n"})
    checkpoint: CheckpointArguments = field(default_factory=CheckpointArguments, metadata={"help": "Checkpoint configuration"})
    ppo_max_token_len_per_gpu: int = field(default=16384, metadata={"help": "Max tokens per GPU"})
    loss_agg_mode: str = field(default="token-mean", metadata={"help": "token-mean, seq-mean-token-sum, seq-mean-token-mean"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OverlongBufferArguments:
    """DAPO-specific overlong buffer configuration for handling sequences longer than max length."""

    enable: bool = field(default=False, metadata={"help": "Enable overlong sequence buffer"})
    len: int = field(default=512, metadata={"help": "Buffer length for overlong sequences"})
    penalty_factor: float = field(default=1.0, metadata={"help": "Penalty factor for overlong sequences"})
    log: bool = field(default=False, metadata={"help": "Enable logging of overlong buffer rewards and penalties"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RewardModelArguments:
    enable: bool = field(default=False, metadata={"help": "Enable reward model"})
    strategy: str = field(default="fsdp", metadata={"help": "Parallel strategy"})
    model: ModelArguments = field(
        default_factory=lambda: ModelArguments(path="~/models/deepseek-llm-7b-chat", enable_gradient_checkpointing=True),
        metadata={"help": "Critic model"},
    )
    fsdp_config: FSDPArguments = field(
        default_factory=lambda: FSDPArguments(wrap_policy={"min_num_params": 0}, param_offload=False),
        metadata={"help": "FSDP configuration"},
    )
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    micro_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Micro-batch size"})
    micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU micro-batch size"})
    max_length: Optional[int] = field(default=None, metadata={"help": "Max sequence length"})
    ulysses_sequence_parallel_size: int = field(default=1, metadata={"help": "Sequence parallel size"})
    use_dynamic_bsz: bool = field(default=False, metadata={"help": "Dynamic batch size"})
    reward_manager: str = field(default="batch", metadata={"help": "Reward management strategy"})
    forward_micro_batch_size_per_gpu: int = field(default=None, metadata={"help": "Forward micro batch size per gpu"})
    forward_micro_batch_size: int = field(default=None, metadata={"help": "Forward micro batch size"})
    forward_max_token_len_per_gpu: int = field(default=32768, metadata={"help": "Forward max token length in per gpu"})
    load_weight: bool = field(default=True)
    launch_reward_fn_async: bool = field(default=False, metadata={"help": "custom reward function executed async on CPU, during log_prob"})
    reward_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    sandbox_fusion: Optional[Dict[str, Any]] = field(default=None)
    overlong_buffer: OverlongBufferArguments = field(default_factory=OverlongBufferArguments, metadata={"help": "DAPO overlong buffer configuration"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KLCtrlArguments:
    type: str = field(default="fixed", metadata={"help": "Type of KL Ctrl, fixed or adaptive"})
    kl_coef: float = field(default=0.001, metadata={"help": "Coef of KL"})
    target_kl: Optional[float] = field(default=0, metadata={"help": "Target KL value"})
    horizon: Optional[float] = field(default=0, metadata={"help": "Horizon of KL"})


@dataclass
class FilterGroupsArguments:
    """DAPO-specific filter groups configuration for dynamic sampling."""

    enable: bool = field(default=False, metadata={"help": "Enable trajectory filtering based on variance"})
    metric: str = field(default="acc", metadata={"help": "Metric used for filtering (acc, seq_final_reward, seq_reward)"})
    max_num_gen_batches: int = field(default=10, metadata={"help": "Maximum generation batches before giving up"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlgorithmArguments:
    gamma: float = field(default=1.0, metadata={"help": "Discount factor"})
    lam: float = field(default=1.0, metadata={"help": "GAE lambda"})
    adv_estimator: str = field(default="gae", metadata={"help": "Advantage estimator"})
    kl_penalty: str = field(default="kl", metadata={"help": "KL penalty type"})
    kl_ctrl: KLCtrlArguments = field(default_factory=KLCtrlArguments)
    use_kl_in_reward: bool = field(default=True, metadata={"help": "Use KL In-Reward"})
    norm_adv_by_std_in_grpo: bool = field(default=True, metadata={"help": "Whether to scale the GRPO advantage"})
    weight_factor_in_cpgd: str = field(default="STD_weight", metadata={"help": "The weighting methods for advantage {STD_weight, clip_filter_like_weight, naive}"})
    algorithm_name: str = field(default="grpo", metadata={"help": "Algorithm name, e.g., grpo, ppo, dapo"})
    filter_groups: FilterGroupsArguments = field(default_factory=FilterGroupsArguments, metadata={"help": "DAPO filter groups configuration"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
