#!/usr/bin/env bash
# ===================================================================================
# ===                       USER CONFIGURATION SECTION                            ===
# ===================================================================================

# --- Model & Data Paths ---
export HOME=${HOME:-/root}
export MODEL_PATH=$HOME/models/Qwen/Qwen2.5-0.5B-Instruct
export TRAIN_DATA_PATH=$HOME/data/gsm8k/train.parquet
export TEST_DATA_PATH=$HOME/data/gsm8k/test.parquet

# --- Output Paths ---
export TENSORBOARD_DIR=tensorboard/qwen2.5_0.5b_grpo_toy
export SIIRL_LOGGING_FILENAME=qwen2.5_0.5b_grpo

# --- Core Training Command ---
# All arguments for the training script are defined here for clarity.
TRAINING_CMD=(
    python3 -m siirl.client.main_dag
    algorithm.adv_estimator=grpo
    data.train_files=$TRAIN_DATA_PATH
    data.val_files=$TEST_DATA_PATH
    data.train_batch_size=128
    data.max_prompt_length=2048
    data.max_response_length=4096
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    actor_rollout_ref.model.path=$MODEL_PATH
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.grad_clip=0.5
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.max_model_len=8192
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.kl_ctrl.kl_coef=0.001
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.logger=['console','tensorboard']
    trainer.project_name=siirl_qwen2.5_0.5b_grpo
    trainer.experiment_name=siirl_qwen2.5_0.5b_grpo_toy
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.save_freq=200
    trainer.test_freq=10
    trainer.total_epochs=30
    trainer.resume_mode=auto
    trainer.max_actor_ckpt_to_keep=1
    trainer.default_local_dir=ckpts/qwen2.5_0.5b/grpo/
    trainer.val_before_train=True
)

# ===================================================================================
# ===                  MAIN EXECUTION LOGIC & INFRASTRUCTURE                      ===
# ===         (Generally, no modifications are needed below this line)            ===
# ===================================================================================

# --- Boilerplate Setup ---
set -e
set -o pipefail
set -x

# --- Main Execution Function ---
main() {
    # Ensure any previous Ray instance is stopped.
    ray stop --force

    # Set environment variables for vLLM if needed
    export VLLM_USE_V1=1

    echo "INFO: Starting single-GPU training task..."

    # Execute the training command. The '$@' allows passing extra command-line args.
    eval "${TRAINING_CMD[@]}" "$@"

    echo "INFO: Training script has finished."
    ray stop --force >/dev/null 2>&1
}

# --- Script Entrypoint ---
main "$@"

