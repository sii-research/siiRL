#!/usr/bin/env bash
# ===================================================================================
# ===    Embodied AI GRPO Training with OpenVLA-OFT on LIBERO-10               ===
# ===================================================================================
# 
# This script trains an OpenVLA-OFT model using GRPO on the LIBERO-10 benchmark.

set -e

# --- Environment Setup (Critical for siiRL) ---
export SIIRL_DIR="${SIIRL_DIR:your_siirl_path}"
export PYTHONPATH="$SIIRL_DIR:/root/LIBERO/:your_vjepa2_path:$PYTHONPATH"

# --- Experiment and Model Definition ---
export DATASET=libero_10
export ALG=grpo
export MODEL_NAME=openvla-oft-7b
export MODEL_TYPE=openvla-oft

# --- Path Definitions (USER PROVIDED) ---
export HOME_PATH=${HOME_PATH:your_home_path}
export TRAIN_DATA_PATH=$HOME_PATH/datasets/vla-oft/libero/$DATASET/train.parquet
export TEST_DATA_PATH=$HOME_PATH/datasets/vla-oft/libero/$DATASET/test.parquet
export MODEL_PATH=$HOME_PATH/models/Haozhan72/Openvla-oft-SFT-libero10-traj1
export VJEPA_MODEL_PATH=$HOME_PATH/models/vjepa2/vitg-384.pt

# Base output paths
export BASE_CKPT_PATH=ckpts
export BASE_TENSORBOARD_PATH=tensorboard

# --- Embodied AI Specific Parameters (from YAML) ---
export ACTION_TOKEN_LEN=7        # 7 dimensions: xyz (3), quaternion (3), gripper (1)
export ACTION_CHUNKS_LEN=8       # OpenVLA-OFT uses 8-step action chunks
export NUM_ENVS=16               # From YAML: actor_rollout_ref.embodied.env.num_envs
export MAX_EPISODE_STEPS=512     # From YAML: actor_rollout_ref.embodied.env.max_steps
export UNNORM_KEY=libero_10      # ALIGNED: Standard key (not _no_noops)

# --- Data and Sampling Parameters (OPTIMIZED - removed duplicates) ---
export VAL_BATCH_SIZE=496                      # Validation batch size (verl: 496)
export MAX_PROMPT_LENGTH=256                   # ALIGNED: Changed from 512 to 256 (verl: 256)
export MAX_RESPONSE_LENGTH=128                 # ALIGNED: Changed from 512 to 128 (verl: 128)

# --- Embodied Sampling Parameters (moved to algorithm.embodied_sampling) ---
export FILTER_ACCURACY=True                    # Enable accuracy-based filtering (verl: True)
export ACCURACY_LOWER_BOUND=0.1                # Only keep prompts with success rate >= 0.1 (verl: 0.1)
export ACCURACY_UPPER_BOUND=0.9                # Only keep prompts with success rate <= 0.9 (verl: 0.9)
export FILTER_TRUNCATED=False                  # Filter truncated episodes (uses env.max_steps)
export OVERSAMPLE_FACTOR=1                     # Oversample factor for filtering (verl: 1)

# --- Training Hyperparameters (from YAML - verl standard) ---
export TRAIN_BATCH_SIZE=64       # From YAML: data.train_batch_size
export PPO_MINI_BATCH_SIZE=32    # From YAML: actor_rollout_ref.actor.ppo_mini_batch_size
export PPO_MICRO_BATCH_SIZE_PER_GPU=8          # ALIGNED: Changed from 1 to 8 (verl: 8)
export LOG_PROB_MICRO_BATCH_SIZE=8             # Log prob micro batch size (verl: 8)
export ROLLOUT_MICRO_BATCH_SIZE=1              # Rollout micro batch size (verl: 1)
export ROLLOUT_N_SAMPLES=8                     # REUSED: Number of samples per prompt (replaces N_SAMPLES)
export PPO_EPOCHS=1                            # From YAML: actor_rollout_ref.actor.ppo_epochs

# Algorithm parameters (verl standard)
export LEARNING_RATE=5e-6        # ALIGNED: Changed from 1e-6 to 5e-6 (verl: 5e-6)
export WEIGHT_DECAY=0.0          # From YAML: actor_rollout_ref.actor.optim.weight_decay
export CLIP_RATIO_HIGH=0.28      # From YAML: actor_rollout_ref.actor.clip_ratio_high
export CLIP_RATIO_LOW=0.2        # From YAML: actor_rollout_ref.actor.clip_ratio_low
export ENTROPY_COEFF=0.0         # ALIGNED: Changed from 0.001 to 0.0 (verl: 0.0)
export TEMPERATURE=1.6           # ALIGNED: Changed from 1.0 to 1.6 (verl: 1.6)
export GAMMA=1.0                 # From YAML: algorithm.gamma (verl standard)
export LAM=1.0                   # From YAML: algorithm.lam (verl standard)
export GRAD_CLIP=1.0             # From YAML: actor_rollout_ref.actor.grad_clip

# --- Image/Video Processing (from YAML) ---
export IMG_SIZE=384              # From YAML: actor_rollout_ref.embodied.img_size
export ENABLE_FP16=True          # From YAML: actor_rollout_ref.embodied.enable_fp16
export EMBEDDING_MODEL_OFFLOAD=False  # From YAML: actor_rollout_ref.embodied.embedding_model_offload
export CENTER_CROP=True          # From YAML: actor_rollout_ref.embodied.center_crop
export NUM_IMAGES_IN_INPUT=1     # CRITICAL: Aligned to 1 (verl: 1, avoids 12-channel error)
export NUM_STEPS_WAIT=10         # NEW: Environment stabilization steps (verl: 10)
export GPU_MEMORY_UTILIZATION=0.9  # NEW: GPU memory utilization (verl: 0.9)

# --- Trainer Configuration (from YAML) ---
export SAVE_FREQ=4               # ALIGNED: Changed from 25 to 4 (verl: 4)
export TEST_FREQ=4               # ALIGNED: Changed from 5 to 4 (verl: 4)
export TOTAL_EPOCHS=1000         # From YAML: trainer.total_epochs
export MAX_CKPT_KEEP=5           # From YAML: trainer.max_actor_ckpt_to_keep
export VAL_BEFORE_TRAIN=True     # From YAML: trainer.val_before_train

# --- Multi-node distributed training ---
export N_GPUS_PER_NODE=8         # From YAML: trainer.n_gpus_per_node
export NNODES=1                  # From YAML: trainer.nnodes
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# --- Environment Variables ---
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GLOO_SOCKET_TIMEOUT=600

# --- Output Paths and Experiment Naming ---
timestamp=$(date +%Y%m%d_%H%M%S)
export CKPT_PATH=${BASE_CKPT_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_${NNODES}nodes
export PROJECT_NAME=siirl_embodied_${DATASET}
export EXPERIMENT_NAME=openvla_oft_grpo_fsdp_verl_aligned
export TENSORBOARD_DIR=${BASE_TENSORBOARD_PATH}/${MODEL_NAME}_${ALG}_${DATASET}/${timestamp}
export SIIRL_LOGGING_FILENAME=${MODEL_NAME}_${ALG}_${DATASET}_${timestamp}

# --- Define the Training Command (aligned with embodied_grpo_dag_trainer.yaml) ---
TRAINING_CMD=(
    python3 -m siirl.client.main_dag
    --config-name=embodied_grpo_dag_trainer
    
    # Data configuration (OPTIMIZED - removed duplicates)
    data.train_files=\$TRAIN_DATA_PATH
    data.val_files=\$TEST_DATA_PATH
    data.train_batch_size=\$TRAIN_BATCH_SIZE
    data.val_batch_size=\$VAL_BATCH_SIZE
    data.max_prompt_length=\$MAX_PROMPT_LENGTH
    data.max_response_length=\$MAX_RESPONSE_LENGTH
    
    # Algorithm configuration (standard GRPO - verl aligned)
    algorithm.workflow_type=embodied
    algorithm.adv_estimator=grpo
    algorithm.gamma=\$GAMMA
    algorithm.lam=\$LAM
    algorithm.norm_adv_by_std_in_grpo=True
    
    # Embodied sampling configuration (aligned with DAPO architecture)
    algorithm.embodied_sampling.filter_accuracy=\$FILTER_ACCURACY
    algorithm.embodied_sampling.accuracy_lower_bound=\$ACCURACY_LOWER_BOUND
    algorithm.embodied_sampling.accuracy_upper_bound=\$ACCURACY_UPPER_BOUND
    algorithm.embodied_sampling.filter_truncated=\$FILTER_TRUNCATED
    algorithm.embodied_sampling.oversample_factor=\$OVERSAMPLE_FACTOR
    
    # Model configuration
    actor_rollout_ref.model.path=\$MODEL_PATH
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    
    # Actor configuration
    actor_rollout_ref.actor.optim.lr=\$LEARNING_RATE
    actor_rollout_ref.actor.optim.weight_decay=\$WEIGHT_DECAY
    actor_rollout_ref.actor.ppo_mini_batch_size=\$PPO_MINI_BATCH_SIZE
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=\$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_epochs=\$PPO_EPOCHS
    actor_rollout_ref.actor.grad_clip=\$GRAD_CLIP
    actor_rollout_ref.actor.clip_ratio_high=\$CLIP_RATIO_HIGH
    actor_rollout_ref.actor.clip_ratio_low=\$CLIP_RATIO_LOW
    actor_rollout_ref.actor.entropy_coeff=\$ENTROPY_COEFF
    actor_rollout_ref.actor.shuffle=True
    
    # Actor FSDP configuration
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.grad_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    
    # Rollout configuration (aligned with verl - REUSES rollout.n instead of data.n_samples)
    actor_rollout_ref.rollout.name=hf
    actor_rollout_ref.rollout.n=\$ROLLOUT_N_SAMPLES
    actor_rollout_ref.rollout.temperature=\$TEMPERATURE
    actor_rollout_ref.rollout.do_sample=True
    actor_rollout_ref.rollout.response_length=512
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=\$LOG_PROB_MICRO_BATCH_SIZE
    actor_rollout_ref.rollout.micro_batch_size=\$ROLLOUT_MICRO_BATCH_SIZE
    actor_rollout_ref.rollout.gpu_memory_utilization=\$GPU_MEMORY_UTILIZATION
    
    # Embodied AI specific configuration
    actor_rollout_ref.embodied.embodied_type=\$MODEL_TYPE
    actor_rollout_ref.embodied.action_token_len=\$ACTION_TOKEN_LEN
    actor_rollout_ref.embodied.action_chunks_len=\$ACTION_CHUNKS_LEN
    actor_rollout_ref.embodied.video_embedding_model_path=\$VJEPA_MODEL_PATH
    actor_rollout_ref.embodied.embedding_img_size=\$IMG_SIZE
    actor_rollout_ref.embodied.embedding_enable_fp16=\$ENABLE_FP16
    actor_rollout_ref.embodied.embedding_model_offload=\$EMBEDDING_MODEL_OFFLOAD
    actor_rollout_ref.embodied.center_crop=\$CENTER_CROP
    actor_rollout_ref.embodied.num_images_in_input=\$NUM_IMAGES_IN_INPUT
    actor_rollout_ref.embodied.unnorm_key=\$UNNORM_KEY
    
    # Environment configuration (aligned with verl - REUSES env.env_name instead of data.task_suite_name)
    actor_rollout_ref.embodied.env.env_type=libero
    actor_rollout_ref.embodied.env.env_name=\$DATASET
    actor_rollout_ref.embodied.env.num_envs=\$NUM_ENVS
    actor_rollout_ref.embodied.env.max_steps=\$MAX_EPISODE_STEPS
    actor_rollout_ref.embodied.env.num_steps_wait=\$NUM_STEPS_WAIT
    actor_rollout_ref.embodied.env.num_trials_per_task=50
    actor_rollout_ref.embodied.env.model_family=openvla
    
    # Reference policy configuration
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=\$LOG_PROB_MICRO_BATCH_SIZE
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    
    # Critic configuration (GRPO doesn't use critic)
    critic.use_critic_model=False
    
    # Trainer configuration
    trainer.total_epochs=\$TOTAL_EPOCHS
    trainer.save_freq=\$SAVE_FREQ
    trainer.test_freq=\$TEST_FREQ
    trainer.max_actor_ckpt_to_keep=\$MAX_CKPT_KEEP
    trainer.logger=['console','tensorboard']
    trainer.project_name=\$PROJECT_NAME
    trainer.experiment_name=\$EXPERIMENT_NAME
    trainer.nnodes=\$NNODES
    trainer.n_gpus_per_node=\$N_GPUS_PER_NODE
    trainer.default_local_dir=\$CKPT_PATH
    trainer.resume_mode=auto
    trainer.val_before_train=\$VAL_BEFORE_TRAIN
)

# ===================================================================================
# ===                          EXECUTION LOGIC                                    ===
# ===================================================================================

set -x
eval "${TRAINING_CMD[@]}" 2>&1 | tee "$CKPT_PATH/logs/training_${timestamp}.log"

