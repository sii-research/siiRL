 #!/usr/bin/env bash
# ===================================================================================
# ===                       USER CONFIGURATION SECTION                            ===
# ===================================================================================

# --- Experiment and Model Definition ---
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH

export DATASET=deepscaler
export ALG=grpo
export MODEL_NAME=qwen2.5-7b
export VLLM_USE_V1=1

# --- Path Definitions ---
export HOME={your_home_path}
export TRAIN_DATA_PATH=$HOME/data/datasets/$DATASET/train.parquet
export TEST_DATA_PATH=$HOME/data/datasets/$DATASET/test.parquet
export MODEL_PATH=$HOME/data/models/Qwen2.5-7B-Instruct

# Base output paths
export BASE_CKPT_PATH=ckpts
export BASE_TENSORBOARD_PATH=tensorboard

# --- GLOO Configuration ---
export GLOO_SOCKET_IFNAME=enp91s0np0
export HCCL_SOCKET_IFNAME=enp91s0np0
export GLOO_SOCKET_TIMEOUT=600
export GLOO_TCP_TIMEOUT=600
export HCCL_CONNECT_TIMEOUT=7200
export GLOO_LOG_LEVEL=INFO

# --- Key Training Hyperparameters ---
export TRAIN_BATCH_SIZE_PER_NODE=1024
export PPO_MINI_BATCH_SIZE_PER_NODE=256
export PPO_MICRO_BATCH_SIZE_PER_GPU=4
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=2048
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.5
export ROLLOUT_TP=4
export ROLLOUT_N=5
export SAVE_FREQ=-1
export TEST_FREQ=5
export TOTAL_EPOCHS=300
export MAX_CKPT_KEEP=5

# --- Distributed Training & Infrastructure ---
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-16}
export NNODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}

# --- Output Paths and Experiment Naming ---
export CKPT_PATH=${BASE_CKPT_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_hybrid_${NNODES}nodes
export PROJECT_NAME=siirl_${DATASET}_${ALG}
export EXPERIMENT_NAME=siirl_${MODEL_NAME}_${ALG}_${DATASET}_experiment
export TENSORBOARD_DIR=${BASE_TENSORBOARD_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_hybrid_tensorboard/dlc_${NNODES}_$timestamp
export SIIRL_LOGGING_FILENAME=${MODEL_NAME}_${ALG}_${DATASET}_hybrid_${NNODES}_$timestamp

# --- Calculated Global Hyperparameters ---
export TRAIN_BATCH_SIZE=$(($TRAIN_BATCH_SIZE_PER_NODE * $NNODES))
export PPO_MINI_BATCH_SIZE=$(($PPO_MINI_BATCH_SIZE_PER_NODE * $NNODES))

# --- Define the Training Command and its Arguments ---
TRAINING_CMD=(
    python3 -m siirl.main_dag
    algorithm.adv_estimator=\$ALG
    data.train_files=\$TRAIN_DATA_PATH
    data.val_files=\$TEST_DATA_PATH
    data.train_batch_size=\$TRAIN_BATCH_SIZE
    data.max_prompt_length=\$MAX_PROMPT_LENGTH
    data.max_response_length=\$MAX_RESPONSE_LENGTH
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.auto_repeat=True
    actor_rollout_ref.model.path=\$MODEL_PATH
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=\$PPO_MINI_BATCH_SIZE
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=\$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.grad_clip=0.3 
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.fsdp_size=16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=\$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.tensor_model_parallel_size=\$ROLLOUT_TP
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=\$ROLLOUT_GPU_MEMORY_UTILIZATION
    actor_rollout_ref.rollout.n=\$ROLLOUT_N
    actor_rollout_ref.rollout.prompt_length=\$MAX_PROMPT_LENGTH  
    actor_rollout_ref.rollout.response_length=\$MAX_RESPONSE_LENGTH
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=\$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.logger=['console','tensorboard']
    trainer.project_name=\$PROJECT_NAME
    trainer.experiment_name=\$EXPERIMENT_NAME
    trainer.n_gpus_per_node=\$N_GPUS_PER_NODE
    trainer.nnodes=\$NNODES
    trainer.save_freq=\$SAVE_FREQ
    trainer.test_freq=\$TEST_FREQ
    trainer.total_epochs=\$TOTAL_EPOCHS
    trainer.resume_mode=auto
    trainer.max_actor_ckpt_to_keep=\$MAX_CKPT_KEEP
    trainer.default_local_dir=\$CKPT_PATH
    trainer.val_before_train=True
    trainer.device=npu
)

# ===================================================================================
# ===                  MAIN EXECUTION LOGIC & INFRASTRUCTURE                      ===
# ===================================================================================

# --- Boilerplate Setup ---
set -e
set -o pipefail
set -x

# --- Infrastructure & Boilerplate Functions ---
start_ray_cluster() {
    local RAY_HEAD_WAIT_TIMEOUT=600
    export RAY_RAYLET_NODE_MANAGER_CONFIG_NIC_NAME=${INTERFACE_NAME}
    export RAY_GCS_SERVER_CONFIG_NIC_NAME=${INTERFACE_NAME}
    export RAY_RUNTIME_ENV_AGENT_CREATION_TIMEOUT_S=1200
    export RAY_GCS_RPC_CLIENT_CONNECT_TIMEOUT_S=120

    local ray_start_common_opts=(
        --num-gpus "$N_GPUS_PER_NODE"
        --object-store-memory 100000000000
        --memory 100000000000
    )

    if [ "$NNODES" -gt 1 ]; then
        if [ "$NODE_RANK" = "0" ]; then
            echo "INFO: Starting Ray head node on $(hostname)..."
            export RAY_ADDRESS="$RAY_MASTER_ADDR:$RAY_MASTER_PORT"
            ray start --head --port="$RAY_MASTER_PORT" --dashboard-port="$RAY_DASHBOARD_PORT" "${ray_start_common_opts[@]}" --system-config='{"gcs_server_request_timeout_seconds": 60, "gcs_rpc_server_reconnect_timeout_s": 60}'
            local start_time=$(date +%s)
            while ! ray health-check --address "$RAY_ADDRESS" &>/dev/null; do
                if [ "$(( $(date +%s) - start_time ))" -ge "$RAY_HEAD_WAIT_TIMEOUT" ]; then echo "ERROR: Timed out waiting for head node. Exiting." >&2; ray stop --force; exit 1; fi
                echo "Head node not healthy yet. Retrying in 5s..."
                sleep 5
            done
            echo "INFO: Head node is healthy."
        else
            local head_node_address="$MASTER_ADDR:$RAY_MASTER_PORT"
            echo "INFO: Worker node $(hostname) waiting for head at $head_node_address..."
            local start_time=$(date +%s)
            while ! ray health-check --address "$head_node_address" &>/dev/null; do
                if [ "$(( $(date +%s) - start_time ))" -ge "$RAY_HEAD_WAIT_TIMEOUT" ]; then echo "ERROR: Timed out waiting for head. Exiting." >&2; exit 1; fi
                echo "Head not healthy yet. Retrying in 5s..."
                sleep 5
            done
            echo "INFO: Head is healthy. Worker starting..."
            ray start --address="$head_node_address" "${ray_start_common_opts[@]}"
        fi
    else
        echo "INFO: Starting Ray in single-node mode..."
        ray start --head "${ray_start_common_opts[@]}"
    fi
}

# --- Main Execution Function ---
main() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    ray stop --force
    echo "Cleaning up residual distributed processes..."
    pkill -f ray || true
    pkill -f siirl.main_dag || true
    pkill -f torchrun || true
    pkill -f vllm || true
    pkill -f hccl || true
    for port in ${MASTER_PORT:-29500} ${RAY_MASTER_PORT:-6379}; do
        for pid in $(lsof -ti :$port); do
            kill -9 $pid || true
        done
    done
    sleep 3
    echo "Cleanup finished."

    export GLOO_SOCKET_TIMEOUT=600
    export GLOO_TCP_TIMEOUT=600
    export GLOO_LOG_LEVEL=DEBUG
    export RAY_MASTER_PORT=${RAY_MASTER_PORT:-6379}
    export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
    export RAY_MASTER_ADDR=$MASTER_ADDR
    
    start_ray_cluster

    if [ "$NNODES" -gt 1 ] && [ "$NODE_RANK" = "0" ]; then
        echo "Waiting for all $NNODES nodes to join..."
        local TIMEOUT=600; local start_time=$(date +%s)
        while true; do
            if [ "$(( $(date +%s) - start_time ))" -ge "$TIMEOUT" ]; then echo "Error: Timeout waiting for nodes." >&2; exit 1; fi
            local ready_nodes=$(ray list nodes --format=json | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")
            if [ "$ready_nodes" -ge "$NNODES" ]; then break; fi
            echo "Waiting... ($ready_nodes / $NNODES nodes ready)"
            sleep 5
        done
        echo "All $NNODES nodes have joined."
    fi

    if [ "$NODE_RANK" = "0" ]; then
        echo "INFO [RANK 0]: Starting main training command."
        eval "${TRAINING_CMD[@]}" "$@"
        echo "INFO [RANK 0]: Training finished."
        sleep 30; ray stop --force >/dev/null 2>&1
    elif [ "$NNODES" -gt 1 ]; then
        local head_node_address="$MASTER_ADDR:$RAY_MASTER_PORT"
        echo "INFO [RANK $NODE_RANK]: Worker active. Monitoring head node at $head_node_address."
        while ray health-check --address "$head_node_address" &>/dev/null; do sleep 15; done
        echo "INFO [RANK $NODE_RANK]: Head node down. Exiting."
    fi

    echo "INFO: Script finished on rank $NODE_RANK."
}

# --- Script Entrypoint ---
main "$@"
