 #!/usr/bin/env bash
# ===================================================================================
# ===                       USER CONFIGURATION SECTION                            ===
# ===================================================================================

# --- Experiment and Model Definition ---
export DATASET=deepscaler #deepscaler
export ALG=grpo
export MODEL_NAME=qwen2.5-32b

# --- Path Definitions ---
export HOME=/workspace/

export HOME={your_home_path}
export TRAIN_DATA_PATH=$HOME/datasets/$DATASET/train.parquet
export TEST_DATA_PATH=$HOME/datasets/$DATASET/test.parquet
export MODEL_PATH=$HOME/models/Qwen2.5-32B-Instruct

# Base output paths
export BASE_CKPT_PATH=$HOME/siirl_ckpts2
export BASE_TENSORBOARD_PATH=tensorboard

# --- Key Training Hyperparameters ---
export TRAIN_BATCH_SIZE_PER_NODE=512 #1024  
export PPO_MINI_BATCH_SIZE_PER_NODE=128
export PPO_MICRO_BATCH_SIZE_PER_GPU=4
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=2048
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export ROLLOUT_TP=4
export ROLLOUT_N=4
export SAVE_FREQ=-1
export TEST_FREQ=10
export TOTAL_EPOCHS=30
export MAX_CKPT_KEEP=5

# --- Multi-node (Multi-machine) distributed training environments ---

# Uncomment the following line and set the correct network interface if needed for distributed backend
# export GLOO_SOCKET_IFNAME=bond0  # Modify as needed

# --- Distributed Training & Infrastructure ---
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
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

# mx gpu env
export MACA_PATH=/opt/maca
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export MACA_CLANG_PATH=$MACA_PATH/mxgpu_llvm/bin
export PATH=${CUDA_PATH}/bin:${MACA_CLANG_PATH}:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/tools/cu-bridge/lib/:${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export PYTORCH_ENABLE_SAME_RAND_A100=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export SET_DEVICE_NUMA_PREFERRED=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1
export MCPYTORCH_DISABLE_PRINT=1
export MAX_JOBS=20
export VLLM_USE_V1=0
unset PYTORCH_CUDA_ALLOC_CONF
export MCCL_ENABLE_FC=0
# export MACA_PRIORITY_QUEUE_POLICY=0xa11
# export MCCL_PCIE_BUFFER_MODE=0
# export MCCL_NET_GDR_LEVEL=SYS

# export MCCL_USE_FILE_TUNING=0
# export MCCL_ALGO=Ring
# export MCCL_DISABLE_MULTI_NODE_FABRIC=1
# export MCCL_DISABLE_OPTIC_LINK_=1

export MCCL_MAX_NCHANNELS=8
export PYTHONUNBUFFERED=1
export MCCL_IB_HCA=mlx5
export MCCL_SOCKET_IFNAME=ens1
export GLOO_SOCKET_IFNAME=ens1
export SOCKET_NIC=ens1

# --- Define the Training Command and its Arguments ---
TRAINING_CMD=(
    python3 -m siirl.client.main_dag
    algorithm.adv_estimator=\$ALG
    data.train_files=\$TRAIN_DATA_PATH
    data.val_files=\$TEST_DATA_PATH
    data.train_batch_size=\$TRAIN_BATCH_SIZE
    data.max_prompt_length=\$MAX_PROMPT_LENGTH
    data.max_response_length=\$MAX_RESPONSE_LENGTH
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.auto_repeat=True
    data.shuffle=False
    actor_rollout_ref.model.path=\$MODEL_PATH
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.actor.ppo_mini_batch_size=\$PPO_MINI_BATCH_SIZE
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$(($PPO_MICRO_BATCH_SIZE_PER_GPU / 1))
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.grad_clip=0.5
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.fsdp_size=64
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=\$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.tensor_model_parallel_size=\$ROLLOUT_TP
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=\$ROLLOUT_GPU_MEMORY_UTILIZATION
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=\$ROLLOUT_N
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$(($PPO_MICRO_BATCH_SIZE_PER_GPU * 4))
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

    # export VLLM_USE_V1=0
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
            # local ready_nodes=$(ray list nodes --format=json | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")
            local ready_nodes=$(ray status | grep "node_" | wc -l)
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

