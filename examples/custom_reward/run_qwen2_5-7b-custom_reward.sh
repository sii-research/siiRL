#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -e
set -o pipefail
# Print commands and their arguments as they are executed for easy debugging.
set -x

# --- Environment Setup ---
# bash /root/install_siirl.sh

# Generate a timestamp for unique directory/file names.
timestamp=$(date +"%Y%m%d_%H%M%S")

# Force stop any existing Ray cluster to ensure a clean start.
ray stop --force

# --- Path and Environment Variable Definitions ---

# Define environment variables for data, model, and checkpoint storage paths.
export DATASET=gsm8k
export ALG=grpo
export MODEL_NAME=qwen2.5-7b
export TRAIN_DATA_PATH=$HOME/data/datasets/$DATASET/train.parquet
export TEST_DATA_PATH=$HOME/data/datasets/$DATASET/test.parquet
export MODEL_PATH=$HOME/data/models/Qwen2.5-7B-Instruct
export CKPT_PATH=ckpts/${MODEL_NAME}_${ALG}_${DATASET}_hybrid_$PET_NNODES
export PROJECT_NAME=siirl_${DATASET}_${ALG}
export EXPERIMENT_NAME=siirl_${MODEL_NAME}_${ALG}_${DATASET}_experiment

# Environment variables for Gloo (used for distributed communication).
#export GLOO_SOCKET_IFNAME=bond0
export GLOO_SOCKET_TIMEOUT=600
export GLOO_TCP_TIMEOUT=600
export GLOO_LOG_LEVEL=DEBUG

# Define paths for TensorBoard and logging outputs.
export TENSORBOARD_DIR=${MODEL_NAME}_${ALG}_${DATASET}_hybrid_tensorboard/dlc_${PET_NNODES}_$timestamp
export SIIRL_LOGGING_FILENAME=${MODEL_NAME}_${ALG}_${DATASET}_hybrid_${PET_NNODES}_$timestamp

# --- Training Hyperparameters ---

export TRAIN_BATCH_SIZE_PER_NODE=1024
export PPO_MINI_BATCH_SIZE_PER_NODE=256
export PPO_MICRO_BATCH_SIZE_PER_GPU=16
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=4096
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.6
export ROLLOUT_TP=2
export ROLLOUT_N=8
export SAVE_FREQ=30
export TEST_FREQ=10
export TOTAL_EPOCHS=30
export MAX_CKPT_KEEP=5

# --- Multi-node (Multi-machine) distributed training environments ---

# Uncomment the following line and set the correct network interface if needed for distributed backend
# export GLOO_SOCKET_IFNAME=bond0  # Modify as needed

# --- Cluster Configuration (Usually no changes needed below) ---

# These variables are typically set by the cluster job scheduler (e.g., Slurm, DLC).
export N_GPUS_PER_NODE=8
export NNODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export VLLM_USE_V1=1

# Calculate the global batch sizes based on the number of nodes.
export TRAIN_BATCH_SIZE=$(($TRAIN_BATCH_SIZE_PER_NODE * $NNODES))
export PPO_MINI_BATCH_SIZE=$(($PPO_MINI_BATCH_SIZE_PER_NODE * $NNODES))

# Ray cluster connection settings.
export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export RAY_MASTER_ADDR=$MASTER_ADDR

# --- Ray Cluster Start Function (Robust for Large Scale) ---

start_ray_cluster() {
    # Set a generous timeout for workers waiting for the head node.
    local RAY_HEAD_WAIT_TIMEOUT=600 # 10 minutes

    # For stability in large clusters, explicitly set Ray to use the same network interface.
    export RAY_RAYLET_NODE_MANAGER_CONFIG_NIC_NAME=$INTERFACE_NAME
    export RAY_GCS_SERVER_CONFIG_NIC_NAME=$INTERFACE_NAME
    export RAY_RUNTIME_ENV_AGENT_CREATION_TIMEOUT_S=1200
    
    # Increase Ray GCS client connection timeout for stability.
    export RAY_GCS_RPC_CLIENT_CONNECT_TIMEOUT_S=120
    
    local ray_start_common_opts=(
        --num-gpus "$N_GPUS_PER_NODE"
        --object-store-memory 100000000000
        --memory 100000000000
    )

    # Multi-node environment
    if [ "$NNODES" -gt 1 ]; then
        # Head node logic (rank 0)
        if [ "$NODE_RANK" = "0" ]; then
            echo "INFO: Starting Ray head node on $(hostname)..."
            
            # The head's address is its own resolved IP
            export RAY_ADDRESS="$RAY_MASTER_ADDR:$RAY_MASTER_PORT"
            
            ray start --head \
                --port="$RAY_MASTER_PORT" \
                --dashboard-port="$RAY_DASHBOARD_PORT" \
                "${ray_start_common_opts[@]}" \
                --system-config='{"gcs_server_request_timeout_seconds": 60, "gcs_rpc_server_reconnect_timeout_s": 60}'
            
            echo "INFO: Ray head started. Waiting for services to become healthy at $RAY_ADDRESS..."
            
            local start_time=$(date +%s)
            while ! ray health-check --address "$RAY_ADDRESS" &>/dev/null; do
                local current_time=$(date +%s)
                local elapsed_time=$((current_time - start_time))
                if [ "$elapsed_time" -ge "$RAY_HEAD_WAIT_TIMEOUT" ]; then
                    echo "ERROR: Timed out after ${RAY_HEAD_WAIT_TIMEOUT}s waiting for the local head node services. Exiting." >&2
                    ray stop --force
                    exit 1
                fi
                echo "Head node services not healthy yet. Retrying in 5 seconds..."
                sleep 5
            done
            echo "INFO: Head node services are healthy."
        
        # Worker node logic (all other ranks)
        else
            # The address to connect to is the master node's address from the job scheduler
            local head_node_address="$MASTER_ADDR:$RAY_MASTER_PORT"
            echo "INFO: Worker node $(hostname) waiting for head node at $head_node_address..."
            
            local start_time=$(date +%s)
            # ROBUST CHECK: Use `ray health-check` to wait for the head.
            while ! ray health-check --address "$head_node_address" &>/dev/null; do
                local current_time=$(date +%s)
                local elapsed_time=$((current_time - start_time))

                if [ "$elapsed_time" -ge "$RAY_HEAD_WAIT_TIMEOUT" ]; then
                    echo "ERROR: Timed out after ${RAY_HEAD_WAIT_TIMEOUT}s waiting for the head node to be healthy. Exiting." >&2
                    exit 1
                fi
                
                echo "Head node at $head_node_address not healthy yet. Retrying in 5 seconds..."
                sleep 5
            done

            echo "INFO: Head node is healthy! Worker node $(hostname) is starting and connecting."
            ray start --address="$head_node_address" \
                "${ray_start_common_opts[@]}" \
                --block # Use --block to keep the script running until the worker is stopped.
        fi
    # Single-node setup
    else
        echo "INFO: Starting Ray in single-node mode..."
        ray start --head "${ray_start_common_opts[@]}"
    fi
}


# --- Training Launch Function ---

start_training() {
    if [ "$NODE_RANK" = "0" ]; then
        python3 -m siirl.client.main_dag \
            algorithm.adv_estimator=grpo \
            data.train_files=$TRAIN_DATA_PATH \
            data.val_files=$TEST_DATA_PATH \
            data.train_batch_size=$TRAIN_BATCH_SIZE \
            data.max_prompt_length=$MAX_PROMPT_LENGTH \
            data.max_response_length=$MAX_RESPONSE_LENGTH \
            data.filter_overlong_prompts=True \
            data.truncation='error' \
            data.shuffle=False \
            actor_rollout_ref.model.path=$MODEL_PATH \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.policy_drift_coeff=0.001 \
            actor_rollout_ref.actor.use_cpgd_loss=True \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.model.use_fused_kernels=False \
            actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.grad_clip=0.5 \
            actor_rollout_ref.actor.clip_ratio=0.2 \
            actor_rollout_ref.actor.kl_loss_coef=0.01 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
            actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
            actor_rollout_ref.rollout.max_model_len=8192 \
            actor_rollout_ref.rollout.enable_chunked_prefill=False \
            actor_rollout_ref.rollout.enforce_eager=False \
            actor_rollout_ref.rollout.free_cache_engine=False \
            actor_rollout_ref.rollout.n=$ROLLOUT_N \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            algorithm.kl_ctrl.kl_coef=0.001 \
            trainer.critic_warmup=0 \
            trainer.logger=['console','tensorboard']  \
            trainer.project_name=$PROJECT_NAME \
            trainer.experiment_name=$EXPERIMENT_NAME \
            trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
            trainer.nnodes=$NNODES \
            trainer.save_freq=$SAVE_FREQ \
            trainer.test_freq=$TEST_FREQ \
            trainer.total_epochs=$TOTAL_EPOCHS \
            trainer.resume_mode=auto \
            trainer.max_actor_ckpt_to_keep=$MAX_CKPT_KEEP \
            trainer.default_local_dir=$CKPT_PATH \
            trainer.val_before_train=True \
            custom_reward_function.path=$HOME/rl/rewardfunc_gsm8k.py \
            custom_reward_function.name=compute_score \
            reward_model.reward_manager=batch $@
    fi
}

# --- Main Execution Logic ---

# Start the Ray cluster (handles both single and multi-node cases).
start_ray_cluster

# This logic should only run on the head node (NODE_RANK=0) in a multi-node setup.
if [ "$NNODES" -gt 1 ] && [ "$NODE_RANK" = "0" ]; then
    echo "Head node is up. Waiting for all $NNODES nodes to join the cluster..."
    TIMEOUT_SECONDS=600

    # This command gets the list of nodes in JSON format and parses it with Python to count them.
    # 'ray list nodes' is the correct and modern way to get this information from the CLI.
    get_ready_nodes_cmd='ray list nodes --limit=5000 --format=json | python3 -c "import sys, json; print(len(json.load(sys.stdin)))"'
    start_time=$(date +%s)
    
    # Loop until the number of ready nodes equals the expected number of nodes.
    while true; do
        # --- Timeout Check ---
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        if [ "$elapsed_time" -ge "$TIMEOUT_SECONDS" ]; then
            echo "Error: Timeout! Waited for ${TIMEOUT_SECONDS} seconds, but not all nodes joined." >&2
            exit 1 # Exit with an error code
        fi

        # Execute the command to get the current count of ready nodes.
        # '2>/dev/null' suppresses errors if the ray client isn't ready yet, preventing script failure.
        ready_nodes=$(eval "$get_ready_nodes_cmd" 2>/dev/null) || ready_nodes=0

        if [ "$ready_nodes" -ge "$NNODES" ]; then
            break # All nodes have joined, exit the loop.
        fi

        echo "Waiting for all worker nodes to register... ($ready_nodes / $NNODES nodes ready)"
        sleep 2
    done

    echo "All $NNODES nodes have successfully joined the cluster."
fi

# --- Script Continuation ---
echo "Node initialization complete. Continuing with main task..."


start_training