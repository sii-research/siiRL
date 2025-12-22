Embodied SRPO Training
======================

Introduction
------------

This guide explains how to perform Embodied AI training using the SRPO algorithm with OpenVLA-oft models on tasks such as LIBERO. Embodied AI training involves an agent interacting with an environment, where the rewards are often based on task success.

This example demonstrates how to perform RL training on an `OpenVLA-oft-7B` model using the SRPO algorithm on the `libero_long` benchmark.

Step 1: Prepare the Environment
-------------------------------

You should use the provided Docker image for Embodied AI training, which contains all necessary dependencies including EGL support for rendering.

**Docker Image**: ``siiai/siirl-vla:libero-egl-cu12.6`` (Available at `Docker Hub <https://hub.docker.com/r/siiai/siirl-vla>`_)

Ensure you have the necessary environment variables set. This includes the path to the `siiRL` repository and any other dependencies.

.. code:: bash

   export SIIRL_DIR="/path/to/siiRL"
   export PYTHONPATH="$SIIRL_DIR:/path/to/LIBERO:/path/to/vjepa:$PYTHONPATH"

Step 2: Prepare the Models
--------------------------

You need the following models:

1.  **SFT Model**: A Supervised Fine-Tuned (SFT) OpenVLA-oft model. You should select the model that corresponds to your specific task. For example, if you are training on `libero_long`, you should use the `Sylvest/OpenVLA-AC-PD-1traj-libero-long` model.

    Here are the recommended Hugging Face models from the `Sylvest collection <https://huggingface.co/collections/Sylvest/srpo>`_:

    - `Sylvest/OpenVLA-AC-PD-1traj-libero-object` (for `libero_object`)
    - `Sylvest/OpenVLA-AC-PD-1traj-libero-spatial` (for `libero_spatial`)
    - `Sylvest/OpenVLA-AC-PD-1traj-libero-goal` (for `libero_goal`)
    - `Sylvest/OpenVLA-AC-PD-1traj-libero-long` (for `libero_long`)

2.  **Visual Encoder**: A visual encoder model V-JEPA is **required** for processing visual observations.

    - Download the V-JEPA 2 model from Hugging Face: `Sylvest/vjepa2-vit-g <https://huggingface.co/Sylvest/vjepa2-vit-g>`_
    
      .. code:: bash

         huggingface-cli download Sylvest/vjepa2-vit-g --local-dir $HOME/models/vjepa2

Set the paths to these resources in your environment or script:

.. code:: bash

   export MODEL_PATH=$HOME/models/Sylvest/OpenVLA-AC-PD-1traj-libero-long
   export VJEPA_MODEL_PATH=$HOME/models/vjepa2/vitg-384.pt

.. note::
   
   You do not need to manually prepare a dataset file. ``siiRL`` will automatically generate the task manifest (Parquet files) based on the environment configuration and save them to the path specified in ``TRAIN_DATA_PATH`` and ``TEST_DATA_PATH``.

Step 3: Configure and Run the Training Script
---------------------------------------------

Embodied AI training requires specific configurations to handle the environment interaction and action spaces.

Key Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Embodied Specifics**:

-   ``actor_rollout_ref.embodied.embodied_type``: The model type (e.g., ``openvla-oft``).
-   ``actor_rollout_ref.embodied.action_token_len``: The dimensionality of the action space (e.g., 7 for xyz + quaternion + gripper).
-   ``actor_rollout_ref.embodied.action_chunks_len``: The number of action steps predicted in one forward pass.
-   ``actor_rollout_ref.embodied.video_embedding_model_path``: Path to the V-JEPA 2 video embedding model (e.g., ``$VJEPA_MODEL_PATH``).

**Environment Configuration**:

-   ``actor_rollout_ref.embodied.env.env_type``: The environment library (e.g., ``libero``).
-   ``actor_rollout_ref.embodied.env.env_name``: The specific task suite name (e.g., ``libero_long``).
-   ``actor_rollout_ref.embodied.env.num_envs``: Number of parallel environments per rollout worker. Default is 16 environments per GPU, and it is not recommended to exceed 16.
-   ``actor_rollout_ref.embodied.env.max_steps``: Maximum steps per episode.

**Algorithm Adjustments**:

-   ``algorithm.embodied_sampling.filter_accuracy``: Enable filtering of prompts based on estimated success rate.
-   ``algorithm.embodied_sampling.accuracy_lower_bound``: Lower threshold for filtering (e.g., 0.1).
-   ``algorithm.embodied_sampling.accuracy_upper_bound``: Upper threshold for filtering (e.g., 0.9).

Complete Training Script
~~~~~~~~~~~~~~~~~~~~~~~~

Below is an example script `run_embodied_srpo.sh` to run SRPO training on `libero_long`.

**Note**: The siiRL repository provides ready-to-use training scripts for all four LIBERO tasks in the `examples/embodied_srpo_trainer/` directory:

-   ``run_openvla_oft_libero_long.sh``
-   ``run_openvla_oft_libero_goal.sh``
-   ``run_openvla_oft_libero_object.sh``
-   ``run_openvla_oft_libero_spatial.sh``

To train on a specific task, modify the following paths in the script to match your actual environment:

-   ``SIIRL_DIR``: Path to the siiRL repository
-   ``VJEPA2_DIR``: Path to the V-JEPA2 repository (for ``PYTHONPATH``)
-   ``HOME_PATH``: Your home directory or base path for models and data
-   ``MODEL_PATH``: Path to the corresponding SFT model for the task
-   ``VJEPA_MODEL_PATH``: Path to the V-JEPA 2 model weights file

**Note**: LIBERO is pre-installed in the Docker image at ``/root/LIBERO/`` and does not need to be modified.

.. code-block:: bash

    #!/usr/bin/env bash
    # ===================================================================================
    # ===    Embodied AI SRPO Training with OpenVLA-OFT on LIBERO-LONG               ===
    # ===================================================================================
    # 

    set -e

    # --- Environment Setup (Critical for siiRL) ---
    export SIIRL_DIR="${SIIRL_DIR:-your_siirl_path}"
    export PYTHONPATH="$SIIRL_DIR:/root/LIBERO/:${VJEPA2_DIR:-your_vjepa2_path}:$PYTHONPATH"

    # --- Experiment and Model Definition ---
    export DATASET=libero_long
    export ALG=srpo
    export MODEL_NAME=openvla-oft-7b
    export MODEL_TYPE=openvla-oft

    # --- Path Definitions (USER PROVIDED) ---
    export HOME_PATH=${HOME_PATH:your_home_path}
    export TRAIN_DATA_PATH=$HOME_PATH/data/train.parquet # generated automatically
    export TEST_DATA_PATH=$HOME_PATH/data/validate.parquet # generated automatically
    export MODEL_PATH=$HOME_PATH/models/Sylvest/OpenVLA-AC-PD-1traj-libero-long
    export VJEPA_MODEL_PATH=$HOME_PATH/models/vjepa2/vitg-384.pt

    # Base output paths
    export BASE_CKPT_PATH=ckpts
    export BASE_TENSORBOARD_PATH=tensorboard

    # --- Embodied AI Specific Parameters ---
    export ACTION_TOKEN_LEN=7        # 7 dimensions: xyz (3), quaternion (3), gripper (1)
    export ACTION_CHUNKS_LEN=8       # OpenVLA-OFT uses 8-step action chunks
    export NUM_ENVS=16               # actor_rollout_ref.embodied.env.num_envs
    export MAX_EPISODE_STEPS=512     # actor_rollout_ref.embodied.env.max_steps

    # --- Data and Sampling Parameters ---
    export VAL_BATCH_SIZE=496                      # Validation batch size
    export MAX_PROMPT_LENGTH=256                   
    export MAX_RESPONSE_LENGTH=128                 

    # --- Embodied Sampling Parameters ---
    export FILTER_ACCURACY=True                    # Enable accuracy-based filtering
    export ACCURACY_LOWER_BOUND=0.1                # Only keep prompts with success rate >= 0.1
    export ACCURACY_UPPER_BOUND=0.9                # Only keep prompts with success rate <= 0.9
    export FILTER_TRUNCATED=False                  # Filter truncated episodes (uses env.max_steps)
    export OVERSAMPLE_FACTOR=1                     # Oversample factor for filtering

    # --- Training Hyperparameters ---
    export TRAIN_BATCH_SIZE=64       # data.train_batch_size
    export PPO_MINI_BATCH_SIZE=4     # actor_rollout_ref.actor.ppo_mini_batch_size
                                     # Note: actual ppo_mini_batch_size = PPO_MINI_BATCH_SIZE * ROLLOUT_N_SAMPLES
    export ROLLOUT_N_SAMPLES=8       # REUSED: Number of samples per prompt
    export PPO_EPOCHS=1              # actor_rollout_ref.actor.ppo_epochs

    # Algorithm parameters
    export LEARNING_RATE=5e-6        
    export WEIGHT_DECAY=0.0          # actor_rollout_ref.actor.optim.weight_decay
    export CLIP_RATIO_HIGH=0.28      # actor_rollout_ref.actor.clip_ratio_high
    export CLIP_RATIO_LOW=0.2        # actor_rollout_ref.actor.clip_ratio_low
    export ENTROPY_COEFF=0.0         
    export TEMPERATURE=1.6          
    export GAMMA=1.0                 
    export LAM=1.0                   
    export GRAD_CLIP=1.0            

    # --- Image/Video Processing ---
    export IMG_SIZE=384              # actor_rollout_ref.embodied.img_size
    export ENABLE_FP16=True          # actor_rollout_ref.embodied.enable_fp16
    export EMBEDDING_MODEL_OFFLOAD=False  # actor_rollout_ref.embodied.embedding_model_offload
    export CENTER_CROP=True          # actor_rollout_ref.embodied.center_crop
    export NUM_IMAGES_IN_INPUT=1     
    export NUM_STEPS_WAIT=10           # Environment stabilization steps

    # --- Trainer Configuration ---
    export SAVE_FREQ=5              
    export TEST_FREQ=5              
    export TOTAL_EPOCHS=1000         # trainer.total_epochs
    export MAX_CKPT_KEEP=5           # trainer.max_actor_ckpt_to_keep
    export VAL_BEFORE_TRAIN=True     # trainer.val_before_train

    # --- Multi-node distributed training ---
    export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
    export NNODES=${PET_NNODES:-1}
    export NODE_RANK=${PET_NODE_RANK:-0}
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
    export EXPERIMENT_NAME=openvla_oft_srpo_fsdp
    export TENSORBOARD_DIR=${BASE_TENSORBOARD_PATH}/${MODEL_NAME}_${ALG}_${DATASET}/${timestamp}
    export SIIRL_LOGGING_FILENAME=${MODEL_NAME}_${ALG}_${DATASET}_${timestamp}

    # --- Define the Training Command ---
    TRAINING_CMD=(
        python3 -m siirl.client.main_dag
        --config-name=embodied_srpo_trainer
        
        # Data configuration
        data.train_files=$TRAIN_DATA_PATH
        data.val_files=$TEST_DATA_PATH
        data.train_batch_size=$TRAIN_BATCH_SIZE
        data.val_batch_size=$VAL_BATCH_SIZE
        data.max_prompt_length=$MAX_PROMPT_LENGTH
        data.max_response_length=$MAX_RESPONSE_LENGTH
        
        # Algorithm configuration
        algorithm.workflow_type=embodied
        algorithm.adv_estimator=grpo
        algorithm.gamma=$GAMMA
        algorithm.lam=$LAM
        algorithm.norm_adv_by_std_in_grpo=True
        
        # Embodied sampling configuration (aligned with DAPO architecture)
        algorithm.embodied_sampling.filter_accuracy=$FILTER_ACCURACY
        algorithm.embodied_sampling.accuracy_lower_bound=$ACCURACY_LOWER_BOUND
        algorithm.embodied_sampling.accuracy_upper_bound=$ACCURACY_UPPER_BOUND
        algorithm.embodied_sampling.filter_truncated=$FILTER_TRUNCATED
        algorithm.embodied_sampling.oversample_factor=$OVERSAMPLE_FACTOR
        
        # Model configuration
        actor_rollout_ref.model.path=$MODEL_PATH
        actor_rollout_ref.model.enable_gradient_checkpointing=True
        
        # Actor configuration
        actor_rollout_ref.actor.optim.lr=$LEARNING_RATE
        actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
        actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS
        actor_rollout_ref.actor.grad_clip=$GRAD_CLIP
        actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH
        actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW
        actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF
        actor_rollout_ref.actor.shuffle=True
        
        # Actor FSDP configuration
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.grad_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
        
        # Rollout configuration
        actor_rollout_ref.rollout.name=hf
        actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLES
        actor_rollout_ref.rollout.temperature=$TEMPERATURE
        actor_rollout_ref.rollout.do_sample=True
        actor_rollout_ref.rollout.response_length=512
        
        # Embodied AI specific configuration
        actor_rollout_ref.embodied.embodied_type=$MODEL_TYPE
        actor_rollout_ref.embodied.action_token_len=$ACTION_TOKEN_LEN
        actor_rollout_ref.embodied.action_chunks_len=$ACTION_CHUNKS_LEN
        actor_rollout_ref.embodied.video_embedding_model_path=$VJEPA_MODEL_PATH
        actor_rollout_ref.embodied.embedding_img_size=$IMG_SIZE
        actor_rollout_ref.embodied.embedding_enable_fp16=$ENABLE_FP16
        actor_rollout_ref.embodied.embedding_model_offload=$EMBEDDING_MODEL_OFFLOAD
        actor_rollout_ref.embodied.center_crop=$CENTER_CROP
        actor_rollout_ref.embodied.num_images_in_input=$NUM_IMAGES_IN_INPUT
        actor_rollout_ref.embodied.unnorm_key=$DATASET
        
        # Environment configuration
        actor_rollout_ref.embodied.env.env_type=libero
        actor_rollout_ref.embodied.env.env_name=$DATASET
        actor_rollout_ref.embodied.env.num_envs=$NUM_ENVS
        actor_rollout_ref.embodied.env.max_steps=$MAX_EPISODE_STEPS
        actor_rollout_ref.embodied.env.num_steps_wait=$NUM_STEPS_WAIT
        actor_rollout_ref.embodied.env.num_trials_per_task=50
        actor_rollout_ref.embodied.env.model_family=openvla
        
        # Critic configuration (SRPO doesn't use critic)
        critic.use_critic_model=False
        
        # Trainer configuration
        trainer.total_epochs=$TOTAL_EPOCHS
        trainer.save_freq=$SAVE_FREQ
        trainer.test_freq=$TEST_FREQ
        trainer.max_actor_ckpt_to_keep=$MAX_CKPT_KEEP
        trainer.logger=['console','tensorboard']
        trainer.project_name=$PROJECT_NAME
        trainer.experiment_name=$EXPERIMENT_NAME
        trainer.nnodes=$NNODES
        trainer.n_gpus_per_node=$N_GPUS_PER_NODE
        trainer.default_local_dir=$CKPT_PATH
        trainer.resume_mode=auto
        trainer.val_before_train=$VAL_BEFORE_TRAIN
    )

    # ===================================================================================
    # ===                          EXECUTION LOGIC                                    ===
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

        export VLLM_USE_V1=1
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

Step 4: Checking the Results
----------------------------

1.  **Logs**: Monitor the console output for training progress and environment interaction stats.
2.  **TensorBoard**: Use TensorBoard to visualize rewards, success rates, and other metrics.

    .. code:: bash

       tensorboard --logdir ./tensorboard

3.  **Checkpoints**: Trained models are saved in the ``ckpts`` directory.

