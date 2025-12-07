Megatron-LM Training Backend
============================================

Introduction
------------

This guide explains how to use the Megatron-LM backend in siiRL for RL training. Megatron-LM is a powerful library for training very large transformer models, and integrating it as a backend allows for efficient 5D parallelism (DP/TP/EP/PP/CP).

This example demonstrates how to fine-tune a `Qwen3-8B` model using the GRPO algorithm with the Megatron-LM as training backend.

Step 1: Prepare the Dataset
---------------------------

First, ensure your dataset is in the required Parquet format. If you are using one of the example datasets like `gsm8k` or `deepscaler`, you can use the provided preprocessing scripts. For example, for `deepscaler`:

.. code:: bash

   cd examples/data_preprocess
   python3 deepscaler.py --local_dir ~/data/deepscaler

This will download and process the dataset, saving `train.parquet` and `test.parquet` in the specified directory.

Step 2: Download the Pre-trained Model
--------------------------------------

You need a base model to start training. For this example, we'll use `Qwen3-8B`. Download it from Hugging Face or ModelScope to a local directory.

.. code:: bash

   # For Hugging Face
   huggingface-cli download Qwen/Qwen3-8B-Instruct --local-dir ~/data/models/Qwen3-8B --local-dir-use-symlinks False
   
   # For ModelScope
   modelscope download Qwen/Qwen3-8B-Instruct --local_dir ~/data/models/Qwen3-8B

Step 3: Configure and Run the Training Script
---------------------------------------------

To use the Megatron-LM backend, you need to modify the training configuration in your run script.

Key Configuration Changes
~~~~~~~~~~~~~~~~~~~~~~~~~

The main change is to set the training strategy to `megatron` and configure its parallelism parameters.

1.  **Set the Strategy**: e.g., in the `TRAINING_CMD` array, set `actor_rollout_ref.actor.strategy=megatron`.
2.  **Configure Parallelism**: Add Megatron-specific settings for 5D parallelism. For a 8B model on a single node with 8 GPUs, you might use 2-way tensor parallelism and 4-way pipeline parallelism, with sequence parallelism enabled.

    .. code-block:: text

        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4
        actor_rollout_ref.actor.megatron.context_parallel_size=1
        actor_rollout_ref.actor.megatron.sequence_parallel=True

3.  **Configure Distributed Optimizer**: Add Megatron-specific settings for distributed optimizer. This allows for memory efficient training with ZeRO-1 optimization and is recommended for large models.

    .. code-block:: text

        actor_rollout_ref.actor.megatron.use_distributed_optimizer=True

4.  **Configure Offloading**: Add Megatron-specific settings for parameter, gradient, and optimizer offload. This allows for parameter, gradient, and optimizer offloading to CPU to save GPU memory.

    .. code-block:: text

        actor_rollout_ref.actor.megatron.param_offload=True
        actor_rollout_ref.actor.megatron.grad_offload=True
        actor_rollout_ref.actor.megatron.optimizer_offload=True

Complete Training Script
~~~~~~~~~~~~~~~~~~~~~~~~

Below is a complete example script, `run_qwen3-8b-megatron.sh`, which is adapted from the standard GRPO script to use the Megatron backend. You will need to create this script yourself or adapt an existing one.

.. code-block:: bash

    #!/usr/bin/env bash
    # ===================================================================================
    # ===                       USER CONFIGURATION SECTION                            ===
    # ===================================================================================

    # --- For debugging
    export HYDRA_FULL_ERROR=1
    export SIIRL_LOG_VERBOSITY=INFO

    # --- Experiment and Model Definition ---
    export DATASET=deepscaler
    export ALG=grpo
    export MODEL_NAME=qwen3-8b

    # --- Path Definitions ---
    export HOME=${HOME:-"/root"} # Set your home path
    export TRAIN_DATA_PATH=$HOME/data/datasets/$DATASET/train.parquet
    export TEST_DATA_PATH=$HOME/data/datasets/$DATASET/test.parquet
    export MODEL_PATH=$HOME/data/models/Qwen3-8B

    # Base output paths
    export BASE_CKPT_PATH=$HOME/ckpts
    export BASE_TENSORBOARD_PATH=$HOME/tensorboard

    # --- Key Training Hyperparameters ---
    export TRAIN_BATCH_SIZE_PER_NODE=128
    export PPO_MINI_BATCH_SIZE_PER_NODE=16
    export PPO_MICRO_BATCH_SIZE_PER_GPU=8
    export MAX_PROMPT_LENGTH=1024
    export MAX_RESPONSE_LENGTH=2048
    export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
    export ROLLOUT_N=8
    export SAVE_FREQ=30
    export TEST_FREQ=10
    export TOTAL_EPOCHS=30
    export MAX_CKPT_KEEP=5

    # ---- Megatron Parallelism Configuration ----
    export ACTOR_REF_TP=2
    export ACTOR_REF_PP=4
    export ACTOR_REF_CP=1
    export ACTOR_REF_SP=True

    # --- Distributed Training & Infrastructure ---
    export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
    export NNODES=${PET_NNODES:-1}
    export NODE_RANK=${PET_NODE_RANK:-0}
    export MASTER_ADDR=${MASTER_ADDR:-localhost}

    # --- Output Paths and Experiment Naming ---
    timestamp=$(date +"%Y%m%d_%H%M%S")
    export CKPT_PATH=${BASE_CKPT_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_megatron_${NNODES}nodes
    export PROJECT_NAME=siirl_${DATASET}_${ALG}
    export EXPERIMENT_NAME=siirl_${MODEL_NAME}_${ALG}_${DATASET}_megatron_experiment
    export TENSORBOARD_DIR=${BASE_TENSORBOARD_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_megatron_tensorboard/dlc_${NNODES}_$timestamp
    export SIIRL_LOGGING_FILENAME=${MODEL_NAME}_${ALG}_${DATASET}_megatron_${NNODES}_$timestamp

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
        actor_rollout_ref.model.path=\$MODEL_PATH
        actor_rollout_ref.model.enable_gradient_checkpointing=True
        
        # --- Megatron Backend Configuration ---
        actor_rollout_ref.actor.strategy=megatron
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=\$ACTOR_REF_TP
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=\$ACTOR_REF_PP
        actor_rollout_ref.actor.megatron.context_parallel_size=\$ACTOR_REF_CP
        actor_rollout_ref.actor.megatron.sequence_parallel=\$ACTOR_REF_SP
        actor_rollout_ref.actor.megatron.use_distributed_optimizer=True
        actor_rollout_ref.actor.megatron.param_dtype=bfloat16
        actor_rollout_ref.actor.megatron.param_offload=False
        
        # --- PPO & Other Hyperparameters ---
        actor_rollout_ref.actor.optim.lr=1e-6
        actor_rollout_ref.actor.ppo_mini_batch_size=\$PPO_MINI_BATCH_SIZE
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=\$PPO_MICRO_BATCH_SIZE_PER_GPU
        actor_rollout_ref.actor.grad_clip=1.0
        
        # --- Rollout (vLLM) Configuration ---
        actor_rollout_ref.rollout.tensor_model_parallel_size=\$ACTOR_REF_TP
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.gpu_memory_utilization=\$ROLLOUT_GPU_MEMORY_UTILIZATION
        actor_rollout_ref.rollout.n=\$ROLLOUT_N
        actor_rollout_ref.rollout.prompt_length=\$MAX_PROMPT_LENGTH  
        actor_rollout_ref.rollout.response_length=\$MAX_RESPONSE_LENGTH
        
        # --- Trainer Configuration ---
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

Step 4: Checking the Results
----------------------------

During training, you can monitor the progress through several means:

1.  **Console Logs**: The console will output detailed logs. Look for initialization messages from the Megatron backend to confirm it's being used. You should see logs pertaining to the setup of 5D parallelism.

2.  **TensorBoard**: If you enabled the `tensorboard` logger, you can monitor training metrics in real-time.
    
    .. code:: bash

       tensorboard --logdir $HOME/tensorboard

    Navigate to the TensorBoard URL in your browser to view metrics such as reward, KL divergence, and loss curves.

3.  **Checkpoints**: Checkpoints will be saved in the directory specified by `CKPT_PATH`. You can use these to resume training or for inference later.
