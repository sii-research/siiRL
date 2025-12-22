.. _quickstart:

=========================================================
Quickstart: GRPO training on GSM8K dataset
=========================================================

Post-train a LLM using GSM8K dataset.

Introduction
------------

.. _hf_dataset_gsm8k: https://huggingface.co/datasets/gsm8k

In this example, we train an LLM to tackle the `GSM8k <hf_dataset_gsm8k>`_ task with function-based rewards. 

Prerequisite:

- the latest version of ``siiRL`` and its dependencies installed following the installation guide. Using the docker image is recommended.

- a GPU with at least 24 GB HBM


Dataset Introduction
--------------------

GSM8k is a math problem dataset. The prompt is an elementary school
problem. The LLM model is asked to solve the math problem. Below is an example:

Prompt

   Katy makes coffee using teaspoons of sugar and cups of water in the
   ratio of 7:13. If she used a total of 120 teaspoons of sugar and cups
   of water, calculate the number of teaspoonfuls of sugar she used.

Solution

   The total ratio representing the ingredients she used to make the
   coffee is 7+13 = <<7+13=20>>20 Since the fraction representing the
   number of teaspoons she used is 7/20, she used 7/20\ *120 =
   <<7/20*\ 120=42>>42 #### 42

Step 1: Prepare the dataset
----------------------------

We preprocess the dataset in parquet format so that (1) it contains necessary fields for computing RL rewards and (2) is faster to read.

.. code-block:: bash

   python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

Step 2: Download a model for post-training
-------------------------------------------

In this example, we start with the ``Qwen2.5-0.5B-Instruct`` model.

.. code-block:: bash

   python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"

Step 3: Perform GRPO training with the instruct model
----------------------------------------------------------------------

**Reward Model/Function**

We use a pre-defined rule-based reward model. We force the model to produce a final
answer following 4 “#” as shown in the solution. We extract the final
answer from both the solution and model's output using regular
expression matching. We assign a reward of 1 to correct
answer, 0.0 to incorrect answer and 0 to no answer. 

For more details, please refer to `siirl/utils/reward_score/gsm8k.py <https://github.com/sii-research/siiRL/blob/main/siirl/utils/reward_score/gsm8k.py>`_.

**Training Script**

Now let's run GRPO training with the dataset and model above. [1]_


Set the ``data.train_files`` ,\ ``data.val_files``, ``actor_rollout_ref.model.path`` and ``critic.model.path`` based on your dataset and model names or paths.

.. code-block:: bash

   python3 -m siirl.client.main_dag \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$HOME/models/Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard']  \
    trainer.project_name=siirl_qwen2.5_0.5b_grpo \
    trainer.experiment_name=siirl_qwen2.5_0.5b_grpo_toy \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir=ckpts/qwen2.5_0.5b/grpo/ \
    trainer.val_before_train=True 2>&1 | tee verl_demo.log

You are expected to see the following logs, indicating training in progress. The key metric ``val/test_score/openai/gsm8k`` is computed every ``trainer.test_freq`` steps:

.. code-block:: bash

    step:1 - training/epoch:1.000 - training/global_step:0.000 - training/rollout_probs_diff_max:0.373 - training/rollout_probs_diff_mean:0.004 - training/rollout_probs_diff_std:0.009 - actor/entropy_loss:0.438 - actor/grad_norm:0.221 - actor/lr:0.000 - actor/pg_clipfrac:0.000 - actor/pg_clipfrac_lower:0.000 - actor/pg_loss:0.003 - actor/ppo_kl:-0.000 - critic/advantages/max:1.789 - critic/advantages/mean:-0.002 - critic/advantages/min:-0.730 - critic/returns/max:1.789 - critic/returns/mean:-0.002 - critic/returns/min:-0.730 - critic/rewards/max:1.000 - critic/rewards/mean:0.013 - critic/rewards/min:0.000 - critic/score/max:1.000 - critic/score/mean:0.013 - critic/score/min:0.000 - perf/cpu_mem_used_gb:11.576 - perf/cpu_memory_used_gb:125.440 - perf/delta_time/actor:72.260 - perf/delta_time/actor_log_prob:10.829 - perf/delta_time/advantage:0.039 - perf/delta_time/compute_core_metrics:0.020 - perf/delta_time/data_loading:1.030 - perf/delta_time/get_data_from_buffer:0.001 - perf/delta_time/get_entry_node:0.000 - perf/delta_time/get_intern_data_actor_old_log_prob:0.000 - perf/delta_time/get_intern_data_actor_train:0.000 - perf/delta_time/get_intern_data_calculate_advantages:0.000 - perf/delta_time/get_intern_data_function_reward:0.000 - perf/delta_time/get_intern_data_reference_log_prob:0.000 - perf/delta_time/get_next_node:0.000 - perf/delta_time/graph_execution:128.358 - perf/delta_time/graph_loop_management:0.001 - perf/delta_time/graph_output_handling:0.002 - perf/delta_time/put_data_to_buffer:0.001 - perf/delta_time/put_intern_data_actor_old_log_prob:0.000 - perf/delta_time/put_intern_data_actor_train:0.000 - perf/delta_time/put_intern_data_calculate_advantages:0.000 - perf/delta_time/put_intern_data_function_reward:0.000 - perf/delta_time/put_intern_data_reference_log_prob:0.000 - perf/delta_time/reduce_metrics:0.036 - perf/delta_time/ref:28.170 - perf/delta_time/reference:28.172 - perf/delta_time/reset_data_buffer:0.038 - perf/delta_time/reset_intern_data_buffer:0.000 - perf/delta_time/reward:0.255 - perf/delta_time/rollout:16.797 - perf/delta_time/step:129.426 - perf/delta_time/step_barrier:0.001 - perf/max_mem_alloc_gb:34.832 - perf/max_mem_rsvd_gb:39.678 - perf/max_memory_allocated_gb:34.832 - perf/max_memory_reserved_gb:39.678 - perf/mfu/actor:0.023 - perf/mfu/actor_log_prob:0.052 - perf/mfu/ref:0.021 - perf/mfu/rollout:0.079 - response_length/clip_ratio:0.610 - response_length/max:256.000 - response_length/mean:232.029 - response_length/min:76.000 - prompt_length/clip_ratio:0.000 - prompt_length/max:189.000 - prompt_length/mean:104.727 - prompt_length/min:66.000 - perf/total_num_tokens:431047.000 - perf/time_per_step:129.426 - perf/throughput:3330.450
    step:2 - training/epoch:1.000 - training/global_step:1.000 - training/rollout_probs_diff_max:0.326 - training/rollout_probs_diff_mean:0.004 - training/rollout_probs_diff_std:0.009 - actor/entropy_loss:0.432 - actor/grad_norm:0.210 - actor/lr:0.000 - actor/pg_clipfrac:0.000 - actor/pg_clipfrac_lower:0.000 - actor/pg_loss:0.004 - actor/ppo_kl:-0.000 - critic/advantages/max:1.789 - critic/advantages/mean:-0.004 - critic/advantages/min:-0.730 - critic/returns/max:1.789 - critic/returns/mean:-0.004 - critic/returns/min:-0.730 - critic/rewards/max:1.000 - critic/rewards/mean:0.013 - critic/rewards/min:0.000 - critic/score/max:1.000 - critic/score/mean:0.013 - critic/score/min:0.000 - perf/cpu_mem_used_gb:11.589 - perf/cpu_memory_used_gb:125.617 - perf/delta_time/actor:72.457 - perf/delta_time/actor_log_prob:10.689 - perf/delta_time/advantage:0.040 - perf/delta_time/compute_core_metrics:0.001 - perf/delta_time/data_loading:0.005 - perf/delta_time/get_data_from_buffer:0.001 - perf/delta_time/get_entry_node:0.000 - perf/delta_time/get_intern_data_actor_old_log_prob:0.000 - perf/delta_time/get_intern_data_actor_train:0.000 - perf/delta_time/get_intern_data_calculate_advantages:0.000 - perf/delta_time/get_intern_data_function_reward:0.000 - perf/delta_time/get_intern_data_reference_log_prob:0.000 - perf/delta_time/get_next_node:0.000 - perf/delta_time/graph_execution:123.794 - perf/delta_time/graph_loop_management:0.001 - perf/delta_time/graph_output_handling:0.002 - perf/delta_time/put_data_to_buffer:0.001 - perf/delta_time/put_intern_data_actor_old_log_prob:0.000 - perf/delta_time/put_intern_data_actor_train:0.000 - perf/delta_time/put_intern_data_calculate_advantages:0.000 - perf/delta_time/put_intern_data_function_reward:0.000 - perf/delta_time/put_intern_data_reference_log_prob:0.000 - perf/delta_time/reduce_metrics:0.001 - perf/delta_time/ref:24.271 - perf/delta_time/reference:24.273 - perf/delta_time/reset_data_buffer:0.005 - perf/delta_time/reset_intern_data_buffer:0.000 - perf/delta_time/reward:0.286 - perf/delta_time/rollout:16.043 - perf/delta_time/step:123.805 - perf/delta_time/step_barrier:0.001 - perf/max_mem_alloc_gb:36.362 - perf/max_mem_rsvd_gb:41.596 - perf/max_memory_allocated_gb:36.362 - perf/max_memory_reserved_gb:41.596 - perf/mfu/actor:0.023 - perf/mfu/actor_log_prob:0.053 - perf/mfu/ref:0.024 - perf/mfu/rollout:0.082 - response_length/clip_ratio:0.595 - response_length/max:256.000 - response_length/mean:230.901 - response_length/min:20.000 - prompt_length/clip_ratio:0.000 - prompt_length/max:215.000 - prompt_length/mean:105.098 - prompt_length/min:65.000 - perf/total_num_tokens:430078.000 - perf/time_per_step:123.805 - perf/throughput:3473.837

Beside, we provides a formatted, easy-to-read summary of core performance metrics on rank 0. This provides a clear, separate view of the most important indicators.

.. code-block:: bash

   ========================= RANK(0): Core Performance Metrics (Step: 1) =========================

   --- ⏱️  Overall Performance ---
   Step Time                   : 129.426 s
   Throughput (tokens/s)       : 3330.45
   Total Tokens in Step        : 431047

   --- 📈 Algorithm Metrics ---
   Actor Entropy               : 0.4380
   Critic Rewards (Mean/Min/Max): 0.013 / 0.000 / 1.000
   Critic Scores (Mean/Min/Max): 0.013 / 0.000 / 1.000

   --- 🔥 Model Flops Utilization (MFU) ---
   Mean MFU                    : N/A
   Actor Training MFU          : 0.023
   Rollout MFU                 : 0.079
   Reference Policy MFU        : 0.021
   Actor LogProb MFU           : 0.052

   --- 💾 Memory Usage ---
   Max GPU Memory Allocated    : 34.83 GB
   Max GPU Memory Reserved     : 39.68 GB
   CPU Memory Used             : 11.58 GB

   --- 📏 Sequence Lengths ---
   Prompt Length (Mean/Max)    : 104.7 / 189
   Response Length (Mean/Max)  : 232.0 / 256

   ==================================================================================

Checkout ``Algorithm Baselines`` page for full training and validation logs for reference.


If you encounter out of memory issues with HBM less than 32GB, enable the following configs would help:

.. code-block:: bash

    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_micro_batch_size_per_gpu=1 \

For the full set of configs, please refer to :ref:`config-explain-page` for detailed explanation and performance tuning.


.. [1] More training script examples for FSDP backend are stored in `examples/ppo_trainer <https://github.com/sii-research/siiRL/tree/main/examples/ppo_trainer>`_ directory.