===================
Configuration Guide
===================

siiRL uses Hydra-based configuration management with dataclass parameters. All configuration parameters are defined in the ``siirl/params/`` directory and can be set via command-line arguments.

Configuration Structure
-----------------------

Parameters are organized into the following modules:

- ``DataArguments``: Data-related parameters (``siirl/params/data_args.py``)
- ``ActorRolloutRefArguments``: Actor, Rollout, and Reference model parameters (``siirl/params/model_args.py``)
- ``CriticArguments``: Critic model parameters (``siirl/params/model_args.py``)
- ``RewardModelArguments``: Reward model parameters (``siirl/params/model_args.py``)
- ``AlgorithmArguments``: RL algorithm parameters (``siirl/params/model_args.py``)
- ``TrainingArguments``: Training configuration (``siirl/params/training_args.py``)
- ``DAGArguments``: DAG workflow parameters (``siirl/params/dag_args.py``)
- ``ProfilerArguments``: Profiling parameters (``siirl/params/profiler_args.py``)

All parameters are combined into the ``SiiRLArguments`` class.

Usage
-----

Parameters are set via command-line arguments using dot notation:

.. code-block:: bash

   python -m siirl.main_dag \
     data.train_files=/path/to/train.parquet \
     data.train_batch_size=512 \
     actor_rollout_ref.model.path=/path/to/model \
     algorithm.adv_estimator=grpo \
     trainer.total_epochs=30

Data Parameters
---------------

Location: ``siirl/params/data_args.py``

.. code-block:: bash

   data.tokenizer=null
   data.train_files=/path/to/train.parquet
   data.val_files=/path/to/val.parquet
   data.prompt_key=prompt
   data.max_prompt_length=512
   data.max_response_length=512
   data.train_batch_size=1024
   data.return_raw_input_ids=False
   data.return_raw_chat=False
   data.return_full_prompt=False
   data.shuffle=True
   data.filter_overlong_prompts=False
   data.filter_overlong_prompts_workers=1
   data.truncation=error
   data.image_key=images
   data.trust_remote_code=True

**Key Parameters:**

- ``data.train_files``: Training data file path (Parquet format, can be list or single file)
- ``data.val_files``: Validation data file path
- ``data.prompt_key``: Field name for prompt in dataset (default: "prompt")
- ``data.max_prompt_length``: Maximum prompt length (left-padded)
- ``data.max_response_length``: Maximum response length for rollout generation
- ``data.train_batch_size``: Training batch size per iteration
- ``data.return_raw_input_ids``: Return original input_ids without chat template (for different RM chat templates)
- ``data.shuffle``: Whether to shuffle data
- ``data.truncation``: Truncation strategy ("error", "left", "right", "middle")
- ``data.trust_remote_code``: Allow remote code execution for tokenizers

Custom Dataset
~~~~~~~~~~~~~~

.. code-block:: bash

   data.custom_cls.path=/path/to/custom_dataset.py
   data.custom_cls.name=MyDatasetClass

- ``data.custom_cls.path``: Path to custom dataset class file
- ``data.custom_cls.name``: Name of the dataset class

Actor/Rollout/Reference Model
------------------------------

Location: ``siirl/params/model_args.py``

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   actor_rollout_ref.hybrid_engine=True
   actor_rollout_ref.model.path=/path/to/model
   actor_rollout_ref.model.external_lib=null
   actor_rollout_ref.model.enable_gradient_checkpointing=False
   actor_rollout_ref.model.enable_activation_offload=False
   actor_rollout_ref.model.trust_remote_code=False
   actor_rollout_ref.model.use_remove_padding=False

- ``actor_rollout_ref.model.path``: Huggingface model path (local or HDFS)
- ``actor_rollout_ref.model.external_lib``: Additional Python packages to import
- ``actor_rollout_ref.model.enable_gradient_checkpointing``: Enable gradient checkpointing
- ``actor_rollout_ref.model.enable_activation_offload``: Enable activation offloading
- ``actor_rollout_ref.model.trust_remote_code``: Allow remote code model loading
- ``actor_rollout_ref.model.use_remove_padding``: Remove padding tokens for efficiency

Actor Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   actor_rollout_ref.actor.strategy=fsdp
   actor_rollout_ref.actor.ppo_mini_batch_size=256
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
   actor_rollout_ref.actor.grad_clip=1.0
   actor_rollout_ref.actor.clip_ratio=0.2
   actor_rollout_ref.actor.entropy_coeff=0.0
   actor_rollout_ref.actor.use_kl_loss=False
   actor_rollout_ref.actor.kl_loss_coef=0.001
   actor_rollout_ref.actor.ppo_epochs=1
   actor_rollout_ref.actor.optim.lr=1e-6

- ``actor.strategy``: Backend strategy ("fsdp" or "megatron")
- ``actor.ppo_mini_batch_size``: Mini-batch size for PPO updates (global across GPUs)
- ``actor.ppo_micro_batch_size_per_gpu``: Micro-batch size per GPU (gradient accumulation)
- ``actor.grad_clip``: Gradient clipping threshold
- ``actor.clip_ratio``: PPO clip ratio
- ``actor.use_kl_loss``: Enable KL loss in actor
- ``actor.kl_loss_coef``: KL loss coefficient (for GRPO)
- ``actor.optim.lr``: Learning rate

Reference Model
~~~~~~~~~~~~~~~

.. code-block:: bash

   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
   actor_rollout_ref.ref.fsdp_config.param_offload=False

- ``ref.log_prob_micro_batch_size_per_gpu``: Micro-batch size for reference log prob computation
- ``ref.fsdp_config.param_offload``: Enable parameter offloading (recommended for models > 7B)

Rollout Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   actor_rollout_ref.rollout.name=vllm
   actor_rollout_ref.rollout.temperature=1.0
   actor_rollout_ref.rollout.top_k=-1
   actor_rollout_ref.rollout.top_p=1.0
   actor_rollout_ref.rollout.tensor_model_parallel_size=2
   actor_rollout_ref.rollout.gpu_memory_utilization=0.5
   actor_rollout_ref.rollout.n=8

- ``rollout.name``: Rollout backend ("vllm", "sglang", "hf")
- ``rollout.temperature``: Sampling temperature
- ``rollout.top_k``: Top-k sampling (-1 for vLLM, 0 for HF)
- ``rollout.top_p``: Top-p sampling
- ``rollout.tensor_model_parallel_size``: Tensor parallelism size (vLLM only)
- ``rollout.gpu_memory_utilization``: GPU memory fraction for vLLM
- ``rollout.n``: Number of responses per prompt (>1 for GRPO/RLOO)

Critic Model
------------

Location: ``siirl/params/model_args.py``

.. code-block:: bash

   critic.enable=True
   critic.model.path=/path/to/critic_model
   critic.ppo_mini_batch_size=256
   critic.ppo_micro_batch_size_per_gpu=8
   critic.optim.lr=1e-5

Most parameters are similar to Actor configuration.

Reward Model
------------

Location: ``siirl/params/model_args.py``

.. code-block:: bash

   reward_model.enable=False
   reward_model.model.path=/path/to/reward_model
   reward_model.model.input_tokenizer=null
   reward_model.micro_batch_size_per_gpu=16
   reward_model.reward_manager=naive

- ``reward_model.enable``: Enable reward model (False = use only custom reward functions)
- ``reward_model.model.input_tokenizer``: Input tokenizer path (if different from policy)
- ``reward_model.reward_manager``: Reward manager type ("naive", "batch", "parallel", "dapo", "embodied")

Custom Reward Function
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   custom_reward_function.path=/path/to/my_reward.py
   custom_reward_function.name=compute_score

- ``custom_reward_function.path``: Path to custom reward function file
- ``custom_reward_function.name``: Function name (default: "compute_score")

See :doc:`/user_interface/reward_interface` for details.

Algorithm Parameters
--------------------

Location: ``siirl/params/model_args.py``

.. code-block:: bash

   algorithm.gamma=1.0
   algorithm.lam=1.0
   algorithm.adv_estimator=grpo
   algorithm.use_kl_in_reward=False
   algorithm.kl_penalty=kl
   algorithm.kl_ctrl.type=fixed
   algorithm.kl_ctrl.kl_coef=0.005
   algorithm.workflow_type=DEFAULT

- ``algorithm.gamma``: Discount factor
- ``algorithm.lam``: GAE lambda (bias-variance tradeoff)
- ``algorithm.adv_estimator``: Advantage estimator ("gae", "grpo", "cpgd", "gspo", "rloo")
- ``algorithm.use_kl_in_reward``: Enable KL penalty in reward
- ``algorithm.kl_penalty``: KL divergence calculation method ("kl", "abs", "mse", "low_var_kl", "full")
- ``algorithm.workflow_type``: Workflow type ("DEFAULT", "DAPO", "EMBODIED")

Training Parameters
-------------------

Location: ``siirl/params/training_args.py``

.. code-block:: bash

   trainer.total_epochs=30
   trainer.project_name=siirl_examples
   trainer.experiment_name=gsm8k
   trainer.logger=['console', 'wandb']
   trainer.nnodes=1
   trainer.n_gpus_per_node=8
   trainer.save_freq=10
   trainer.val_before_train=True
   trainer.test_freq=2

- ``trainer.total_epochs``: Number of training epochs
- ``trainer.project_name``: Project name (for logging)
- ``trainer.experiment_name``: Experiment name (for logging)
- ``trainer.logger``: Logger types (["console", "wandb", "tensorboard", "mlflow"])
- ``trainer.nnodes``: Number of nodes
- ``trainer.n_gpus_per_node``: Number of GPUs per node
- ``trainer.save_freq``: Checkpoint saving frequency (by iteration)
- ``trainer.val_before_train``: Run validation before training
- ``trainer.test_freq``: Validation frequency (by iteration)

DAG Parameters
--------------

Location: ``siirl/params/dag_args.py``

.. code-block:: bash

   dag.custom_pipeline_fn=null

- ``dag.custom_pipeline_fn``: Custom pipeline function path (e.g., "module:function")

See :doc:`/user_interface/pipeline_interface` for custom pipeline details.

Complete Example
----------------

GRPO Training
~~~~~~~~~~~~~

.. code-block:: bash

   python -m siirl.main_dag \
     algorithm.adv_estimator=grpo \
     algorithm.workflow_type=DEFAULT \
     data.train_files=/path/to/gsm8k/train.parquet \
     data.train_batch_size=512 \
     data.max_prompt_length=2048 \
     data.max_response_length=4096 \
     actor_rollout_ref.model.path=/path/to/model \
     actor_rollout_ref.actor.optim.lr=1e-6 \
     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
     actor_rollout_ref.rollout.name=vllm \
     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
     actor_rollout_ref.rollout.n=8 \
     custom_reward_function.path=siirl/user_interface/rewards_interface/custom_gsm8k_reward.py \
     custom_reward_function.name=compute_score \
     trainer.total_epochs=30 \
     trainer.n_gpus_per_node=8 \
     trainer.save_freq=10

PPO Training
~~~~~~~~~~~~

.. code-block:: bash

   python -m siirl.main_dag \
     algorithm.adv_estimator=gae \
     critic.enable=True \
     data.train_files=/path/to/data.parquet \
     actor_rollout_ref.model.path=/path/to/model \
     actor_rollout_ref.actor.optim.lr=1e-6 \
     actor_rollout_ref.rollout.name=vllm \
     critic.optim.lr=1e-5 \
     trainer.total_epochs=30

DAPO Training
~~~~~~~~~~~~~

.. code-block:: bash

   python -m siirl.main_dag \
     algorithm.workflow_type=DAPO \
     algorithm.adv_estimator=grpo \
     algorithm.filter_groups.enable=True \
     algorithm.filter_groups.metric=seq_final_reward \
     data.train_files=/path/to/data.parquet \
     actor_rollout_ref.model.path=/path/to/model \
     trainer.total_epochs=30

Parameter Reference
-------------------

For the complete parameter definitions, see:

- ``siirl/params/data_args.py`` - Data parameters
- ``siirl/params/model_args.py`` - Model, algorithm parameters
- ``siirl/params/training_args.py`` - Training parameters
- ``siirl/params/dag_args.py`` - DAG workflow parameters
- ``siirl/params/profiler_args.py`` - Profiler parameters


