Ascend NPU
==========

SiiRL is also supports for Huawei's Ascend NPU devices. This guide has been tested with the following hardware:
- Atlas 200T A2 Box16

Installation Process
--------------------

Core Environment Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure your environment meets these core software version requirements:

+---------------------+------------+
| Software            | Version    |
+---------------------+------------+
| Python              | == 3.10    |
+---------------------+------------+
| CANN                | == 8.1.RC1 |
+---------------------+------------+
| PyTorch             | == 2.5.1   |
+---------------------+------------+
| torch_npu           | == 2.5.1   |
+---------------------+------------+
| mindspeed(Optional) | == 0.12.1  |
+---------------------+------------+
Recommended Base Image
^^^^^^^^^^^^^^^^^^^^^^

For a smoother setup, we strongly recommend using our pre-built Docker image, which includes all necessary dependencies. Please note this pre-built docker image contains torch, torch-npu, vLLM and vLLM-Ascend packages, after pulling it you only need to install siiRL framework from source.

.. code-block:: bash

    docker pull crispig/verl_npu:cann8.1rc1-py3.10-torch2.5.1-vllm-ascend0.7.3.post1-250616

Compiling vLLM and vllm-ascend [Optional]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Proper integration of vLLM within siiRL requires compiling both `vllm` and `vllm-ascend` from source. Follow the steps below, paying close attention to the instructions specific to your hardware.

.. note::
    We recommend using the latest version of vllm v0.9.2 and vllm-ascend v0.9.0rc2, which support setting use_remove_padding=True.

.. code-block:: bash
    
    # vllm
    git clone -b v0.9.2 --depth 1 https://github.com/vllm-project/vllm.git
    cd vllm
    pip install -r requirements-build.txt

    # For Atlas 200T A2 Box16
    VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/

.. code-block:: bash
    
    # vllm-ascend
    git clone -b v0.9.0rc2 --depth 1 https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    export COMPILE_CUSTOM_KERNELS=1
    python setup.py install

SiiRL Installation
^^^^^^^^^^^^^^^^^^

Finally, install the siiRL framework itself. DO NOT use the pip install command to install siiRL, it will cause dependency conflicts.

.. code-block:: bash

    git clone https://github.com/sii-research/siiRL.git
    cd siirl
    pip install -e .

Third-Party Library Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please be aware of the following specific requirements and limitations for certain libraries on Ascend hardware:

+--------------+---------------+
| Software     | Description   |
+--------------+---------------+
| transformers | v4.52.4       |
+--------------+---------------+
| flash_attn   | not supported |
+--------------+---------------+
| liger-kernel | not supported |
+--------------+---------------+
| tensordict   | 0.8.3 (ARM)   |
+--------------+---------------+

1.  Using `--flash_attention_2` through `transformers` is supported (requires `transformers` version >= 4.52.0).
2.  Flash Attention acceleration via the `flash_attn` package is not supported.
3.  `liger-kernel` is not supported.
4.  For ARM servers, `tensordict` version 0.8.3 is required. You can manually install it after the main dependencies are installed.
5.  For x86 servers, the CPU version of `torchvision` must be installed.

.. code-block:: bash

    pip install torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu

Verification with a Quick Start Example
---------------------------------------

To ensure your setup is correct, we recommend performing a quick test run. The following example trains a Qwen2.5-0.5B model on the GSM8k dataset using the GRPO algorithm.

1.  **Prepare the Dataset**
    First, download and preprocess the GSM8k dataset. The provided script will convert it to the Parquet format required by the framework.

.. code-block:: bash

    python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

2.  **Run the Training Job**
    Next, execute the training command below. Ensure you have set the `VLLM_ATTENTION_BACKEND` environment variable.

.. code-block:: bash

    set -x

    python3 -m siirl.client.main_dag \
        algorithm.adv_estimator=grpo \
        data.train_files=/datasets/gsm8k/train.parquet\
        data.val_files=/datasets/gsm8k/teset.parquet \
        data.train_batch_size=1024 \
        data.max_prompt_length=1024 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=/models/Qwen2.5-7B-Instruct \
        actor_rollout_ref.actor.optim.lr=5e-8 \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='siirl_grpo_example_gsm8k' \
        trainer.experiment_name='qwen2_7b_function_rm' \
        trainer.n_gpus_per_node=16 \
        trainer.nnodes=$NNODES \
        trainer.save_freq=-1 \
        trainer.test_freq=5 \
        trainer.total_epochs=300 \
        trainer.device=npu $@

(Optional) Setting Up MindSpeed Training Backend Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Refer to the MindSpeed README <https://gitee.com/ascend/MindSpeed>_ for instructions on installing the MindSpeed acceleration library, recommended versions: MindSpeed Core 0.12.1, Megatron-LM 0.12.2.

.. warning::

   Please Be sure to install **megatron-core** via ``pip install``.  
   Using ``PYTHONPATH`` to point to megatron will crash the program.

Enable siirl worker model ``strategy`` and set it to ``megatron``. For example: ``actor_rollout_ref.actor.strategy=megatron``.

Custom MindSpeed parameters can be passed through the override_transformer_config option. For instance, to enable FA for the actor model, you can use:
``+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True``.

MindSpeed provides the same support for siiRL and verl. For more feature details, please refer to the MindSpeed+verl documentation. <https://gitee.com/ascend/MindSpeed/blob/master/docs/user-guide/verl.md>_.
