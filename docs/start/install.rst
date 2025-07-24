Installation
============

siiRL provides three primary installation methods. We **strongly recommend** using the Docker image for the most reliable and hassle-free experience.

* :ref:`Method 1: Install from Docker Image (Recommended) <install-docker>`
* :ref:`Method 2: Install from PyPI (pip) <install-pip>`
* :ref:`Method 3: Install from Source (Custom Environment) <install-source>`
Requirements
------------

- **Python**: Version >= 3.10
- **CUDA**: Version >= 12.1

Currently, siiRL supports the following configurations are available:

- **FSDP** for training.
- **SGLang** and **vLLM** for rollout generation.

.. _install-docker:
Method 1: Install from docker image
-------------------------

The stable image is ``siiai/siirl-base:vllm0.8.5.post1-sglang0.4.6.post5-cu124``. This images contains the latest version of inference and training framework and its dependencies.

.. _install-pip:
Method 2: Install from PIP
-----------------

We provide prebuilt python wheels for Linux. Install siiRL with the following command:

.. code:: bash

    # Install siiRL with vLLM
    pip install siirl[vllm]

    # Then, install required high-performance dependencies for siiRL
    pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
    pip install flash-attn==2.7.3 --no-build-isolation   

.. _install-source:
Method 3: Install from custom environment
---------------------------------------------

We recommend to use docker images for convenience. However, if your environment is not compatible with the docker image, you can also install siirl in a python environment.

Install dependencies
::::::::::::::::::::

1. First of all, to manage environment, we recommend using conda:

.. code:: bash

   conda create -n siirl python==3.10
   conda activate siirl

2. Install python packages

.. note::
    The following commands are an example for an environment with CUDA 12.4.
    If you are using a different CUDA version, you must adjust the package versions and index URLs accordingly, especially for torch, flashinfer, and flash-attn.
    
.. code:: bash

    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
    pip install flash-attn==2.7.3 --no-build-isolation
    pip install accelerate codetiming datasets dill hydra-core pandas wandb loguru tensorboard qwen_vl_utils
    pip install 'ray[default]>=2.47.1'
    pip install opentelemetry-exporter-prometheus==0.47b0


3. Then, execute the following commands to install vLLM and SGLang:

.. code:: bash

    pip install vllm==0.8.5.post1

Install siirl
::::::::::::::

For installing the latest version of siirl, the best way is to clone and
install it from source. Then you can modify our code to customize your
own post-training jobs.

.. code:: bash

   git clone https://github.com/sii-research/siiRL.git
   cd siirl
   pip install -e .

