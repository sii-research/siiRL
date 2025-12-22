DeepScaleR Example with PPO
=============================

Introduction
------------

This example demonstrates how to fine-tune a Large Language Model for advanced mathematical reasoning using the **DeepScaleR** dataset.

**Paper:** https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2.

**Dataset:** https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset

The core idea is to leverage Reinforcement Learning (RL), specifically Proximal Policy Optimization (PPO), to teach the model not just to find the correct answer, but to follow a logical, step-by-step reasoning process. This is achieved by rewarding the model based on the correctness of its final answer, which is extracted from a structured output.

Dataset Overview
----------------

The DeepScaleR dataset consists of challenging mathematical problems. Each sample includes a question (`problem`), a detailed reasoning path (`solution`), and a final answer enclosed in a `\\boxed{}` block (`answer`).

**An example from DeepScaleR:**

**Prompt:**
   "Let $a_n=6^{n}+8^{n}$. Determine the remainder upon dividing $a_ {83}$ by $49$."

**Solution:**
   "$6^{83} + 8^{83} = (6+8)(6^{82}-6^{81}8+\\ldots-8^{81}6+8^{82})$\n Becuase $7|(6+8)$, we only consider $6^{82}-6^{81}8+\\ldots-8^{81}6+8^{82} \\pmod{7}$\n$6^{82}-6^{81}8+\\ldots-8^{81}6+8^{82} \\equiv (-1)^{82} - (-1)^{81}+ \\ldots - (-1)^1 + 1 = 83 \\equiv 6 \\pmod{7}$\n$6^{83} + 8^{83} \\equiv 14 \\cdot 6 \\equiv \\boxed{035} \\pmod{49}$"

**Answer:**
   `35`

Step 1: Prepare the Dataset
---------------------------

First, preprocess the DeepScaleR dataset into the required Parquet format. Our framework includes a script for this purpose.

.. code:: bash

   cd examples/data_preprocess
   python3 deepscaler.py --local_dir ~/data/deepscaler

This will download the dataset from Hugging Face, process it, and save `train.parquet` and `test.parquet` files in the `~/data/deepscaler` directory.

Step 2: Download the Pre-trained Model
--------------------------------------

You need a base model to start the PPO training. In this example, we use `Qwen2.5-7B-Instruct`. There are several ways to make the model available to the trainer:

- **Recommended: Download via CLI:** Use tools like `huggingface-cli` or `modelscope` to download the model to a local directory. This gives you more control.

  .. code:: bash

     # For Hugging Face
     huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ~/data/models/Qwen2.5-7B-Instruct --local-dir-use-symlinks False
     
     # For ModelScope
     modelscope download Qwen/Qwen2.5-7B-Instruct --local_dir ~/data/models/Qwen2.5-7B-Instruct

- **Automatic Download:** You can also specify the Hugging Face model name (e.g., `Qwen/Qwen2.5-7B-Instruct`) directly in the `actor_rollout_ref.model.path` and `critic.model.path` fields of your run script. The framework will attempt to download it automatically on the first run.

Step 3: Perform PPO Training
----------------------------

With the data and model ready, you can now launch the PPO training job.

**Reward Function**

For this task, we use a simple but effective rule-based reward function. The framework's default reward mechanism will be used, which performs an exact match between the model's generated answer and the `ground_truth` from the dataset.
- The model is prompted to provide its final answer inside a `\\boxed{...}` block.
- The reward function checks if the content inside the generated `\\boxed{}` matches the ground truth answer.
- A correct match receives a positive reward (e.g., 1.0), while an incorrect match or a malformed response receives zero reward.

**Training Script**

Below is a complete training script based on `examples/ppo_trainer/run_qwen2_5-7b.sh`. It is configured for a single-node, multi-GPU setup. You should adapt paths like `HOME` to your environment.

.. literalinclude:: ../../examples/ppo_trainer/run_qwen2_5-7b.sh
   :language: bash
   :caption: examples/ppo_trainer/run_qwen2_5-7b.sh
