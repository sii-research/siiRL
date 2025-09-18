Multi-Agent Example with PPO
=============================

Introduction
------------

This guide details how to fine-tune multi-agent with Marft(Multi-Agent Reinforcement Fine-Tuning) with DeepScaleR.

**Paper:** https://arxiv.org/pdf/2504.16129.

**Dataset:** https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset

The core idea is to leverage Reinforcement Learning (RL) in Multi-Agent with Multi-Turn with Env, to teach the model not just to find the correct answer, but to follow a logical, step-by-step reasoning process. This is achieved by rewarding the model based on the correctness of its final answer, which is extracted from a structured output.

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

You need a base model to start the PPO training. In this example, we use `Qwen2.5-3B-Instruct`. There are several ways to make the model available to the trainer:

- **Recommended: Download via CLI:** Use tools like `huggingface-cli` or `modelscope` to download the model to a local directory. This gives you more control.

  .. code:: bash

     # For Hugging Face
     huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ~/data/models/Qwen2.5-3B-Instruct --local-dir-use-symlinks False
     
     # For ModelScope
     modelscope download Qwen/Qwen2.5-3B-Instruct --local_dir ~/data/models/Qwen2.5-3B-Instruct

- **Automatic Download:** You can also specify the Hugging Face model name (e.g., `Qwen/Qwen2.5-3B-Instruct`) directly in the `actor_rollout_ref.model.path` and `critic.model.path` fields of your run script. The framework will attempt to download it automatically on the first run.


Step 3: Prepare the WorkerFlow and script
-----------------------------------------

You need to specified the WorkerFlow path of Multi-agent algo with param `dag.workflow_path`. We have put a predefined yaml file at `examples/experimental/marft/config/workflow_marft.yaml`.
This chapter will explain params of agent in workerflow and script.

**Agent Group**

Each node has an agent_group parameter, which defines which corresponding agent the node belongs to. Note that the agent_group of all node must be a sequence of consecutive integers and start from 0.

**Agent Options**

Each node has an agent_options parameter, which is valid for Rollout and Actor nodes. In `workflow_marft.yaml`, you can see parameters defined like this:

    .. code:: bash

      - node_id: "rollout"
        node_type: "MODEL_INFERENCE"
        node_role: "ROLLOUT"
        agent_options:
            obs_with_env: true
            process_path: examples/experimental/marft/config/process.py
            pre_process_kwargs: 
                pre_chat_template: "<|im_start|>system: Two LLM agents (Reasoner -> Actor) collaborate step-by-step to solve math problems. You are the **Reasoner**: Analyze the original problem, historical actions, and reflection data (if provided) to determine the critical next step. Guide the Actor by providing concise reasoning for the optimal operation.<|im_end|>\n <|im_start|> problem: ${prompt} <|im_end|>\n <|im_start|> reasoner:  "
            post_process_kwargs:
                post_chat_template: " <|im_start|> reasoner: "
            env_path: [examples/experimental/marft/config/math_env.py:MathEnv]


      - node_id: "actor_train"
        node_type: "MODEL_TRAIN"
        node_role: "ACTOR"
        agent_options:
            train_cycle: 15

      - node_id: "critic_2_value"
        node_type: "MODEL_TRAIN"
        node_role: "CRITIC"
        agent_group: 1
        only_forward_compute: true
        agent_options:
            share_instance: 0


    
`obs_with_env` defines whether the rollout node interacts with the environment. If set to true, the node will attempt to get the initial observation from the environment and, after generating content, obtain the next observation from the environment.

`process_path` defines the path to a Python file that will be executed by rollout node. This script contains two functions: pre_process and post_process, as each agent may have its own data processing method.

`pre_process_kwargs` and `post_process_kwargs` are used to pass parameters to the pre-processing and post-processing functions, respectively.

`pre_process func` takes the following parameters:  tokenizer, prompt_id(type:List[int]), obs from environment(type:List[int]) and pre_process_kwargs it returns a new prompt_id(type:List[int]) that will be used in rollout generation.

`post_process func` takes the following parameters: tokenizer, prompt_id (type: List[int]), response_id (type: List[int]), and post_process_kwargs. It returns a new prompt_id that will be used in the next generation turn.

`train_cycle` determines the training interval of the actor node.

`env_path` defines the path to the environment file and class name of the environment, splited by `:`, like file_path:class_name. The class should have reset and step methods. Step method should get actions and ground_truth, then return a tuple of (obs, reward, done).

`share_instance` specifies the node sharing instance associated with the target agent_group; for example, a critic shared among multiple agents.

**Script Params**

`rollout.agent.rewards_with_env` defines whether the rollout node interacts with the environment to obtain rewards. If set to true, it will attempt to get reward from env after generating content.

`rollout.multi_turn.max_assistant_turns` defines the maximum number of turns.

`rollout.multi_turn.use_all_traj` specifies that all multi-turn trajectories generated by multi-agents in each round are required for training.

`algorithm.share_reward_in_agent` defines whether to share rewards among all agents.Becuase in Marft, only last agent has reward node, so pre-agent will not have reward.If set to true, pre-agent will have same reward as last agent. Otherwise, pre-agent will have zero reward.


Step 4: Perform PPO Training
----------------------------

With the data and model ready, you can now launch the PPO training job.

**Reward Function**

For this task, we use a simple but effective rule-based reward function. The framework's default reward mechanism will be used, which performs an exact match between the model's generated answer and the `ground_truth` from the dataset.
- The model is prompted to provide its final answer inside a `\\boxed{...}` block.
- The reward function checks if the content inside the generated `\\boxed{}` matches the ground truth answer.
- A correct match receives a positive reward (e.g., 1.0), while an incorrect match or a malformed response receives zero reward.

**Training Script**

Below is a complete training script based on `examples/experimental/marft/run_qwen2_5-3b_marft.sh`. It is configured for a single-node, multi-GPU setup. You should adapt paths like `HOME` to your environment.

.. literalinclude:: ../../examples/experimental/marft/run_qwen2_5-3b_marft.sh
   :language: bash
   :caption: examples/experimental/marft/run_qwen2_5-3b_marft.sh

