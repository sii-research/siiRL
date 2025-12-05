Implementing Reward Functions for Datasets
===========================================

In Reinforcement Learning for LLMs, the reward function is a critical component that guides the model's learning process. It quantitatively evaluates the quality of a generated response, signaling what constitutes a "good" or "bad" output. Our framework provides a flexible system for defining these rewards, supporting both pre-implemented logic for common datasets and fully customized functions for specific tasks.

The RewardManager
-----------------

The ``RewardManager`` is the central hub for reward computation. As defined in `siirl/scheduler/reward.py`, its primary role is to orchestrate the scoring of generated responses by invoking a specified scoring function. Different managers, like `NaiveRewardManager` or `BatchRewardManager`, offer different strategies for handling this process. This design is consistent with the `verl` framework's architecture. [1]_

The typical workflow is as follows:
1. The manager receives a `DataProto` object, which is a batch containing all necessary information.
2. It extracts relevant fields, such as the model's generated text (`solution_strs`) and the reference answer (`ground_truth`).
3. It passes this data to a designated scoring function (`compute_score_fn`) to calculate the reward for each item in the batch.

This design allows the core training loop to remain agnostic to the specifics of reward calculation, which are neatly encapsulated within the manager and its scoring function.

Reward Function Implementations
-------------------------------

You can define reward logic in two ways: by using our pre-built functions or by creating your own.

Pre-implemented Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

For standard benchmarks, we provide ready-to-use reward functions in the `siirl/utils/reward_score/` directory. These cover datasets like `GSM8K` and `MATH`, implementing their standard evaluation logic. For instance, the `GSM8K` scorer extracts the final numerical answer and compares it to the ground truth.

Customized Functions
~~~~~~~~~~~~~~~~~~~~

For novel tasks or custom evaluation criteria, you can supply your own reward function. This is configured via two parameters: `custom_reward_function.path` and `custom_reward_function.name`.

Let's consider a practical example from the `run_qwen2_5-7b-custom_reward.sh` script, which uses a batch-processing reward function for efficiency.

**1. Configuration in the script:**

The script specifies the path to the custom code, the function to use, and selects the `BatchRewardManager` to execute it.

.. code-block:: bash

   # ... other configurations ...
   python3 -m siirl.main_dag \
       # ...
       custom_reward_function.path=$HOME/rl/rewardfunc_gsm8k.py \
       custom_reward_function.name=compute_score \
       reward_model.reward_manager=batch \
       # ...

**2. Implementation of the reward function:**

The corresponding `rewardfunc_gsm8k.py` file implements the `compute_score` function. This function is designed to process an entire batch of solutions at once, which is significantly more efficient than processing them one by one.

.. code:: python

   import re

   def extract_solution(solution_str, method="strict"):
       # ... (logic to extract the final answer from text)
       # For example, finds the number after "####"
       if method == "strict":
           solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
           if solution is None: return None
           final_answer = solution.group(0).split("#### ")[1].replace(",", "")
           return final_answer
       # ... other extraction logic ...

   def compute_score(data_sources, solution_strs, ground_truths, extra_infos, method="strict", score=1.0, **kwargs):
       """
       Computes scores for a batch of solutions.
       """
       scores = []
       for solution_str, ground_truth in zip(solution_strs, ground_truths):
           answer = extract_solution(solution_str=solution_str, method=method)
           if answer is not None and answer == ground_truth:
               scores.append(score)
           else:
               scores.append(0.0)
       return scores

The function signature should accept lists of `solution_strs` and `ground_truths`. You can also pass custom parameters from your configuration, like `method` or `score`, by defining them under `custom_reward_function.reward_kwargs`. This allows you to easily experiment with different reward schemes without changing the code.

.. [1] https://verl.readthedocs.io/en/latest/preparation/reward_function.html