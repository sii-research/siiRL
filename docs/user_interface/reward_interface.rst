================
Reward Interface
================

Custom reward functions allow you to score model-generated responses. Simply write a Python function and specify its path in configuration.

**Official Example:** ``siirl/user_interface/rewards_interface/custom_gsm8k_reward.py``

Architecture Overview
---------------------

::

                           Reward Computation Flow
   ==============================================================================

   +------------------+     +-------------------+     +------------------+
   |  Rollout Node    |     |   Reward Node     |     | Advantage Node   |
   |  (Generation)    |---->|   (Scoring)       |---->|  (Normalization) |
   +------------------+     +-------------------+     +------------------+
                                    |
                                    v
                            +---------------+
                            | RewardManager |
                            +---------------+
                                    |
           +------------------------+------------------------+
           |                        |                        |
           v                        v                        v
   +---------------+        +---------------+        +---------------+
   | Naive Reward  |        | Batch Reward  |        | Custom Reward |
   | (Rule-based)  |        | (Model-based) |        | (User-defined)|
   +---------------+        +---------------+        +---------------+
                                                            |
                                                            v
                                                    +---------------+
                                                    | compute_score |
                                                    | (data_source, |
                                                    |  solution_str,|
                                                    |  ground_truth,|
                                                    |  extra_info)  |
                                                    +-------+-------+
                                                            |
                                                            v
                                                    +---------------+
                                                    | Returns float |
                                                    | score [0, 1]  |
                                                    +---------------+

   ==============================================================================

   Custom Reward Function Integration:

   Configuration                          Runtime
   +---------------------------+          +---------------------------+
   | custom_reward_function:   |          | RewardManager loads       |
   |   path: /path/to/file.py  |  ----->  | compute_score function    |
   |   name: compute_score     |          | and calls it per sample   |
   +---------------------------+          +---------------------------+

Quick Start
-----------

Step 1: Write Reward Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a Python file with ``compute_score`` function:

.. code-block:: python

   # my_reward.py

   def compute_score(data_source, solution_str, ground_truth, extra_info):
       """
       Custom reward function

       Args:
           data_source (str): Dataset source identifier (e.g., "openai/gsm8k")
           solution_str (str): Model generated text
           ground_truth (str): Correct answer
           extra_info (dict): Additional information (optional)

       Returns:
           float: Score (typically 0-1)
       """
       # Your scoring logic
       if solution_str == ground_truth:
           return 1.0
       else:
           return 0.0

Step 2: Configuration
~~~~~~~~~~~~~~~~~~~~~

**Command Line:**

.. code-block:: bash

   python -m siirl.main_dag \
     custom_reward_function.path=/path/to/my_reward.py \
     custom_reward_function.name=compute_score

Official Example: GSM8K
-----------------------

**File:** ``siirl/user_interface/rewards_interface/custom_gsm8k_reward.py``

.. code-block:: python

   import re

   def extract_solution(solution_str, method="strict"):
       """Extract answer from solution"""
       if method == "strict":
           # Requires #### <answer> format
           solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
           if solution is None:
               return None
           final_answer = solution.group(1).replace(",", "")
           return final_answer
       elif method == "flexible":
           # Extract last number
           answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
           if len(answer) == 0:
               return None
           for final_answer in reversed(answer):
               if final_answer not in ["", "."]:
                   return final_answer
       return None

   def compute_score(data_source, solution_str, ground_truth, extra_info):
       """
       GSM8K scoring function

       Checks format and compares answer
       """
       method = "strict"
       format_score = 0.0
       score = 1.0

       answer = extract_solution(solution_str, method=method)

       if answer is None:
           return 0  # Format error
       elif answer == ground_truth:
           return score  # Correct answer
       else:
           return format_score  # Correct format but wrong answer

**Usage:**

.. code-block:: bash

   python -m siirl.main_dag \
     custom_reward_function.path=siirl/user_interface/rewards_interface/custom_gsm8k_reward.py \
     custom_reward_function.name=compute_score \
     data.train_files=/path/to/gsm8k.parquet

Custom Examples
---------------

Example 1: Keyword Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_score(data_source, solution_str, ground_truth, extra_info):
       """Keyword-based reward"""
       score = 0.0

       # Check keywords
       keywords = ["because", "therefore", "thus"]
       for keyword in keywords:
           if keyword in solution_str.lower():
               score += 0.3

       # Length check
       words = len(solution_str.split())
       if 50 <= words <= 200:
           score += 0.4

       return min(score, 1.0)

Example 2: Regex Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import re

   def compute_score(data_source, solution_str, ground_truth, extra_info):
       """Regex-based format validation"""
       # Extract numeric answer
       match = re.search(r"答案[是为][:：]\s*(\d+)", solution_str)

       if match is None:
           return 0.0  # Incorrect format

       answer = match.group(1)
       if answer == ground_truth:
           return 1.0  # Correct
       else:
           return 0.1  # Correct format but wrong answer

Example 3: Multi-stage Scoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import re

   def compute_score(data_source, solution_str, ground_truth, extra_info):
       """Multi-stage scoring: format + reasoning + correctness"""
       score = 0.0

       # Stage 1: Format check (0.2 points)
       if "####" in solution_str:
           score += 0.2

       # Stage 2: Reasoning steps (0.3 points)
       steps = solution_str.count('\n')
       if steps >= 3:
           score += 0.3

       # Stage 3: Answer correctness (0.5 points)
       answer_match = re.search(r"#### ([\-0-9\.]+)", solution_str)
       if answer_match:
           answer = answer_match.group(1)
           if answer == ground_truth:
               score += 0.5

       return score

Example 4: Multiple Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_score(data_source, solution_str, ground_truth, extra_info):
       """Route to different scoring functions based on data_source"""
       if data_source == "gsm8k":
           return score_gsm8k(solution_str, ground_truth)
       elif data_source == "math":
           return score_math(solution_str, ground_truth)
       else:
           return 0.0

   def score_gsm8k(solution_str, ground_truth):
       # GSM8K specific logic
       pass

   def score_math(solution_str, ground_truth):
       # MATH specific logic
       pass

Function Specification
----------------------

Required Signature
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_score(data_source, solution_str, ground_truth, extra_info):
       """
       Args:
           data_source (str): Dataset source
           solution_str (str): Model generated response
           ground_truth (str): Correct answer
           extra_info (dict): Additional information

       Returns:
           float: Score value
       """
       pass

Important Notes
~~~~~~~~~~~~~~~

1. **Function Name:** Can be customized, specify via ``custom_reward_function.name``
2. **Return Type:** Must return ``float``, typically in [0, 1] range
3. **Error Handling:** Recommended to catch exceptions and return default value (e.g., 0.0)
4. **Parameter Order:** Must follow the signature order
