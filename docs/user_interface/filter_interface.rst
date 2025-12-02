================
Filter Interface
================

Filter interface is used for dynamic sampling and data filtering in Pipelines.

**Location:** ``siirl/user_interface/filter_interface/``

Built-in Filters
----------------

DAPO Dynamic Sampling
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``siirl/user_interface/filter_interface/dapo.py``

**Function:** ``dynamic_sampling()``

Filters zero-variance sample groups (all correct or all incorrect).

**How it works:**

1. Group samples by uid (prompt)
2. Calculate variance for each group
3. Filter groups with variance = 0

**Configuration:**

.. code-block:: bash

   python -m siirl.main_dag \
     algorithm.workflow_type=DAPO \
     algorithm.filter_groups.enable=true \
     algorithm.filter_groups.metric=seq_final_reward

**Usage in Pipeline:**

.. code-block:: python

   pipeline.add_node(
       "dynamic_sampling",
       func="siirl.user_interface.filter_interface.dapo:dynamic_sampling",
       deps=["function_reward"],
       node_type=NodeType.COMPUTE,
       node_role=NodeRole.DYNAMIC_SAMPLING
   )

**Returned Metrics:**

- ``dapo_sampling/kept_trajectories_ratio``
- ``dapo_sampling/kept_groups``
- ``dapo_sampling/total_groups``

Embodied AI Sampling
~~~~~~~~~~~~~~~~~~~~

**Location:** ``siirl/user_interface/filter_interface/embodied.py``

**Function:** ``embodied_local_rank_sampling()``

Filters Embodied AI data based on task completion and accuracy.

**Features:**

- Task verification
- Accuracy-based filtering
- Truncated trajectory filtering

**Configuration:**

.. code-block:: bash

   python -m siirl.main_dag \
     algorithm.workflow_type=EMBODIED \
     algorithm.embodied_sampling.filter_accuracy=true \
     algorithm.embodied_sampling.filter_truncated=true \
     algorithm.embodied_sampling.accuracy_lower_bound=0.0 \
     algorithm.embodied_sampling.accuracy_upper_bound=1.0 \
     actor_rollout_ref.embodied.env.max_steps=100

**Usage in Pipeline:**

.. code-block:: python

   pipeline.add_node(
       "dynamic_sampling",
       func="siirl.user_interface.filter_interface.embodied:embodied_local_rank_sampling",
       deps=["rollout_actor"],
       node_type=NodeType.COMPUTE,
       node_role=NodeRole.DYNAMIC_SAMPLING
   )

Custom Filter
-------------

Basic Template
~~~~~~~~~~~~~~

.. code-block:: python

   from siirl.params import SiiRLArguments
   from siirl.dag_worker.data_structures import NodeOutput
   from siirl.data_coordinator.sample import filter_tensordict
   import torch

   def my_custom_filter(
       config: SiiRLArguments,
       batch,
       **kwargs
   ) -> NodeOutput:
       """Custom filter function"""

       # Get data
       rewards = batch.batch["rewards"]

       # Create filter mask
       mask = rewards > threshold  # Boolean tensor

       # Apply filter
       filtered_batch = filter_tensordict(batch, mask)

       # Collect metrics
       metrics = {
           "filter/kept_ratio": mask.sum().item() / len(mask)
       }

       return NodeOutput(batch=filtered_batch, metrics=metrics)

Example: Reward Threshold Filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def reward_threshold_filter(
       config: SiiRLArguments,
       batch,
       **kwargs
   ) -> NodeOutput:
       """Filter samples below reward threshold"""

       rewards = batch.batch["rewards"]
       threshold = config.algorithm.filter_threshold

       # Create mask
       mask = rewards > threshold

       # Apply filter
       from siirl.data_coordinator.sample import filter_tensordict
       filtered_batch = filter_tensordict(batch, mask)

       # Metrics
       metrics = {
           "filter/kept_ratio": mask.sum().item() / len(mask),
           "filter/threshold": threshold
       }

       return NodeOutput(batch=filtered_batch, metrics=metrics)

**Configuration:**

.. code-block:: bash

   python -m siirl.main_dag \
     algorithm.filter_threshold=0.5

**Usage in Pipeline:**

.. code-block:: python

   pipeline.add_node(
       "reward_filter",
       func="my_module:reward_threshold_filter",
       deps=["function_reward"],
       node_type=NodeType.COMPUTE,
       node_role=NodeRole.DYNAMIC_SAMPLING
   )

Example: Group Variance Filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from collections import defaultdict

   def group_variance_filter(
       config: SiiRLArguments,
       batch,
       **kwargs
   ) -> NodeOutput:
       """Filter groups with low variance"""

       rewards = batch.batch["rewards"]
       uids = batch.batch["uid"]

       # Group by uid
       uid_to_rewards = defaultdict(list)
       for i, uid in enumerate(uids):
           uid_key = int(uid) if hasattr(uid, 'item') else uid
           uid_to_rewards[uid_key].append(rewards[i].item())

       # Calculate std for each group
       min_std = config.algorithm.min_group_std
       kept_uids = {
           uid for uid, r in uid_to_rewards.items()
           if torch.std(torch.tensor(r)).item() >= min_std
       }

       # Create mask
       mask = torch.tensor([
           (int(uids[i]) if hasattr(uids[i], 'item') else uids[i]) in kept_uids
           for i in range(len(uids))
       ], dtype=torch.bool)

       # Apply filter
       from siirl.data_coordinator.sample import filter_tensordict
       filtered_batch = filter_tensordict(batch, mask)

       metrics = {
           "filter/kept_groups": len(kept_uids),
           "filter/total_groups": len(uid_to_rewards)
       }

       return NodeOutput(batch=filtered_batch, metrics=metrics)
