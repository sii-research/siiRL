============
Pipeline API
============

Pipeline is a declarative Python API for defining training workflows. Each Pipeline consists of Nodes connected through dependencies to form a DAG.

Architecture Overview
---------------------

::

                            Pipeline Architecture
   ==============================================================================

   +------------------+                      +------------------+
   |    Pipeline      |     .build()         |   TaskGraph      |
   |    (Builder)     | ------------------> |     (DAG)        |
   +------------------+                      +------------------+
   | - pipeline_id    |                      | - graph_id       |
   | - description    |                      | - nodes: Dict    |
   | - _nodes: Dict   |                      | - adj: Dict      |
   +------------------+                      | - rev_adj: Dict  |
                                             +------------------+
                                                     |
                                                     | executed by
                                                     v
                                             +------------------+
                                             |   DAGWorker      |
                                             |   (per GPU)      |
                                             +------------------+

   ==============================================================================

   Built-in Pipelines Comparison:

   +----------+------------------------------------------------------------------+
   | Pipeline | Nodes Flow                                                       |
   +----------+------------------------------------------------------------------+
   | GRPO     | rollout -> reward -> advantage -> old_log -> ref_log -> train    |
   +----------+------------------------------------------------------------------+
   | PPO      | rollout -> reward -> value -> advantage -> old_log -> ref_log    |
   |          |         -> train_actor -> train_critic                           |
   +----------+------------------------------------------------------------------+
   | DAPO     | rollout -> reward -> dynamic_sampling -> advantage -> old_log    |
   |          |         -> ref_log -> train                                      |
   +----------+------------------------------------------------------------------+
   | Embodied | rollout -> embodied_sampling -> reward -> advantage -> old_log   |
   | SRPO     |         -> ref_log -> train                                      |
   +----------+------------------------------------------------------------------+

Basic Usage
-----------

Creating a Pipeline
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from siirl.execution.dag.pipeline import Pipeline
   from siirl.execution.dag.node import NodeType, NodeRole

   pipeline = Pipeline("my_pipeline", "Description")

   # Add nodes (supports chaining)
   pipeline.add_node(
       "node_id",
       func="module:function",  # or "module:Class.method"
       deps=["dependency_node_ids"],
       node_type=NodeType.COMPUTE,
       node_role=NodeRole.DEFAULT
   ).add_node(
       "next_node",
       func="module:another_function",
       deps=["node_id"],
       node_type=NodeType.MODEL_TRAIN,
       node_role=NodeRole.ACTOR
   )

   # Build TaskGraph
   task_graph = pipeline.build()

Node Parameters
~~~~~~~~~~~~~~~

- ``node_id``: Unique identifier
- ``func``: Function path (``"module:function"`` or ``"module:Class.method"``)
- ``deps``: List of dependency node IDs
- ``node_type``: MODEL_INFERENCE / MODEL_TRAIN / COMPUTE / DATA_LOAD
- ``node_role``: ROLLOUT / ACTOR / CRITIC / REFERENCE / REWARD / ADVANTAGE / DYNAMIC_SAMPLING / DEFAULT
- ``only_forward_compute``: Forward only (default False)

Built-in Pipelines
------------------

siiRL provides 4 built-in pipelines in ``siirl/execution/dag/builtin_pipelines.py``:

GRPO Pipeline
~~~~~~~~~~~~~

**Workflow:** rollout → reward → advantage → old_log_prob → ref_log_prob → train_actor

**Usage:**

.. code-block:: bash

   python -m siirl.main_dag \
     algorithm.adv_estimator=grpo

PPO Pipeline
~~~~~~~~~~~~

**Workflow:** rollout → reward → critic_value → advantage → old_log_prob → ref_log_prob → train_actor → train_critic

**Key Difference:** Adds value function and critic training

**Usage:**

.. code-block:: bash

   python -m siirl.main_dag \
     algorithm.adv_estimator=gae \
     critic.enable=true

DAPO Pipeline
~~~~~~~~~~~~~

**Workflow:** rollout → reward → dynamic_sampling → advantage → old_log_prob → ref_log_prob → train_actor

**Key Feature:** Filters zero-variance sample groups

**Usage:**

.. code-block:: bash

   python -m siirl.main_dag \
     algorithm.workflow_type=DAPO \
     algorithm.filter_groups.enable=true

Embodied GRPO Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

**Workflow:** rollout → embodied_sampling → reward → advantage → old_log_prob → ref_log_prob → train_actor

**Key Feature:** Embodied AI specific filtering

**Usage:**

.. code-block:: bash

   python -m siirl.main_dag \
     algorithm.workflow_type=EMBODIED

Custom Pipeline Definition
---------------------------

Define Custom Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from siirl.execution.dag.pipeline import Pipeline
   from siirl.execution.dag.task_graph import TaskGraph
   from siirl.execution.dag.node import NodeType, NodeRole

   def my_custom_pipeline() -> TaskGraph:
       pipeline = Pipeline("my_pipeline", "My workflow")

       pipeline.add_node(
           "rollout_actor",
           func="siirl.dag_worker.dagworker:DAGWorker.generate",
           deps=[],
           node_type=NodeType.MODEL_INFERENCE,
           node_role=NodeRole.ROLLOUT
       ).add_node(
           "my_custom_node",
           func="my_module:my_function",
           deps=["rollout_actor"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.DEFAULT
       )

       return pipeline.build()

Custom Node Function
~~~~~~~~~~~~~~~~~~~~

Node functions must follow this signature:

.. code-block:: python

   from siirl.dag_worker.data_structures import NodeOutput

   def my_function(batch, config=None, **kwargs) -> NodeOutput:
       """
       Args:
           batch: Input data (TensorDict)
           config: Global configuration
           **kwargs: Additional arguments

       Returns:
           NodeOutput(batch=processed_batch, metrics={})
       """
       # Process batch
       processed_batch = process(batch)

       # Collect metrics
       metrics = {"metric_name": value}

       return NodeOutput(batch=processed_batch, metrics=metrics)

Use Custom Pipeline
~~~~~~~~~~~~~~~~~~~

**Command Line:**

.. code-block:: bash

   python -m siirl.main_dag \
     dag.custom_pipeline_fn="my_module:my_custom_pipeline"


