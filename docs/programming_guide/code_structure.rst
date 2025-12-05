===============
Code Structure
===============

This document describes the code structure and architecture of siiRL.

Directory Structure
-------------------

.. code-block:: text

   siirl/
   ├── main_dag.py                   # Main entry point
   ├── dag_worker/                   # DAG Worker implementation
   ├── execution/                    # Execution engine
   ├── engine/                       # Model engine
   ├── data_coordinator/             # Data coordination
   ├── params/                       # Configuration parameters
   ├── environment/                  # Environment abstraction
   └── user_interface/               # User interface

Core Modules
------------

dag_worker/
~~~~~~~~~~~

DAG execution unit, one worker per GPU.

.. code-block:: text

   dag_worker/
   ├── dagworker.py              # Core Worker class (~1320 lines)
   ├── core_algos.py             # RL algorithm implementations
   ├── dag_utils.py              # Utility functions
   ├── checkpoint_manager.py     # Checkpoint management
   ├── metrics_collector.py      # Metrics collection
   ├── metric_aggregator.py      # Metrics aggregation
   ├── validator.py              # Validation logic
   ├── constants.py              # Constants
   └── data_structures.py        # Data structures

**Responsibilities:**

- Execute TaskGraph nodes
- Manage model Workers (Actor/Critic/Rollout/Reference/Reward)
- Data flow and caching
- Metrics collection and reporting
- Checkpoint saving and loading

execution/
~~~~~~~~~~

Execution engine for DAG definition, scheduling, and metrics aggregation.

.. code-block:: text

   execution/
   ├── dag/                      # DAG definition
   │   ├── task_graph.py         # TaskGraph class
   │   ├── node.py               # Node class
   │   ├── builtin_pipelines.py  # Built-in Pipelines
   │   ├── pipeline.py           # Pipeline Builder API
   │   ├── config_loader.py      # Configuration loader
   │   └── task_loader.py        # Task loader
   ├── scheduler/                # Task scheduling
   │   ├── task_scheduler.py     # Task scheduler
   │   ├── launch.py             # Ray launcher
   │   ├── process_group_manager.py  # Process group manager
   │   ├── graph_updater.py      # Graph updater
   │   ├── reward.py             # Reward scheduler
   │   ├── enums.py              # Enum definitions
   │   └── resource_manager.py   # Resource manager
   ├── metric_worker/            # Distributed metrics aggregation
   │   ├── metric_worker.py      # MetricWorker
   │   └── utils.py
   └── rollout_flow/             # Rollout flow
       ├── multi_agent/          # Multi-agent support
       └── multiturn/            # Multi-turn interaction

**Responsibilities:**

- DAG definition and validation
- Task scheduling and resource allocation
- Distributed metrics collection
- Multi-agent/multi-turn interaction flow

engine/
~~~~~~~

Model execution engine containing all model workers.

.. code-block:: text

   engine/
   ├── actor/                    # Actor models
   │   ├── base.py
   │   ├── dp_actor.py           # FSDP Actor
   │   ├── megatron_actor.py     # Megatron Actor
   │   └── embodied_actor.py     # Embodied Actor
   ├── critic/                   # Critic models
   │   ├── base.py
   │   ├── dp_critic.py
   │   └── megatron_critic.py
   ├── rollout/                  # Rollout engine
   │   ├── base.py
   │   ├── vllm_rollout/         # vLLM backend
   │   ├── sglang_rollout/       # SGLang backend
   │   ├── hf_rollout.py         # HuggingFace backend
   │   └── embodied_rollout.py   # Embodied Rollout
   ├── reward_model/             # Reward models
   ├── reward_manager/           # Reward managers
   │   ├── naive.py              # Simple reward
   │   ├── batch.py              # Batch Reward Model
   │   ├── parallel.py           # Parallel Reward Model
   │   ├── dapo.py               # DAPO Reward
   │   └── embodied.py           # Embodied Reward
   ├── sharding_manager/         # Sharding management
   ├── base_worker/              # Worker base classes
   ├── fsdp_workers.py           # FSDP Worker
   └── megatron_workers.py       # Megatron Worker

**Responsibilities:**

- Training and inference for Actor/Critic/Rollout/Reference/Reward models
- Support for FSDP and Megatron backends
- Support for vLLM/SGLang/HuggingFace inference backends

data_coordinator/
~~~~~~~~~~~~~~~~~

Data coordinator for distributed data management.

.. code-block:: text

   data_coordinator/
   ├── data_buffer.py            # Distributed data buffer
   ├── dataloader/               # Data loading
   │   ├── data_loader_node.py
   │   ├── partitioned_dataset.py
   │   ├── embodied_preprocess.py
   │   └── vision_utils.py
   ├── protocol.py               # Data protocol
   └── sample.py                 # Sampling logic

**Responsibilities:**

- Distributed data buffering (per-server)
- Data loading (per-GPU)
- Data redistribution and load balancing

params/
~~~~~~~

Parameter configuration using Hydra.

.. code-block:: text

   params/
   ├── __init__.py               # SiiRLArguments
   ├── parser.py                 # Configuration parser
   ├── data_args.py              # Data parameters
   ├── model_args.py             # Model parameters
   ├── training_args.py          # Training parameters
   ├── dag_args.py               # DAG parameters
   ├── embodied_args.py          # Embodied parameters
   └── profiler_args.py          # Profiler parameters

environment/
~~~~~~~~~~~~

Environment abstraction for Embodied AI and multi-agent systems.

.. code-block:: text

   environment/
   └── embodied/
       ├── base.py               # Environment base class
       ├── venv.py               # Vectorized environment
       └── adapters/             # Environment adapters
           └── libero.py         # Libero adapter

user_interface/
~~~~~~~~~~~~~~~

User-defined interfaces.

.. code-block:: text

   user_interface/
   ├── filter_interface/
   │   ├── dapo.py               # DAPO dynamic sampling
   │   └── embodied.py           # Embodied data filtering
   └── rewards_interface/
       └── custom_gsm8k_reward.py  # Custom reward example

**Purpose:** Provides interfaces for user-defined node functions.

Data Structures
---------------

NodeOutput
~~~~~~~~~~

Return value from node execution.

.. code-block:: python

   @dataclass
   class NodeOutput:
       batch: Any              # Data batch
       metrics: Dict = None    # Metrics
       info: Dict = None       # Additional info

Node
~~~~

DAG node definition.

.. code-block:: python

   @dataclass
   class Node:
       node_id: str                    # Node ID
       node_type: NodeType             # Node type
       node_role: NodeRole             # Node role
       dependencies: List[str]         # Dependency nodes
       executable: Callable            # Executable function
       executable_ref: str             # Function path
       only_forward_compute: bool      # Forward only

Enumerations
~~~~~~~~~~~~

**NodeType:**

.. code-block:: python

   class NodeType(Enum):
       MODEL_INFERENCE = "model_inference"
       MODEL_TRAIN = "model_train"
       COMPUTE = "compute"
       DATA_LOAD = "data_load"

**NodeRole:**

.. code-block:: python

   class NodeRole(Enum):
       ROLLOUT = "rollout"
       ACTOR = "actor"
       CRITIC = "critic"
       REFERENCE = "reference"
       REWARD = "reward"
       ADVANTAGE = "advantage"
       DYNAMIC_SAMPLING = "dynamic_sampling"
       DEFAULT = "default"

**AdvantageEstimator:**

.. code-block:: python

   class AdvantageEstimator(Enum):
       GRPO = "grpo"
       GAE = "gae"
       CPGD = "cpgd"
       GSPO = "gspo"

**WorkflowType:**

.. code-block:: python

   class WorkflowType(Enum):
       DEFAULT = "DEFAULT"
       DAPO = "DAPO"
       EMBODIED = "EMBODIED"

Execution Flow
--------------

Startup Flow (main_dag.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   1. Parse configuration (parse_config)
   2. Load Pipeline (load_pipeline)
   3. Initialize DataBuffer (init_data_coordinator)
   4. Initialize MetricWorker
   5. Task scheduling (TaskScheduler)
   6. Launch Ray cluster (RayTrainer)
   7. Create DAGWorker (one per GPU)
   8. Execute training (DAGWorker.execute_task_graph)

DAGWorker Execution Flow
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   1. Initialize Workers (Actor/Critic/Rollout/Reference/Reward)
   2. Initialize DataLoader
   3. Initialize Validator
   4. Load Checkpoint (if exists)
   5. Training loop:
      - Load data
      - Execute nodes in topological order
      - Collect metrics
      - Save Checkpoint
      - Validate (if needed)

Node Execution Flow
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   1. DAGWorker gets node's executable function
   2. Call function with current batch
   3. Function processes data, returns NodeOutput
   4. Update batch, pass to next node
   5. Collect node metrics

Key Concepts
------------

TaskGraph
~~~~~~~~~

Directed Acyclic Graph representing training workflow.

**Core Methods:**

- ``add_node()``: Add node
- ``build_adjacency_lists()``: Build adjacency lists
- ``validate_graph()``: Validate DAG
- ``get_execution_order()``: Get topological sort

Pipeline
~~~~~~~~

Declarative API for building TaskGraph.

**Core Methods:**

- ``add_node()``: Add node (supports chaining)
- ``build()``: Build and validate TaskGraph

DAGWorker Class
~~~~~~~~~~~~~~~

Execution unit per GPU.

**Core Methods:**

- ``generate()``: Rollout generation
- ``compute_reward()``: Compute reward
- ``compute_advantage()``: Compute advantage
- ``compute_old_log_prob()``: Old policy log prob
- ``compute_ref_log_prob()``: Reference model log prob
- ``compute_value()``: Value function (PPO)
- ``train_actor()``: Train actor
- ``train_critic()``: Train critic (PPO)

Configuration Parameters
------------------------

Main Configuration Groups
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   algorithm:
     adv_estimator: grpo  # grpo/gae/cpgd/gspo
     workflow_type: DEFAULT  # DEFAULT/DAPO/EMBODIED

   data:
     train_files: /path/to/train.parquet
     train_batch_size: 512
     max_prompt_length: 2048
     max_response_length: 4096

   actor_rollout_ref:
     model:
       path: /path/to/model
     actor:
       optim:
         lr: 1e-6
       ppo_mini_batch_size: 256
     rollout:
       name: vllm  # vllm/sglang/hf
       tensor_model_parallel_size: 2
       n: 8  # GRPO group size

   trainer:
     n_gpus_per_node: 8
     nnodes: 1
     total_epochs: 30
     save_freq: 10

   dag:
     custom_pipeline_fn: null  # Custom Pipeline

Extension Points
----------------

Custom Pipeline
~~~~~~~~~~~~~~~

Add new functions in ``siirl/execution/dag/builtin_pipelines.py``.

Custom Node Functions
~~~~~~~~~~~~~~~~~~~~~

Implement functions following the signature:

.. code-block:: python

   def my_node(batch, config=None, **kwargs) -> NodeOutput:
       return NodeOutput(batch=batch, metrics={})

Custom Reward Manager
~~~~~~~~~~~~~~~~~~~~~

Add new classes in ``siirl/engine/reward_manager/``.

Custom Environment
~~~~~~~~~~~~~~~~~~

Add new environment classes in ``siirl/environment/``.
