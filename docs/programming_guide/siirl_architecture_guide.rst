=======================================
siiRL Complete Architecture Guide
=======================================

.. note::
   **Target Audience**: This document assumes no prior knowledge of siiRL, but expects basic familiarity with Python, PyTorch, and reinforcement learning concepts.
   We will systematically explain siiRL's design philosophy, architecture implementation, and core algorithms from the ground up.

Table of Contents
=================

- :ref:`sec1_overview`
- :ref:`sec2_design_philosophy`
- :ref:`sec3_main_entry`
- :ref:`sec4_dag_planner`
- :ref:`sec5_dag_worker`
- :ref:`sec6_data_coordinator`
- :ref:`sec7_engine`
- :ref:`sec8_core_algorithms`
- :ref:`sec9_execution_flow`
- :ref:`sec10_configuration`
- :ref:`sec11_extension_guide`

----

.. _sec1_overview:

1. siiRL Architecture Overview
==============================

1.1 What is siiRL?
------------------

**siiRL** (Shanghai Innovation Institute RL Framework) is a novel **fully distributed reinforcement learning framework** designed to break the scaling barriers in LLM post-training. By eliminating the centralized controller common in other frameworks, siiRL achieves:

- **Near-Linear Scalability**: The multi-controller paradigm eliminates central bottlenecks by distributing control logic and data management across all workers
- **SOTA Throughput**: Fully distributed dataflow architecture minimizes communication and I/O overhead
- **Flexible DAG-Defined Pipeline**: Decouples algorithmic logic from physical hardware, enabling rapid experimentation

1.2 System Architecture and Data Flow
-------------------------------------

**System Architecture Diagram**:

.. figure:: https://github.com/sii-research/siiRL/raw/main/asset/overview.png
   :width: 100%
   :alt: siiRL Architecture Overview
   :align: center
   
   **Figure 1.1**: siiRL System Architecture showing the three core components: DAG Planner, DAG Workers, and Data Coordinator

**Complete Training Step Sequence Diagram**:

The following sequence diagram shows the complete data flow for a single GRPO training step:

::

      User          MainRunner       DAGWorker      DataCoordinator     Engine
     (YAML)         (Planner)       (per GPU)        (Singleton)       Workers
        |               |               |                 |               |
   ============================================================================
   | INITIALIZATION PHASE                                                     |
   ============================================================================
        |               |               |                 |               |
        | 1. Config     |               |                 |               |
        |-------------->|               |                 |               |
        |               |               |                 |               |
        |               | 2. load_pipeline() + TaskScheduler.schedule()   |
        |               |------------------------------------------------>|
        |               |               |                 |               |
        |               | 3. Create DAGWorkers (one per GPU)              |
        |               |-------------->|                 |               |
        |               |               |                 |               |
        |               |               | 4. init_graph() |               |
        |               |               |    Load models  |               |
        |               |               |-------------------------------->|
        |               |               |                 |               |
   ============================================================================
   | TRAINING LOOP (per step)                                                 |
   ============================================================================
        |               |               |                 |               |
        |               |               | 5. DataLoader   |               |
        |               |               |    .run()       |               |
        |               |               |<----------------|               |
        |               |               | batch (prompts) |               |
        |               |               |                 |               |
        |               |               | 6. Node: rollout_actor          |
        |               |               |-------------------------------->|
        |               |               |     Rollout.generate_sequences()|
        |               |               |<--------------------------------|
        |               |               | batch + responses               |
        |               |               |                 |               |
        |               |               | 7. Node: function_reward        |
        |               |               |    compute_reward()             |
        |               |               |---------------->|               |
        |               |               | batch + scores  |               |
        |               |               |                 |               |
        |               |               | 8. Node: calculate_advantages   |
        |               |               |    compute_advantage()          |
        |               |               |    (GRPO group normalization)   |
        |               |               |                 |               |
        |               |               | 9. put_data_to_buffers()        |
        |               |               |    (if DP size changes)         |
        |               |               |---------------->|               |
        |               |               |                 | ray.put()     |
        |               |               |                 |               |
        |               |               | 10. get_data_from_buffers()     |
        |               |               |<----------------|               |
        |               |               | redistributed batch             |
        |               |               |                 |               |
        |               |               | 11. Node: actor_old_log_prob    |
        |               |               |-------------------------------->|
        |               |               |     Actor.compute_log_prob()    |
        |               |               |<--------------------------------|
        |               |               | batch + old_log_probs           |
        |               |               |                 |               |
        |               |               | 12. Node: reference_log_prob    |
        |               |               |-------------------------------->|
        |               |               |   Reference.compute_ref_log_prob|
        |               |               |<--------------------------------|
        |               |               | batch + ref_log_probs           |
        |               |               |                 |               |
        |               |               | 13. Node: actor_train           |
        |               |               |-------------------------------->|
        |               |               |     Actor.update_actor()        |
        |               |               |     - Forward pass              |
        |               |               |     - Compute policy loss       |
        |               |               |     - Backward pass             |
        |               |               |     - Optimizer step            |
        |               |               |<--------------------------------|
        |               |               | metrics                         |
        |               |               |                 |               |
        |               |               | 14. sync_weights_actor_to_rollout
        |               |               |-------------------------------->|
        |               |               |     ShardingManager.sync()      |
        |               |               |                 |               |
        |               |               | 15. Log metrics + checkpoint    |
        |               |               |                 |               |
   ============================================================================
   | REPEAT for next training step                                            |
   ============================================================================

**Data Flow Summary**:

::

                              GRPO Single Step Data Flow
   ==============================================================================
   
   DataLoader
       |
       | batch: {prompts, attention_mask, index}
       v
   +---------------------+
   | rollout_actor       | DAGWorker.generate()
   | (MODEL_INFERENCE)   | -> Rollout.generate_sequences()
   +----------+----------+
              | + {responses, response_ids, response_mask}
              v
   +---------------------+
   | function_reward     | DAGWorker.compute_reward()
   | (COMPUTE)           | -> RewardManager.compute_reward()
   +----------+----------+
              | + {token_level_scores, token_level_rewards}
              v
   +---------------------+
   | calculate_advantages| DAGWorker.compute_advantage()
   | (COMPUTE)           | -> compute_grpo_outcome_advantage()
   +----------+----------+ Group by prompt -> Normalize (score - mean)/std
              | + {advantages}
              v
   +---------------------+
   | actor_old_log_prob  | DAGWorker.compute_old_log_prob()
   | (MODEL_TRAIN)       | -> Actor.compute_log_prob()
   | only_forward=True   |
   +----------+----------+
              | + {old_log_probs}
              v
   +---------------------+
   | reference_log_prob  | DAGWorker.compute_ref_log_prob()
   | (MODEL_TRAIN)       | -> Reference.compute_ref_log_prob()
   +----------+----------+
              | + {ref_log_prob}
              v
   +---------------------+
   | actor_train         | DAGWorker.train_actor()
   | (MODEL_TRAIN)       | -> Actor.update_actor()
   +----------+----------+ policy_loss = -advantages * clip(ratio)
              |
              | metrics: {loss, clipfrac, kl, lr, ...}
              v
   +---------------------+
   | sync_weights        | ShardingManager.sync_weights_actor_to_rollout()
   +---------------------+                                            

1.3 Core Component Responsibilities
-----------------------------------

.. list-table:: siiRL Core Components
   :header-rows: 1
   :widths: 20 20 60

   * - Component
     - Process/Actor
     - Core Responsibilities
   * - **DAG Planner**
     - MainRunner Actor
     - Parse user-defined DAG workflows, generate execution plans, assign tasks to workers
   * - **DAG Worker**
     - One Actor per GPU
     - Core execution unit responsible for model initialization, task execution, data flow management
   * - **Data Coordinator**
     - Global Singleton Actor
     - Manage distributed data lifecycle including data loading and intermediate data redistribution
   * - **TaskScheduler**
     - Inside MainRunner
     - Split and assign TaskGraph to each DAG Worker
   * - **ProcessGroupManager**
     - Inside MainRunner
     - Manage creation and configuration of distributed communication groups (TP/PP/DP)
   * - **MetricWorker**
     - Standalone Actor
     - Distributed metrics collection and aggregation

1.4 Why is siiRL Different?
---------------------------

**Problems with Traditional Frameworks**:

1. **Single-Controller Bottleneck**: All data flows through a single node, causing I/O and communication overhead
2. **Rigid Algorithm Pipelines**: Modifying workflows requires deep modifications to framework source code

**siiRL's Solutions**:

.. list-table:: siiRL Design Advantages
   :header-rows: 1
   :widths: 25 35 40

   * - Traditional Frameworks
     - siiRL DistFlow
     - Advantage
   * - Centralized Controller
     - Multi-Controller Paradigm
     - Eliminates single-point bottleneck, near-linear scaling
   * - Hard-coded Workflows
     - DAG-Defined Pipeline
     - Declarative configuration, no code modification needed
   * - Centralized Data Management
     - Distributed Data Coordinator
     - Avoids OOM, parallelizes data loading

----

.. _sec2_design_philosophy:

2. DistFlow Design Philosophy
=============================

2.1 Fully Distributed Architecture
----------------------------------

The core idea of DistFlow is **"no central coordinator"**. Each DAG Worker is an independent execution unit with its own:

- Data loader (partitioned dataset)
- Model instances (Actor/Critic/Rollout/Reference/Reward)
- Task execution graph (subgraph of TaskGraph)
- Local data cache

2.2 Three-Layer Architecture Design
-----------------------------------

::

   ┌─────────────────────────────────────────────────────────────────┐
   │                     User Configuration Layer (YAML/Python)      │
   │   - workflow_grpo.yaml: Define algorithm DAG                    │
   │   - config.yaml: Model, data, training parameters               │
   └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                     Execution Scheduling Layer (DAG Planner)     │
   │   - TaskScheduler: Task assignment                              │
   │   - ProcessGroupManager: Communication group management          │
   │   - GraphUpdater: Configuration injection                       │
   └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                     Distributed Execution Layer (DAG Workers)    │
   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
   │   │Worker 0  │  │Worker 1  │  │Worker 2  │  │Worker N  │       │
   │   │ (GPU 0)  │  │ (GPU 1)  │  │ (GPU 2)  │  │ (GPU N)  │       │
   │   └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
   └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                     Data Coordination Layer (Data Coordinator)   │
   │   - Distributed DataLoader: Partitioned data loading            │
   │   - Distributed DataBuffer: Intermediate data redistribution    │
   └─────────────────────────────────────────────────────────────────┘

2.3 Core Design Principles
--------------------------

.. list-table:: DistFlow Design Principles
   :header-rows: 1
   :widths: 25 75

   * - Principle
     - Description
   * - **Worker Autonomy**
     - Each DAG Worker is a fully independent execution unit, not dependent on central coordination
   * - **Data Locality**
     - Data is processed locally as much as possible, reducing cross-node transfers
   * - **Declarative Workflows**
     - Algorithm logic is declared via DAG, decoupled from execution engine
   * - **Unified Sample Protocol**
     - All intermediate data uses Sample/SampleInfo protocol, supporting flexible routing
   * - **Late Binding**
     - Configuration is injected into nodes at runtime, supporting dynamic adjustment

----

.. _sec3_main_entry:

3. Program Entry and Startup Flow
=================================

3.1 main_dag.py Explained
-------------------------

``main_dag.py`` is the entry point of siiRL, but unlike traditional frameworks, its role is a **launcher** rather than an executor.

.. code-block:: python
   :caption: siirl/main_dag.py Core Structure

   def main() -> None:
       """Main entry: Initialize Ray cluster, parse config, start MainRunner"""
       
       # 1. Initialize Ray cluster
       if not ray.is_initialized():
           ray.init(runtime_env={"env_vars": RAY_RUNTIME_ENV_VARS})
       
       # 2. Parse configuration
       siirl_args = parse_config()
       
       # 3. Start main orchestration Actor
       runner = MainRunner.remote()
       ray.get(runner.run.remote(siirl_args))

3.2 MainRunner Actor
--------------------

``MainRunner`` is the "brain" of the system, responsible for orchestrating the entire training workflow:

.. code-block:: python
   :caption: MainRunner.run() Core Flow

   @ray.remote(num_cpus=MAIN_RUNNER_CPU_RESERVATION)
   class MainRunner:
       def run(self, siirl_args: SiiRLArguments) -> None:
           # 1. Initialize DataCoordinator
           data_coordinator_handle = init_data_coordinator(
               num_buffers=siirl_args.trainer.nnodes,
               ppo_mini_batch_size=siirl_args.actor_rollout_ref.actor.ppo_mini_batch_size,
               world_size=siirl_args.trainer.nnodes * siirl_args.trainer.n_gpus_per_node
           )
           
           # 2. Load and configure workflow DAG
           workflow_taskgraph = load_pipeline(siirl_args)
           update_task_graph_node_configs(workflow_taskgraph, siirl_args)
           
           # 3. Schedule tasks to each worker
           task_scheduler = TaskScheduler(siirl_args.trainer.nnodes, 
                                          siirl_args.trainer.n_gpus_per_node)
           rank_taskgraph_mapping = task_scheduler.schedule_and_assign_tasks([workflow_taskgraph])
           
           # 4. Create process groups
           process_group_manager = ProcessGroupManager(total_workers, rank_taskgraph_mapping)
           
           # 5. Create metric worker
           metric_worker_handle = MetricWorker.remote()
           
           # 6. Initialize and start DAG Workers
           trainer = RayTrainer(config=siirl_args, ...)
           trainer.init_workers()
           trainer.start_workers()

3.3 Startup Flow Sequence Diagram
---------------------------------

::

   main()
      │
      ├── ray.init()                          ← Initialize Ray cluster
      │
      ├── parse_config()                      ← Parse YAML configuration
      │
      └── MainRunner.run()
              │
              ├── init_data_coordinator()     ← Create global DataCoordinator
              │
              ├── load_pipeline()             ← Load DAG definition
              │       │
              │       └── grpo_pipeline()     ← Return TaskGraph
              │
              ├── TaskScheduler.schedule()    ← Assign tasks to each rank
              │
              ├── ProcessGroupManager()       ← Create communication group specs
              │
              ├── RayTrainer.init_workers()   ← Create DAG Worker Actors
              │       │
              │       └── DAGWorker.__init__() × N_workers
              │
              └── RayTrainer.start_workers()  ← Start training loop
                      │
                      └── DAGWorker.execute_task_graph() × N_workers

----

.. _sec4_dag_planner:

4. DAG Planner Deep Dive
========================

The DAG Planner is siiRL's "scheduling brain", responsible for converting user-defined high-level workflows into executable distributed tasks.

**Pipeline Architecture Overview**:

The following diagram shows how the core data structures relate to each other and how a Pipeline is built and executed:

::

                           Pipeline Data Structure Relationships
   ==============================================================================
   
                                 +------------------+
                                 |    Pipeline      |
                                 |    (Builder)     |
                                 +------------------+
                                 | - pipeline_id    |
                                 | - description    |
                                 | - _nodes: Dict   |
                                 +--------+---------+
                                          |
                                          | .build()
                                          v
                                 +------------------+
                                 |   TaskGraph      |
                                 |     (DAG)        |
                                 +------------------+
                                 | - graph_id       |
                                 | - nodes: Dict    |
                                 | - adj: Dict      |
                                 | - rev_adj: Dict  |
                                 +--------+---------+
                                          |
                                          | contains multiple
                                          v
         +----------------+    +----------------+    +----------------+
         |     Node       |    |     Node       |    |     Node       |  ...
         +----------------+    +----------------+    +----------------+
         | - node_id      |    | - node_id      |    | - node_id      |
         | - node_type    |    | - node_type    |    | - node_type    |
         | - node_role    |    | - node_role    |    | - node_role    |
         | - dependencies |    | - dependencies |    | - dependencies |
         | - executable   |    | - executable   |    | - executable   |
         | - config       |    | - config       |    | - config       |
         +----------------+    +----------------+    +----------------+
   
   ==============================================================================
   
   NodeType (from node.py)             NodeRole (from node.py)
   +------------------------+          +------------------------+
   | COMPUTE                |          | DEFAULT                |
   | DATA_LOAD              |          | ACTOR                  |
   | ENV_INTERACT           |          | ADVANTAGE              |
   | MODEL_INFERENCE        |          | CRITIC                 |
   | MODEL_TRAIN            |          | ROLLOUT                |
   | PUT_TO_BUFFER          |          | REFERENCE              |
   | GET_FROM_BUFFER        |          | REWARD                 |
   | BARRIER_SYNC           |          | DYNAMIC_SAMPLING       |
   | CUSTOM                 |          +------------------------+
   +------------------------+

**Pipeline Building Flow**:

::

                            How Pipeline is Built and Executed
   ================================================================================
   
   Step 1: User Defines Pipeline (Python Code)
   --------------------------------------------
   
       pipeline = Pipeline("grpo_training_pipeline")
       
       pipeline.add_node("rollout_actor", func="...:DAGWorker.generate", deps=[])
              .add_node("function_reward", func="...:DAGWorker.compute_reward", ...)
              .add_node("calculate_advantages", func="...:DAGWorker.compute_advantage", ...)
              .add_node("actor_old_log_prob", func="...:DAGWorker.compute_old_log_prob", ...)
              .add_node("reference_log_prob", func="...:DAGWorker.compute_ref_log_prob", ...)
              .add_node("actor_train", func="...:DAGWorker.train_actor", ...)
   
                                            |
                                            | pipeline.build()
                                            v
   
   Step 2: Build TaskGraph (Validation + Adjacency Lists)
   ------------------------------------------------------
   
       TaskGraph                          Adjacency Lists (adj)
       +--------------------+             +------------------------------------------+
       | graph_id: "grpo.." |             | rollout_actor      -> [function_reward]  |
       |                    |             | function_reward    -> [calculate_adv.]   |
       | nodes: {           |             | calculate_adv.     -> [actor_old_log]    |
       |   "rollout_actor", |             | actor_old_log      -> [reference_log]    |
       |   "function_reward"|             | reference_log      -> [actor_train]      |
       |   "calculate_adv.",|             | actor_train        -> []                 |
       |   ...              |             +------------------------------------------+
       | }                  |
       +--------------------+
                                            |
                                            | TaskScheduler.schedule()
                                            v
   
   Step 3: TaskScheduler Assigns to Workers
   ----------------------------------------
   
       +------------------------------------------------------------------------+
       |  TaskScheduler                                                         |
       |                                                                        |
       |  Input: TaskGraph + num_workers                                        |
       |                                                                        |
       |  1. discover_and_split_parallel_paths(graph) -> Split parallel branches|
       |  2. Apportion workers to subgraphs (param_aware / even)                |
       |  3. Assign each worker a TaskGraph copy                                |
       |                                                                        |
       |  Output: Dict[rank, TaskGraph] (rank_taskgraph_mapping)                |
       +------------------------------------------------------------------------+
   
                          +-------------------------------------------+
                          |           rank_taskgraph_mapping          |
                          +-------------------------------------------+
                          |  rank 0  ->  TaskGraph (copy)             |
                          |  rank 1  ->  TaskGraph (copy)             |
                          |  rank 2  ->  TaskGraph (copy)             |
                          |  ...     ->  ...                          |
                          |  rank N  ->  TaskGraph (copy)             |
                          +-------------------------------------------+
                                            |
                                            | DAGWorker receives TaskGraph
                                            v
   
   Step 4: DAGWorker Executes TaskGraph
   ------------------------------------
   
       +------------------------------------------------------------------------+
       |  DAGWorker.execute_task_graph()                                        |
       |                                                                        |
       |  for each training step:                                               |
       |      1. batch = DataLoader.run()                                       |
       |      2. entry_nodes = taskgraph.get_entry_nodes()  # [rollout_actor]   |
       |      3. node_queue = entry_nodes                                       |
       |                                                                        |
       |      while node_queue:                                                 |
       |          cur_node = node_queue.pop(0)                                  |
       |                                                                        |
       |          # Execute node's function                                     |
       |          output = cur_node.run(batch=batch, _dag_worker_instance=self) |
       |                                                                        |
       |          # Resolves executable_ref to actual function:                 |
       |          # "siirl.dag_worker.dagworker:DAGWorker.generate"             |
       |          #  -> DAGWorker.generate(self, batch, ...)                    |
       |                                                                        |
       |          # Get downstream nodes and add to queue                       |
       |          next_nodes = taskgraph.get_downstream_nodes(cur_node.node_id) |
       |          node_queue.extend(next_nodes)                                 |
       |                                                                        |
       |          # If DP size changes between nodes, use DataCoordinator       |
       |          put_data_to_buffers() / get_data_from_buffers()               |
       +------------------------------------------------------------------------+

**Execution Order Example (GRPO)**:

::

                            GRPO Pipeline Execution Order
   ================================================================================
   
   Topological Order:
   
     +------------------+      +------------------+      +---------------------+
     |  rollout_actor   |----->| function_reward  |----->|calculate_advantages |
     |  (Inference)     |      |    (Compute)     |      |      (Compute)      |
     |                  |      |                  |      |                     |
     |  NodeRole:       |      |  NodeRole:       |      |  NodeRole:          |
     |  ROLLOUT         |      |  REWARD          |      |  ADVANTAGE          |
     +------------------+      +------------------+      +----------+----------+
                                                                    |
         +----------------------------------------------------------+
         |
         v
     +---------------------+      +---------------------+      +------------------+
     | actor_old_log_prob  |----->| reference_log_prob  |----->|   actor_train    |
     |   (Forward Only)    |      |   (Forward Only)    |      |     (Train)      |
     |                     |      |                     |      |                  |
     |  NodeRole: ACTOR    |      |  NodeRole: REFERENCE|      |  NodeRole: ACTOR |
     |  only_forward=True  |      |                     |      |                  |
     +---------------------+      +---------------------+      +------------------+
   
   Data flows through each node, accumulating fields in the batch:
   
     batch: {prompts}
        |
        v rollout_actor
     batch: {prompts, responses, response_ids, response_mask}
        |
        v function_reward  
     batch: {..., token_level_scores, token_level_rewards}
        |
        v calculate_advantages
     batch: {..., advantages}
        |
        v actor_old_log_prob
     batch: {..., old_log_probs}
        |
        v reference_log_prob
     batch: {..., ref_log_prob}
        |
        v actor_train
     metrics: {loss, clipfrac, kl, ...}

4.1 Pipeline API
----------------

siiRL provides a clean Pipeline API for users to define training pipelines directly in Python:

.. code-block:: python
   :caption: siirl/execution/dag/pipeline.py

   class Pipeline:
       """Declarative Pipeline Builder"""
       
       def __init__(self, pipeline_id: str, description: str = ""):
           self.pipeline_id = pipeline_id
           self._nodes: Dict[str, Dict[str, Any]] = {}
       
       def add_node(
           self,
           node_id: str,
           func: Union[str, Callable],  # Function path or direct Callable
           deps: Optional[List[str]] = None,
           **kwargs
       ) -> "Pipeline":
           """Add node with method chaining support"""
           self._nodes[node_id] = {
               "func": func,
               "deps": deps or [],
               "kwargs": kwargs
           }
           return self  # Support method chaining
       
       def build(self) -> TaskGraph:
           """Build and validate TaskGraph"""
           task_graph = TaskGraph(graph_id=self.pipeline_id)
           # ... create nodes, build adjacency lists, validate DAG
           return task_graph

4.2 Built-in Pipeline Definitions
---------------------------------

siiRL provides four built-in pipeline definitions in ``siirl/execution/dag/builtin_pipelines.py``:

**4.2.1 GRPO Pipeline (grpo_pipeline)**

Standard GRPO (Group Relative Policy Optimization) training workflow:

.. code-block:: python
   :caption: siirl/execution/dag/builtin_pipelines.py - GRPO Pipeline

   def grpo_pipeline() -> TaskGraph:
       """
       Standard GRPO (Group Relative Policy Optimization) pipeline.

       Workflow:
           1. rollout_actor: Generate sequences using the policy model
           2. function_reward: Compute rewards for generated sequences
           3. calculate_advantages: Calculate advantage estimates
           4. actor_old_log_prob: Compute log probabilities with old policy (forward only)
           5. reference_log_prob: Compute log probabilities with reference model
           6. actor_train: Train the actor model
       """
       pipeline = Pipeline("grpo_training_pipeline", "Standard GRPO workflow")

       pipeline.add_node(
           "rollout_actor",
           func="siirl.dag_worker.dagworker:DAGWorker.generate",
           deps=[],
           node_type=NodeType.MODEL_INFERENCE,
           node_role=NodeRole.ROLLOUT
       ).add_node(
           "function_reward",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
           deps=["rollout_actor"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.REWARD
       ).add_node(
           "calculate_advantages",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
           deps=["function_reward"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.ADVANTAGE
       ).add_node(
           "actor_old_log_prob",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_old_log_prob",
           deps=["calculate_advantages"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.ACTOR,
           only_forward_compute=True
       ).add_node(
           "reference_log_prob",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_ref_log_prob",
           deps=["actor_old_log_prob"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.REFERENCE
       ).add_node(
           "actor_train",
           func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
           deps=["reference_log_prob"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.ACTOR
       )

       return pipeline.build()

**4.2.2 PPO Pipeline (ppo_pipeline)**

Standard PPO with Critic model and GAE advantage estimation:

.. code-block:: python
   :caption: siirl/execution/dag/builtin_pipelines.py - PPO Pipeline

   def ppo_pipeline() -> TaskGraph:
       """
       Standard PPO (Proximal Policy Optimization) pipeline.

       Workflow:
           1. rollout_actor: Generate sequences using the policy model
           2. function_reward: Compute rewards for generated sequences
           3. compute_value: Compute value function estimates (forward only)
           4. calculate_advantages: Calculate GAE (Generalized Advantage Estimation)
           5. actor_old_log_prob: Compute log probabilities with old policy (forward only)
           6. reference_log_prob: Compute log probabilities with reference model
           7. actor_train: Train the actor model
           8. critic_train: Train the critic (value) model
       """
       pipeline = Pipeline("ppo_training_pipeline", "Standard PPO workflow")

       pipeline.add_node(
           "rollout_actor",
           func="siirl.dag_worker.dagworker:DAGWorker.generate",
           deps=[],
           node_type=NodeType.MODEL_INFERENCE,
           node_role=NodeRole.ROLLOUT
       ).add_node(
           "function_reward",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
           deps=["rollout_actor"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.REWARD
       ).add_node(
           "compute_value",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_value",
           deps=["function_reward"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.CRITIC,
           only_forward_compute=True
       ).add_node(
           "calculate_advantages",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
           deps=["compute_value"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.ADVANTAGE
       ).add_node(
           "actor_old_log_prob",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_old_log_prob",
           deps=["calculate_advantages"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.ACTOR,
           only_forward_compute=True
       ).add_node(
           "reference_log_prob",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_ref_log_prob",
           deps=["actor_old_log_prob"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.REFERENCE
       ).add_node(
           "actor_train",
           func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
           deps=["reference_log_prob"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.ACTOR
       ).add_node(
           "critic_train",
           func="siirl.dag_worker.dagworker:DAGWorker.train_critic",
           deps=["actor_train"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.CRITIC
       )

       return pipeline.build()

**4.2.3 DAPO Pipeline (dapo_pipeline)**

DAPO (Data-Augmented Policy Optimization) with dynamic sampling filtering:

.. code-block:: python
   :caption: siirl/execution/dag/builtin_pipelines.py - DAPO Pipeline

   def dapo_pipeline() -> TaskGraph:
       """
       DAPO (Data-Augmented Policy Optimization) pipeline.

       DAPO is a variant of GRPO with dynamic sampling filtering based on metric variance.
       The key difference is that after computing rewards, we filter out trajectory groups
       with zero variance (all correct or all incorrect) as they provide no learning signal.

       Workflow:
           1. rollout_actor: Generate sequences using the policy model
           2. function_reward: Compute rewards for generated sequences
           3. dynamic_sampling: DAPO-specific filtering based on metric variance
           4. calculate_advantages: Calculate advantage estimates
           5. actor_old_log_prob: Compute log probabilities with old policy (forward only)
           6. reference_log_prob: Compute log probabilities with reference model
           7. actor_train: Train the actor model
       """
       pipeline = Pipeline("dapo_training_pipeline", "DAPO workflow")

       pipeline.add_node(
           "rollout_actor",
           func="siirl.dag_worker.dagworker:DAGWorker.generate",
           deps=[],
           node_type=NodeType.MODEL_INFERENCE,
           node_role=NodeRole.ROLLOUT
       ).add_node(
           "function_reward",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
           deps=["rollout_actor"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.REWARD
       ).add_node(
           "dynamic_sampling",
           func="siirl.user_interface.filter_interface.dapo.dynamic_sampling",
           deps=["function_reward"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.DYNAMIC_SAMPLING
       ).add_node(
           "calculate_advantages",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
           deps=["dynamic_sampling"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.ADVANTAGE
       ).add_node(
           "actor_old_log_prob",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_old_log_prob",
           deps=["calculate_advantages"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.ACTOR,
           only_forward_compute=True
       ).add_node(
           "reference_log_prob",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_ref_log_prob",
           deps=["actor_old_log_prob"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.REFERENCE
       ).add_node(
           "actor_train",
           func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
           deps=["reference_log_prob"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.ACTOR
       )

       return pipeline.build()

**4.2.4 Embodied SRPO Pipeline (embodied_srpo_pipeline)**

Embodied AI SRPO training with data filtering and VJEPA-based reward computation:

.. code-block:: python
   :caption: siirl/execution/dag/builtin_pipelines.py - Embodied SRPO Pipeline

   def embodied_srpo_pipeline() -> TaskGraph:
       """
       Embodied AI GRPO training pipeline with data filtering and VJEPA-based reward computation.

       Workflow:
           1. rollout_actor: Environment rollout with embodied AI agent
           2. embodied_sampling: Data verification and filtering
           3. data_rebalance: Data rebalancing across workers (after filtering)
           4. compute_reward: VJEPA-based reward computation
           5. calculate_advantages: Calculate advantages (GRPO group-based)
           6. actor_old_log_prob: Compute old actor log probabilities (forward only)
           7. reference_log_prob: Compute reference model log probabilities
           8. actor_train: Actor training with GRPO
       """
       pipeline = Pipeline(
           "embodied_grpo_training_pipeline",
           "Embodied AI GRPO training workflow with data filtering and VJEPA-based reward computation."
       )

       pipeline.add_node(
           "rollout_actor",
           func="siirl.dag_worker.dagworker:DAGWorker.generate",
           deps=[],
           node_type=NodeType.MODEL_INFERENCE,
           node_role=NodeRole.ROLLOUT
       ).add_node(
           "dynaminc_sampling",
           func="siirl.user_interface.filter_interface.embodied.embodied_local_rank_sampling",
           deps=["rollout_actor"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.DYNAMIC_SAMPLING
       ).add_node(
           "compute_reward",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
           deps=["dynaminc_sampling"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.REWARD
       ).add_node(
           "calculate_advantages",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
           deps=["compute_reward"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.ADVANTAGE
       ).add_node(
           "actor_old_log_prob",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_old_log_prob",
           deps=["calculate_advantages"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.ACTOR,
           only_forward_compute=True
       ).add_node(
           "reference_log_prob",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_ref_log_prob",
           deps=["actor_old_log_prob"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.REFERENCE
       ).add_node(
           "actor_train",
           func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
           deps=["reference_log_prob"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.ACTOR
       )

       return pipeline.build()

**Pipeline Comparison Table**:

.. list-table:: Built-in Pipeline Comparison
   :header-rows: 1
   :widths: 15 45 40

   * - Pipeline
     - Key Difference
     - Use Case
   * - **GRPO**
     - Group-based advantage normalization
     - Reasoning tasks, math problems
   * - **PPO**
     - Critic model + GAE advantage estimation
     - General RL tasks with value function
   * - **DAPO**
     - Dynamic sampling to filter zero-variance groups
     - Challenging tasks with sparse rewards
   * - **Embodied SRPO**
     - Environment interaction + VJEPA reward + dynamic sampling
     - Robotics, embodied AI tasks

4.3 Node Data Structure
-----------------------

Each DAG node is represented by the ``Node`` class:

.. code-block:: python
   :caption: siirl/execution/dag/node.py

   class NodeType(Enum):
       """Define the types of nodes in the DAG."""
       COMPUTE = "COMPUTE"                    # General computing task
       DATA_LOAD = "DATA_LOAD"                # Load data from DataLoader
       ENV_INTERACT = "ENV_INTERACT"          # Interact with the environment
       MODEL_INFERENCE = "MODEL_INFERENCE"    # Model inference (Rollout)
       MODEL_TRAIN = "MODEL_TRAIN"            # Model training
       PUT_TO_BUFFER = "PUT_TO_BUFFER"        # Put data into the distributed buffer
       GET_FROM_BUFFER = "GET_FROM_BUFFER"    # Get data from the distributed buffer
       BARRIER_SYNC = "BARRIER_SYNC"          # Global synchronization point
       CUSTOM = "CUSTOM"                      # User-defined node type

   class NodeRole(Enum):
       """Define the roles that a node plays in a distributed RL framework."""
       DEFAULT = "DEFAULT"                # Default role
       ACTOR = "ACTOR"                    # Actor model (policy)
       ADVANTAGE = "ADVANTAGE"            # Advantage computation
       CRITIC = "CRITIC"                  # Critic model (value function)
       ROLLOUT = "ROLLOUT"                # Rollout inference engine
       REFERENCE = "REFERENCE"            # Reference model (for KL)
       REWARD = "REWARD"                  # Reward computation
       DYNAMIC_SAMPLING = "DYNAMIC_SAMPLING"  # Dynamic sampling in databuffer (DAPO/Embodied)

   class NodeStatus(Enum):
       """Define the execution status of a DAG node."""
       PENDING = "PENDING"      # Waiting for dependencies to complete
       READY = "READY"          # Dependencies completed, ready to execute
       RUNNING = "RUNNING"      # Currently executing
       COMPLETED = "COMPLETED"  # Execution completed successfully
       FAILED = "FAILED"        # Execution failed
       SKIPPED = "SKIPPED"      # Skipped

   class Node:
       """Represents a node (task unit) in the DAG."""
       
       def __init__(
           self,
           node_id: str,
           node_type: NodeType,
           node_role: NodeRole = NodeRole.DEFAULT,
           only_forward_compute: bool = False,  # Forward only, no weight update
           agent_group: int = 0,                # Multi-agent scenario grouping
           dependencies: Optional[List[str]] = None,
           config: Optional[Dict[str, Any]] = None,
           executable_ref: Optional[str] = None,  # Function path "module:Class.method"
           filter_plugin: Optional[Callable] = None,  # Filter function for data
           agent_options: AgentArguments = None,
           retry_limit: int = 0,
       ):
           self.node_id = node_id
           self.node_type = node_type
           self.node_role = node_role
           self.only_forward_compute = only_forward_compute
           self.agent_group = agent_group
           self.dependencies = dependencies or []
           self.config = config or {}
           self.executable_ref = executable_ref
           self.retry_limit = retry_limit
           self._executable: Optional[Callable] = None
           self.status = NodeStatus.PENDING
           
           # Resolve executable function from path
           if self.executable_ref:
               self._resolve_executable()
       
       def _resolve_executable(self) -> None:
           """Dynamically import and obtain the executable function.
           
           Supports two formats:
           1. "module.path:ClassName.method" - imports module.path, gets ClassName.method
           2. "module.path.function" - imports module.path, gets function
           """
           if ":" in self.executable_ref:
               module_path, attr_path = self.executable_ref.split(":", 1)
               module = importlib.import_module(module_path)
               obj = module
               for attr_name in attr_path.split("."):
                   obj = getattr(obj, attr_name)
               self._executable = obj
           else:
               module_path, function_name = self.executable_ref.rsplit(".", 1)
               module = importlib.import_module(module_path)
               self._executable = getattr(module, function_name)
       
       def run(self, **kwargs) -> Any:
           """Execute the task of the node."""
           if self.executable:
               return self.executable(**kwargs)

4.4 TaskGraph Data Structure
----------------------------

``TaskGraph`` represents the entire training workflow as a DAG:

.. code-block:: python
   :caption: siirl/execution/dag/task_graph.py

   class TaskGraph:
       """Directed Acyclic Graph representing training workflow"""
       
       def __init__(self, graph_id: str):
           self.graph_id = graph_id
           self.nodes: Dict[str, Node] = {}       # node_id -> Node
           self.adj: Dict[str, List[str]] = {}    # Forward adjacency: node -> dependents
           self.rev_adj: Dict[str, List[str]] = {} # Reverse adjacency: node -> dependencies
       
       def add_node(self, node: Node) -> None:
           """Add node to graph"""
           self.nodes[node.node_id] = node
           self._update_adj_for_node(node)
       
       def get_topological_sort(self) -> List[str]:
           """Topological sort using Kahn's algorithm"""
           # ... implement Kahn's algorithm
       
       def validate_graph(self) -> Tuple[bool, Optional[str]]:
           """Validate DAG validity (no cycles, dependencies exist)"""
           # 1. Check all dependencies exist
           # 2. Use topological sort to detect cycles
           try:
               self.get_topological_sort()
               return True, None
           except ValueError as e:
               return False, str(e)
       
       def get_entry_nodes(self) -> List[Node]:
           """Get entry nodes (no dependencies)"""
           return [node for node_id, node in self.nodes.items() 
                   if not self.rev_adj.get(node_id)]
       
       def get_downstream_nodes(self, node_id: str) -> List[Node]:
           """Get downstream nodes"""
           return self.get_dependents(node_id)

4.5 TaskScheduler
-----------------

``TaskScheduler`` is responsible for assigning TaskGraph to each worker:

.. code-block:: python
   :caption: siirl/execution/scheduler/task_scheduler.py

   class TaskScheduler:
       """Task Scheduler: Assign TaskGraph to Workers"""
       
       def __init__(self, num_physical_nodes: int, gpus_per_node: int):
           self.num_physical_nodes = num_physical_nodes
           self.gpus_per_node = gpus_per_node
           self.num_workers = num_physical_nodes * gpus_per_node
           
           # State variables
           self.worker_to_graph_assignment: Dict[int, Optional[TaskGraph]] = {}
           self.node_active_worker_count: Dict[int, int] = defaultdict(int)
           self.node_free_gpus: Dict[int, List[int]] = defaultdict(list)
       
       def schedule_and_assign_tasks(
           self,
           original_task_graphs: List[TaskGraph],
           apportion_strategy: str = "param_aware",  # or "even"
           consider_node_cohesion: bool = True,      # Same task on same physical node
           consider_node_load: bool = True,          # Prefer lower load nodes
       ) -> Dict[int, Optional[TaskGraph]]:
           """Schedule tasks to each worker"""
           
           # 1. Split original graphs into irreducible subgraphs
           all_subgraphs = []
           for graph in original_task_graphs:
               subgraphs = discover_and_split_parallel_paths(graph)
               all_subgraphs.extend(subgraphs)
           
           # 2. Estimate subgraph sizes and sort
           subgraphs_with_sizes = sorted(
               [(sg, estimate_graph_model_params(sg)) for sg in all_subgraphs],
               key=lambda x: x[1],
               reverse=True
           )
           
           # 3. Apportion worker counts
           workers_per_task = self._apportion_workers_to_tasks(
               subgraphs_with_sizes,
               self.num_workers,
               apportion_strategy
           )
           
           # 4. Place workers (considering cohesion and load balancing)
           for task_graph, _ in subgraphs_with_sizes:
               num_workers = workers_per_task[task_graph.graph_id]
               for _ in range(num_workers):
                   best_worker = self._find_best_worker(
                       task_graph, consider_node_cohesion, consider_node_load
                   )
                   self.worker_to_graph_assignment[best_worker] = task_graph
           
           return self.worker_to_graph_assignment

**Scheduling Strategy Comparison**:

.. list-table:: Scheduling Strategies
   :header-rows: 1
   :widths: 20 40 40

   * - Strategy
     - Description
     - Use Case
   * - **even**
     - Distribute workers evenly among tasks
     - Similar task workloads
   * - **param_aware**
     - Distribute based on model parameter ratio
     - Large variance in task sizes

4.6 Task Graph Splitting (task_loader.py)
-----------------------------------------

The ``task_loader.py`` module provides utilities for analyzing and splitting complex TaskGraphs:

.. code-block:: python
   :caption: siirl/execution/dag/task_loader.py

   def discover_and_split_parallel_paths(src_task_graph: TaskGraph) -> List[TaskGraph]:
       """
       Discovers and splits a TaskGraph into irreducible subgraphs by iteratively
       identifying and splitting fan-out and re-converging parallel paths.
       
       Args:
           src_task_graph: The original TaskGraph to be analyzed and split
       
       Returns:
           List[TaskGraph]: A list of irreducible subgraph TaskGraph objects
       """
       # 1. Try to split by fan-out to distinct exits
       graphs_after_fan_out = split_by_fan_out_to_exits(current_graph, iteration_counter)
       
       # 2. If no fan-out split, try to split by re-converging paths
       graphs_after_reconverge = split_by_reconverging_paths(current_graph, iteration_counter)
       
       # 3. If no split possible, graph is irreducible
       return final_irreducible_graphs

This enables automatic parallelization of independent pipeline branches across different worker groups.

----

.. _sec5_dag_worker:

5. DAG Worker Deep Dive
=======================

DAG Worker is the core execution unit of siiRL, with one DAG Worker running per GPU.

5.1 DAGWorker Class Structure
-----------------------------

.. code-block:: python
   :caption: siirl/dag_worker/dagworker.py

   class DAGWorker(Worker):
       """DAG Execution Unit, one instance per GPU"""
       
       def __init__(
           self,
           config: SiiRLArguments,
           process_group_manager: ProcessGroupManager,
           taskgraph_mapping: Dict[int, TaskGraph],
           data_coordinator: ray.actor.ActorHandle,
           metric_worker: ray.actor.ActorHandle,
       ):
           # Configuration
           self.config = config
           self.process_group_manager = process_group_manager
           self.taskgraph_mapping = taskgraph_mapping
           self.data_coordinator = data_coordinator
           
           # State
           self.global_steps = 0
           self.workers: Dict[str, Any] = {}  # Node role -> Worker instance
           self.multi_agent_group: Dict[int, Dict[NodeRole, Any]] = defaultdict(dict)
           self.process_groups: Dict[str, ProcessGroup] = {}
           self.internal_data_cache: Dict[str, Any] = {}
           
           # Initialize
           self._initialize_worker()

5.2 Initialization Flow
-----------------------

DAGWorker initialization is divided into two phases:

**Phase 1: _initialize_worker() in __init__**

.. code-block:: python

   def _initialize_worker(self):
       """Initialize all Worker components"""
       
       # 1. Validate rank and get assigned TaskGraph
       self._rank = get_and_validate_rank()
       self.taskgraph = get_taskgraph_for_rank(self._rank, self.taskgraph_mapping)
       
       # 2. Set up distributed environment
       self._setup_distributed_environment()
       
       # 3. Initialize Tokenizer
       self._setup_tokenizers()
       
       # 4. Initialize DataLoader
       self._setup_dataloader()
       
       # 5. Initialize Reward Manager
       self._setup_reward_managers()
       
       # 6. Create role -> Worker class mapping
       self._setup_role_worker_mapping()
       
       # 7. Instantiate node Workers
       self._initialize_node_workers()

**Phase 2: init_graph() method**

.. code-block:: python

   def init_graph(self):
       """Load model weights, restore checkpoint"""
       
       # 1. Load model weights to GPU
       self._load_model_weights()
       
       # 2. Set up weight sharing (Actor-Rollout)
       self._setup_sharding_manager()
       
       # 3. Initialize async rollout (if configured)
       self._setup_async_rollout()
       
       # 4. Initialize multi-agent loop (if configured)
       self._setup_multi_agent_loop()
       
       # 5. Initialize validator
       self._init_validator()
       
       # 6. Initialize checkpoint manager and restore
       self._init_checkpoint_manager()
       self.global_steps = self.checkpoint_manager.load_checkpoint()
       
       # 7. Global synchronization
       dist.barrier(self._gather_group)

5.3 Training Loop
-----------------

.. code-block:: python
   :caption: DAGWorker Training Loop Core Logic

   def execute_task_graph(self):
       """Main entry: Execute DAG training pipeline"""
       
       # Optional pre-training validation
       if self.config.trainer.val_before_train:
           self.validator.validate(global_step=self.global_steps)
       
       # Main training loop
       self._run_training_loop()
   
   def _run_training_loop(self):
       """Main training loop"""
       
       for epoch in range(self.config.trainer.total_epochs):
           for batch_idx in range(self.dataloader.num_train_batches):
               # Execute one training step
               ordered_metrics = self._run_training_step(epoch, batch_idx)
               self.global_steps += 1
               
               # Save checkpoint
               if self.global_steps % self.config.trainer.save_freq == 0:
                   self.checkpoint_manager.save_checkpoint(self.global_steps)
               
               # Execute validation
               if self.global_steps % self.config.trainer.test_freq == 0:
                   self.validator.validate(global_step=self.global_steps)
               
               # Log metrics
               if self._rank == 0 and self.logger:
                   self.logger.log(data=ordered_metrics, step=self.global_steps)

5.4 Single Training Step Execution
----------------------------------

.. code-block:: python
   :caption: _run_training_step() Explained

   def _run_training_step(self, epoch: int, batch_idx: int) -> Optional[Dict]:
       """Execute a single training step"""
       
       # 1. Get data from DataLoader
       batch = preprocess_dataloader(
           self.dataloader.run(epoch=epoch, is_validation_step=False),
           self.config.actor_rollout_ref.rollout.n
       )
       
       # 2. Get DAG entry nodes
       node_queue = self.taskgraph.get_entry_nodes()
       entry_node_id = node_queue[0].node_id
       visited_nodes = set()
       
       # 3. Graph traversal execution
       while node_queue:
           cur_node = node_queue.pop(0)
           if cur_node.node_id in visited_nodes:
               continue
           visited_nodes.add(cur_node.node_id)
           
           # 3.1 Get node's DP/TP/PP info
           cur_dp_size, cur_dp_rank, cur_tp_rank, cur_tp_size, cur_pp_rank, cur_pp_size = \
               self._get_node_dp_info(cur_node)
           
           # 3.2 Non-entry nodes get data from buffer
           if cur_node.node_id != entry_node_id:
               batch = self.get_data_from_buffers(
                   key=cur_node.node_id,
                   cur_dp_size=cur_dp_size,
                   cur_dp_rank=cur_dp_rank
               )
           
           # 3.3 Execute node
           if cur_node.executable and batch is not None:
               node_output = cur_node.run(
                   batch=batch,
                   config=self.config,
                   process_group=self._get_node_process_group(cur_node),
                   agent_group=self.multi_agent_group[cur_node.agent_group],
                   _dag_worker_instance=self
               )
           else:
               node_output = NodeOutput(batch=batch)
           
           # 3.4 Process output, pass to downstream nodes
           if next_nodes := self.taskgraph.get_downstream_nodes(cur_node.node_id):
               next_node = next_nodes[0]
               next_dp_size = self._get_node_dp_info(next_node)[0]
               
               # If DP size changes, need DataCoordinator for redistribution
               self.put_data_to_buffers(
                   key=next_node.node_id,
                   data=node_output.batch,
                   source_dp_size=cur_dp_size,
                   dest_dp_size=next_dp_size
               )
               
               # Add downstream nodes to queue
               for n in next_nodes:
                   if n.node_id not in visited_nodes:
                       node_queue.append(n)
       
       # 4. Clean up caches
       self._cleanup_step_buffers()
       
       # 5. Collect and return metrics
       return self._collect_metrics()

5.5 Node Execution Methods
--------------------------

DAGWorker provides a series of node execution methods, each corresponding to a node role:

.. code-block:: python
   :caption: Node Execution Methods

   # Rollout: Generate sequences
   def generate(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
       """Generate sequences using the Rollout model"""
       agent_group = kwargs.pop("agent_group")
       is_embodied = self.config.actor_rollout_ref.model.model_type == "embodied"
       
       if is_embodied:
           return self.generate_embodied_mode(agent_group, batch, **kwargs)
       
       if self.rollout_mode == 'sync':
           gen_output = agent_group[NodeRole.ROLLOUT].generate_sequences(batch)
           batch = batch.update(gen_output)
           return NodeOutput(batch=batch, metrics=gen_output["metrics"])
       else:
           return self.generate_async_mode(batch)
   
   # Reward: Compute rewards
   def compute_reward(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
       """Compute rewards for generated sequences"""
       reward_tensor, extra_infos = compute_reward(batch, self.reward_fn)
       batch["token_level_scores"] = reward_tensor
       
       if config.algorithm.use_kl_in_reward:
           batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl_in_reward, ...)
       else:
           batch["token_level_rewards"] = batch["token_level_scores"]
       
       return NodeOutput(batch=batch, metrics=metrics)
   
   # Advantage: Compute advantages
   def compute_advantage(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
       """Compute GAE/GRPO/CPGD advantages"""
       return NodeOutput(
           batch=compute_advantage(
               batch,
               adv_estimator=config.algorithm.adv_estimator,
               gamma=config.algorithm.gamma,
               lam=config.algorithm.lam,
               norm_adv_by_std_in_grpo=config.algorithm.norm_adv_by_std_in_grpo
           )
       )
   
   # Actor Forward: Compute old policy log prob
   def compute_old_log_prob(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
       """Compute log probabilities before policy update"""
       agent_group = kwargs.pop("agent_group")
       processed_data = agent_group[NodeRole.ACTOR].compute_log_prob(batch)
       return NodeOutput(batch=processed_data, metrics=processed_data.get("metrics", {}))
   
   # Reference: Compute reference model log prob
   def compute_ref_log_prob(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
       """Compute reference model log probabilities"""
       agent_group = kwargs.pop("agent_group")
       processed_data = agent_group[NodeRole.REFERENCE].compute_ref_log_prob(batch)
       return NodeOutput(batch=processed_data, metrics=processed_data["metrics"])
   
   # Actor Train: Train Actor model
   def train_actor(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
       """Execute Actor model training step"""
       agent_group = kwargs.pop("agent_group")
       processed_data = agent_group[NodeRole.ACTOR].update_actor(batch)
       return NodeOutput(batch=processed_data, metrics=processed_data["metrics"])
   
   # Critic Train: Train Critic model (PPO)
   def train_critic(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
       """Execute Critic model training step"""
       agent_group = kwargs.pop("agent_group")
       processed_data = agent_group[NodeRole.CRITIC].update_critic(batch)
       return NodeOutput(batch=processed_data, metrics=processed_data["metrics"])

----

.. _sec6_data_coordinator:

6. Data Coordinator Deep Dive
=============================

Data Coordinator is the core component of siiRL's fully distributed data management.

6.1 Design Philosophy
---------------------

**Why do we need Data Coordinator?**

In traditional frameworks, all intermediate data (Rollout outputs, Reward results, etc.) must pass through a central controller for redistribution, causing severe I/O bottlenecks. siiRL's Data Coordinator adopts a different design:

1. **Store only metadata and references**: Actual data is stored in Ray Object Store
2. **Support flexible sampling strategies**: Custom sampling via filter_plugin
3. **Automatic load balancing**: Optimize sequence length distribution via balance_partitions

6.2 DataCoordinator Implementation
----------------------------------

.. code-block:: python
   :caption: siirl/data_coordinator/data_buffer.py

   @ray.remote
   class DataCoordinator:
       """Global singleton data coordination Actor"""
       
       def __init__(self, nnodes: int, ppo_mini_batch_size: int, world_size: int):
           self.nnodes = nnodes
           self.ppo_mini_batch_size = ppo_mini_batch_size
           self.world_size = world_size
           
           # Efficiently store metadata and references using deque
           self._sample_queue: deque[Tuple[SampleInfo, ray.ObjectRef]] = deque()
           self.lock = asyncio.Lock()
           self._cache = []
       
       async def put_batch(
           self, 
           sample_infos: List[SampleInfo], 
           sample_refs: List[ray.ObjectRef],
           caller_node_id: Optional[str] = None
       ):
           """Register a batch of sample references and metadata"""
           
           # Inject caller node ID (for subsequent routing)
           if caller_node_id is None:
               caller_node_id = ray.get_runtime_context().get_node_id()
           
           for i in range(len(sample_infos)):
               if sample_infos[i].node_id is None:
                   sample_infos[i].node_id = caller_node_id
           
           async with self.lock:
               self._sample_queue.extend(zip(sample_infos, sample_refs))
       
       async def get_batch(
           self,
           batch_size: int,
           dp_rank: int,
           filter_plugin: Optional[Callable[[SampleInfo], bool]] = None,
           balance_partitions: Optional[int] = None
       ) -> List[ray.ObjectRef]:
           """Get a batch of sample ObjectRefs"""
           
           async with self.lock:
               # 1. If cached, return directly
               if len(self._cache) > 0:
                   return self._cache[dp_rank]
               
               # 2. No filter, use efficient FIFO
               if not filter_plugin:
                   batch_items = []
                   while self._sample_queue:
                       item = self._sample_queue.popleft()
                       batch_items.append(item)
                   
                   # Apply length balancing
                   if balance_partitions and balance_partitions > 1:
                       batch_refs = self._apply_length_balancing(batch_items, balance_partitions)
                   else:
                       batch_refs = [item[1] for item in batch_items]
                   
                   self._cache = batch_refs
                   return self._cache[:batch_size]
               
               # 3. With filter, execute filtering
               else:
                   potential_items = [item for item in self._sample_queue 
                                      if filter_plugin(item[0])]
                   
                   global_batch_size = batch_size * balance_partitions
                   if len(potential_items) < global_batch_size:
                       return []
                   
                   potential_items = potential_items[:global_batch_size]
                   
                   # Remove selected items from queue
                   refs_to_remove = {item[1] for item in potential_items}
                   self._sample_queue = deque(
                       item for item in self._sample_queue if item[1] not in refs_to_remove
                   )
                   
                   # Apply length balancing and cache
                   if balance_partitions and balance_partitions > 1:
                       batch_refs = self._apply_length_balancing(potential_items, balance_partitions)
                   else:
                       batch_refs = [item[1] for item in potential_items]
                   
                   for rank in range(balance_partitions):
                       self._cache.append(batch_refs[rank * batch_size: (rank + 1) * batch_size])
                   
                   return self._cache[dp_rank]

6.3 SampleInfo Metadata
-----------------------

.. code-block:: python
   :caption: siirl/data_coordinator/sample.py

   @dataclass
   class SampleInfo:
       """Sample metadata for routing and sampling"""
       
       sum_tokens: int = 0          # Total tokens (prompt + response)
       prompt_length: int = 0       # Prompt length
       response_length: int = 0     # Response length
       uid: str = ""                # Unique identifier
       node_id: Optional[str] = None  # Source node ID
       dict_info: Dict[str, Any] = field(default_factory=dict)  # Extended info
           # Common fields:
           # - 'key': Target node ID
           # - 'source_dp_size': Source DP size

6.4 Length Balancing Algorithm
------------------------------

Data Coordinator uses LPT (Longest Processing Time) algorithm to optimize sample distribution:

.. code-block:: python

   def _apply_length_balancing(
       self,
       batch_items: List[Tuple[SampleInfo, ray.ObjectRef]],
       k_partitions: int
   ) -> List[ray.ObjectRef]:
       """Reorder samples using LPT algorithm to balance sequence lengths across workers"""
       
       # 1. Extract each sample's length
       seqlen_list = [item[0].sum_tokens for item in batch_items]
       
       # 2. Calculate workloads
       workload_lst = calculate_workload(seqlen_list)
       
       # 3. Partition using Karmarkar-Karp algorithm
       global_partition_lst = get_seqlen_balanced_partitions(
           workload_lst, k_partitions=self.world_size, equal_size=True
       )
       
       # 4. Sort within each partition (smaller batches at ends to reduce PP bubbles)
       for idx, partition in enumerate(global_partition_lst):
           partition.sort(key=lambda x: (workload_lst[x], x))
           ordered_partition = partition[::2] + partition[1::2][::-1]
           global_partition_lst[idx] = ordered_partition
       
       # 5. Organize sample refs by partition order
       reordered_refs = []
       for partition in global_partition_lst:
           for original_idx in partition:
               reordered_refs.append(batch_items[original_idx][1])
       
       return reordered_refs

6.5 DAGWorker Data Flow Operations
----------------------------------

.. code-block:: python
   :caption: Data flow methods in DAGWorker

   def put_data_to_buffers(
       self,
       key: str,
       data: TensorDict,
       source_dp_size: int,
       dest_dp_size: int,
       enforce_buffer: bool = False
   ):
       """Put data into DataCoordinator"""
       
       # Same source and dest DP size and not forcing buffer, use local cache
       if source_dp_size == dest_dp_size and not enforce_buffer:
           self.internal_data_cache[key] = data
       else:
           # Convert to Sample list
           samples = Dict2Samples(data)
           
           # Create metadata
           sample_infos = []
           for sample in samples:
               sample_infos.append(SampleInfo(
                   sum_tokens=int(sample.attention_mask.sum()),
                   uid=str(sample.uid),
                   dict_info={'key': key, 'source_dp_size': source_dp_size}
               ))
           
           # Upload to Ray Object Store
           sample_refs = [ray.put(sample) for sample in samples]
           
           # Register with DataCoordinator
           caller_node_id = ray.get_runtime_context().get_node_id()
           self.data_coordinator.put_batch.remote(sample_infos, sample_refs, caller_node_id)
   
   def get_data_from_buffers(
       self,
       key: str,
       cur_dp_size: int,
       cur_dp_rank: int
   ) -> Optional[TensorDict]:
       """Get data from DataCoordinator"""
       
       # Check local cache first
       if key in self.internal_data_cache:
           return self.internal_data_cache.pop(key)
       
       # Define filter function
       def key_filter(sample_info: SampleInfo) -> bool:
           return sample_info.dict_info.get('key') == key
       
       # Calculate adjusted batch size
       rollout_n = self.config.actor_rollout_ref.rollout.n
       adjusted_batch_size = int(self.config.data.train_batch_size * rollout_n / cur_dp_size)
       
       # Get from DataCoordinator
       sample_refs = ray.get(self.data_coordinator.get_batch.remote(
           adjusted_batch_size,
           cur_dp_rank,
           filter_plugin=key_filter,
           balance_partitions=cur_dp_size
       ))
       
       if not sample_refs:
           return None
       
       # Get actual data and collate
       samples = ray.get(sample_refs)
       return Samples2Dict(samples)

----

.. _sec7_engine:

7. Engine Model Execution
=========================

The Engine module contains all model Worker implementations, supporting both FSDP and Megatron training backends.

7.1 Engine Module Structure
---------------------------

::

   engine/
   ├── actor/                    # Actor models
   │   ├── base.py               # Base class
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
   │   ├── naive.py              # Simple Reward
   │   ├── parallel.py           # Parallel Reward Model
   │   ├── dapo.py               # DAPO Reward
   │   └── embodied.py           # Embodied Reward
   ├── sharding_manager/         # Weight sharding management
   │   ├── base.py
   │   ├── fsdp_hf.py
   │   ├── fsdp_sglang.py
   │   ├── fsdp_vllm.py
   │   ├── megatron_sglang.py
   │   └── megatron_vllm.py
   ├── fsdp_workers.py           # FSDP Worker factory
   └── megatron_workers.py       # Megatron Worker factory

7.2 Worker Base Class
---------------------

All model Workers inherit from a unified base class:

.. code-block:: python
   :caption: siirl/engine/base_worker/base/base_worker.py

   class Worker:
       """Abstract base class for all Workers"""
       
       @property
       def world_size(self) -> int:
           """Get global world size"""
           if not dist.is_initialized():
               return 1
           return dist.get_world_size()
       
       def init_model(self):
           """Initialize model weights (implemented by subclasses)"""
           raise NotImplementedError

7.3 Actor Worker
----------------

Actor Worker is responsible for policy model training:

.. code-block:: python
   :caption: siirl/engine/actor/dp_actor.py (simplified)

   class FSDPActor(Actor):
       """FSDP Distributed Actor"""
       
       def __init__(self, config, process_group: ProcessGroup):
           self.config = config
           self.process_group = process_group
           
           # Model related
           self.model = None
           self.optimizer = None
           self.scheduler = None
       
       def init_model(self):
           """Initialize model, optimizer, scheduler"""
           
           # 1. Load model
           self.model = self._load_model()
           
           # 2. Apply FSDP wrapping
           self.model = FSDP(
               self.model,
               sharding_strategy=ShardingStrategy.FULL_SHARD,
               process_group=self.process_group,
               mixed_precision=...,
           )
           
           # 3. Create optimizer
           self.optimizer = create_optimizer(self.model, self.config.actor.optim)
           
           # 4. Create learning rate scheduler
           self.scheduler = create_scheduler(self.optimizer, self.config.actor.optim)
       
       def compute_log_prob(self, batch: TensorDict) -> TensorDict:
           """Compute log probabilities (forward pass, no weight update)"""
           
           with torch.no_grad():
               outputs = self.model(
                   input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"],
               )
               
               log_probs = compute_log_prob_from_logits(
                   outputs.logits, batch["responses"], batch["response_mask"]
               )
           
           batch["old_log_probs"] = log_probs
           return batch
       
       def update_actor(self, batch: TensorDict) -> TensorDict:
           """Execute Actor training step"""
           
           metrics = {}
           total_loss = 0.0
           
           for _ in range(self.config.actor.ppo_epochs):
               # Forward pass
               outputs = self.model(
                   input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"],
               )
               
               # Compute current log probabilities
               log_probs = compute_log_prob_from_logits(
                   outputs.logits, batch["responses"], batch["response_mask"]
               )
               
               # Compute policy loss
               pg_loss, pg_clipfrac, ppo_kl, _ = compute_policy_loss(
                   old_log_prob=batch["old_log_probs"],
                   log_prob=log_probs,
                   advantages=batch["advantages"],
                   response_mask=batch["response_mask"],
                   cliprange=self.config.actor.clip_ratio,
               )
               
               # Compute entropy loss
               entropy_loss = compute_entropy_loss(outputs.logits, batch["response_mask"])
               
               # Total loss
               loss = pg_loss - self.config.actor.entropy_coef * entropy_loss
               
               # Backward pass
               self.optimizer.zero_grad()
               loss.backward()
               
               # Gradient clipping
               if self.config.actor.max_grad_norm:
                   torch.nn.utils.clip_grad_norm_(
                       self.model.parameters(), self.config.actor.max_grad_norm
                   )
               
               # Optimizer step
               self.optimizer.step()
               self.scheduler.step()
               
               total_loss += loss.item()
           
           metrics["actor/loss"] = total_loss / self.config.actor.ppo_epochs
           metrics["actor/pg_clipfrac"] = pg_clipfrac.item()
           metrics["actor/ppo_kl"] = ppo_kl.item()
           
           batch["metrics"] = metrics
           return batch

7.4 Rollout Worker
------------------

Rollout Worker is responsible for sequence generation:

.. code-block:: python
   :caption: siirl/engine/rollout/vllm_rollout/vllm_rollout.py (simplified)

   class VLLMRollout:
       """vLLM Inference Backend"""
       
       def __init__(self, config, process_group: ProcessGroup):
           self.config = config
           self.process_group = process_group
           
           # vLLM LLM instance
           self.llm = None
           self.tokenizer = None
       
       def init_model(self):
           """Initialize vLLM engine"""
           
           from vllm import LLM, SamplingParams
           
           self.llm = LLM(
               model=self.config.model.path,
               tensor_parallel_size=self.config.rollout.tensor_model_parallel_size,
               trust_remote_code=True,
               dtype=self.config.model.dtype,
           )
           
           self.tokenizer = self.llm.get_tokenizer()
       
       def generate_sequences(self, batch: TensorDict) -> TensorDict:
           """Generate sequences"""
           
           from vllm import SamplingParams
           
           # Build sampling parameters
           sampling_params = SamplingParams(
               n=self.config.rollout.n,  # GRPO group size
               temperature=self.config.rollout.temperature,
               top_p=self.config.rollout.top_p,
               max_tokens=self.config.data.max_response_length,
           )
           
           # Prepare prompts
           prompts = batch["prompts"]  # List[str] or List[List[int]]
           
           # Generate
           outputs = self.llm.generate(prompts, sampling_params)
           
           # Process outputs
           all_responses = []
           all_response_ids = []
           
           for output in outputs:
               for completion in output.outputs:
                   all_responses.append(completion.text)
                   all_response_ids.append(completion.token_ids)
           
           # Update batch
           batch["responses"] = all_responses
           batch["response_ids"] = torch.tensor(all_response_ids)
           batch["metrics"] = {
               "rollout/avg_response_length": np.mean([len(r) for r in all_response_ids])
           }
           
           return batch

7.5 Sharding Manager
--------------------

Sharding Manager is responsible for weight synchronization between Actor and Rollout:

.. code-block:: python
   :caption: siirl/engine/sharding_manager/fsdp_vllm.py (simplified)

   class FSDPVLLMShardingManager:
       """Weight synchronization between FSDP Actor and vLLM Rollout"""
       
       def __init__(self, actor: FSDPActor, rollout: VLLMRollout, process_group: ProcessGroup):
           self.actor = actor
           self.rollout = rollout
           self.process_group = process_group
       
       def sync_weights_actor_to_rollout(self):
           """Sync Actor weights to Rollout"""
           
           # 1. Gather full weights from FSDP
           with FSDP.state_dict_type(
               self.actor.model,
               StateDictType.FULL_STATE_DICT,
               FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
           ):
               state_dict = self.actor.model.state_dict()
           
           # 2. Broadcast to all ranks
           dist.broadcast_object_list([state_dict], src=0, group=self.process_group)
           
           # 3. Update vLLM model weights
           self.rollout.load_weights(state_dict)

----

.. _sec8_core_algorithms:

8. Core Algorithm Implementation
================================

8.1 Advantage Estimators
------------------------

siiRL supports multiple advantage estimation methods:

.. code-block:: python
   :caption: siirl/dag_worker/core_algos.py

   # Registry decorator
   ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}
   
   def register_adv_est(name_or_enum: str | AdvantageEstimator):
       """Register an advantage estimator"""
       def decorator(fn):
           name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
           ADV_ESTIMATOR_REGISTRY[name] = fn
           return fn
       return decorator
   
   @register_adv_est(AdvantageEstimator.GAE)
   def compute_gae_advantage_return(
       token_level_rewards: torch.Tensor,  # (bs, response_length)
       values: torch.Tensor,               # (bs, response_length)
       response_mask: torch.Tensor,        # (bs, response_length)
       gamma: float,
       lam: float,
   ):
       """GAE (Generalized Advantage Estimation) for PPO"""
       with torch.no_grad():
           nextvalues = 0
           lastgaelam = 0
           advantages_reversed = []
           gen_len = token_level_rewards.shape[-1]
           
           for t in reversed(range(gen_len)):
               delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
               lastgaelam_ = delta + gamma * lam * lastgaelam
               
               # Skip padding tokens
               nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
               lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam
               
               advantages_reversed.append(lastgaelam)
           
           advantages = torch.stack(advantages_reversed[::-1], dim=1)
           returns = advantages + values
           advantages = masked_whiten(advantages, response_mask)
       
       return advantages, returns
   
   @register_adv_est(AdvantageEstimator.GRPO)
   def compute_grpo_outcome_advantage(
       token_level_rewards: torch.Tensor,  # (bs, response_length)
       response_mask: torch.Tensor,        # (bs, response_length)
       index: np.ndarray,                  # Index for grouping
       epsilon: float = 1e-6,
       norm_adv_by_std_in_grpo: bool = True,
   ):
       """GRPO (Group Relative Policy Optimization)"""
       scores = token_level_rewards.sum(dim=-1)  # Sequence-level rewards
       
       id2score = defaultdict(list)
       id2mean = {}
       id2std = {}
       
       with torch.no_grad():
           bsz = scores.shape[0]
           
           # Group by prompt
           for i in range(bsz):
               idx_key = int(index[i].item()) if isinstance(index[i], torch.Tensor) else int(index[i])
               id2score[idx_key].append(scores[i])
           
           # Compute group mean and std
           for idx in id2score:
               if len(id2score[idx]) == 1:
                   id2mean[idx] = torch.tensor(0.0)
                   id2std[idx] = torch.tensor(1.0)
               elif len(id2score[idx]) > 1:
                   scores_tensor = torch.stack(id2score[idx])
                   id2mean[idx] = torch.mean(scores_tensor)
                   id2std[idx] = torch.std(scores_tensor)
           
           # Normalize
           for i in range(bsz):
               idx_key = int(index[i].item()) if isinstance(index[i], torch.Tensor) else int(index[i])
               if norm_adv_by_std_in_grpo:
                   scores[i] = (scores[i] - id2mean[idx_key]) / (id2std[idx_key] + epsilon)
               else:  # Dr.GRPO
                   scores[i] = scores[i] - id2mean[idx_key]
           
           scores = scores.unsqueeze(-1) * response_mask
       
       return scores, scores

8.2 Policy Loss Functions
-------------------------

siiRL supports multiple policy loss functions:

.. code-block:: python
   :caption: siirl/dag_worker/core_algos.py

   POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}
   
   def register_policy_loss(name: str):
       """Register a policy loss function"""
       def decorator(func: PolicyLossFn) -> PolicyLossFn:
           POLICY_LOSS_REGISTRY[name] = func
           return func
       return decorator
   
   @register_policy_loss("vanilla")
   def compute_policy_loss_vanilla(
       old_log_prob: torch.Tensor,
       log_prob: torch.Tensor,
       advantages: torch.Tensor,
       response_mask: torch.Tensor,
       loss_agg_mode: str = "token-mean",
       config: Optional[ActorArguments] = None,
       rollout_is_weights: torch.Tensor | None = None,
   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
       """Standard PPO policy loss (dual-clip)"""
       
       clip_ratio = config.clip_ratio
       clip_ratio_low = config.clip_ratio_low or clip_ratio
       clip_ratio_high = config.clip_ratio_high or clip_ratio
       clip_ratio_c = config.clip_ratio_c
       
       negative_approx_kl = log_prob - old_log_prob
       ratio = torch.exp(negative_approx_kl)
       ppo_kl = masked_mean(-negative_approx_kl, response_mask)
       
       # Standard PPO clipping
       pg_losses1 = -advantages * ratio
       pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
       clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
       pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
       
       # Dual clipping (negative advantage scenario)
       pg_losses3 = -advantages * clip_ratio_c
       clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
       pg_clipfrac_lower = masked_mean(
           torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
       )
       
       pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
       
       # Apply importance weights
       if rollout_is_weights is not None:
           pg_losses = pg_losses * rollout_is_weights
       
       pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)
       
       return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
   
   @register_policy_loss("cpgd")
   def compute_policy_loss_cpgd(...):
       """CPGD policy loss (direct log_prob clipping)"""
       ...
   
   @register_policy_loss("gspo")
   def compute_policy_loss_gspo(...):
       """GSPO policy loss (sequence-level importance ratio)"""
       ...
   
   @register_policy_loss("gpg")
   def compute_policy_loss_gpg(...):
       """GPG policy loss (REINFORCE style)"""
       ...

8.3 KL Penalty
--------------

.. code-block:: python

   class AdaptiveKLController:
       """Adaptive KL Controller"""
       
       def __init__(self, init_kl_coef, target_kl, horizon):
           self.value = init_kl_coef
           self.target = target_kl
           self.horizon = horizon
       
       def update(self, current_kl, n_steps):
           proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
           mult = 1 + proportional_error * n_steps / self.horizon
           self.value *= mult
   
   def apply_kl_penalty(data: TensorDict, kl_ctrl, kl_penalty="kl"):
       """Apply KL penalty to token-level rewards"""
       
       kld = kl_penalty_fn(data["old_log_probs"], data["ref_log_prob"], kl_penalty)
       kld = kld * data["response_mask"]
       beta = kl_ctrl.value
       
       data["token_level_rewards"] = data["token_level_scores"] - beta * kld
       
       current_kl = masked_mean(kld, data["response_mask"]).item()
       kl_ctrl.update(current_kl=current_kl, n_steps=data.batch_size[0])
       
       return data, {"actor/reward_kl_penalty": current_kl, "actor/kl_coef": beta}

----

.. _sec9_execution_flow:

9. Complete Execution Flow
==========================

9.1 GRPO Training Flow
----------------------

Using GRPO as an example, showing the complete training flow:

::

   ┌──────────────────────────────────────────────────────────────────────────────┐
   │                          GRPO Single Step Training Flow                       │
   └──────────────────────────────────────────────────────────────────────────────┘
   
   [1. Data Loading]
       │
       │  DataLoader.run() → batch (prompts, attention_mask, ...)
       │
       ▼
   [2. Rollout Generation]  ───────────────────────────────────────────────────────
       │
       │  DAGWorker.generate()
       │      │
       │      ├── Prepare generation batch
       │      ├── rollout_worker.generate_sequences(batch)
       │      │       │
       │      │       ├── vLLM/SGLang/HF inference
       │      │       └── Return responses, response_ids
       │      │
       │      └── Update batch: responses, response_mask
       │
       │  Output: batch with responses (bs * n_samples, seq_len)
       │
       ▼
   [3. Reward Computation]  ──────────────────────────────────────────────────────
       │
       │  DAGWorker.compute_reward()
       │      │
       │      ├── reward_fn.score(batch) → token_level_scores
       │      │
       │      ├── (Optional) Apply KL penalty:
       │      │       kl = old_log_prob - ref_log_prob
       │      │       token_level_rewards = token_level_scores - β * kl
       │      │
       │      └── Otherwise: token_level_rewards = token_level_scores
       │
       │  Output: batch with token_level_rewards
       │
       ▼
   [4. Advantage Computation]  ───────────────────────────────────────────────────
       │
       │  DAGWorker.compute_advantage()
       │      │
       │      └── compute_grpo_outcome_advantage()
       │              │
       │              ├── Compute sequence-level scores: scores = rewards.sum(dim=-1)
       │              ├── Group by prompt
       │              ├── Compute group mean and std
       │              └── Normalize: (scores - mean) / std
       │
       │  Output: batch with advantages
       │
       ▼
   [5. Actor Forward]  ───────────────────────────────────────────────────────────
       │
       │  DAGWorker.compute_old_log_prob()
       │      │
       │      └── actor_worker.compute_log_prob(batch)
       │              │
       │              ├── Forward pass (no_grad)
       │              └── Compute old_log_probs
       │
       │  Output: batch with old_log_probs
       │
       ▼
   [6. Reference Forward]  ───────────────────────────────────────────────────────
       │
       │  DAGWorker.compute_ref_log_prob()
       │      │
       │      └── reference_worker.compute_ref_log_prob(batch)
       │              │
       │              ├── Forward pass (no_grad)
       │              └── Compute ref_log_prob
       │
       │  Output: batch with ref_log_prob
       │
       ▼
   [7. Actor Training]  ──────────────────────────────────────────────────────────
       │
       │  DAGWorker.train_actor()
       │      │
       │      └── actor_worker.update_actor(batch)
       │              │
       │              ├── for _ in range(ppo_epochs):
       │              │       │
       │              │       ├── Forward pass → log_probs
       │              │       ├── Compute policy loss:
       │              │       │       pg_loss = -advantages * clipped_ratio
       │              │       ├── Compute entropy loss
       │              │       ├── Total loss = pg_loss - entropy_coef * entropy
       │              │       ├── Backward pass
       │              │       └── Optimizer step
       │              │
       │              └── Return metrics
       │
       │  Output: batch with metrics
       │
       ▼
   [8. Sync Weights]
       │
       │  sharding_manager.sync_weights_actor_to_rollout()
       │
       ▼
   [Done: Continue to next step]

9.2 PPO Training Flow
---------------------

PPO adds Critic model and GAE computation compared to GRPO:

::

   GRPO flow + the following additional steps:
   
   [3.5. Value Computation] (After Reward, before Advantage)
       │
       │  DAGWorker.compute_value()
       │      │
       │      └── critic_worker.compute_values(batch)
       │              │
       │              ├── Forward pass (no_grad)
       │              └── Compute values
       │
       │  Output: batch with values
   
   [4. Advantage Computation] (Uses GAE instead of GRPO)
       │
       │  compute_gae_advantage_return()
       │      │
       │      ├── Reverse iterate through response tokens
       │      ├── Compute TD-error: δ = r + γV(s') - V(s)
       │      └── GAE: A = δ + γλA'
   
   [7.5. Critic Training] (After Actor training)
       │
       │  DAGWorker.train_critic()
       │      │
       │      └── critic_worker.update_critic(batch)
       │              │
       │              ├── Forward pass → vpreds
       │              ├── Compute Value loss:
       │              │       vf_loss = clipped_mse(vpreds, returns)
       │              ├── Backward pass
       │              └── Optimizer step

----

.. _sec10_configuration:

10. Configuration Parameters
============================

10.1 Configuration File Structure
---------------------------------

siiRL uses Hydra for configuration management, with main configuration groups:

.. code-block:: yaml
   :caption: Configuration File Structure

   # algorithm: Algorithm configuration
   algorithm:
     adv_estimator: grpo        # grpo/gae/cpgd/gspo
     workflow_type: DEFAULT     # DEFAULT/DAPO/EMBODIED
     gamma: 1.0                 # Discount factor
     lam: 0.95                  # GAE lambda
     use_kl_in_reward: false    # Whether to use KL penalty in reward
     norm_adv_by_std_in_grpo: true
     
     kl_ctrl:
       type: fixed              # fixed/adaptive
       kl_coef: 0.001
   
   # data: Data configuration
   data:
     train_files: /path/to/train.parquet
     train_batch_size: 512
     max_prompt_length: 2048
     max_response_length: 4096
     num_loader_workers: 4
   
   # actor_rollout_ref: Model configuration
   actor_rollout_ref:
     model:
       path: /path/to/model
       dtype: bfloat16
       trust_remote_code: true
     
     actor:
       strategy: fsdp           # fsdp/megatron
       clip_ratio: 0.2
       entropy_coef: 0.01
       ppo_epochs: 1
       ppo_mini_batch_size: 256
       max_grad_norm: 1.0
       
       optim:
         lr: 1e-6
         weight_decay: 0.01
         scheduler: cosine_with_warmup
         warmup_ratio: 0.1
     
     rollout:
       name: vllm                # vllm/sglang/hf
       tensor_model_parallel_size: 2
       n: 8                      # GRPO group size
       temperature: 1.0
       top_p: 1.0
       mode: sync                # sync/async
   
   # trainer: Trainer configuration
   trainer:
     n_gpus_per_node: 8
     nnodes: 1
     total_epochs: 30
     save_freq: 10
     test_freq: 5
     val_before_train: false
     critic_warmup: 0
     
     project_name: my_project
     experiment_name: grpo_training
     logger: wandb             # wandb/tensorboard/console
   
   # dag: DAG configuration
   dag:
     custom_pipeline_fn: null   # Custom Pipeline function path
     enable_perf: false
     backend_threshold: 32

10.2 Key Parameter Descriptions
-------------------------------

.. list-table:: Key Configuration Parameters
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``algorithm.adv_estimator``
     - grpo
     - Advantage estimator (grpo/gae/cpgd/gspo)
   * - ``algorithm.workflow_type``
     - DEFAULT
     - Workflow type (DEFAULT/DAPO/EMBODIED)
   * - ``data.train_batch_size``
     - 512
     - Global training batch size
   * - ``actor_rollout_ref.rollout.n``
     - 8
     - GRPO samples per prompt
   * - ``actor_rollout_ref.actor.clip_ratio``
     - 0.2
     - PPO clipping ratio
   * - ``actor_rollout_ref.actor.ppo_epochs``
     - 1
     - PPO epochs per training step
   * - ``actor_rollout_ref.rollout.tensor_model_parallel_size``
     - 1
     - Rollout TP size
   * - ``trainer.save_freq``
     - 10
     - Checkpoint save frequency (steps)
   * - ``trainer.test_freq``
     - 5
     - Validation frequency (steps)

10.3 How to Add New Configuration Items
---------------------------------------

siiRL uses Python dataclasses for configuration management. Here's how to add new configuration items:

**Step 1: Identify the Configuration Group**

Configuration is organized into the following groups in ``siirl/params/``:

::

   siirl/params/
   ├── __init__.py              # Exports all argument classes
   ├── training_args.py         # TrainingArguments, SiiRLArguments (root)
   ├── model_args.py            # ActorArguments, RolloutArguments, AlgorithmArguments, etc.
   ├── data_args.py             # DataArguments
   ├── dag_args.py              # DagArguments
   ├── profiler_args.py         # ProfilerArguments
   └── embodied_args.py         # EmbodiedArguments

**Step 2: Add a New Field to the Appropriate Dataclass**

Example: Adding a new ``max_retry_count`` field to ``TrainingArguments``:

.. code-block:: python
   :caption: siirl/params/training_args.py

   from dataclasses import dataclass, field
   from typing import Optional
   
   @dataclass
   class TrainingArguments:
       # Existing fields...
       total_epochs: int = field(default=30, metadata={"help": "Total training epochs"})
       save_freq: int = field(default=-1, metadata={"help": "Checkpoint frequency"})
       
       # Add your new field here
       max_retry_count: int = field(
           default=3,
           metadata={"help": "Maximum retry count for failed training steps"}
       )

**Step 3: Add a New Argument Group (if needed)**

If adding a completely new category, create a new dataclass and register it in ``SiiRLArguments``:

.. code-block:: python
   :caption: siirl/params/my_custom_args.py (new file)

   from dataclasses import dataclass, field
   from typing import Dict, Any
   
   @dataclass
   class MyCustomArguments:
       """Custom arguments for new feature."""
       
       enable_feature: bool = field(
           default=False,
           metadata={"help": "Enable the custom feature"}
       )
       feature_threshold: float = field(
           default=0.5,
           metadata={"help": "Threshold for the custom feature"}
       )
       feature_config: Dict[str, Any] = field(
           default_factory=dict,
           metadata={"help": "Additional configuration for the feature"}
       )
       
       def to_dict(self) -> Dict[str, Any]:
           from dataclasses import asdict
           return asdict(self)

Then register in ``SiiRLArguments``:

.. code-block:: python
   :caption: siirl/params/training_args.py

   from siirl.params.my_custom_args import MyCustomArguments
   
   @dataclass
   class SiiRLArguments:
       data: DataArguments = field(default_factory=DataArguments)
       actor_rollout_ref: ActorRolloutRefArguments = field(default_factory=ActorRolloutRefArguments)
       # ... existing fields ...
       
       # Add your new argument group
       my_custom: MyCustomArguments = field(default_factory=MyCustomArguments)

**Step 4: Export in __init__.py**

.. code-block:: python
   :caption: siirl/params/__init__.py

   from .my_custom_args import MyCustomArguments
   
   __all__ = [
       # ... existing exports ...
       "MyCustomArguments",
   ]

**Step 5: Use in YAML Configuration**

After adding the new fields, you can use them in your YAML configuration:

.. code-block:: yaml
   :caption: config.yaml

   trainer:
     total_epochs: 30
     save_freq: 10
     max_retry_count: 5  # Your new field
   
   my_custom:  # Your new argument group
     enable_feature: true
     feature_threshold: 0.7
     feature_config:
       key1: value1
       key2: value2

**Step 6: Access in Code**

.. code-block:: python

   def my_function(config: SiiRLArguments):
       # Access top-level trainer config
       max_retry = config.trainer.max_retry_count
       
       # Access your custom argument group
       if config.my_custom.enable_feature:
           threshold = config.my_custom.feature_threshold
           extra_config = config.my_custom.feature_config

**Configuration Hierarchy**:

::

   SiiRLArguments (root)
   ├── data: DataArguments
   │   ├── train_files
   │   ├── train_batch_size
   │   └── ...
   ├── actor_rollout_ref: ActorRolloutRefArguments
   │   ├── model: ModelArguments
   │   ├── actor: ActorArguments
   │   │   ├── strategy
   │   │   ├── clip_ratio
   │   │   ├── optim: OptimizerArguments
   │   │   └── ...
   │   ├── rollout: RolloutArguments
   │   └── ref: RefArguments
   ├── critic: CriticArguments
   ├── reward_model: RewardModelArguments
   ├── algorithm: AlgorithmArguments
   │   ├── adv_estimator
   │   ├── workflow_type
   │   └── kl_ctrl: KLCtrlArguments
   ├── trainer: TrainingArguments
   ├── custom_reward_function: CustomRewardArguments
   ├── dag: DagArguments
   └── profiler: ProfilerArguments

----

.. _sec11_extension_guide:

11. Extension Guide
===================

11.1 Custom Pipeline
--------------------

Users can define custom Pipelines:

.. code-block:: python
   :caption: examples/custom_pipeline_example/custom_pipeline.py

   from siirl.execution.dag.pipeline import Pipeline
   from siirl.execution.dag.node import NodeType, NodeRole
   from siirl.execution.dag.task_graph import TaskGraph
   
   def my_custom_pipeline() -> TaskGraph:
       """Custom training pipeline"""
       pipeline = Pipeline("my_custom_pipeline", "My custom RL workflow")
       
       # Add custom nodes
       pipeline.add_node(
           "rollout_actor",
           func="siirl.dag_worker.dagworker:DAGWorker.generate",
           deps=[],
           node_type=NodeType.MODEL_INFERENCE,
           node_role=NodeRole.ROLLOUT
       ).add_node(
           "custom_reward",
           func="my_module.custom_reward:compute_custom_reward",  # Custom function
           deps=["rollout_actor"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.REWARD
       ).add_node(
           "calculate_advantages",
           func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
           deps=["custom_reward"],
           node_type=NodeType.COMPUTE,
           node_role=NodeRole.ADVANTAGE
       ).add_node(
           "actor_train",
           func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
           deps=["calculate_advantages"],
           node_type=NodeType.MODEL_TRAIN,
           node_role=NodeRole.ACTOR
       )
       
       return pipeline.build()

Specify in configuration:

.. code-block:: yaml

   dag:
     custom_pipeline_fn: "my_module.custom_pipeline:my_custom_pipeline"

11.2 Custom Reward Function
---------------------------

.. code-block:: python
   :caption: siirl/user_interface/rewards_interface/custom_reward.py

   from siirl.dag_worker.data_structures import NodeOutput
   from tensordict import TensorDict
   
   def compute_custom_reward(batch: TensorDict, config, **kwargs) -> NodeOutput:
       """Custom Reward computation function"""
       
       # Get generated responses
       responses = batch["responses"]
       prompts = batch["prompts"]
       
       # Custom reward logic
       rewards = []
       for prompt, response in zip(prompts, responses):
           # Implement your reward function
           score = my_scoring_function(prompt, response)
           rewards.append(score)
       
       # Convert to token-level rewards
       token_level_rewards = torch.zeros_like(batch["attention_mask"])
       for i, score in enumerate(rewards):
           # Assign sequence-level reward to last token
           token_level_rewards[i, -1] = score
       
       batch["token_level_scores"] = token_level_rewards
       batch["token_level_rewards"] = token_level_rewards
       
       metrics = {"reward/mean_score": np.mean(rewards)}
       
       return NodeOutput(batch=batch, metrics=metrics)

11.3 Custom Advantage Estimator
-------------------------------

.. code-block:: python
   :caption: Registering Custom Advantage Estimator

   from siirl.dag_worker.core_algos import register_adv_est
   from siirl.execution.scheduler.enums import AdvantageEstimator
   
   @register_adv_est("my_custom_adv")  # Or use enum
   def compute_my_custom_advantage(
       token_level_rewards: torch.Tensor,
       response_mask: torch.Tensor,
       **kwargs
   ):
       """Custom Advantage estimation"""
       
       # Implement your advantage estimation logic
       advantages = ...
       returns = ...
       
       return advantages, returns

11.4 Custom Policy Loss
-----------------------

.. code-block:: python
   :caption: Registering Custom Policy Loss

   from siirl.dag_worker.core_algos import register_policy_loss
   
   @register_policy_loss("my_custom_loss")
   def compute_my_custom_policy_loss(
       old_log_prob: torch.Tensor,
       log_prob: torch.Tensor,
       advantages: torch.Tensor,
       response_mask: torch.Tensor,
       loss_agg_mode: str = "token-mean",
       config = None,
       rollout_is_weights = None,
   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
       """Custom policy loss"""
       
       # Implement your policy loss logic
       pg_loss = ...
       pg_clipfrac = ...
       ppo_kl = ...
       pg_clipfrac_lower = ...
       
       return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

----

Appendix A: Code File Navigation
================================

::

   siirl/
   ├── main_dag.py                           # Main entry point
   ├── dag_worker/                           # DAG Worker module
   │   ├── dagworker.py                      # Core Worker class (~1320 lines)
   │   ├── core_algos.py                     # RL algorithm implementations
   │   ├── dag_utils.py                      # Utility functions
   │   ├── checkpoint_manager.py             # Checkpoint management
   │   ├── validator.py                      # Validation logic
   │   ├── metrics_collector.py              # Metrics collection
   │   └── data_structures.py                # Data structure definitions
   ├── execution/                            # Execution engine
   │   ├── dag/                              # DAG definitions
   │   │   ├── __init__.py                   # Module exports
   │   │   ├── task_graph.py                 # TaskGraph class
   │   │   ├── node.py                       # Node/NodeType/NodeRole/NodeStatus classes
   │   │   ├── pipeline.py                   # Pipeline Builder API
   │   │   ├── builtin_pipelines.py          # Built-in Pipelines (GRPO/PPO/DAPO/Embodied)
   │   │   └── task_loader.py                # Graph splitting utilities
   │   ├── scheduler/                        # Scheduler
   │   │   ├── task_scheduler.py             # Task scheduling
   │   │   ├── process_group_manager.py      # Process group management
   │   │   ├── launch.py                     # Ray launcher
   │   │   └── enums.py                      # Enum definitions
   │   └── metric_worker/                    # Distributed metrics
   │       └── metric_worker.py              # MetricWorker Actor
   ├── engine/                               # Model execution engine
   │   ├── actor/                            # Actor Workers
   │   ├── critic/                           # Critic Workers
   │   ├── rollout/                          # Rollout Workers (vLLM/SGLang/HF)
   │   ├── reward_model/                     # Reward Model Workers
   │   ├── reward_manager/                   # Reward Managers (naive/parallel/dapo/embodied)
   │   └── sharding_manager/                 # Weight sharding management (FSDP/Megatron)
   ├── data_coordinator/                     # Data coordinator
   │   ├── data_buffer.py                    # DataCoordinator Actor
   │   ├── dataloader/                       # Distributed DataLoader
   │   ├── protocol.py                       # Data protocol
   │   └── sample.py                         # Sample/SampleInfo
   ├── user_interface/                       # User extension interfaces
   │   ├── filter_interface/                 # Filtering plugins
   │   │   ├── dapo.py                       # DAPO dynamic sampling
   │   │   └── embodied.py                   # Embodied dynamic sampling
   │   └── rewards_interface/                # Custom reward interfaces
   ├── params/                               # Configuration parameters
   │   ├── __init__.py                       # SiiRLArguments
   │   ├── parser.py                         # Configuration parser
   │   ├── data_args.py                      # Data parameters
   │   ├── model_args.py                     # Model parameters
   │   └── training_args.py                  # Training parameters
   └── utils/                                # Utilities
       ├── checkpoint/                       # Checkpoint utilities
       ├── logger/                           # Logging utilities
       ├── model_utils/                      # Model utilities
       └── reward_score/                     # Reward computation

----

Summary
=======

This document provides a comprehensive guide to siiRL's architecture implementation, including:

1. **Architecture Overview**: siiRL's position in distributed RL systems and core advantages
2. **DistFlow Design Philosophy**: Fully distributed, multi-controller paradigm design
3. **Program Entry**: main_dag.py and MainRunner startup flow
4. **DAG Planner**: Pipeline API, TaskGraph, TaskScheduler implementation
5. **DAG Worker**: Core execution unit initialization, training loop, node execution
6. **Data Coordinator**: Distributed data management and length balancing algorithm
7. **Engine**: Actor/Critic/Rollout/Reference/Reward Worker implementations
8. **Core Algorithms**: Advantage estimators, Policy Loss function implementations
9. **Execution Flow**: Complete GRPO/PPO training flows
10. **Configuration**: Key configuration parameters explained
11. **Extension Guide**: Custom Pipeline, Reward, Advantage, Policy Loss

By reading this document, readers should gain a deep understanding of siiRL's design philosophy and implementation details, providing a solid foundation for future development, optimization, and extension work.

**References**:

- siiRL Paper: `DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training <https://arxiv.org/abs/2507.13833>`__
- Official Documentation: `https://siirl.readthedocs.io/ <https://siirl.readthedocs.io/>`__
- GitHub Repository: `https://github.com/sii-research/siiRL <https://github.com/sii-research/siiRL>`__
