=====================================
siiRL: The DistFlow Programming Guide
=====================================

siiRL evolves from the design principles of earlier RL frameworks but introduces **DistFlow**, a fundamentally new architecture designed to overcome critical bottlenecks in scalability and flexibility. This guide explains the design philosophy of DistFlow and how its principles are realized in the siiRL codebase.

Paper: `DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training <https://arxiv.org/pdf/2507.13833>`__

Motivation: Overcoming the Limits of Centralized Control
---------------------------------------------------------

Previous-generation RLHF frameworks, while functional, were often built on a hybrid-controller architecture. In this model, a single, central process orchestrated the high-level algorithm logic and, most critically, managed the entire data lifecycle. This included the initial loading of massive datasets and the collection and redistribution of all intermediate data (rollouts, rewards, advantages) between computational stages.

This centralized design created two fundamental limitations that hindered large-scale research:

1.  **The Single-Controller Bottleneck**: Forcing all data to flow through a single node creates severe I/O and communication overhead. As the number of GPUs and the data volume (e.g., in multi-modal or long-context tasks) increase, this central controller is easily overwhelmed, leading to system instability, out-of-memory (OOM) errors, and a hard ceiling on scalability.
2.  **Rigid Algorithmic Pipelines**: The computational workflow in these systems was often hard-coded, tightly coupling the algorithmic logic with the execution engine. Modifying the pipeline—for instance, to experiment with a new reward calculation method or a different policy loss—required deep changes to the core framework source code, dramatically slowing down the cycle of research and innovation.

**DistFlow** was designed from the ground up to solve these problems by adopting a **fully distributed, multi-controller paradigm**. There is no central orchestrator. Instead, data management and execution logic are decentralized across all workers, eliminating the central bottleneck and providing a flexible, DAG-defined interface for algorithm design.

The DistFlow Architecture
-------------------------

DistFlow is composed of three core components that work in concert to deliver a scalable and flexible RL training system.


.. figure:: https://github.com/sii-research/siiRL/raw/main/asset/overview.png
   :alt: The DistFlow Execution Diagram
   :align: center

   The DistFlow architecture, where a user-defined DAG is decomposed by the DAG Planner and executed by distributed DAG Workers, with data flow managed by the Data Coordinator.

1.  **The DAG Planner**
    The DAG Planner is the "brain" of the system. It takes a high-level description of the RL algorithm, defined by the user as a **Directed Acyclic Graph (DAG)** in a YAML file. Its primary job is to translate this logical workflow into a concrete execution plan. It intelligently decomposes the global DAG into a series of smaller, sequential task chains, one for each `DAG Worker`. This process considers the user-selected execution mode (e.g., serial for colocated setups, parallel for distributed clusters) to ensure efficient, contention-free execution.

2.  **The DAG Worker**
    The `DAG Worker` is the fundamental execution unit of the framework. Each worker is an independent actor, typically bound to a single GPU. It receives its assigned task chain from the DAG Planner and executes it from start to finish. The worker is entirely self-sufficient; it is responsible for its own environment setup, model initialization, and task execution, making the system highly modular and scalable.

3.  **The Data Coordinator**
    To eliminate the data bottleneck, DistFlow introduces the `Data Coordinator`, a high-level abstraction for the entire data lifecycle. It is composed of two specialized, distributed components:
    - **Distributed Dataloader**: Each `DAG Worker` has its own dataloader. Before training begins, the dataset is partitioned, and each dataloader only loads the specific shard of data its corresponding worker will need. This prevents OOM errors from loading a massive dataset on a single node and parallelizes data loading.
    - **Distributed Databuffer**: This component manages the dynamic flow of *intermediate* data between different stages of the RL workflow (e.g., from generation to evaluation). It is aware of the parallelism strategies of each stage and automatically handles the complex data redistribution (shuffling and re-partitioning) required when transitioning between stages with different numbers of data-parallel workers.

Codebase Walkthrough: How DistFlow is Implemented
-------------------------------------------------

The distributed philosophy of DistFlow is directly reflected in the structure of the codebase.

**The Entrypoint: `siirl.main_dag.py`**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unlike previous frameworks where the entrypoint contained the entire PPO loop, the role of `main_dag.py` in siiRL is much simpler. It acts as a **launcher**. Its primary responsibilities are:
1.  Initializing the distributed environment (Ray).
2.  Creating shared resources like the `ProcessGroupManager` and `DataBuffer`s.
3.  Loading the user's workflow DAG from the YAML file and passing it to the `DAGPlanner`.
4.  Instantiating and launching the `DAGWorker` actors, providing each with its unique, decomposed `TaskGraph`.

The algorithmic logic does not live here; it lives in the DAG configuration and is executed by the workers.

**The Core Executor: `siirl.workers.dag_worker.dagworker.py`**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `DAGWorker` is the primary execution engine. When initialized, it receives its own `TaskGraph` object. Its main `run` loop iterates through the nodes of this graph and executes them sequentially.

The `DAGWorker` internally maps the `node_type` and `node_role` from the DAG configuration to specific internal methods (e.g., `_execute_model_train`, `_execute_model_inference`, `_execute_compute_node`). This is where the abstract DAG definition is translated into concrete actions like running model inference, calculating advantages, or updating model weights.

.. code-block:: python
   
   # Inside DAGWorker's main loop (simplified)
   for node in self.taskgraph.get_nodes():
       # Gather inputs from dependency nodes' outputs
       inputs = self._gather_node_inputs(node)

       if node.node_type == NodeType.MODEL_TRAIN:
           output = self._execute_model_train(node, **inputs)
       elif node.node_type == NodeType.MODEL_INFERENCE:
           output = self._execute_model_inference(node, **inputs)
       elif node.node_type == NodeType.COMPUTE:
           output = self._execute_compute_node(node, **inputs)
       
       # Store output for dependent nodes
       self._update_internal_data_cache(node.node_id, output)


**The Power of Abstraction: `workflow_*.yaml`**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This design means a user can define a complex algorithm like GRPO entirely in YAML without writing any new Python code for the orchestration logic.

.. code-block:: yaml

   # from workflow_grpo.yaml
   nodes:
     - node_id: "rollout_actor"
       node_type: "MODEL_INFERENCE"
       node_role: "ROLLOUT"
       dependencies: []

     - node_id: "function_reward"
       node_type: "COMPUTE"
       node_role: "REWARD"
       dependencies:
         - "rollout_actor"
     
     # ... and so on

The `DAGWorker` reads this structure and executes it, handling all the underlying distributed communication and resource management automatically.

Key Takeaways
-------------
- **Fully Distributed**: siiRL's DistFlow architecture eliminates the single-controller bottleneck by making every worker a self-sufficient orchestrator with its own data management.
- **Flexibility through Abstraction**: The RL algorithm's logic is defined in a high-level DAG configuration, completely decoupling it from the physical execution engine. This makes it easy to design and test new algorithms.
- **Scalability and Efficiency**: By distributing all tasks, the framework is designed to scale linearly and efficiently, even in data-intensive, large-scale scenarios.