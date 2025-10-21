# Python-Based Pipeline API

## Overview

siiRL now supports defining training pipelines directly in Python code, making workflows more transparent and easier to customize. This new API coexists with the legacy YAML-based configuration for backward compatibility.

## Key Features

✅ **Explicit Function Binding**: See exactly which functions execute at each pipeline step
✅ **No Hidden Logic**: No more guessing which method is called based on `NodeType`/`NodeRole`
✅ **Easy Customization**: Mix built-in and custom functions seamlessly
✅ **Type Safe**: Full IDE support with autocompletion and type checking
✅ **Backward Compatible**: Existing YAML configs continue to work

## Quick Start

### Option 1: Use Built-in Pipelines (Recommended for Standard Algorithms)

```bash
# Just run with your algorithm - built-in pipeline is used automatically
python3 -m siirl.main_dag \
    algorithm.adv_estimator=grpo \
    data.train_files=/path/to/data \
    actor_rollout_ref.model.path=/path/to/model \
    ...
```

### Option 2: Use Custom Pipeline

```bash
# Specify your custom pipeline function
python3 -m siirl.main_dag \
    algorithm.adv_estimator=grpo \
    dag.custom_pipeline_fn="examples.custom_pipeline_example.custom_grpo:grpo_with_custom_reward" \
    data.train_files=/path/to/data \
    ...
```

## Built-in Pipelines

### GRPO Pipeline

```python
from siirl.execution.dag.builtin_pipelines import grpo_pipeline

# The pipeline definition is clear and explicit:
# 1. rollout_actor: DAGWorker.generate
# 2. function_reward: DAGWorker.compute_reward
# 3. calculate_advantages: DAGWorker.compute_advantage
# 4. actor_old_log_prob: DAGWorker.compute_old_log_prob (forward only)
# 5. reference_log_prob: DAGWorker.compute_ref_log_prob
# 6. actor_train: DAGWorker.train_actor

graph = grpo_pipeline()
```

### PPO Pipeline

```python
from siirl.execution.dag.builtin_pipelines import ppo_pipeline

# PPO adds critic training:
# 1. rollout_actor: DAGWorker.generate
# 2. function_reward: DAGWorker.compute_reward
# 3. compute_value: DAGWorker.compute_value (forward only)
# 4. calculate_advantages: DAGWorker.compute_advantage (GAE)
# 5. actor_old_log_prob: DAGWorker.compute_old_log_prob (forward only)
# 6. reference_log_prob: DAGWorker.compute_ref_log_prob
# 7. actor_train: DAGWorker.train_actor
# 8. critic_train: DAGWorker.train_critic

graph = ppo_pipeline()
```

## Custom Pipeline Examples

### Example 1: Custom Reward Function

```python
# examples/custom_pipeline_example/custom_grpo.py

from siirl.execution.dag.pipeline import Pipeline
from siirl.data_coordinator import DataProto
from siirl.dag_worker.data_structures import NodeOutput

def grpo_with_custom_reward():
    """Replace reward function while keeping rest standard."""
    pipeline = Pipeline("grpo_custom_reward")

    # Standard rollout
    pipeline.add_node(
        "rollout_actor",
        func="siirl.dag_worker.dagworker:DAGWorker.generate",
        deps=[]
    )

    # YOUR custom reward function
    pipeline.add_node(
        "custom_reward",
        func="examples.custom_pipeline_example.custom_grpo:my_custom_reward_fn",
        deps=["rollout_actor"]
    )

    # Standard training
    pipeline.add_node(
        "calculate_advantages",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
        deps=["custom_reward"]
    ).add_node(
        "actor_train",
        func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
        deps=["calculate_advantages"]
    )

    return pipeline.build()

def my_custom_reward_fn(batch: DataProto, **kwargs) -> NodeOutput:
    """Your custom reward logic."""
    # Implement your scoring logic here
    responses = batch.non_tensor_batch.get("responses", [])
    rewards = compute_my_rewards(responses)
    batch.batch["rewards"] = rewards
    return NodeOutput(batch=batch, metrics={})
```

### Example 2: Dual Reward Pipeline (Parallel Execution)

```python
def grpo_with_dual_rewards():
    """Compute two rewards in parallel and combine them."""
    pipeline = Pipeline("grpo_dual_rewards")

    pipeline.add_node("rollout", func="siirl.dag_worker.dagworker:DAGWorker.generate", deps=[])

    # Two rewards computed in parallel
    pipeline.add_node(
        "accuracy_reward",
        func="examples.custom_pipeline_example.custom_grpo:compute_accuracy_reward",
        deps=["rollout"]
    ).add_node(
        "style_reward",
        func="examples.custom_pipeline_example.custom_grpo:compute_style_reward",
        deps=["rollout"]
    )

    # Combine rewards
    pipeline.add_node(
        "combine_rewards",
        func="examples.custom_pipeline_example.custom_grpo:combine_dual_rewards",
        deps=["accuracy_reward", "style_reward"]
    )

    # Standard training
    pipeline.add_node("advantage", func="...", deps=["combine_rewards"])
    pipeline.add_node("train", func="...", deps=["advantage"])

    return pipeline.build()
```

## API Reference

### Pipeline Class

```python
from siirl.execution.dag.pipeline import Pipeline

pipeline = Pipeline(
    pipeline_id="my_pipeline",
    description="Optional description"
)

pipeline.add_node(
    node_id="unique_node_name",
    func="module.path:ClassName.method",  # or direct callable
    deps=["parent_node1", "parent_node2"],  # optional
    config=NodeConfig(...),  # optional
    only_forward_compute=True  # optional kwargs
)

graph = pipeline.build()  # Returns TaskGraph
```

### Function Path Format

Two ways to specify functions:

1. **String Path** (recommended for built-in functions):
   ```python
   func="siirl.dag_worker.dagworker:DAGWorker.generate"
   func="siirl.dag_worker.dagworker:DAGWorker.compute_reward"
   ```

2. **Direct Callable** (for custom functions):
   ```python
   func=my_custom_function
   ```

## Migration Guide

### Before (YAML - Hidden Logic)

```yaml
# siirl/global_config/config/workflow_grpo.yaml
nodes:
  - node_id: "rollout_actor"
    node_type: "MODEL_INFERENCE"  # Which function does this call? 🤔
    node_role: "ROLLOUT"
    dependencies: []
```

**Problem**: User has to remember that `node_type=MODEL_INFERENCE` + `node_role=ROLLOUT` → calls `DAGWorker.generate`

### After (Python - Explicit)

```python
# examples/my_pipeline.py
from siirl.execution.dag.pipeline import Pipeline

def my_grpo():
    pipeline = Pipeline("grpo")

    pipeline.add_node(
        "rollout_actor",
        func="siirl.dag_worker.dagworker:DAGWorker.generate",  # Crystal clear! ✅
        deps=[]
    )

    return pipeline.build()
```

## Configuration Priority

When loading pipelines, siiRL checks in this order:

1. **Custom Pipeline** (`dag.custom_pipeline_fn`): User-specified Python function
2. **Built-in Pipeline**: Automatically selected based on `algorithm.adv_estimator`
3. **Legacy YAML** (`dag.workflow_path`): Backward compatibility

## File Structure

```
siirl/
├── execution/dag/
│   ├── pipeline.py              # Core Pipeline API
│   ├── builtin_pipelines.py     # Built-in pipelines (GRPO, PPO, DAPO)
│   └── config_loader.py         # Legacy YAML loader (kept for compatibility)
│
examples/
└── custom_pipeline_example/
    ├── custom_grpo.py           # Example custom pipelines
    └── README.md                # This file
```

## Running Examples

```bash
# Visualize all example pipelines
cd examples/custom_pipeline_example
python custom_grpo.py

# This generates visualizations in ./pipeline_visualizations/
```

## Tips

1. **Start with built-in pipelines**: They work out of the box for standard algorithms
2. **Customize incrementally**: Copy a built-in pipeline and modify only what you need
3. **Use explicit function paths**: Makes debugging much easier
4. **Test your pipeline**: Run `pipeline.build()` to catch errors early

## Backward Compatibility

All existing training scripts using YAML configs will continue to work without any changes. The new Python API is purely additive.

## Support

- See `examples/custom_pipeline_example/custom_grpo.py` for complete examples
- Check `siirl/execution/dag/builtin_pipelines.py` for built-in implementations
- Refer to `CLAUDE.md` for detailed architecture documentation
