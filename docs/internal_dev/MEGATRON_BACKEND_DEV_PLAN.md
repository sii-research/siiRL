# Megatron Backend Integration Dev. Plan

## Overview
This doc. outlines the development plan to enable full Megatron backend support in siiRL. Currently, siiRL only supports FSDP as a training backend, but has extensive Megatron foundation code (ported from veRL) that needs to be integrated into the DAG execution system.

## Current Code Analysis

### What Exists (ported from veRL) ✅
- **Megatron Foundation Code**: Complete implementation in `siirl/workers/megatron_workers.py`
  - `ActorRolloutRefWorker(MegatronWorker)`: Hybrid worker supporting actor/rollout/reference roles
  - `CriticWorker(MegatronWorker)`: Critic model implementation
  - `RewardModelWorker(MegatronWorker)`: Reward model implementation

- **Megatron-Specific Components**:
  - `MegatronPPOActor`: Complete PPO actor implementation with pipeline/tensor parallelism
  - `MegatronPPOCritic`: Critic implementation with Megatron backend
  - Megatron utilities in `utils/megatron/`: optimizer, TP, SP, PP, etc.
  - Checkpoint managers: `MegatronCheckpointManager`
  - Model definitions: Llama and Qwen2 architectures in `models/llama/megatron/` and `models/qwen2/megatron/`

- **Rollout Integration**:
  - vLLM rollout support with Megatron: `MegatronVLLMShardingManager`
  - Weight resharding between training and inference engines
  - Memory management with model/optimizer offloading

### What's Missing ❌
- **DAG Integration**: No `strategy="megatron"` support in DAG execution system
- **Process Group Integration**: Megatron initialization not integrated with siiRL's `ProcessGroupManager`
- **Configuration Pipeline**: Strategy selection logic in `InitializationMixin` only supports FSDP
- **Configuration Templates**: No Megatron-specific config examples or validation
- **E2E Large Model Tunning**: No production readniess for large MoE model's RL training, such as 235B, 671B et al.

## Implementation Plan

Each step works with standalone unit tests.

### Step 1: Enable basic Megatron backend selection in DAG system

**Tasks**:
1. **Strategy Registration**
   - Add `MEGATRON_STRATEGIES = ["megatron"]` and Update strategy validation logic

2. **Worker Class Mapping**
   - Modify DAGWorker `InitializationMixin`
   - Map node roles to appropriate Megatron worker classes

3. **Configuration Validation**
   - Add Megatron-specific config validation
   - Ensure required Megatron parameters are present

### Step 2: Integrate Megatron with siiRL's resource management and launching system

**Notes**: Megatron manages its own internal parallelism groups via `mpu.initialize_model_parallel()`. The DAG-level process groups serve as boundaries, and we need to ensure Megatron's total parallelism (TP × PP × DP) fits within the assigned process group size. In addition, we need to ensure compatibility with existing Ray-based distributed setup.

**Tasks**:
1. **Process Group Integration**
   - Updated Megatron worker signatures to accept `process_group` parameter
   - Added process group storage and rank/world_size tracking
   - Megatron's `mpu.initialize_model_parallel()` creates internal groups within process boundaries

2. **Resource Allocation**
   - Update `RayTrainer` in `scheduler/launch.py` for Megatron requirements
   - Handle tensor_model_parallel_size and pipeline_model_parallel_size
   - Validate GPU allocation matches parallelism requirements (TP × PP × DP ≤ process_group_size)

3. **Configuration Templates** 
   - Create `megatron_ppo_dag_trainer.yaml` config template
   - Add Megatron-specific parameter documentation
   - Provide example configurations for different model sizes

### Step 3: Ensure robust rollout integration (vLLM only, by far we leave SGLang alone)

**Tasks**:
1. **vLLM Integration Validation**
   - Verify `MegatronVLLMShardingManager` works correctly, no precision issue.
   - Test weight resharding between training and inference

2. **Memory Management**
   - Validate/implement proper model/optimizer offloading for Megatron? (not sure, need revisit)
   - Optimize memory usage during rollout transitions
   - Handle large model scenarios (235B+ parameters)

3. **Performance Optimization**
   - Probably, tune communication, optimize tensor resharding operations
   - Benchmark against FSDP backend

### Step 4: Testing & Documentation 

**Tasks**:
1. **End-to-end Testing**
   - Create test suite for Megatron backend
   - Test various model sizes (7B, 13B, 70B, 235B, 671B?)
   - Validate training convergence matches FSDP results (brefily with Wandb)

2. **Documentation**

3. **Examples**
   - Create example training scripts
   - Add model-specific configurations (Might provide scaling guidelines)

## Notices:

1. Need to maintain compatibility with existing workflows, i.e., FSDP, SGLang et al.
2. Need critical benchmarking before release.
3. Provide sensible defaults and validation.