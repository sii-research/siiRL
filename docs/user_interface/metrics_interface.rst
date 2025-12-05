=================
Metrics Interface
=================

Custom metrics allow you to track and aggregate any quantitative measures during training and validation. siiRL provides a distributed, Ray-based metrics system that automatically handles aggregation across all workers using various reduction operations (mean, max, min, sum).

Architecture Overview
---------------------

::

                         Distributed Metrics Architecture
   ==============================================================================

   DAGWorker 0        DAGWorker 1        DAGWorker 2        DAGWorker N
   +-----------+      +-----------+      +-----------+      +-----------+
   | compute   |      | compute   |      | compute   |      | compute   |
   | metrics   |      | metrics   |      | metrics   |      | metrics   |
   +-----+-----+      +-----+-----+      +-----+-----+      +-----+-----+
         |                  |                  |                  |
         v                  v                  v                  v
   +-----+-----+      +-----+-----+      +-----+-----+      +-----+-----+
   | Metric    |      | Metric    |      | Metric    |      | Metric    |
   | Client    |      | Client    |      | Client    |      | Client    |
   +-----+-----+      +-----+-----+      +-----+-----+      +-----+-----+
         |                  |                  |                  |
         +------------------+------------------+------------------+
                                    |
                                    v
                         +-------------------+
                         |   MetricWorker    |  (Ray Actor - Singleton)
                         |   (Aggregator)    |
                         +-------------------+
                         | - Collect metrics |
                         | - Wait for all    |
                         |   workers         |
                         | - Aggregate:      |
                         |   mean/max/min/   |
                         |   sum             |
                         +--------+----------+
                                  |
                                  v
                         +-------------------+
                         |  Final Metrics    |
                         | (to Logger/WandB) |
                         +-------------------+

   ==============================================================================

   Metrics Data Flow:

   +-------------+     +----------------+     +----------------+     +--------+
   | TensorDict  | --> | compute_*      | --> | MetricClient   | --> | Metric |
   | (batch)     |     | _metric()      |     | .submit_metric |     | Worker |
   +-------------+     +----------------+     +----------------+     +--------+
                              |
                              v
                       +-------------+
                       | Dict[str,   |
                       |   float]    |
                       | {name: val} |
                       +-------------+

   ==============================================================================

**Key Files:**

- ``siirl/execution/metric_worker/metric_worker.py`` - Ray-based distributed metrics aggregation
- ``siirl/utils/metrics/metric_utils.py`` - Core metric computation functions
- ``siirl/execution/metric_worker/utils.py`` - Aggregation function utilities

Quick Start
-----------

Method 1: Extending Core Metrics Functions (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1:** Create your metric computation function in ``metric_utils.py``

.. code-block:: python

   # Add to siirl/utils/metrics/metric_utils.py

   def compute_custom_data_metrics(data: TensorDict) -> Dict[str, float]:
       """Custom metrics computed from batch data"""
       metrics = {}

       # Token-level accuracy
       if "correct_tokens" in data and "attention_mask" in data:
           correct = data["correct_tokens"].float()
           mask = data["attention_mask"].float()
           accuracy = (correct * mask).sum() / mask.sum()
           metrics["custom/token_accuracy/mean"] = accuracy.item()

       # Response quality score
       if "responses" in data and "response_mask" in data:
           response_quality = compute_response_quality_score(data)
           metrics["custom/response_quality/mean"] = response_quality.mean().item()
           metrics["custom/response_quality/max"] = response_quality.max().item()
           metrics["custom/response_quality/min"] = response_quality.min().item()

       return metrics

   def compute_response_quality_score(data: TensorDict) -> torch.Tensor:
       """Helper function to compute response quality"""
       responses = data["responses"]
       response_mask = data["response_mask"]

       # Example: vocabulary diversity score
       unique_tokens_per_response = []
       for i in range(responses.shape[0]):
           response_tokens = responses[i][response_mask[i].bool()]
           unique_count = len(torch.unique(response_tokens))
           unique_tokens_per_response.append(unique_count)

       return torch.tensor(unique_tokens_per_response, device=responses.device).float()

**Step 2:** Submit metrics using MetricClient

.. code-block:: python

   # Usage in your training loop
   from siirl.execution.metric_worker.metric_worker import MetricClient

   # In your DAG worker or training script
   custom_metrics = compute_custom_data_metrics(batch)
   metric_client.submit_metric(custom_metrics, world_size)

Current Metrics System
----------------------

Built-in Metrics Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following tables list all built-in metrics provided by siiRL.

**Data Metrics** (from ``compute_data_metric`` in ``metric_utils.py``):

.. list-table:: Critic Metrics
   :header-rows: 1
   :widths: 40 60

   * - Metric Name
     - Description
   * - ``critic/score/mean|max|min``
     - Sequence-level scores from token-level scores
   * - ``critic/rewards/mean|max|min``
     - Sequence-level rewards from token-level rewards
   * - ``critic/advantages/mean|max|min``
     - Advantages (masked by response_mask)
   * - ``critic/returns/mean|max|min``
     - Returns (masked by response_mask)
   * - ``critic/values/mean|max|min``
     - Value function estimates (if available)
   * - ``critic/vf_explained_var``
     - Explained variance of value function

.. list-table:: Response Analysis Metrics
   :header-rows: 1
   :widths: 40 60

   * - Metric Name
     - Description
   * - ``response/length/mean|max|min``
     - Response token lengths
   * - ``response/clip_ratio/mean``
     - Proportion hitting max response length
   * - ``response/correct_length/mean|max|min``
     - Lengths for responses with reward > 0.5
   * - ``response/wrong_length/mean|max|min``
     - Lengths for responses with reward ≤ 0.5

.. list-table:: Prompt Analysis Metrics
   :header-rows: 1
   :widths: 40 60

   * - Metric Name
     - Description
   * - ``prompt/length/mean|max|min``
     - Prompt token lengths
   * - ``prompt/clip_ratio/mean``
     - Proportion hitting max prompt length

.. list-table:: System & Multi-turn Metrics
   :header-rows: 1
   :widths: 40 60

   * - Metric Name
     - Description
   * - ``perf/process_cpu_mem_used_gb``
     - CPU memory usage per process
   * - ``num_turns/min|max|mean``
     - Statistics for multi-turn conversations

**Timing Metrics** (from ``compute_timing_metrics``):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric Name
     - Description
   * - ``timing_s/{stage}``
     - Raw timing in seconds for each stage
   * - ``timing_per_token_ms/{stage}``
     - Per-token timing in milliseconds

Stages: ``gen``, ``ref``, ``values``, ``adv``, ``update_critic``, ``update_actor``

**Throughput Metrics** (from ``compute_throughout_metrics``):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric Name
     - Description
   * - ``perf/total_num_tokens``
     - Total tokens processed
   * - ``perf/time_per_step``
     - Time per training step
   * - ``perf/throughput``
     - Tokens per second per GPU

**Validation Metrics** (from ``process_validation_metrics``):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric Name
     - Description
   * - ``val-core/{data_source}/{var}/mean@N``
     - Mean across N samples
   * - ``val-core/{data_source}/{var}/best@N/mean|std``
     - Bootstrap best-of-N statistics
   * - ``val-core/{data_source}/{var}/worst@N/mean|std``
     - Bootstrap worst-of-N statistics
   * - ``val-core/{data_source}/{var}/maj@N/mean|std``
     - Bootstrap majority voting statistics
   * - ``val/test_score/{data_source}``
     - Test score per data source

Custom Metrics Implementation
-----------------------------

Method 1: Custom Data Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extend the data metrics computed from training batches:

.. code-block:: python

   # Add to metric_utils.py
   def compute_custom_training_metrics(data: TensorDict) -> Dict[str, float]:
       """Custom training-specific metrics"""
       metrics = {}

       # Policy entropy (exploration measure)
       if "policy_logits" in data:
           logits = data["policy_logits"]
           probs = torch.softmax(logits, dim=-1)
           entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
           response_mask = data.get("response_mask", torch.ones_like(entropy))

           # Only compute entropy for response tokens
           masked_entropy = entropy * response_mask.float()
           valid_entropy = masked_entropy.sum() / response_mask.sum()

           metrics["training/policy_entropy/mean"] = valid_entropy.item()

       # Gradient norm tracking
       if "grad_norm" in data:
           metrics["training/grad_norm/mean"] = data["grad_norm"].item()

       # Loss convergence tracking
       if "loss_values" in data:
           loss_values = data["loss_values"]
           metrics["training/loss/mean"] = loss_values.mean().item()
           metrics["training/loss/std"] = loss_values.std().item()

       return metrics

   # Usage in MetricClient.compute_local_data_metric
   def compute_local_data_metric(self, data: TensorDict, world_size: int):
       # Standard metrics
       standard_metrics = compute_data_metric(data)

       # Add custom metrics
       custom_metrics = compute_custom_training_metrics(data)

       # Combine and submit
       all_metrics = {**standard_metrics, **custom_metrics}
       self.submit_metric(all_metrics, world_size)

Method 2: Custom Validation Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add custom validation metrics with bootstrap sampling:

.. code-block:: python

   # Add to metric_utils.py
   def compute_custom_validation_metrics(
       data_sources: list[str],
       sample_inputs: list[str],
       infos_dict: dict[str, list],
       sample_turns: list[int]
   ) -> dict[str, float]:
       """Custom validation metrics with bootstrap analysis"""

       # Extract custom fields from infos_dict
       custom_metrics = {}

       if "custom_score" in infos_dict:
           # Group by data source
           source_scores = defaultdict(list)
           for i, source in enumerate(data_sources):
               source_scores[source].append(infos_dict["custom_score"][i])

           # Compute statistics per source
           for source, scores in source_scores.items():
               if len(scores) > 0:
                   custom_metrics[f"val/custom_score/{source}/mean"] = np.mean(scores)
                   custom_metrics[f"val/custom_score/{source}/std"] = np.std(scores)

                   # Bootstrap sampling for confidence intervals
                   if len(scores) > 1:
                       bootstrap_results = bootstrap_metric(
                           data=scores,
                           subset_size=min(5, len(scores)),
                           reduce_fns=[np.mean, np.max, np.min],
                           n_bootstrap=1000
                       )
                       custom_metrics[f"val/custom_score/{source}/bootstrap_mean"] = bootstrap_results[0][0]
                       custom_metrics[f"val/custom_score/{source}/bootstrap_mean_std"] = bootstrap_results[0][1]

       # Conversation quality for multi-turn
       if "conversation_quality" in infos_dict and len(sample_turns) > 0:
           quality_by_turns = defaultdict(list)
           for i, turns in enumerate(sample_turns):
               if i < len(infos_dict["conversation_quality"]):
                   quality_by_turns[turns].append(infos_dict["conversation_quality"][i])

           for turn_count, qualities in quality_by_turns.items():
               if len(qualities) > 0:
                   custom_metrics[f"val/conversation_quality/turns_{turn_count}/mean"] = np.mean(qualities)

       return custom_metrics

   # Usage in MetricClient.process_local_validation_metrics
   def process_local_validation_metrics(self, data_sources, sample_inputs, infos_dict, sample_turns, world_size):
       # Standard validation metrics
       standard_metrics = process_validation_metrics(data_sources, sample_inputs, infos_dict, sample_turns)

       # Add custom validation metrics
       custom_metrics = compute_custom_validation_metrics(data_sources, sample_inputs, infos_dict, sample_turns)

       # Combine and submit
       all_metrics = {**standard_metrics, **custom_metrics}
       self.submit_metric(all_metrics, world_size)

Method 3: Custom Aggregation Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create custom aggregation functions for specialized reduction operations:

.. code-block:: python

   # Add to execution/metric_worker/utils.py
   def MedianMetric(metrics: List[Metric]):
       """Custom median aggregation"""
       values = [v for metric in metrics
                for v in (metric.value if isinstance(metric.value, list) else [metric.value])]
       return float(torch.median(torch.tensor(values)).item())

   def PercentileMetric(percentile: float):
       """Custom percentile aggregation factory"""
       def _percentile_metric(metrics: List[Metric]):
           values = [v for metric in metrics
                    for v in (metric.value if isinstance(metric.value, list) else [metric.value])]
           return float(torch.quantile(torch.tensor(values), percentile / 100.0).item())
       return _percentile_metric

   # Update MetricFunc to handle custom aggregations
   def MetricFunc(name: str):
       if "median" in name:
           return MedianMetric
       elif "p95" in name:
           return PercentileMetric(95)
       elif "p99" in name:
           return PercentileMetric(99)
       elif "min" in name:
           return MinMetric
       elif "max" in name:
           return MaxMetric
       elif "sum" in name or "total" in name:
           return SumMetric
       else:
           return MeanMetric

   # Usage: name your metrics to trigger specific aggregations
   metrics = {
       "custom/latency/median": latency_values,  # Will use MedianMetric
       "custom/score/p95": score_values,         # Will use 95th percentile
   }

Method 4: Complex Custom Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more sophisticated metrics requiring multiple computation steps:

.. code-block:: python

   # Add to metric_utils.py
   def compute_advanced_metrics(data: TensorDict) -> Dict[str, float]:
       """Advanced metrics requiring complex computation"""
       metrics = {}

       # Sequence coherence analysis
       if "responses" in data and "attention_mask" in data:
           coherence_scores = compute_sequence_coherence(data)
           metrics.update({
               "analysis/coherence/mean": coherence_scores.mean().item(),
               "analysis/coherence/std": coherence_scores.std().item(),
               "analysis/coherence/median": coherence_scores.median().item(),
           })

       # Token transition analysis
       if "responses" in data:
           transition_metrics = analyze_token_transitions(data)
           metrics.update(transition_metrics)

       # Reward distribution analysis
       if "token_level_rewards" in data:
           reward_dist_metrics = analyze_reward_distribution(data)
           metrics.update(reward_dist_metrics)

       return metrics

   def compute_sequence_coherence(data: TensorDict) -> torch.Tensor:
       """Compute coherence score for each sequence"""
       responses = data["responses"]
       attention_mask = data["attention_mask"]
       batch_size = responses.shape[0]

       coherence_scores = []
       for i in range(batch_size):
           # Extract valid tokens for this sequence
           valid_length = attention_mask[i].sum().item()
           sequence = responses[i][:valid_length]

           # Compute local coherence (e.g., token transition smoothness)
           if len(sequence) > 1:
               # Simplified coherence: variance in token values
               coherence = 1.0 / (1.0 + torch.var(sequence.float()).item())
           else:
               coherence = 1.0

           coherence_scores.append(coherence)

       return torch.tensor(coherence_scores, device=responses.device)

   def analyze_token_transitions(data: TensorDict) -> Dict[str, float]:
       """Analyze patterns in token transitions"""
       responses = data["responses"]
       response_mask = data.get("response_mask", torch.ones_like(responses))

       # Count unique transitions
       unique_transitions = set()
       total_transitions = 0

       for i in range(responses.shape[0]):
           response_tokens = responses[i][response_mask[i].bool()]
           if len(response_tokens) > 1:
               for j in range(len(response_tokens) - 1):
                   transition = (response_tokens[j].item(), response_tokens[j+1].item())
                   unique_transitions.add(transition)
                   total_transitions += 1

       diversity_ratio = len(unique_transitions) / max(total_transitions, 1)

       return {
           "analysis/transition_diversity/mean": diversity_ratio,
           "analysis/unique_transitions/total": len(unique_transitions),
           "analysis/total_transitions/total": total_transitions,
       }

Integration with Training Workflow
----------------------------------

MetricClient Usage Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``MetricClient`` provides the main interface for submitting metrics:

.. code-block:: python

   from siirl.execution.metric_worker.metric_worker import MetricClient, MetricWorker

   # Initialize metric worker and client
   metric_worker = MetricWorker.remote()
   await metric_worker.start.remote()
   metric_client = MetricClient(metric_worker)

   # During training loop
   for step, batch in enumerate(dataloader):
       # ... training logic ...

       # Submit standard metrics
       metric_client.compute_local_data_metric(batch, world_size)

       # Submit custom metrics
       custom_metrics = compute_advanced_metrics(batch)
       metric_client.submit_metric(custom_metrics, world_size)

       # Submit timing metrics
       timing_data = {"step": step_time, "forward": forward_time}
       metric_client.compute_local_timing_metrics(batch, timing_data, world_size)

       # Wait for metrics to be processed
       metric_client.wait_submit()

   # Get final aggregated results
   final_metrics = metric_client.wait_final_res()

Ray-based Distributed Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system uses Ray actors for distributed metrics processing:

**MetricWorker Actor:**
- Runs asynchronously to collect metrics from all workers
- Aggregates metrics when all processes have submitted values
- Supports different aggregation functions (mean, max, min, sum)
- Automatically handles timing metric renaming (``timing_s/`` → ``perf/delta_time/``)

**Aggregation Logic:**
- Metrics are collected in a queue until all workers (``world_size``) submit
- Each metric triggers computation when the expected number of submissions is reached
- Final results are stored and returned when requested

Special Metric Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some metrics require special aggregation logic:

.. code-block:: python

   # In metric_worker.py
   Special_Metric = {
       "graph_output_handling": MaxMetric,  # Only rollout_tp 0 contributes
   }

Custom metrics can be added to this dictionary for specialized handling.

Advanced Examples
-----------------

Example 1: Model Performance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_model_performance_metrics(data: TensorDict, model_outputs: dict) -> Dict[str, float]:
       """Comprehensive model performance analysis"""
       metrics = {}

       # Attention pattern analysis
       if "attention_weights" in model_outputs:
           attention_weights = model_outputs["attention_weights"]

           # Attention concentration (how focused is attention)
           attention_entropy = -torch.sum(
               attention_weights * torch.log(attention_weights + 1e-9), dim=-1
           )
           metrics["model/attention_entropy/mean"] = attention_entropy.mean().item()

           # Attention on different token types
           if "attention_mask" in data:
               prompt_attention = attention_weights[:, :, :-data["responses"].shape[-1]]
               response_attention = attention_weights[:, :, -data["responses"].shape[-1]:]

               metrics["model/prompt_attention_ratio/mean"] = (
                   prompt_attention.sum() / attention_weights.sum()
               ).item()

       # Hidden state analysis
       if "hidden_states" in model_outputs:
           hidden_states = model_outputs["hidden_states"]

           # Representation diversity
           layer_norms = torch.norm(hidden_states, dim=-1)
           metrics["model/hidden_norm/mean"] = layer_norms.mean().item()
           metrics["model/hidden_norm/std"] = layer_norms.std().item()

       return metrics

Example 2: Conversation Quality Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_conversation_quality_metrics(data: TensorDict) -> Dict[str, float]:
       """Multi-dimensional conversation quality assessment"""
       metrics = {}

       if "responses" not in data or "prompts" not in data:
           return metrics

       responses = data["responses"]
       prompts = data["prompts"]
       response_mask = data.get("response_mask", torch.ones_like(responses))

       batch_size = responses.shape[0]
       quality_scores = []

       for i in range(batch_size):
           # Extract actual tokens (remove padding)
           response_tokens = responses[i][response_mask[i].bool()]
           prompt_tokens = prompts[i]

           # Length appropriateness (not too short, not too long)
           response_length = len(response_tokens)
           length_score = compute_length_appropriateness(response_length)

           # Vocabulary richness
           unique_tokens = len(torch.unique(response_tokens))
           vocab_score = min(unique_tokens / response_length, 1.0) if response_length > 0 else 0

           # Repetition penalty
           repetition_score = compute_repetition_score(response_tokens)

           # Overall quality
           quality = 0.3 * length_score + 0.3 * vocab_score + 0.4 * repetition_score
           quality_scores.append(quality)

       quality_tensor = torch.tensor(quality_scores, device=responses.device)

       return {
           "conversation/quality/mean": quality_tensor.mean().item(),
           "conversation/quality/std": quality_tensor.std().item(),
           "conversation/quality/min": quality_tensor.min().item(),
           "conversation/quality/max": quality_tensor.max().item(),
       }

   def compute_length_appropriateness(length: int, target_length: int = 50) -> float:
       """Compute how appropriate the response length is"""
       if length == 0:
           return 0.0
       ratio = length / target_length
       if ratio <= 1.0:
           return ratio  # Shorter is better than longer
       else:
           return 1.0 / ratio  # Penalize overly long responses

   def compute_repetition_score(tokens: torch.Tensor) -> float:
       """Compute score based on repetition patterns"""
       if len(tokens) <= 1:
           return 1.0

       # Count repeated bigrams
       bigrams = set()
       repeated_bigrams = 0

       for i in range(len(tokens) - 1):
           bigram = (tokens[i].item(), tokens[i+1].item())
           if bigram in bigrams:
               repeated_bigrams += 1
           else:
               bigrams.add(bigram)

       # Higher repetition = lower score
       repetition_ratio = repeated_bigrams / (len(tokens) - 1)
       return 1.0 - repetition_ratio

Configuration and Best Practices
---------------------------------

Metric Naming Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow these conventions for consistent metric organization:

.. code-block:: text

   # Training metrics
   training/{category}/{metric_name}/{aggregation}

   # Validation metrics
   val/{category}/{data_source}/{metric_name}
   val-core/{data_source}/{variable}/{metric_name}
   val-aux/{category}/{metric_name}

   # Performance metrics
   perf/{metric_name}

   # Analysis metrics
   analysis/{category}/{metric_name}/{aggregation}

   # Model introspection
   model/{component}/{metric_name}/{aggregation}

Aggregation Selection
~~~~~~~~~~~~~~~~~~~~~

Choose aggregation methods based on metric semantics:

- **mean**: Default for most metrics (accuracy, loss, etc.)
- **max**: For peak values (max memory, worst-case latency)
- **min**: For best-case scenarios (min loss, fastest response)
- **sum/total**: For cumulative values (total tokens, total time)
- **median**: For robust central tendency (when outliers matter)
- **p95/p99**: For percentile-based SLA metrics

Error Handling
~~~~~~~~~~~~~~

Always implement robust error handling:

.. code-block:: python

   def compute_safe_custom_metrics(data: TensorDict) -> Dict[str, float]:
       """Example of safe metric computation"""
       metrics = {}

       try:
           # Check data availability
           if "required_field" not in data:
               return metrics

           # Handle empty tensors
           values = data["required_field"]
           if values.numel() == 0:
               return metrics

           # Compute metrics with numerical stability
           mean_val = torch.mean(values.float())
           if torch.isfinite(mean_val):
               metrics["custom/metric/mean"] = mean_val.item()

       except Exception as e:
           # Log error but don't crash training
           print(f"Error computing custom metrics: {e}")
           return {}

       return metrics

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Batch Processing**: Compute metrics on entire batches, not individual samples
- **Device Placement**: Keep tensors on the same device as input data
- **Memory Management**: Avoid accumulating large tensors across steps
- **Async Processing**: Use Ray actors for non-blocking metrics aggregation
- **Selective Computation**: Only compute expensive metrics when needed

Debugging Custom Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os

   def debug_custom_metrics(data: TensorDict, metrics: Dict[str, float]):
       """Debug utility for custom metrics"""
       if os.environ.get("DEBUG_METRICS", "0") == "1":
           print(f"Data keys: {list(data.keys())}")
           print(f"Data shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in data.items()]}")
           print(f"Computed metrics: {metrics}")

           # Check for common issues
           for name, value in metrics.items():
               if not isinstance(value, (int, float)):
                   print(f"WARNING: Metric {name} has invalid type {type(value)}")
               elif not np.isfinite(value):
                   print(f"WARNING: Metric {name} is not finite: {value}")

File Structure Summary
----------------------

.. code-block:: text

   siirl/execution/metric_worker/
   ├── metric_worker.py          # Ray actor for distributed aggregation
   │   ├── MetricWorker          # Ray remote actor class
   │   └── MetricClient          # Client interface
   └── utils.py                  # Aggregation functions
       ├── Metric                # Dataclass for metric values
       ├── MetricFunc            # Function selection logic
       ├── MeanMetric            # Mean aggregation
       ├── MaxMetric             # Maximum aggregation
       ├── MinMetric             # Minimum aggregation
       └── SumMetric             # Sum aggregation

   siirl/utils/metrics/
   └── metric_utils.py           # Core metric computation
       ├── compute_data_metric           # Standard training metrics
       ├── compute_timing_metrics        # Timing analysis
       ├── compute_throughout_metrics    # Throughput analysis
       ├── process_validation_metrics    # Validation with bootstrap
       ├── bootstrap_metric             # Bootstrap sampling utility
       └── aggregate_validation_metrics  # Parallel validation processing

This architecture provides a scalable, flexible foundation for comprehensive metrics collection in distributed training environments.