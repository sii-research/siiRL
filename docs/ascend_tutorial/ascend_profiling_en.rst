Data Collection on Ascend Devices Based on the FSDP Backend
====================================

Last updated: 08/14/2025.

This is a tutorial for using GRPO to collect data on Ascend devices based on the FSDP backend.

Configuration
----

- Global Collection Control: Use the configuration items in siirl/client/config/ppo_trainer.yaml to control the default collection mode.

Control collection parameters using parameters in ppo_trainer.yaml:

- enable: Whether to enable performance profiling.
- save_path: The path to save collected data.

- level: Collection level—options include level_none, level0, level1, and level2.
- level_none: Disables all level-based data collection (turns off profiler_level).
- level0: Collects high-level application data, low-level NPU data, and operator execution details on the NPU.
- level1: Adds CANN layer AscendCL data and AI Core performance metrics on the NPU based on level0.
- level2: Adds CANN layer Runtime data and AI CPU metrics based on level1.

- with memory: Enables memory analysis (defaults to True).
- record shapes: Enables recording of tensor shapes (defaults to False).
- with npu: Enables collection of device-side performance data (defaults to True).
- with cpu: Enables collection of host-side performance data (defaults to True).
- with module: Enables recording of framework-level Python call stack information.
- with stack: Enables recording of operator call stack information.
- analysis: Enables automatic data analysis.
- discrete: Enables discrete mode, collecting performance data for each stage separately (defaults to False).

- roles: Collection stage - used in conjunction with the discrete parameter. Options include:

generate, compute_reward, compute_old_log_prob, compute_ref_log_prob, compute_value, compute_advantage,

train_critic, train_actor

- all_ranks: Whether to collect data from all ranks.

- ranks: List of ranks for which to collect data. If empty, no data is collected.

- profile_steps: List of collection steps. For example, [2, 4] indicates that steps 2 and 4 will be collected. If set to null, no data is collected.

Example
----
Disable collection
~~~~~~~~~~~~~~~~~~~~
.. code:: yaml

  profiler:
    enable: False # disable profile

End-to-end collection
~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

  profiler:
    steps: [1, 2, 5]
    discrete: False

The run_qwen2_5-7b-npu-e2e_prof.sh script is provided in examples/grpo_trainer for reference.

Discrete mode collection
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

  profiler:
    discrete: True
    roles:['generate', 'train_actor']

The discrete mode acquisition script run_qwen2_5-7b-npu-discrete_prof.sh is provided in examples/grpo_trainer for reference.

Visualization
------

The acquired data is stored in the user-defined save_path and can be visualized using the MindStudio Insight tool，
you can refer to <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>.


If the analysis parameter is set to False, offline analysis is required after collection:

.. code:: python

        import argparse
        from torch_npu.profiler.profiler import analyse

        parser = argparse.ArgumentParser()
        parser.add_argument("--path", type=str, default="facebook/opt-125m")

        if __name__ == "__main__":
         args = parser.parse_args()
         path = args.path
