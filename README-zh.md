<div align="center">
  <img src="asset/sii.png" width="100%"/>
  <br>
</div>
<br>

<h1 align="center">
siiRL: Shanghai Innovation Institute RL Framework for Advanced LLMs and Multi-Agent Systems
</h1>

<p align="center">
| <a href="https://arxiv.org/abs/2507.13833"><b>📄 论文</b></a> | 
| <a href="https://siirl.readthedocs.io/en/latest/index.html"><b>📚 文档</b></a> |
| <a href="asset/siiRL-feishu-group.png">
    <img src="asset/logo-feishu.png" alt="Feishu Group QR Code" height="15" /> 
    <b> 飞书群</b>
  </a> 
| <a href="asset/siiRL-wechat-group.png">
    <img src="asset/logo-wechat.png" alt="Wechat Group QR Code" height="15" /> 
    <b> 微信群</b>
  </a> 
| <a href="README.md"><b> English</b></a> |
</p>

**siiRL** 是一个新型的、**完全分布式的强化学习 (RL) 框架**，旨在突破大语言模型 (LLM) 后训练中的扩展性瓶颈，由**上海创智学院**的研究人员开发。

通过移除其他框架中常见的中心化控制器，siiRL 实现了**近线性的扩展能力**、**显著的吞吐量提升**以及**前所未有的灵活性**，为基于强化学习的 LLM 开发带来了全新的可能性。

---

## 🚀 亮点

+ **近线性扩展能力**: 多控制器模式通过将控制逻辑和数据管理分布到所有工作节点，消除了中心化瓶颈，从而实现了在数千张 GPU 上的近线性扩展。

+ **业界领先的吞吐量 (SOTA)**: 完全分布式的数据流架构最大限度地减少了通信和 I/O 开销，在数据密集型场景中实现了业界领先的吞吐量。

+ **灵活的 DAG 定义流水线**: 将您的算法逻辑与物理硬件解耦。通过 siiRL，您可以将复杂的 RL 工作流定义为一个简单的有向无环图 (DAG)，从而实现快速、经济且无需编写代码的实验。

+ **跨硬件兼容性**: siiRL 现已正式支持华为昇腾 (Ascend) NPU，为在不同硬件平台上进行训练和推理提供了高性能的替代方案。

+ **经过验证的性能与稳定性**: 在 7B 到 72B 尺寸的模型上进行了广泛的基准测试，siiRL 在各种任务中均表现出卓越的性能。其优势在长上下文和多模态训练等数据密集型工作负载中尤为明显。

---

## 📰 最新动态

* **[2025/07]**: 我们很开心向开源社区发布 siiRL！欢迎查阅我们的[论文](https://arxiv.org/abs/2507.13833)，深入了解其架构和评测。

* **[2025/07]**: siiRL 现已支持 [CPGD](https://arxiv.org/pdf/2505.12504v1)，这是一种通过正则化大幅度的策略更新来增强 RL 训练稳定性和性能的算法。

* **[2025/07]**: siiRL 为 LaMAS 新增了 [MARFT](https://arxiv.org/pdf/2504.16129) 支持，可通过 Flex-POMDP 对 LLM 多智能体进行强化学习微调。

---

## 💡 架构概览

siiRL 是一个为大规模集群设计的完全分布式强化学习框架。siiRL 采用多控制器模式，将所有计算和数据流均匀地分派到每个 GPU。siiRL 由三个主要组件构成：DAG Planner，DAG Workers 和 Data Coordinator.

<div align="center">
  <img src="asset/overview.png" width="650px" alt="siiRL 架构概览">
  <p><i>图 1. siiRL 架构概览。</i></p>
</div>

siiRL 是一个**完全分布式、多控制器的架构**。

关键组件包括：
* **DAG Planner**: 将用户定义的逻辑工作流 (DAG) 转换为序列化、可供每个工作节点执行的流水线。
* **DAG Workers**: 核心执行单元，每个工作节点绑定到单个 GPU，独立运行其分配的任务。
* **Data Coordinator**: 一组分布式组件（`分布式数据加载器`和`分布式数据缓冲区`），无需中央协调器即可管理从初始加载到中间数据再分配的整个数据生命周期。

## 🧪 实验评测

我们对 siiRL 的性能和扩展性进行了全面评测，并与业界领先的 RL 框架 verl 进行了比较。实验表明，siiRL 在所有指标上均表现出卓越的性能。

### 端到端吞吐量
在标准的 PPO 和 GRPO 算法下，siiRL 的吞吐量全面超越了基线系统。特别是在数据密集度更高的 GRPO 算法下，siiRL 通过其完全分布式的架构有效解决了数据瓶颈，实现了高达 **2.62 倍**的性能提升。

<p align="center">
<img src="asset/ppo_performance_comparison.png" width="80%" alt="PPO 算法性能对比"/>
<br>
<em>图 2: 使用 PPO 算法的端到端性能对比</em>
</p>
<p align="center">
<img src="asset/grpo_performance_comparison.png" width="80%" alt="GRPO 算法性能对比"/>
<br>
<em>图 3: 使用 GRPO 算法的端到端性能对比</em>
</p>

### 大规模扩展性
siiRL 展示了近线性的扩展能力，可平滑扩展至 1024 张 GPU。相比之下，基线框架由于其单点数据瓶颈导致的 OOM (内存不足) 错误，在相同条件下运行失败。在基线系统所能支持的最大批量大小下，siiRL 的性能优势可高达 **7 倍**。

<p align="center">
<img src="asset/scaling_trend_new.png" width="80%" alt="siiRL 扩展性测试"/>
<br>
<em>图 4: siiRL 在 VLM 模型上的扩展性测试</em>
</p>

<p align="center">
<img src="asset/batch_size_total_throughput_final.png" width="80%" alt="VLM 任务性能对比"/>
<br>
<em>图 5: 在基线系统最大负载下，VLM 任务的性能对比</em>
</p>

### 长上下文性能
在处理长上下文任务时，数据传输开销成为主要瓶颈。siiRL 的分布式数据流设计使其性能优势随着上下文长度的增加而愈发明显，实现了高达 **2.03 倍**的吞吐量提升，并成功运行了基线系统无法处理的 72B 模型长上下文任务。

<p align="center">
<img src="asset/context_length_comparison_with_oom_label.png" width="80%" alt="长上下文性能对比"/>
<br>
<em>图 6: 长上下文场景下的性能对比</em>
</p>

### 模型收敛性
实验证实，siiRL 的性能优化并未以牺牲模型精度为代价。在超参数相同的情况下，siiRL 的奖励和熵收敛曲线与基线系统完全一致，同时将总训练时间**减少了 21%**。

<p align="center">
<img src="asset/reward_and_entropy_comparison_final.png" width="45%" alt="收敛曲线对比"/>
<br>
<em>图 7: 模型收敛曲线对比</em>
</p>

---

## 📚 相关资源

<a href="https://siirl.readthedocs.io/en/latest/index.html"><b>使用文档</b></a>

- <a href="https://siirl.readthedocs.io/en/latest/start/install.html"><b>安装指南</b></a>

- <a href="https://siirl.readthedocs.io/en/latest/start/quickstart.html"><b>快速入门: 运行 PPO/GRPO</b></a>

---

## 🗓️ 未来计划

siiRL 仍在积极开发中。我们对未来充满期待，并致力于在两个关键方向上扩展框架的功能：推进多智能体支持和优化基础框架。

### 🚀 增强的多智能体能力
我们灵活的 DAG 设计为复杂的多智能体系统提供了天然且强大的基础。我们计划将其作为最主要的特性，主要包括：

* **[ ]** 复杂的智能体交互工作流。
* **[ ]** 扩展多智能体强化学习 (MARL) 算法支持。
* **[ ]** 丰富的环境交互接口。

### 🔧 基础框架增强
我们将持续努力提升核心系统的性能、效率和扩展性。主要优先事项包括：

* **[ ]** 集成 Megatron-LM 和 SGLang。
* **[ ]** 优化模型浮点运算利用率 (MFU)。
* **[ ]** 解决 Rollout 阶段的长尾问题。
* **[ ]** 扩展硬件支持。

---

## 🙏 致谢

我们首先要感谢开源 RL 框架 [verl](https://github.com/volcengine/verl)，我们使用它作为评测的主要基线系统。我们特别感谢其分层的 API 设计；我们复用了 verl 中的 `3DParallelWorker` 基类来管理 siiRL 中的系统组件。

siiRL 的构建也离不开其他优秀的开源项目。我们衷心感谢 PyTorch、Ray、vLLM、vLLM-Ascend 和 SGLang 团队的杰出工作。

我们的工作旨在解决研究过程中发现的扩展性挑战，并希望 siiRL 能为社区的共同进步做出积极贡献。

---

## 🖋️ 如何引用

如果您在研究中发现 siiRL 有用，请考虑引用我们的论文。

```bibtex
@misc{wang2025distflowfullydistributedrl,
      title={DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training}, 
      author={Zhixin Wang and Tianyi Zhou and Liming Liu and Ao Li and Jiarui Hu and Dian Yang and Jinlong Hou and Siyuan Feng and Yuan Cheng and Yuan Qi},
      year={2025},
      eprint={2507.13833},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={[https://arxiv.org/abs/2507.13833](https://arxiv.org/abs/2507.13833)}, 
}