---
tags: [learning]
subfield: training-infra
created: 2026-03-05
---

# Training Infra 概念图谱

> 目标：理解"如何把 LLM 训练这件事分摊到成千上万张 GPU 上，高效跑完"
> 你的起点：LLaMA 3.1 论文，理解训练流程，但没深入 infra 层

---

## 核心问题是什么？

训练一个大模型（如 LLaMA 3.1 405B）需要的显存远超单张 GPU 的容量，而且需要极高的计算量。核心挑战：

1. **单张 GPU 装不下**：405B 参数 × 2 bytes（bf16）= 810GB，A100/H100 只有 80GB HBM
2. **训练要快**：训练周期可能长达数月，GPU 集群的利用率直接决定成本
3. **通信是瓶颈**：GPU 之间要同步梯度/激活值，网络带宽往往是限制因素

---

## 核心概念

### 第一层：并行化的三个维度（3D Parallelism）

这是 Training Infra 最核心的概念，你在 LLaMA 3.1 论文里看到的那些并行方式都在这里。

**Data Parallelism（DP，数据并行）**
- 最简单：把同一个模型复制到每张 GPU 上，每张 GPU 处理不同的数据
- 每步结束后，所有 GPU 同步梯度（All-Reduce）
- 问题：模型太大，单张 GPU 装不下时失效
- 变体：**ZeRO**（DeepSpeed）把参数、梯度、优化器状态也分散到各 GPU，让每张 GPU 只保存一部分，从逻辑上等同于 DP 但显存效率极高

**Tensor Parallelism（TP，张量并行）**
- 把单个矩阵运算（如 Linear 层）切分到多张 GPU 上并行计算
- 优点：可以处理单张 GPU 放不下的层
- 缺点：每次计算完需要 All-Reduce，通信频繁，必须用 NVLink（同机器内）效果才好
- Megatron-LM 是 TP 的代表实现

**Pipeline Parallelism（PP，流水线并行）**
- 把模型不同的层分配给不同的 GPU，像流水线一样
- GPU 0 处理 Layer 1-8，GPU 1 处理 Layer 9-16……
- 问题：有"气泡"（bubble），等待前一个 GPU 完成才能开始，GPU 有空闲
- 优化：GPipe / PipeDream / 1F1B schedule 都是在减少 bubble

**Expert Parallelism（EP，专家并行）**
- 专门为 MoE（Mixture of Experts）设计
- 不同专家放在不同 GPU 上，token 路由到对应 GPU 处理
- DeepSeek-V3 大量使用，也是 LLaMA 3.1 没有重点使用的

### 第二层：具体实现

**Megatron-LM**：NVIDIA 开发，TP + PP 的工业级实现，LLaMA 3.1 的训练基础设施就是基于它。

**ZeRO（Zero Redundancy Optimizer）**：DeepSpeed 的核心技术，三个阶段：
- ZeRO-1：分散优化器状态（Adam 的 momentum/variance）
- ZeRO-2：再分散梯度
- ZeRO-3：连参数也分散，通信量更大但显存最省

**Activation Checkpointing（梯度检查点）**：
- 正向传播时不保存中间激活值，反向传播时重新计算
- 省显存但代价是重新计算（约增加 33% 的计算量）
- 几乎所有大模型训练都开着

**Mixed Precision Training（混合精度）**：
- 用 bf16/fp16 做 forward + backward（省显存、快），用 fp32 维护参数副本（保精度）
- 你从半导体角度理解：H100 的 bf16 Tensor Core throughput 是 fp32 的 2x，所以混精度是"用对了硬件"

### 第三层：通信与网络

**All-Reduce**：每张 GPU 计算出本地梯度后，需要对所有 GPU 的梯度求和（或平均），确保每张 GPU 上的参数保持一致。是 Data Parallel 中的核心通信操作。

**Ring All-Reduce**：把 All-Reduce 实现为环形通信，每张 GPU 只和相邻 GPU 通信，总通信量不随 GPU 数量线性增长。

**InfiniBand vs NVLink**：
- NVLink：同一机器内 GPU 之间的高速互联（NVLink 4.0 = 900 GB/s）
- InfiniBand（IB）：跨机器的网络（HDR = 200 Gbps，NDR = 400 Gbps）
- 你的半导体背景：NVLink 是 NVIDIA 私有协议，Spectrum-X 是 NVIDIA 的以太网方案，这是他们的竞争壁垒

**RDMA（远程直接内存访问）**：GPU 可以直接读写另一台机器的显存，绕过 CPU，InfiniBand 和 ROCEv2 都支持。

### 第四层：控制平面（MAST/训练管理）

你提到的 LLaMA 3.1 论文中的"MAST"，应该是 Meta 的 **MAST（Meta's Advanced Scheduling and Training）** 系统，是训练的控制平面：

- 负责调度训练任务到 GPU 集群
- 处理故障恢复（某张 GPU 挂了怎么办——checkpoint + restart）
- 监控训练状态（loss 曲线是否正常）
- 管理 checkpoint 存储

这类系统通常不发论文，但在实际工业训练中极其重要。DeepSeek-V3 论文里也有对他们训练管理系统的描述。

---

## 和 Inference 的关键区别

| | Training | Inference |
|---|---|---|
| 计算模式 | 需要梯度、反向传播 | 只有前向传播 |
| 显存需求 | 参数 + 梯度 + 优化器状态 + 激活值 | 参数 + KV Cache |
| 批大小 | 大（几千到几百万 tokens/step） | 小（单个用户请求） |
| 延迟要求 | 不敏感（整体训练时间） | 敏感（用户体验） |
| 主要挑战 | 显存容量、通信瓶颈 | 显存带宽、调度效率 |