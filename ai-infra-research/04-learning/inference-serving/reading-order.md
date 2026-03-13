---
tags: [learning]
subfield: inference-serving
created: 2026-03-05
---

# Inference Serving 精选阅读路径

> 针对你的背景（半导体分析师 + LLaMA 3.1）定制，估计 4-6 周完成

---

## 阶段 0：热身（第 1 周，读博客不读论文）

在读论文前，先用博客建立直觉。

1. **vLLM Blog: "Efficient Memory Management for LLM Serving with PagedAttention"**
   - https://blog.vllm.ai/2023/06/20/vllm.html
   - 为什么读：图文并茂讲清楚了 KV Cache 碎片化问题和 PagedAttention 的直觉，读完你就懂了 80% 的 inference serving 背景知识
   - 读完能做到：解释为什么传统方式浪费显存

2. **LMSYS Blog: "Fast LLM Serving with vLLM"**
   - 同上链接，配合 vLLM 的 benchmark 结果看
   - 重点看他们的性能对比图，感受量级

3. **Anthropic / OpenAI 的 system card**（选读）
   - 了解真实部署场景对推理系统的要求

---

## 阶段 1：奠基论文（第 2-3 周）

### 必读 ①：Orca（Continuous Batching 的起点）
- **论文**：*Orca: A Distributed Serving System for Transformer-Based Generative Models*
- **发表**：OSDI 2022
- **为什么重要**：提出了 iteration-level scheduling（即 Continuous Batching），现在所有主流推理框架都用这个思路
- **关键概念**：iteration-level scheduling vs request-level scheduling
- **读法**：重点读 Introduction、Section 3（设计）、Evaluation，跳过实现细节
- **读完能做到**：解释为什么 continuous batching 比 static batching 好

### 必读 ②：vLLM/PagedAttention（现代推理的基础）
- **论文**：*Efficient Memory Management for Large Language Model Serving with PagedAttention*
- **发表**：SOSP 2023
- **为什么重要**：解决了 KV Cache 碎片化问题，vLLM 成为业界标准
- **关键概念**：PagedAttention、KV Cache 碎片化、copy-on-write（prefix caching 的前身）
- **读法**：Figure 2（显存碎片化示意图）和 Figure 3（PagedAttention 示意图）是理解核心的关键
- **链接**：https://arxiv.org/abs/2309.06180

---

## 阶段 2：关键优化技术（第 3-4 周）

### 必读 ③：FlashAttention 三部曲 ✅ 已完成（通过 Claude 对话 + 总结笔记）
- **论文**：*FlashAttention* (NeurIPS 2022) + *FlashAttention-2* (ICLR 2024) + *FlashAttention-3* (2024)
- **笔记**：[[flash-attention]]
- **掌握程度**：理解 Tiling + Online Softmax 原理，FA1→FA2 循环顺序优化逻辑，未读论文原文细节
- **链接**：https://arxiv.org/abs/2205.14135 (FA1), https://arxiv.org/abs/2307.08691 (FA2)

### 必读 ③-b：FlashAttention-4 ✅ 已完成（原论文精读）
- **论文**：*FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling* (arXiv 2603.05451, 2026)
- **笔记**：[[flash-attention-4]]
- **掌握程度**：精读全文（20 页 + 附录），理解非对称硬件扩展问题、Roofline 分析（Table 1/3）、前向流水线 ping-pong + 软件模拟 exp + 条件 rescaling、反向 TMEM + 2-CTA MMA、LPT 调度、CuTe-DSL 实现、所有 Figure/Table
- **链接**：https://arxiv.org/abs/2603.05451

### 选读 ④：Speculative Decoding ✅ 已完成
- **论文**：*Fast Inference from Transformers via Speculative Decoding*（Google, ICML 2023）
- **笔记**：[[speculative-decoding]]
- **掌握程度**：理解核心机制（小模型猜+大模型验）、rejection sampling 完整数学推导、与 RFT 的统一视角
- **链接**：https://arxiv.org/abs/2211.17192

### 必读 ⑤：SGLang ✅ 已完成
- **论文**：*SGLang: Efficient Execution of Structured Language Model Programs*
- **发表**：NeurIPS 2024
- **笔记**：[[sglang]]
- **掌握程度**：理解核心 thesis（前端语言 + 后端运行时 co-design）、RadixAttention（Radix Tree 跨请求 KV Cache 复用）、FSM Constrained Decoding（预编译 token mask）
- **链接**：https://arxiv.org/abs/2312.07104

---

## 阶段 3：前沿方向（第 5-6 周）

### 必读 ⑥：Prefill-Decode 解耦
- **论文 A**：*DistServe* ✅ 已完成（OSDI 2024）
  - **笔记**：[[distserve]]
  - **掌握程度**：理解核心 idea（P/D 解耦 + 各自独立优化并行策略）、goodput 指标、Algorithm 1/2 的区别（高/低跨节点带宽）、Simulator 驱动的配置搜索
  - https://arxiv.org/abs/2401.09670
- **论文 B**：*Mooncake* ✅ 已完成（Moonshot AI / Kimi，FAST 2025 Best Paper）
  - **笔记**：[[mooncake]]
  - **掌握程度**：理解 KVCache-centric 架构（三资源池）、链式哈希前缀匹配、Layer-wise 流式传输、分层存储（VRAM/DRAM/SSD）、预测性 Early Rejection、CPP 并行策略
  - https://arxiv.org/abs/2407.00079
- **为什么重要**：这是 2024-2025 年推理领域最核心的研究方向之一，OSDI/MLSys 有大量 follow-up

### 必读 ⑥-b：Attention-FFN 解耦（MoE 专属）
- **论文**：*MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism* ✅ 已完成
  - **笔记**：[[megascale-infer]]
  - **掌握程度**：精读全文（17 页），理解 MoE 稀疏性导致 FFN memory-bound 的核心问题、Attention-FFN 解耦架构（Figure 3）、Ping-Pong Pipeline Parallelism 三约束条件、Deployment Plan Search（Algorithm 1）、异构部署（H20+L40S）、M2N 通信库设计（vs NCCL 吞吐 4.2× / P99 -96.2%）、所有 Figure/Table
  - https://arxiv.org/abs/2504.02263
- **和 ⑥ 的关系**：DistServe/Mooncake 做 P/D 解耦（请求级），MegaScale-Infer 做 Attention/FFN 解耦（层内级），两者互补

### 选读 ⑦：DeepSeek 推理系统
- **DeepSeek-V3 Technical Report**（2024）的推理部分
  - 你了解 DeepSeek-V3，现在可以深入看它的推理架构（MLA、MoE 对推理的影响）
  - 特别关注他们的 MLA（Multi-head Latent Attention）如何减少 KV Cache

### 选读 ⑧：NVIDIA Dynamo
- **Dynamo Blog/文档**
- NVIDIA 开源推理框架，关注它如何实现 P/D 解耦 + 调度

---

## 读完这些之后你能做什么

- 看到 OSDI/MLSys inference serving 的论文 title，大致能判断它在解决什么问题
- 在会议上看到 vLLM / SGLang / DistServe 的作者，能问出"为什么选择 X 而不是 Y"这样的问题
- 理解 DeepSeek 技术报告中推理部分的技术选择
- 给你的投研工作提供更深的系统视角（为什么某些 inference 创业公司的方向是对的/错的）