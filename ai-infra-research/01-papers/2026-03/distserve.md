---
title: "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"
tags: [inference-serving, disaggregation, prefill-decode, scheduling]
subfield: inference-serving
venue: "OSDI 2024"
date: 2026-03-09
authors: [Yinmin Zhong, Shengyu Liu, Junda Chen, Jianxi Ye, Ji Qi, Yibo Zhu, Haibin Lin, Xuanzhe Liu, Xin Jin]
institution: [Peking University, Sea AI Lab]
url: "https://arxiv.org/abs/2401.09670"
status: 已读
rating: ⭐⭐⭐⭐
---

# DistServe：Prefill-Decode 解耦

## 一句话总结

Prefill 和 Decode 的硬件瓶颈完全不同（compute-bound vs memory-bandwidth-bound），混跑互相干扰且并行策略被绑定。DistServe 把它们拆到不同 GPU 上，各自独立优化并行策略和资源分配，用 Simulator 搜索最优配置，最大化 goodput（SLO 约束下的有效吞吐）。

---

![[Pasted image 20260309153328.png]]
## 核心问题：为什么混跑不行？

### 干扰（Interference）

Prefill 和 Decode 在同一 batch 中，GPU 必须等最慢的那个完成。只要混入 1 个 Prefill 请求，整批 Decode 的完成时间大幅跳升，TPOT 变差；反过来 Decode 也拖慢 Prefill，TTFT 变差。干扰是双向的。

### 耦合（Coupling）

混跑系统两个阶段被迫共享同一组 GPU 和并行策略，但两者偏好完全不同：

```
             Prefill                    Decode
瓶颈        Compute-bound              Memory-bandwidth-bound
Batch 偏好   小 batch（加大无收益）      大 batch（提升利用率）
并行偏好     TP（降低单次执行时间）      PP/复制（扩容量、堆 batch）
SLO 指标     TTFT（首 token 延迟）      TPOT（每 token 生成时间）
```

混跑 = 两头不讨好，为了同时满足 TTFT 和 TPOT，只能过度配置 GPU。

---

## 核心 Idea：Disaggregation + Per-phase Optimization

```
请求到达
   ↓
┌──────────────────┐     KV Cache 传输     ┌──────────────────┐
│  Prefill Instance │ ─────────────────→   │  Decode Instance  │
│  (独立 GPU 组)    │                       │  (独立 GPU 组)    │
│  独立选 TP/PP     │                       │  独立选 TP/PP     │
│  优化目标: TTFT   │                       │  优化目标: TPOT   │
└──────────────────┘                       └──────────────────┘
```

拆分后：
- 干扰消失：两阶段不再争抢同一 GPU 时间片
- 资源独立伸缩：各自选最优并行策略
- 代价：需要传输 KV Cache（实验证明 < 0.1% 端到端延迟）

---

## Goodput：论文提出的核心指标

之前的系统优化总 throughput（token/s），但线上业务真正关心的是 SLO 约束下的有效吞吐：

**per-GPU goodput** = 在 TTFT/TPOT 双约束、达到目标 SLO attainment（如 90%）的前提下，每张 GPU 能承载的最大请求速率（req/s/GPU）。

这直接决定"每次请求成本"。

---

## 两套 Placement 算法

核心区分：KV Cache 传输走哪条路？

### Algorithm 1（High Node-Affinity）

适用：跨节点带宽够大（InfiniBand ~800Gbps），KV Cache 跨节点传输开销可忽略。

思路 = 分而治之：
1. 独立为 Prefill 找最优 (TP, PP)
2. 独立为 Decode 找最优 (TP, PP)
3. 按流量比例各自复制实例

两阶段搜索完全独立，约束最少，效果最好。

### Algorithm 2（Low Node-Affinity）

适用：跨节点带宽弱（普通以太网 ~25Gbps），只有节点内 NVLink 够快。

约束 = Prefill 的第 k 个 pipeline stage 和 Decode 的第 k 个 pipeline stage **必须在同一节点内**：

```
同一节点（8 GPUs）：
┌────────────────────────────────────────────┐
│  Prefill stage k (4 GPUs) → NVLink → Decode stage k (4 GPUs)  │
└────────────────────────────────────────────┘
```

代价：
- 两者必须用相同的 PP 数（stage 对齐）
- 同节点 GPU 要同时塞两阶段，内存更紧张
- 变成联合搜索，不能独立优化

### 两者都依赖 Simulator

用工作负载分布（输入/输出长度、到达率）做事件仿真，估计 SLO attainment，枚举+二分找最大 goodput。仿真与真实系统误差 < 2%。

---

## 关键实验结论

- 相比 vLLM/DeepSpeed-MII，DistServe 能多服务 **7.4× 请求量**或支持 **12.6× 更严格的 SLO**
- 消融实验：vLLM++ (混跑+调并行) ≈ vLLM → 光调并行没用，**拆分本身是主要增益来源**
- DistServe-High > DistServe-Low → 网络条件越好，拆分收益上限越高
- 175B 模型实际配置：Prefill TP=3 PP=3, Decode TP=4 PP=3 → 同一模型两阶段用不同并行切法，这在混跑系统中不可能

---

## 我的理解

- Idea 很简单清晰：P 和 D 瓶颈不同，分开各自优化。工程上的核心挑战是 placement 算法（怎么选并行配置 + 怎么放置）和 KV Cache 传输
- Goodput 是一个比 throughput 更贴近业务的指标，值得记住
- 这篇论文和 Meta 2023 OCP 的 workload diversity 雷达图完美呼应：Prefill 和 Decode 的资源需求画像几乎完全不同
- 和 SGLang 的关系：SGLang 优化的是"跨请求的 KV Cache 复用"（RadixAttention），DistServe 优化的是"Prefill 和 Decode 的资源分配"——两者正交，可以组合
- 后续 Mooncake（Moonshot AI / Kimi）在工业界验证了这个思路的可行性

---

## 关联笔记

- [[vllm-pagedattention]] — vLLM 解决单请求内 KV Cache 碎片化，DistServe 更进一步解决 P/D 混跑的干扰问题
- [[orca-continuous-batching]] — Orca 的 continuous batching 是混跑范式的起点，DistServe 指出混跑的局限并提出解耦
- [[sglang]] — SGLang 优化跨请求 KV Cache 复用（RadixAttention），与 DistServe 的 P/D 解耦正交互补
- [[flash-attention]] — FlashAttention 优化单次 Attention 计算效率，DistServe 优化系统级资源分配，层次不同
- [[AI-Infra-Cheatsheet]] — Prefill = compute-bound, Decode = memory-bandwidth-bound 的分析在 §三 Transformer 推理完整流程 中有详细说明

---

*学习方式：alphaxiv 论文讲解 + Claude 对话深入理解*
*最后更新：2026-03-09*
