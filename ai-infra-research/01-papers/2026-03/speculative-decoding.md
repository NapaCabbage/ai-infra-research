---
title: "Fast Inference from Transformers via Speculative Decoding"
tags: [inference-serving, decoding, sampling]
subfield: inference-serving
venue: "ICML 2023"
date: 2026-03-06
authors: [Yaniv Leviathan, Matan Kalman, Yossi Matias]
institution: [Google]
url: "https://arxiv.org/abs/2211.17192"
status: 已读
rating: ⭐⭐⭐⭐
---

# Speculative Decoding：用小模型"猜"、大模型"验"的免费加速

## 一句话总结

小模型快速生成 K 个候选 token，大模型一次 forward pass 批量验证，用 rejection sampling 保证输出分布与纯大模型**精确一致**——加速推理但不牺牲任何质量。

---

## 核心 Idea

```
传统 Decode：大模型生成 K 个 token → K 次 forward pass → 慢

Speculative Decoding：
  1. 小模型生成 K 个 token        → 很快（小模型便宜）
  2. 大模型验证这 K 个 token       → 1 次 forward pass（≈ 生成 1 个 token 的成本）
  3. 接受正确的，拒绝错误的         → rejection sampling 保证分布精确
  → 好的情况下 1 次大模型调用得到 K 个 token
```

---

## 为什么验证 K 个 token ≈ 生成 1 个 token 的成本？

这就是 Prefill vs Decode 的区别：

```
生成 1 个 token（Decode）：
  读全部模型权重 → 算 1 个 token → memory-bound，带宽是瓶颈

验证 K 个 token（本质是 Prefill）：
  读全部模型权重 → 同时算 K 个 token → 矩阵乘法并行

关键：两种情况都要读一遍全部模型权重
     权重搬运是固定成本，K 个 token 的额外计算被带宽瓶颈"遮住了"
     只要 K 不太大（通常 4-8），耗时几乎不变
```

> 参考 [[AI-Infra-Cheatsheet]] 公式 3：Decode 速度 ≈ 带宽 / 权重大小。验证时权重只读一遍，K 个 token 并行计算。

---

## Rejection Sampling：为什么能保证分布精确一致

### 算法

```
小模型（分布 q）生成 K 个 token：x1, x2, ..., xK
大模型（分布 p）对这 K 个 token 跑一次 forward pass

对每个 token xi：
  如果 p(xi) ≥ q(xi) → 直接接受
  如果 p(xi) < q(xi) → 以概率 p(xi)/q(xi) 接受，否则拒绝

一旦在位置 i 拒绝：
  从修正分布 norm(max(p - q, 0)) 重新采样一个 token
  丢弃 i 之后所有 token
```

### 修正分布是什么意思？（用具体数字）

假设词表只有 {A, B, C}：

```
大模型 p：  A=0.5,  B=0.3,  C=0.2
小模型 q：  A=0.2,  B=0.6,  C=0.2
                         ↑ 小模型对 B 过度自信

小模型采出 B（q=0.6），大模型觉得 B 只值 0.3
→ 以概率 p/q = 0.3/0.6 = 50% 接受
→ 运气不好，被拒绝了

修正分布：
  p - q：    A=0.3,  B=-0.3,  C=0.0
  max(,0)：  A=0.3,  B=0,     C=0
  归一化：   A=1.0,  B=0,     C=0
  → 采样结果一定是 A
```

直觉：小模型"透支"了 B 的概率，拒绝后从"大模型想要但小模型给少了"的部分补回来。

### 数学证明：为什么最终分布 = p

对任意 token x，被选中的总概率 = 被接受的概率 + 拒绝后被补采样到的概率：

```
P(选中 x) = q(x) · min(1, p(x)/q(x))                         ← 直接被接受
           + [总拒绝概率] × max(p(x)-q(x), 0) / Σ max(p-q, 0) ← 拒绝后补采样到
```

**当 p(x) < q(x) 时**（小模型过度自信的 token）：
```
第一项 = q(x) · p(x)/q(x) = p(x)
第二项中 max(p(x)-q(x), 0) = 0 → 第二项 = 0
总计 = p(x) ✓
```

**当 p(x) ≥ q(x) 时**（小模型低估的 token）：
```
第一项 = q(x) · 1 = q(x)    ← 不够，还差 p(x) - q(x)

需要证明第二项 = p(x) - q(x)：
  总拒绝概率 = Σ_y q(y) · (1 - p(y)/q(y))   （只对 q(y) > p(y) 的 y）
             = Σ_y (q(y) - p(y))             （对 q(y) > p(y) 的 y）
             = Σ_z max(q(z)-p(z), 0)

  关键数学事实：因为 Σp = Σq = 1（都是概率分布），所以
  "q 比 p 多出来的总量" = "p 比 q 多出来的总量"
  即：Σ max(q-p, 0) = Σ max(p-q, 0)

  所以：总拒绝概率 = Σ max(p-q, 0) = 修正分布的归一化常数

  代入第二项：
  = Σ max(p-q, 0) × (p(x)-q(x)) / Σ max(p-q, 0)
  = p(x) - q(x)

总计 = q(x) + p(x) - q(x) = p(x) ✓
```

**结论**：不管 q 长什么样，最终分布都精确等于 p。q 只影响接受率（→ 速度），不影响正确性。

---

## 与 Rejection Fine-tuning (RFT) 的联系

我在学习中的一个发现：Speculative Decoding 和 Rejection Fine-tuning 背后**都是 rejection sampling**。

```
                     Speculative Decoding          Rejection Fine-tuning (RFT)
采样源 q             小模型（快）                    当前策略模型
目标分布 p           大模型（慢）                    "高质量回答"的隐式分布
rejection 标准       p(x)/q(x) 概率比              reward > threshold
目的                 加速推理（不改变质量）           提升模型质量（用好样本训练）
数学保证             精确等价于 p                    近似（threshold 是 heuristic）
```

本质都是"生成一堆候选 → 按标准筛选"。Speculative Decoding 用精确概率比，所以保证分布一致；RFT 用 reward threshold 筛选，是近似的。

DeepSeek-R1 的训练就大量使用了 rejection sampling：让模型生成很多 chain-of-thought，用最终答案正确性筛选好的样本，再做 SFT。

> **统一视角**：Rejection sampling 是一个通用的"从易采样分布模拟目标分布"的工具。推理加速（Speculative Decoding）和训练优化（RFT/RLHF）都是它的应用场景。

---

## 我的理解

- 核心 idea 确实简单：小模型猜、大模型验、rejection sampling 保证正确性
- 关键工程 insight 是"验证 K 个 token ≈ 1 次 forward pass"——这依赖于 Decode 是 memory-bound 的事实（Prefill 可以并行），和之前学的 GPU 知识直接关联
- Rejection sampling 的数学之美：一个 1960 年代的统计方法，在 2022 年成为 LLM 加速的核心工具
- 实际限制：需要有一个好的 draft model（接受率高才有加速），小模型太差就退化成普通 Decode

---

## 关联笔记

- [[flash-attention]] — FA 优化 Attention 计算，Speculative Decoding 优化 Decode 的调用次数，两者互补
- [[orca-continuous-batching]] — Orca 的 continuous batching 可以和 speculative decoding 结合
- [[gpu-architecture-basics]] — 理解"验证 K 个 token ≈ 1 次 forward pass"需要理解 Prefill(compute-bound) vs Decode(memory-bound)
- [[AI-Infra-Cheatsheet]] — 公式 3 直接解释了为什么验证几乎免费

---

*学习方式：读论文 + Claude 对话深入理解 rejection sampling 数学推导*
*最后更新：2026-03-06*
