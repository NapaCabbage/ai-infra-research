---
title: "FlashAttention 三部曲 (FA1/FA2/FA3)"
tags: [inference-serving, attention, kernel-optimization]
subfield: inference-serving
venue: "NeurIPS 2022 / ICML 2024 (FA2) / 2024 (FA3)"
date: 2026-03-06
authors: [Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré]
institution: [Stanford, Princeton, Together AI]
url: "https://arxiv.org/abs/2205.14135"
status: 已读
rating: ⭐⭐⭐⭐⭐
---

# FlashAttention 三部曲：Tiling + Online Softmax 消灭 HBM 瓶颈

## 一句话总结

标准 Attention 反复把 N×N score 矩阵搬进搬出 HBM，FlashAttention 用**分块（Tiling）+ 增量式 Softmax（Online Softmax）**让数据留在 SRAM 里算完再写回，省掉了最大的 HBM 瓶颈。

---

## 核心问题

**瓶颈不是计算量，而是内存搬运量。**

标准 Attention 的计算流程：
```
S = Q × K^T          → 写 N×N 矩阵到 HBM
P = softmax(S)       → 读回 S，写 P 到 HBM
O = P × V            → 读回 P，写 O 到 HBM
```

N×N 的 score 矩阵（N = sequence length）要在 HBM 里写两次、读两次。N = 4096 时，这个矩阵有 16M 个元素——比 Q、K、V 本身大得多。Attention 是 **memory-bandwidth-bound**，不是 compute-bound。

---

## 两个关键技术

### 技术 1：Tiling（分块）

把 Q、K、V 切成小块，每次只把一小块加载到 SRAM（~128KB，带宽 19 TB/s），在 SRAM 里完成 Q·K^T → softmax → ×V 的完整计算，**N×N 的 score 矩阵从不完整出现在 HBM 中**。

```
Q 切成 Br 行的块：Q1, Q2, Q3, ...
K/V 切成 Bc 行的块：K1/V1, K2/V2, ...

双层循环：
  外层：遍历 K/V 的块（FA1）或 Q 的块（FA2）
  内层：遍历另一个
  每次只在 SRAM 里算一个 Br × Bc 的小 score 块
```

> 这和 [[gpu-architecture-basics]] 第九节讲的矩阵乘法 Tiling 是同一个原理——把数据搬到快 6 倍的 SRAM 里复用，减少 HBM 访问。

### 技术 2：Online Softmax（增量式 softmax）

Tiling 带来一个难题：softmax 需要知道**整行的最大值和求和**，但我们一次只看到一小块，怎么办？

答案是利用指数的数学性质 `e^(a+b) = e^a · e^b`，维护三个 running 统计量，每来一个新 block 就动态修正之前的结果：

```
每个新 K/V block 进来时：
  m_new = max(m_old, max(当前block的scores))    ← 更新最大值
  corr  = exp(m_old − m_new)                     ← 修正系数（≤ 1）
  l     = l × corr + Σexp(scores − m_new)        ← 修正累积分母
  O     = O × corr + exp(scores − m_new) × V     ← 修正累积输出

全部 block 扫完后：
  output = O / l
```

本质就是：之前算的用了旧的最大值 m_old，新 block 可能让最大值变大了，所以要把之前的结果乘一个 `exp(m_old - m_new)` 缩小回去。**初中指数运算的性质，但工程意义巨大。**

> 来源：Milakov & Gimelshein, NVIDIA 2018 → Tri Dao 2022 将其与 Tiling 结合 → FlashAttention

---

## FA1 vs FA2 vs FA3

| | FA1 (2022) | FA2 (2023) | FA3 (2024) |
|---|---|---|---|
| **循环顺序** | 外层 K/V，内层 Q | **外层 Q，内层 K/V** | 同 FA2 |
| **SM 通信** | 需要跨 SM reduction | **零跨 SM 通信** | 流水线并行 |
| **GPU 利用率** | ~35% | **~73%** | 更高 |
| **核心贡献** | Tiling + Online Softmax | 改循环顺序 | Hopper 硬件优化 |
| **FA3 新特性** | — | — | Warp Specialization, WGMMA, TMA |

→ FA4 (2026) 见独立笔记 [[flash-attention-4]]：针对 Blackwell 非对称扩展，软件模拟 exp + 条件 rescaling + TMEM + 2-CTA MMA，达到 1613 TFLOPs/s。

### FA1 → FA2：为什么换循环顺序就能翻倍利用率？

这是我最初没理解的点，记录下来：

**FA1（外层 K/V，内层 Q）的问题：**
```
for 每个 K/V 块:          ← K/V 固定在 SRAM
  for 每个 Q 块:          ← Q 轮流进来
    算局部 attention → 累加到 O[这个Q块]
```
- 每个 Q 块的输出 O 被**每个 K/V 块**更新一次 → O 要反复从 HBM 读改写
- 不同 SM 可能处理同一个 Q 块的不同 K/V 部分 → 最后需要**跨 SM 做 reduction**（等待同步）
- SM 经常在等别人 → 利用率只有 35%

**FA2（外层 Q，内层 K/V）的优势：**
```
for 每个 Q 块:            ← Q 固定在 SRAM，O 也在 SRAM！
  for 每个 K/V 块:        ← K/V 轮流进来
    算局部 attention → 累加到 O（在 SRAM 里！）
  写回 O                  ← 只写 HBM 一次！
```
- 每个 SM "承包" 一个 Q 块从头算到尾，O 始终在 SRAM 累加
- **零跨 SM 通信** → SM 永远在干活，不用等任何人
- 利用率直接翻倍到 73%

> 和 [[orca-continuous-batching]] 的思路相通——减少协调开销，让每个计算单元尽量独立工作。

---

## 思想传承

```
Welford 1962        在线统计量（running mean/variance），不存所有数据
      ↓
Milakov 2018        Online Softmax，一次遍历算出 softmax
      ↓
Tri Dao 2022        Tiling + Online Softmax → FlashAttention
      ↓
Tri Dao 2023        改循环顺序 → FA2（零跨SM通信）
      ↓
Tri Dao 2024        Hopper 硬件特性 → FA3
      ↓
Tri Dao 2026        Blackwell 非对称扩展 → FA4（见 [[flash-attention-4]]）
```

---

## 我的理解（学习笔记）

**最初的错误理解**：
- ❌ 以为 FA1 是 "QK 外层，V 内层" → 实际上 K 和 V 始终绑定（因为 softmax(QK^T) 算完就要立刻乘 V），区别在于 K/V 块 vs Q 块 谁在外层
- ❌ 没理解为什么 FA1 利用率低 → 核心是 O 的 HBM 反复读写 + 跨 SM reduction 的同步开销

**现在的理解**：
- FlashAttention 的 idea 不难：分块 + 动态缩放 softmax，两个技术单独看都很朴素
- 真正难的是**工程实现**：SRAM 容量极小（128KB/SM），要精确算好 tile 大小让 Q/K/V 块都放得下；还要处理 causal mask、非对齐序列长度等 edge case
- FA2 的核心贡献是"换个循环顺序"——但要**意识到**循环顺序会影响跨 SM 通信，需要深刻理解 GPU 执行模型（参见 [[gpu-architecture-basics]]）
- 这再次印证了 AI Infra 论文的规律：**idea 简洁，难度在软硬件协同优化**

---

## 关联笔记

- [[vllm-pagedattention]] — PagedAttention 管理 KV Cache 显存，FlashAttention 优化 Attention 计算本身，两者互补
- [[orca-continuous-batching]] — Orca 在调度层减少协调，FA2 在 kernel 层减少协调，思路相通
- [[gpu-architecture-basics]] — 第九节 Tiling 是理解 FA 的直接前置；§2.4 SRAM vs HBM 带宽差距解释了 FA 为什么能快
- [[AI-Infra-Cheatsheet]] — KV Cache 公式、GPU 带宽数字可以验证 FA 的收益

---

*学习方式：Claude 对话 + FlashAttention 三部曲总结笔记（未读原论文全文）*
*最后更新：2026-03-06*
