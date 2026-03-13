---
title: "Reducing Activation Recomputation in Large Transformer Models"
tags: [training-infra, activation-checkpointing, sequence-parallelism, tensor-parallelism, memory-optimization]
subfield: training-infra
venue: "MLSys 2023 (arXiv 2022)"
date: 2026-03-11
authors: [Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, Bryan Catanzaro]
institution: [NVIDIA]
url: "https://arxiv.org/abs/2205.05198"
status: 已读
rating: ⭐⭐⭐⭐⭐
---

# 减少大 Transformer 模型的激活重算

## 一句话总结

提出 Sequence Parallelism（把 TP 管不到的 LayerNorm/Dropout 按序列维度切分）和 Selective Activation Recomputation（只重算"大显存、低计算"的 attention 内部操作）。两者组合使激活显存降低 5×，训练吞吐提升 ~30%，MFU 从 42% 提升到 56%，几乎消除了 full recomputation 的 30-40% 额外计算开销。

---

## 核心问题

Megatron TP 只并行了 Attention 和 MLP **内部**的矩阵乘法。LayerNorm 和 Dropout 没有被 TP 切分，每张 GPU 都存完整激活值。这部分贡献了 10sbh 的显存（公式 2），不被 t 除。

对于大模型，未被切分的激活值反而是显存主要来源。

---

## 激活显存拆解（§4.1）

每层每个 Transformer block 的激活显存（无并行，单位 bytes）：

```
Attention 块：11sbh + 5as²b
  - QKV 投影输入：2sbh，dropout mask：sbh
  - Q, K 存储：4asbh（= 4sbh，因为拆头后每头 d=h/a）
  - Softmax 输出：2as²b
  - Softmax dropout mask：as²b
  - Attention over V：2as²b + 2sbh

MLP 块：19sbh
  - 两个线性层输入：2sbh + 8sbh
  - GeLU 输入：sbh
  - Dropout mask：sbh

LayerNorm（2个）：4sbh

总计：sbh(34 + 5as/h)     ← 公式 1
```

---

## 贡献 1：Sequence Parallelism（§4.2.2）

### 核心观察

LayerNorm 和 Dropout 是逐元素操作，各 token 互相独立。TP 按 h 维度切分 Attention/MLP，那 LayerNorm/Dropout 可以按 **s 维度**切分。

### 新布局（Figure 5）

```
原来（TP only, Figure 4）：
  [LayerNorm全量] → f → [Attention(TP)] → [Dropout全量] → f̄ → ...
  10sbh 不被 t 除

现在（TP + SP, Figure 5）：
  [LayerNorm(SP,按s切)] → g → [Attention(TP,按h切)] → [Dropout(SP,按s切)] → ḡ → ...
  全部被 t 除
```

### 零额外通信的秘密

原来 TP 用 All-Reduce（= Reduce-Scatter + All-Gather）。SP 把它拆开：
- g（SP→TP）= All-Gather（forward）/ Reduce-Scatter（backward）
- ḡ（TP→SP）= Reduce-Scatter（forward）/ All-Gather（backward）

原来每层 2 次 All-Reduce = 2×(RS+AG)。现在 4 次单独操作 = 2×RS + 2×AG。**总通信量完全一样**。

### 显存效果

```
TP only：  sbh(10 + 24/t + 5as/ht)    ← 10sbh 没被除
TP + SP：  sbh/t × (34 + 5as/h)       ← 全部被 t 除
```

t=8 时，激活显存减少约 **50%**。

---

## 贡献 2：Selective Activation Recomputation（§5）

### 核心观察

一层里各操作的"显存 vs 计算"比例差异巨大：

| 操作 | 显存占比 | 重算计算成本 |
|---|---|---|
| QK^T, Softmax, Softmax dropout, Attention×V | 5as²b（大，和 s² 成正比） | 低（逐元素操作为主） |
| 线性层输入、GeLU 输入等 | 34sbh/t（相对小） | 高（矩阵乘法） |

### 策略

只丢弃并重算 attention 内部的大 tensor（5as/h 项），保留其余所有激活值：

```
Full recomputation：全丢全重算 → 显存 = 2sbhL，但 30-40% 额外计算
Selective recomputation：只丢 attention 内部 → 显存 = 34sbhL/t，只多 2-3% 计算

GPT-3 (175B): 5as/h = 5×96×2048/12288 = 80 >> 34
  → attention 占激活显存 70%
  → selective recompute 只多 2.7% FLOPs，省了 70% 激活显存
```

---

## 所有配置对比（Table 2）

| 配置 | 每层激活显存 | 说明 |
|---|---|---|
| 无并行 | sbh(34 + 5as/h) | 基准 |
| TP only | sbh(10 + 24/t + 5as/ht) | 10sbh 未被 t 除 |
| **TP + SP** | **sbh/t(34 + 5as/h)** | 全部被 t 除 |
| TP + Selective | sbh(10 + 24/t) | 丢了 attention 大 tensor |
| **TP + SP + Selective** | **34sbh/t** | 最优：全除 + 丢 attention |
| Full Recompute | 2sbh | 最省但 30-40% 额外计算 |

---

## 关键实验结果

### Figure 7：显存节省

SP 和 Selective 各自减少约 50% 激活显存，组合后减少 **~5×**（降到 tensor-parallel baseline 的 ~20%）。这是 full recompute 显存的 ~2× 倍，但几乎无额外计算。

### Table 4：单层耗时（22B 模型）

| 配置 | Forward | Backward | 总计 | 开销 |
|---|---|---|---|---|
| Baseline（无 recompute） | 7.7ms | 11.9ms | 19.6ms | — |
| SP（无 recompute） | 7.2ms | 11.8ms | 19.0ms | **-3%** |
| Full Recompute | 7.7ms | 19.5ms | 27.2ms | 39% |
| Selective Recompute | 7.7ms | 13.2ms | 20.9ms | 7% |
| **Selective + SP** | **7.2ms** | **13.1ms** | **20.3ms** | **4%** |

SP 本身还加速了 forward（LayerNorm/Dropout 只算 1/t 的数据）。Selective + SP 组合只有 **4% 开销**，而 full recompute 是 39%。

### Table 5：端到端吞吐

| 模型 | Full Recompute | Present Work | 提升 | MFU |
|---|---|---|---|---|
| 22B | 1.42s | 1.10s | 29% | 41.5% |
| 175B | 18.13s | 13.75s | 31.8% | 51.4% |
| 530B | 49.05s | 37.83s | 29.7% | 56.0% |
| 1T | 94.42s | 71.49s | 32.1% | **56.3%** |

530B 模型 MFU 从 42.1%（full recompute）→ **54.2%**（present work on 2240 GPUs）。

---

## 关键 Figure/Table 索引

| 图表 | 内容 | 重要性 |
|---|---|---|
| Figure 1 | 参数+optimizer vs 激活显存对比，present work 大幅缩小激活 | ⭐⭐⭐⭐ |
| Figure 4 | TP only 的 Transformer 层（LayerNorm/Dropout 未切分） | ⭐⭐⭐⭐⭐ 核心 |
| Figure 5 | TP + SP 的 Transformer 层（全部切分） | ⭐⭐⭐⭐⭐ 核心 |
| Figure 6 | MLP 块的 SP+TP 详细数据流（g 和 ḡ 操作） | ⭐⭐⭐⭐ |
| Figure 7 | 各技术组合的显存占比 | ⭐⭐⭐⭐ |
| Figure 8 | Forward/backward/recompute 时间分解 | ⭐⭐⭐⭐ |
| Table 2 | 所有配置的激活显存公式汇总 | ⭐⭐⭐⭐⭐ |
| Table 4 | 单层耗时对比（selective 只多 4%） | ⭐⭐⭐⭐ |
| Table 5 | 端到端吞吐和 MFU | ⭐⭐⭐⭐⭐ |

---

## 和其他论文的关系

- **← Megatron 2019**：TP 切法直接复用，SP 是 TP 的自然补充（TP 管 h，SP 管 s）
- **← Megatron 3D (2021)**：那篇的 52% MFU 用的是 full recompute；加了本文技术后能到 56%
- **→ LLaMA 3.1**：训练时使用了 SP + Selective Recomputation
- **→ DeepSeek-V3**：同样使用这些技术，加上 FP8 进一步降低显存
- **通信原语联系**：All-Reduce = Reduce-Scatter + All-Gather 这个拆分是 SP 零额外通信的数学基础
