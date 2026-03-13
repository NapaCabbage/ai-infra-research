---
title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
tags: [training-infra, tensor-parallelism, distributed-training, transformer]
subfield: training-infra
venue: "arXiv 1909.08053 (2019)"
date: 2026-03-11
authors: [Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro]
institution: [NVIDIA]
url: "https://arxiv.org/abs/1909.08053"
status: 已读
rating: ⭐⭐⭐⭐⭐
---

# Megatron-LM：用模型并行训练数十亿参数的语言模型

## 一句话总结

Tensor Parallelism 的奠基之作。提出了 Transformer MLP 和 Self-Attention 的高效切分方式：列切第一个矩阵 + 行切第二个矩阵，使得每个 Transformer 层只需 4 次 All-Reduce（forward 2 + backward 2）。在 512 张 V100 上训练 8.3B GPT-2，达到 76% 扩展效率。

---

## 核心问题

2019 年背景：模型越来越大（BERT-Large 340M，GPT-2 1.5B），单张 GPU 放不下或训练太慢。Data Parallelism 要求每张卡放一份完整模型，模型太大就失效。

**核心挑战：怎么把 Transformer 的单层计算切分到多张 GPU 上，同时尽量少通信？**

---

## 核心方法：Transformer 的 TP 切法（§3）

### MLP 切分（Figure 3a）

Transformer FFN 有两个矩阵运算：

```
Y = GeLU(X · A) · B
```

切法：
1. **矩阵 A 按列切（Column Parallel）**：A 切成 [A₁, A₂]，分到 GPU 0 和 GPU 1
   - 每张 GPU 算 GeLU(X · Aᵢ)
   - 关键：GeLU 是逐元素操作，按列切后可以**独立计算**，不需要通信
2. **矩阵 B 按行切（Row Parallel）**：B 切成 B₁ / B₂
   - 每张 GPU 算完后做一次 **All-Reduce** 合并结果

**为什么巧妙**：整个 MLP 只需要 1 次 All-Reduce。如果切法不对（比如 A 按行切），GeLU 就不能独立算，需要额外通信。

### Self-Attention 切分（Figure 3b）

Multi-Head Attention 天然适合切分——本来就有多个 head。

- Q、K、V 投影矩阵按列切（每张 GPU 负责一部分 head）
- 每张 GPU 独立算自己的 head 的 attention
- 输出投影矩阵 O 按行切
- 最后一次 **All-Reduce** 合并

同样只需 **1 次 All-Reduce**。

### 每层总通信量（Figure 4）

一个 Transformer 层 = Self-Attention + MLP：
- **Forward**：2 次 All-Reduce（Attention 1 次 + MLP 1 次）
- **Backward**：2 次 All-Reduce（反向传播对称）
- **总计：4 次 All-Reduce per layer**

→ 这就是 TP 必须在 NVLink 节点内使用的原因：每层都通信 4 次，跨节点 InfiniBand 带宽撑不住。

---

## Embedding 层的处理

词表矩阵 V×H 也很大（如 GPT-2 的 50257×1024）。切法：
- 按词表维度切分到各 GPU，每张 GPU 只存一部分词表
- forward 结束后做 All-Reduce 合并 logits
- Cross-entropy loss 可以在各 GPU 上并行计算

---

## 扩展性分析

### Figure 1 & 5：Scaling Efficiency

| 模型大小 | TP degree | GPU 数 | 效率 |
|---|---|---|---|
| 1.2B | 1 (纯 DP) | 32 | ~96% |
| 2.5B | 2 | 64 | ~93% |
| 4.2B | 4 | 128 | ~85% |
| 8.3B | 8 | 512 | **76%** |

关键观察：
- TP 1→2 效率几乎无损（NVLink 带宽够用）
- TP 4→8 效率下降明显——All-Reduce 通信量随 TP degree 线性增长
- Weak scaling（模型和 GPU 数一起增大）比 strong scaling 效率高

### Table 1：模型配置

| 模型 | 参数量 | 层数 | 隐藏维度 | Attention heads | TP | DP | 总 GPU |
|---|---|---|---|---|---|---|---|
| 1.2B | 1.2B | 24 | 2048 | 32 | 1 | 32 | 32 |
| 2.5B | 2.5B | 40 | 2560 | 32 | 2 | 32 | 64 |
| 4.2B | 4.2B | 48 | 3072 | 32 | 4 | 32 | 128 |
| 8.3B | 8.3B | 72 | 3072 | 24 | 8 | 64 | 512 |

---

## 模型质量结果

### GPT-2 结果（Table 2-3）

8.3B GPT-2 在 WikiText-103 上的 perplexity = **10.8**，当时 SOTA。证明了 TP 训练出的大模型质量确实好，不只是参数多。

### BERT 结果（Table 4-5）

3.9B BERT 在多个下游任务上超 BERT-Large。

### Figure 7：LayerNorm 位置的重要发现

原始 BERT 用 post-LayerNorm（残差连接之后）。发现模型变大后 post-LN 训练不稳定。改为 **pre-LayerNorm**（残差连接之前）后解决。

→ 这个发现影响深远：后来 GPT-3、LLaMA 全系列都用 pre-LN（LLaMA 用的 RMSNorm 本质一样）。

---

## 混合并行策略（Figure 8，Appendix B）

TP 只能在节点内（NVLink），跨节点用 DP。组合方式：
- 节点内 8 张 GPU 做 TP（切分单层计算）
- 跨节点做 DP（不同节点处理不同数据，梯度 All-Reduce）

这种 TP + DP 组合是后续 3D Parallelism（TP + PP + DP）的前身。2021 年 Megatron 扩展论文加入了 PP，完成了 3D。

---

## 关键 Figure/Table 索引

| 图表 | 内容 | 重要性 |
|---|---|---|
| Figure 3a | MLP 切分：列切 A + 行切 B + 1 次 All-Reduce | ⭐⭐⭐⭐⭐ 核心 |
| Figure 3b | Self-Attention 切分：按 head 切 + 1 次 All-Reduce | ⭐⭐⭐⭐⭐ 核心 |
| Figure 4 | 完整 Transformer 层：4 次 All-Reduce（f/g 算子） | ⭐⭐⭐⭐⭐ |
| Figure 1 | 8.3B 模型 scaling 到 512 GPU 的效率曲线 | ⭐⭐⭐⭐ |
| Figure 5 | Weak scaling vs Strong scaling 对比 | ⭐⭐⭐⭐ |
| Figure 7 | Pre-LN vs Post-LN 对训练稳定性的影响 | ⭐⭐⭐⭐ |
| Figure 8 | TP + DP 混合并行的节点拓扑 | ⭐⭐⭐ |
| Table 1 | 模型配置（层数、隐藏维度、TP/DP 设置） | ⭐⭐⭐ |
| Table 2-3 | GPT-2 结果（WikiText perplexity SOTA） | ⭐⭐⭐ |

---

## 和其他论文的关系

- **→ Megatron 2021（SC）**：在此基础上加入 Pipeline Parallelism，完成 3D Parallelism（阅读路径必读②）
- **→ ZeRO（DeepSpeed）**：解决同一问题（模型太大）的另一条路线。TP 切计算，ZeRO 切状态
- **→ Activation Checkpointing（MLSys 2023）**：Megatron 团队后续工作，解决激活值显存占用（阅读路径必读⑤）
- **→ LLaMA 3.1**：训练基础设施基于 Megatron-LM，TP + PP + DP 的 3D 并行
- **→ DeepSeek-V3**：在 Megatron 的基础上加了 EP（Expert Parallelism）
- **→ MegaScale-Infer**：Expert Node 的 TP 切分直接使用 Megatron 的 MLP 切法

---

## 我的理解

### MLP 切分的具体数据流（对话校准）

以 H=4, FFN 升维到 8, TP=2 为例：

```
输入 X = [1,2,3,4]（1×4），每张 GPU 都有完整的 X

A（4×8 升维矩阵）按列切：A₁ = 左4列(4×4) → GPU0, A₂ = 右4列(4×4) → GPU1
B（8×4 降维矩阵）按行切：B₁ = 上4行(4×4) → GPU0, B₂ = 下4行(4×4) → GPU1

GPU 0: Z₀ = GeLU(X · A₁) · B₁ = 1×4 的部分和    ← 全部本地计算
GPU 1: Z₁ = GeLU(X · A₂) · B₂ = 1×4 的部分和    ← 全部本地计算

All-Reduce: Z = Z₀ + Z₁                          ← 唯一的通信
```

**为什么加起来就是对的？** 矩阵乘法的本质是"乘完再加"。每个输出元素 = 一行点乘一列 = 一串乘积的求和。切成两半各自算，最后加起来，和完整计算数学上完全等价：

```
完整计算：GeLU(X·A) · B = [Y₁, Y₂] · [B₁; B₂] = Y₁·B₁ + Y₂·B₂ = Z₀ + Z₁
                           列拼接      行堆叠      分块乘法展开
```

### All-Reduce 同步的是什么？

**是激活值（Z₀ 和 Z₁），不是权重**。TP 和 DP 的 All-Reduce 对象不同：

- **TP 的 All-Reduce**：同步激活值（每层的中间计算结果），每层都做 → 必须 NVLink
- **DP 的 All-Reduce**：同步梯度，整个 backward 做一次 → InfiniBand 就够

### Self-Attention 切分的数据流

Multi-Head Attention 天然按 head 切分，比 MLP 更自然：

```
假设 8 个 head, TP=2:

GPU 0: head 1-4 → Wq₁,Wk₁,Wv₁ 按列切（管4个head的投影）
GPU 1: head 5-8 → Wq₂,Wk₂,Wv₂

各自独立算 attention（零通信）

输出投影 Wo 按行切（和 MLP 的 B 一模一样的道理）：
GPU 0: Z₀ = [head1...4] · Wo₁   ← 部分和
GPU 1: Z₁ = [head5...8] · Wo₂   ← 部分和

All-Reduce: Z = Z₀ + Z₁         ← 唯一的通信
```

Attention 和 MLP 模式完全对称：内部本地算，最后一个矩阵按行切，All-Reduce 求和。

### 为什么列切 A + 行切 B 是最优切法？

关键在于 **GeLU 的性质**。GeLU 是逐元素的非线性操作：
- 如果 A 按列切，X·A₁ 和 X·A₂ 得到的是完整结果的不同列 → GeLU 可以独立算 ✓
- 如果 A 按行切，X₁·A₁ 得到的是部分和 → 必须先 All-Reduce 求完整和，才能算 GeLU ✗

所以 **列切第一个矩阵是为了避免非线性操作前的通信**。第二个矩阵 B 按行切是自然配合（列切的输出 × 行切的矩阵 = 部分和 → All-Reduce）。

### 通信量分析

假设隐藏维度 H，序列长度 S，batch 大小 B，TP degree T：
- 每次 All-Reduce 的数据量 = B × S × H（即输出激活值的大小）
- 每层 4 次 All-Reduce → 总通信量 = 4 × L × B × S × H
- 对比 DP 的 All-Reduce（同步梯度）：只在所有层算完后做一次，频率低得多
- 这就是 TP 必须 NVLink、DP 可以 InfiniBand 的原因

### 为什么 TP=8 是实际上限？

2019 年 DGX-2 有 8 张 V100 + NVLink 互联。超过 8 就要跨节点，带宽从 NVLink 的 300 GB/s 降到 InfiniBand 的 25 GB/s（12× 差距）。所以 TP degree 被物理拓扑限制在单节点 GPU 数量。

到 2024 年的 DGX H100（8×H100 + NVLink 4.0 = 900 GB/s），TP=8 仍然是常用上限。LLaMA 3.1 405B 就用 TP=8。
