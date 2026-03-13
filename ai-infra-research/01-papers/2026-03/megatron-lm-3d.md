---
title: "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"
tags: [training-infra, 3d-parallelism, pipeline-parallelism, tensor-parallelism, distributed-training]
subfield: training-infra
venue: "SC 2021"
date: 2026-03-11
authors: [Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, Matei Zaharia]
institution: [NVIDIA, Stanford University, Microsoft Research]
url: "https://arxiv.org/abs/2104.04473"
status: 已读
rating: ⭐⭐⭐⭐⭐
---

# Megatron-LM 3D Parallelism：高效组合 TP + PP + DP 训练万亿参数模型

## 一句话总结

系统分析了 TP、PP、DP 三种并行的交互与 tradeoff，提出 PTD-P（Pipeline + Tensor + Data Parallelism）组合策略和 interleaved 1F1B 调度，在 3072 张 A100 上训练 1T 参数模型达 502 petaFLOP/s（52% 峰值利用率），比 ZeRO-3 高 70% 吞吐。

---

## 核心问题

Megatron 2019 只有 TP + DP，最大 8.3B。模型继续增长（GPT-3 175B），TP=8 是节点内上限，DP 不切模型 → 放不下。

**核心挑战：TP、PP、DP 怎么组合才能在数千 GPU 上高效训练？三种并行之间有复杂交互，不是"都用上"就行。**

---

## 三种并行的分工与组合（Figure 2）

| 并行方式 | 切什么 | 通信类型 | 通信带宽 | 适用范围 |
|---|---|---|---|---|
| TP (t) | 单层的矩阵 | All-Reduce | NVLink (高) | 节点内 |
| PP (p) | 层（跨节点） | 点对点 Send/Recv | InfiniBand (中) | 跨节点 |
| DP (d) | 数据 | All-Reduce 梯度 | InfiniBand (中) | 跨节点 |

总 GPU 数 n = t × p × d。例如 1T 模型 3072 张 A100：t=8, p=64, d=6。

---

## Pipeline 调度策略（核心贡献）

### GPipe 调度（Figure 3）

所有 forward 做完 → 所有 backward 做完 → pipeline flush。

- Bubble 比例 = (p-1)/m（p = pipeline stages, m = micro-batch 数）
- 缺点：要 m ≫ p 才能压小 bubble，但同时存 m 个 micro-batch 的激活值 → 显存爆炸

### 1F1B 调度（Figure 4 上半）

PipeDream-Flush：warm-up 阶段逐个注入 forward → 稳态阶段每做 1 个 forward 紧跟 1 个 backward → cool-down 清理。

- Bubble 大小不变：(p-1)/m
- 优势：in-flight micro-batch 数最多 = p（不是 m）→ **显存大幅减少**

### Interleaved 1F1B 调度（Figure 4 下半，本文新提出）

**每张 GPU 负责多个不连续的层块（model chunks）**。

例如 4 GPU, 8 层, v=2 chunks per GPU：
- 原来：GPU 0 管 layer 1-2，GPU 1 管 layer 3-4 ...
- Interleaved：GPU 0 管 layer 1,2 和 layer 5,6；GPU 1 管 layer 3,4 和 layer 7,8

效果：
- Bubble 比例 = **(1/v) × (p-1)/m** → 缩小 v 倍
- 代价：通信量增加 v 倍（send/recv 次数多了）
- 需要 m 是 p×v 的整数倍

Figure 12 显示 interleaved + scatter/gather 在大 batch 时比 non-interleaved 高 10%+。

---

## Scatter/Gather 通信优化（Figure 9）

**问题**：TP + PP 组合时，同一 pipeline stage 内 8 张 GPU（TP）持有相同的激活值。朴素 pipeline send/recv → 8 倍冗余跨节点通信。

**优化**：
1. 发送端：scatter，每张 GPU 只通过 InfiniBand 发 1/8 的 tensor
2. 接收端：all-gather，用 NVLink 内拼回完整 tensor

跨节点通信量减少到 **bsh/t**（原来是 bsh），用快速 NVLink 替代慢速 InfiniBand 冗余。Figure 18 显示最高 11% 吞吐提升。

---

## 三条指导原则（Takeaways）

**Takeaway #1：TP 先用满节点，PP 再跨节点扩展**
- TP 用 NVLink（快），PP 用 InfiniBand 点对点（慢但够用）
- 最佳：t = 节点内 GPU 数（8），然后用 PP 扩展

**Takeaway #2：TP × PP 刚好让模型放进显存，剩余 GPU 给 DP**
- DP 用来扩大训练规模（更多数据并行副本）

**Takeaway #3：Microbatch size 是需要调优的超参数**
- 影响算力利用率（大 → 更 compute-bound）
- 影响 bubble（大 batch → m 多 → bubble 小）
- 影响显存
- 最优值因模型而异（Figure 7-8）

---

## 关键实验结果

### Table 1：Weak Scaling 到 1T

| 参数量 | 层数 | 隐藏维度 | TP | PP | GPU 数 | teraFLOP/s per GPU | 峰值占比 |
|---|---|---|---|---|---|---|---|
| 1.7B | 24 | 2304 | 1 | 1 | 32 | 137 | 44% |
| 18.4B | 40 | 6144 | 8 | 1 | 256 | 135 | 43% |
| 145.6B | 80 | 12288 | 8 | 8 | 1536 | 148 | 47% |
| 530B | 105 | 20480 | 8 | 35 | 2520 | 163 | 52% |
| **1T** | **128** | **25600** | **8** | **64** | **3072** | **163** | **52%** |

1T 模型：502 petaFLOP/s 聚合吞吐，预计训练约 3 个月。

### Table 2 & Figure 10：PTD-P vs ZeRO-3

同样的 175B 和 530B 模型，PTD-P 比 ZeRO-3 高 **70%** 吞吐。原因：ZeRO-3 不用 TP，全靠跨节点通信切分状态。GPU 数翻倍时 PTD-P 效率基本不变，ZeRO-3 下降明显。

### Figure 13-15：最佳并行配置

TP=8 + PP + DP 始终优于其他组合。单独增大 PP（减少 DP）→ bubble 增大、吞吐下降。单独增大 TP 超过节点 → 跨节点 All-Reduce 拖累性能。

---

## 其他优化

### Activation Recomputation（§3.5, Figure 17）

- 小 batch：吞吐降低最多 33%（额外 forward 的代价）
- 大 batch：有了 activation recomputation 反而吞吐更高（因为能用更大 batch，bubble 更小）
- 最优 checkpoint 间隔 c = √(l × A_intermediate / A_input)，实际中每 1-2 层 checkpoint

### Operator Fusion（§4.2）

将 bias + GeLU + bias + dropout 融合为单个 kernel，175B 模型吞吐提升 19%。

### 数据布局优化（§4.2）

从 [b, s, a, h] 改为 [s, b, a, h]，避免内存密集的 transpose 操作，启用 strided batched GEMM。

---

## 关键 Figure/Table 索引

| 图表 | 内容 | 重要性 |
|---|---|---|
| Figure 2 | TP + PP 组合示意（节点内 TP，跨节点 PP） | ⭐⭐⭐⭐⭐ 核心 |
| Figure 3 | GPipe 调度（全 forward → 全 backward） | ⭐⭐⭐⭐ |
| Figure 4 | 1F1B vs Interleaved 1F1B 调度对比 | ⭐⭐⭐⭐⭐ 核心 |
| Figure 5 | TP 切分复习（= Megatron 2019 Figure 3） | ⭐⭐⭐ |
| Figure 6 | Bubble 比例随 DP degree 变化 | ⭐⭐⭐ |
| Figure 7-8 | Microbatch size 对吞吐的影响 | ⭐⭐⭐⭐ |
| Figure 9 | Scatter/Gather 通信优化 | ⭐⭐⭐⭐ |
| Figure 10 | PTD-P vs ZeRO-3 吞吐对比 | ⭐⭐⭐⭐⭐ |
| Figure 12 | Interleaved vs Non-interleaved 吞吐对比 | ⭐⭐⭐⭐ |
| Figure 13-15 | TP/PP/DP 不同组合的吞吐对比 | ⭐⭐⭐⭐ |
| Table 1 | 1B → 1T weak scaling 全景 | ⭐⭐⭐⭐⭐ |
| Table 2 | PTD-P vs ZeRO-3 详细对比 | ⭐⭐⭐⭐ |

---

## 和其他论文的关系

- **← Megatron 2019**：TP 切法直接复用，本文加入 PP 和系统性组合分析
- **← PipeDream-Flush**：1F1B 调度来自此工作，本文在此基础上提出 interleaved 版本
- **→ ZeRO（DeepSpeed）**：不同路线。ZeRO 切状态但不切计算，PTD-P 切计算。实测 PTD-P 高 70%
- **→ LLaMA 3.1**：训练基础设施直接基于 Megatron PTD-P
- **→ DeepSeek-V3**：在 3D Parallelism 基础上加 EP（4D）
- **→ MegaScale-Infer 的 ping-pong**：本质上也是 interleaved schedule（交替执行不同任务隐藏延迟）

---

## FLOP 计算公式（Appendix）

每 Transformer 层每 iteration 的 FLOPs（含 activation recomputation 的额外 forward）：

```
96Bslh² × (1 + s/(6h))
```

其中 B=batch, s=seq_len, l=layers, h=hidden_dim。

加上 logit 层后总 FLOPs：

```
96Bslh² × (1 + s/(6h) + V/(16lh))
```

训练时间估算：End-to-end training time ≈ 8TP / (nX)

其中 T=tokens, P=参数量, n=GPU数, X=每GPU吞吐(FLOP/s)。

---

## 我的理解（对话校准）

### Bubble 比例推导

从任一 stage 视角：要等前面的 stage 完成 forward，又等后面的 stage 回传 backward，总共 idle = (p-1) × (t_f + t_b)。

```
GPipe 时间线（p=4, m=8）：

Stage 1: [F1][F2]...[F8]________________________[B8]...[B1]
Stage 2: ____[F1][F2]...[F8]________________[B8]...[B1]
Stage 3: ________[F1]...[F8]________[B8]...[B1]
Stage 4: ____________[F1]...[F8][B8]...[B1]____________

不管在哪个 stage，idle 都是 (p-1) 个单位。

Bubble 比例 = (p-1) × (t_f + t_b) / [m × (t_f + t_b)] = (p-1)/m
```

p=2 ping-pong 式，m=4 → bubble = 1/4 = 25%。m 越大 bubble 越小。

### Interleaved 1F1B 具体理解

最小例子：8 层，2 GPU，v=2 chunks/GPU，m=4。

```
普通（v=1）：
  GPU 0: layer 1-4（一个大 chunk），GPU 1: layer 5-8
  micro-batch 旅程：GPU0(4层) → GPU1(4层)
  每步耗时 = t_f

Interleaved（v=2）：
  GPU 0: layer 1,2 + layer 5,6（两个小 chunk）
  GPU 1: layer 3,4 + layer 7,8
  micro-batch 旅程：GPU0(2层) → GPU1(2层) → GPU0(2层) → GPU1(2层)
  每步耗时 = t_f/2
```

Bubble 发生在 warm-up（GPU 1 等 GPU 0 完成第一个 chunk）：
- 普通：等 t_f（算 4 层）
- Interleaved：等 t_f/2（只算 2 层）→ 等待缩短 v 倍

Bubble = (1/v) × (p-1)/m。代价：通信次数多 v 倍（数据在 GPU 间来回弹）。

Figure 4 深色/浅色 = 第一个 chunk / 第二个 chunk，同一 micro-batch 在同一 GPU 上出现两次。

### Scatter/Gather 为什么能减少通信

关键：TP 组内 8 张 GPU 持有**同一份完整激活值**。

```
无优化：8 张 GPU 各自通过 InfiniBand 发完整 tensor → 8× 冗余
  InfiniBand 流量 = 8D

有 scatter/gather：
  发送端：每张 GPU 只发 1/8 → InfiniBand 流量 = D（减少 8 倍）
  接收端：NVLink All-Gather 拼回完整 tensor（快且在节点内）
```

不是"要传的数据变少了"，而是**消除了冗余传输**。Megatron 2019 没有这个优化是因为当时没有 PP，不存在跨节点的 pipeline send/recv。

### 52% MFU 的拆解

52% = 实测吞吐 / A100 BF16 峰值（163/312 teraFLOP/s）。剩余 48% 的来源：

- **Pipeline bubble**：~10-15%（1T 模型 p=64, interleaved 后缩小）
- **通信开销**：TP 每层 4 次 All-Reduce + PP send/recv + DP 梯度同步，无法完全被遮盖
- **Activation recomputation**：额外 ~33% 的 forward 计算不算进 MFU
- **非 GEMM 操作**：LayerNorm、Softmax、Dropout、optimizer step 等 memory-bound 操作
- **Kernel launch 等碎片开销**

即使完美消除 bubble 和通信，理论 MFU 上限约 **60-65%**（activation recomputation 和 non-GEMM 不可消除）。2024 年 DeepSeek-V3 报 ~55%（H800, FP8+DualPipe），接近此上限。

### 三种并行的通信成本对比

| | TP | PP | DP |
|---|---|---|---|
| 同步什么 | 激活值 | 激活值（点对点） | 梯度 |
| 频率 | 每层每 micro-batch | 每 micro-batch 每 stage pair | 每 batch 一次 |
| 每次数据量 | 8bsh×(t-1)/t per layer | bsh/t（scatter/gather 后） | ~2×参数量×2(d-1)/d |
| 在 critical path 上？ | ✅ 不可重叠 | 部分可重叠 | ✅ 大部分可与 backward 重叠 |
| 带宽需求 | 最高 → NVLink | 中等 → InfiniBand | 最低（可遮盖）→ InfiniBand |

排序：**TP（带宽需求最高）> PP（中等）> DP（可遮盖，最低）**。这就是 Takeaway #1 的数学来源。

### 控制变量实验总结（Figure 13-15）

**TP vs PP（Figure 13，162B, 64 GPU）**：TP=8 + PP=8 最优。TP>8 跨节点 All-Reduce 太慢，TP<8 节点没用满且 PP stage 多 bubble 大。

**PP vs DP（Figure 14，5.9B, 64 GPU）**：PP 越多吞吐越低（bubble 增大）。PP 只用到模型放得下的最小值，剩余全给 DP。

**TP vs DP（Figure 15，5.9B, 64 GPU）**：小模型 TP=2 就够，TP 不应超过需要的最小值。
