---
title: "DeepSeek-V3 Technical Report — Section 3: Training Infrastructure"
tags: [training-infra, moe, pipeline-parallelism, fp8, expert-parallelism, communication-optimization]
subfield: training-infra
venue: "arXiv 2024 (DeepSeek-AI)"
date: 2026-03-12
authors: [DeepSeek-AI]
institution: [DeepSeek]
url: "https://arxiv.org/abs/2412.19437"
status: 已读（Section 3）
rating: ⭐⭐⭐⭐⭐
---

# DeepSeek-V3 训练基础设施

## 一句话总结

671B MoE 模型在 2048 张 H800 上训练 14.8T tokens，**不用 TP**，靠 DualPipe（双向流水线遮盖通信）+ FP8 混合精度训练 + 精细 All-to-All 通信优化 + 极致内存管理，把训练成本压到同期模型的 1/5~1/10（约 557 万美元）。

---

## 训练配置总览

```
硬件：2048 × NVIDIA H800 GPU（节点内 NVLink 160 GB/s，节点间 IB 50 GB/s）
并行策略：
  PP = 16（流水线并行，DualPipe 双向调度）
  EP = 64（专家并行，跨 8 节点）
  ZeRO-1 DP（只切 optimizer states）
  TP = 0 ← 不用 Tensor Parallelism！

对比 LLaMA 3.1 405B：TP=8 + PP=16 + DP（标准 Megatron PTD-P）
```

**不用 TP 的意义**：TP 每层 2 次 All-Reduce 在关键路径上不可遮盖，是通信开销最高的并行方式。DeepSeek-V3 通过 FP8 省显存 + 精细内存优化，让单卡可以放得下完整的一层，从而消灭了 TP 这个最贵的通信开销。

---

## 创新 1：DualPipe（§3.2.1）

### 要解决的问题

MoE 模型的 EP 通信量极大：每个 MoE block 需要 2 次 All-to-All（dispatch + combine），计算与通信比约 1:1。传统 PP（1F1B）无法遮盖这么重的通信。

### 核心思路：细粒度拆分 + 计算通信重叠

把每个 micro-batch chunk 拆成更细的子步骤：

```
Forward chunk:  ATTN(F) → DISPATCH(F) → MLP(F) → COMBINE(F)
                 计算       通信          计算      通信

Backward chunk（更细，拆出 input backward 和 weight backward）:
  ATTN(B) → ATTN(W) → DISPATCH(B) → MLP(B) → MLP(W) → COMBINE(B)
   计算       计算        通信          计算      计算       通信
  + PP 通信
```

然后把一对 forward chunk 和 backward chunk 交错排列，让一个 chunk 的**通信**和另一个 chunk 的**计算**在时间上完全重叠。GPU 分配一部分 SM 给通信（~20/132 个 SM），剩余给计算，两者并行执行。

### 双向流水线调度（Figure 5）

DualPipe 从流水线**两端**同时喂 micro-batch：

```
方向 1（左→右）: micro-batch 从 Device 0 进入，forward 走向 Device 7
方向 2（右→左）: micro-batch 从 Device 7 进入，forward 走向 Device 0
```

中间的 GPU 同时处理两个方向的 chunk → 一个方向的计算可以遮盖另一个方向的通信。

### 与其他 PP 方法对比（Table 2）

| 方法 | Bubble | 参数副本 | 激活显存 |
|---|---|---|---|
| 1F1B | (PP-1)(F+B) | 1× | PP |
| ZB1P | (PP-1)(F+B-2W) | 1× | PP |
| DualPipe | **(PP/2-1)(F&B+B-3W)** | **2×** | **PP+1** |

- Bubble 显著更小
- 代价：2× 参数（双向调度需要两份权重副本来维护梯度）
- EP 下每卡参数量不大，2× 可接受

---

## 创新 2：跨节点 All-to-All 通信优化（§3.2.2）

### IB + NVLink 分层利用

```
节点间 IB: 50 GB/s    节点内 NVLink: 160 GB/s（3.2× IB）

策略：
  1. 限制每个 token 最多路由到 4 个节点（减少 IB 流量）
  2. 两跳传输：
     token → IB 发到目标节点的"同 index GPU" → NVLink 转到目标 expert 所在 GPU
  3. IB 和 NVLink 通信完全重叠
  → 每 token 平均可选 3.2 experts/node，实际选 8 个 expert 可分布在最多 4 个节点
```

### SM 分配

只用 20 个 SM（H800 共 132 个，占 15%）做通信：

```
20 SM → 10 个通信 channel：
  Dispatch: (1) IB 发送, (2) IB→NVLink 转发, (3) NVLink 接收
  Combine:  (1) NVLink 发送, (2) NVLink→IB 转发+累加, (3) IB 接收+累加
  warp 数量按实际负载动态调整
```

用自定义 PTX 指令精调通信 chunk size，减少 L2 cache 干扰。

---

## 创新 3：FP8 混合精度训练（§3.3）

### 混合精度策略（Figure 6）

```
FP8 计算（理论算力翻倍）：
  三大 GEMM: Fprop / Dgrad / Wgrad → FP8 输入，FP32 累加，BF16 输出

保留高精度（BF16/FP32）：
  Embedding, Output Head, MoE Gating, LayerNorm, Attention → 对精度敏感

存储策略：
  Master weights: FP32 → ZeRO-1 分片
  Weight gradients: FP32
  Optimizer states (Adam m/v): BF16（而非传统 FP32，通过分片减小 overhead）
  激活缓存: FP8（省一半显存）
```

### Fine-Grained Quantization（Figure 7a）

传统 per-tensor 量化：一个 outlier 拖垮整个 tensor。解决方案：

```
激活：1×128 tile 分组量化（每 token 每 128 channel 独立缩放）
权重：128×128 block 分组量化
→ outlier 只影响所在 tile/block，不拖累全局
```

### Increasing Accumulation Precision（Figure 7b）

```
H800 Tensor Core FP8 GEMM 内部累加精度 ≈ 14 bit（不是完整 FP32）
K 大时误差严重（K=4096 时最大相对误差 ~2%）

解法：每 N_c=128 个元素（= 4 次 WGMMA 指令）提升到 CUDA Core 做 FP32 累加
H800 上两个 warpgroup 交替执行 MMA 和 promotion → 吞吐几乎不受影响
```

### 其他 FP8 细节

- **E4M3 统一**：所有 GEMM 都用 E4M3（不像之前 forward E4M3 + backward E5M2）
- **Online quantization**：实时计算每个 tile/block 的 max 值并缩放（不用 delayed scaling）
- **低精度通信**：MoE dispatch 前把激活量化到 FP8 传输，减少 All-to-All 带宽需求

### 效果

与 BF16 baseline 相比，FP8 训练 loss 相对误差始终低于 0.25%（在训练随机性范围内）。

---

## 创新 4：极致内存优化（§3.2.3）

不用 TP → 每张 GPU 承载更多 → 需要极致省显存：

| 优化手段 | 节省了什么 | 代价 |
|---|---|---|
| RMSNorm + MLA up-projection 重算 | 不存这些激活，backward 重算 | 少量额外计算 |
| EMA 放 CPU | 指数移动平均参数存 CPU，异步更新 | 零 GPU 显存开销 |
| Embedding + Output Head 共享 PP rank | DualPipe 把最浅层和最深层放同一个 rank | 物理共享参数和梯度 |

---

## 对硬件的建议（§3.5）

DeepSeek 对 GPU 厂商提出两个建议，视角很有价值：

1. **通信协处理器**：All-to-All 通信占用 20 个 SM（15% 计算资源），浪费了宝贵的 Tensor Core。希望有专用协处理器（类似 NVIDIA SHARP）统一 IB + NVLink 网络，释放 SM 给计算。
2. **Tensor Core FP8 累加精度**：H800 的 14-bit 累加是硬伤，建议提高到完整 FP32。当前的 workaround（每 128 元素提升到 CUDA Core）虽然有效但增加了编程复杂度。

---

## 关键数字

```
训练规模：2048 H800, 14.8T tokens
训练成本：2.788M H800 GPU hours ≈ $5.576M
并行策略：PP=16, EP=64, ZeRO-1 DP, 无 TP
FP8 vs BF16：loss 相对误差 < 0.25%
通信 SM：20/132（15%），其余 112 个做计算
每 token 路由：8 experts，最多跨 4 个节点
```

---

## 我的理解（讨论记录）

### MoE 为什么需要 All-to-All

EP=64 意味着 256 个 expert 分散在 64 张 GPU 上，每张只存 4 个 expert。token 被 router 指定去的 expert 大概率不在当前 GPU → 必须物理发送过去。每张 GPU 都要给不同 GPU 发 token、同时接收来自不同 GPU 的 token → 这就是 All-to-All 的定义。

All-to-All 的"transpose"本质：数据组织维度从"按来源 GPU"变为"按目标 expert"。Combine 是反向操作。

### Combine 为什么不可省略

一个 token 的 8 个 expert 结果散落在不同 GPU 上，必须回传给原 GPU 做加权求和 + 残差连接（H_out = H_mid + weighted_sum）。只有原 GPU 持有 H_mid。

### DualPipe 的 2× 参数

不是因为存了不同层的权重，而是双向调度需要两份参数副本来维护梯度。EP 下每卡参数量不大，多存一份可接受。

### MLP = FFN

同一个东西两个名字。Multi-Layer Perceptron 是经典神经网络结构（升维 → 非线性 → 降维），Transformer 论文叫 FFN，GPT/LLaMA 代码叫 MLP。

### 待深入理解

- [ ] DualPipe Figure 5 的完整调度细节（哪个时刻哪个 device 做什么）
- [ ] FP8 fine-grained quantization 的数学细节（per-group scaling factor 如何与 GEMM 结合）
- [ ] Warpgroup-level MMA 的硬件机制（需要 GPU 架构背景）
- [ ] 2× 参数副本的具体梯度管理机制

---

*创建：2026-03-12 | 关联：[[megatron-lm]] [[megatron-lm-3d]] [[reducing-activation-recomputation]]*
