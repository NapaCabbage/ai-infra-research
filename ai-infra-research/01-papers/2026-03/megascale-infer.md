---
title: "MegaScale-Infer: Serving MoE at Scale with Disaggregated Expert Parallelism"
tags: [inference-serving, moe, disaggregation, expert-parallelism]
subfield: inference-serving
venue: "arXiv 2504.02263 (ByteDance Seed, 2025)"
date: 2026-03-10
authors: [Ruidong Zhu, Ziheng Jiang, Chao Jin, Peng Wu, Cesar A. Stuardo, et al.]
institution: [ByteDance Seed, Peking University]
url: "https://arxiv.org/abs/2504.02263"
status: 已读
rating: ⭐⭐⭐⭐⭐
---

# MegaScale-Infer：Attention-FFN 解耦服务大规模 MoE

## 一句话总结

MoE 的稀疏激活让 FFN 从 compute-intensive 退化为 memory-intensive，MegaScale-Infer 把 Attention 和 FFN 拆到不同 GPU 上（Attention 用 Data Parallelism 复制多份，Expert 用 Expert Parallelism），通过 **ping-pong pipeline** 遮盖空闲和通信，配合自研 **M2N 通信库**（比 NCCL 吞吐高 4.2×、尾延迟低 96.2%），同构集群解码吞吐提升 1.90×，异构集群每美元吞吐提升 3.24×。已在字节跳动近万张 GPU 生产部署。

---

## 核心问题：MoE 稀疏性让 FFN 变"闲"

### Dense vs MoE 的 GPU 利用率差异（Figure 1）

| | Dense Model | MoE Model | MegaScale-Infer |
|---|---|---|---|
| **Attention** | memory-intensive（利用率低）| 同左 | 同左，但用高带宽 GPU |
| **FFN** | compute-intensive（利用率高）| **也变成 memory-intensive** | 重新变回 compute-intensive |
| **原因** | — | 稀疏激活：每 expert 只分到 batch×topk/#expert 个 token | 多 Attention 副本汇聚请求→expert batch size 倍增 |

### 为什么 Attention 在 decoding 阶段是 memory-bound？

常见误解：Attention 都是矩阵乘（Q×K），应该是 compute-bound。实际上在 **decoding 阶段**，每步只生成 1 个新 token，Q 是一个 (1, h) 的向量，要和所有历史 token 的 KV Cache 做点积。KV Cache 可能有几千到几万个 token，每个都要从显存读出来。**算的很少（向量×矩阵），读的很多（整个 KV Cache）**→ memory-bound。

Prefill 阶段则是 compute-bound，因为一次性处理整个 prompt 的所有 token，Q 矩阵很大。

### 为什么 FFN 在 dense model 里是 compute-bound？

FFN 做的事：input (b, h) × weight (h, 4h) → 激活 → × weight (4h, h) → output。Weight 矩阵对 batch 里所有 token **共享**——读一次 weight，做 b 次乘法。batch 越大，每读一次 weight 做的计算越多，arithmetic intensity ≈ b。

A100 的 roofline 平衡点：312 TFLOPS / 2 TB/s = **156**。所以 batch≥156 时 FFN 就是 compute-bound。Dense model 做到 batch=156 很容易。

### MoE 的问题：每个 expert 分到的 token 太少

MoE 不是说 expert 闲置不算了——每个 expert 确实在算，问题是**分到的 token 数量太少**，arithmetic intensity 下降。

**关键公式**：
- Dense: `util = min(B/F × b, 1)`
- MoE: `util = min(topk/#expert × B/F × b, 1)`

以 Mixtral 8x22B 为例：8 个 expert，top-2 选路，batch=156。

```
每个 token 独立经过 Gate 网络，各自选出 2 个 expert
（Token 0 → expert 2,5；Token 1 → expert 0,7；Token 2 → expert 2,3……）

总 "token-expert 配对" 数：156 × 2 = 312
均匀分散到 8 个 expert：312 / 8 = 39 个 token/expert
```

每个 expert 的 weight 大小一样，都要从显存读进来，但只服务 39 个 token → arithmetic intensity ≈ 39，远低于平衡点 156 → **memory-bound**。不是因为不做升维降维，升维降维都做了，只是 batch 太小，同一份 weight 只被复用了 39 次而不是 156 次。

---

## 系统架构（Figure 3）

```
┌─────────────────────┐        M2N / N2M        ┌──────────────────────┐
│   Attention Nodes    │  ◄──── IB/Eth ────►   │    Expert Nodes      │
│   (M 个，复制)        │                         │   (N 个，各存1 expert) │
│                      │                         │                      │
│  Attn Params + KV$   │                         │  Expert i's Params   │
│  ┌──┬──┬──┬──┐      │                         │  ┌──┬──┬──┬──┐      │
│  │G0│G1│G2│G3│ TP   │                         │  │G0│G1│G2│G3│ TP   │
│  └──┴──┴──┴──┘      │                         │  └──┴──┴──┴──┘      │
└─────────────────────┘                         └──────────────────────┘
         │                                                │
         └──────── Ping-Pong Pipeline Parallel ──────────┘
```

### 为什么要复制 Attention？不是重复数据？

4 个 Attention 副本各自服务**不同的请求**，不是重复数据：

```
Attention 副本 0：处理请求 0-127（各自有不同的 prompt、不同的 KV Cache）
Attention 副本 1：处理请求 128-255
Attention 副本 2：处理请求 256-383
Attention 副本 3：处理请求 384-511
```

"复制"指的是 **Attention 的模型参数**（QKV weight、output projection weight）在 4 个副本上是一样的——这部分确实冗余。但 Attention 参数量很小，真正占显存的是 KV Cache。拆成 4 份后每个副本只存 1/4 请求的 KV Cache，显存压力反而更小。

Expert 参数量很大（8 个 expert 的 FFN weight），但不需要存 KV Cache，也不需要复制——所有 512 个请求的 token 根据 Gate 选路汇聚过来就行。

**本质是利用了 Attention 和 FFN 的不对称性**：Attention 参数小但状态（KV Cache）大且不可共享，FFN 参数大但无状态且可共享。拆开后各自按照自己的特点部署，比绑在一起更高效。

对比传统 MoE（Figure 2）：
- 传统 Expert Parallelism：每 GPU 存部分 expert，需要两次 **All2All**（发 token + 收结果）
- MegaScale-Infer：Attention 和 Expert 完全分离，All2All → **M2N 点对点通信**

---

## Ping-Pong Pipeline Parallelism（§4.1, Figure 4）

### "遮盖"（overlap / hide latency）概念

"遮盖"不是某个特定算法，而是一个基础的系统设计思想：**当一个操作在等待时，让另一个操作同时跑，这样等待时间就"看不见"了。**

类比：煮饭 20 分钟 + 炒菜 10 分钟。串行 = 30 分钟；按下电饭煲后趁煮饭的时候炒菜 = 20 分钟。炒菜的 10 分钟被"遮盖"在煮饭的 20 分钟里了。

这个思想在计算机系统里无处不在：CPU 流水线用后续指令遮盖前序指令的访存延迟，GPU 用大量线程遮盖显存访问延迟，FlashAttention-4 用 softmax 和 MMA 在两个 tile 之间交替执行遮盖串行依赖。

### 为什么拆开后需要 ping-pong

拆开之后最直接的问题：Attention 算完了要等 Expert 算完才能进下一层，Expert 算完了要等 Attention 算完才有新数据。如果不做优化：

```
Attention:  [算512个token的Attn]  [闲着等Expert]           [算下一层]
通信A→E:                          [发给Expert]
Expert:     [闲着等Attention]      [算512个token的FFN]      [闲着]
通信E→A:                                                   [发回来]
```

大量时间浪费在等待和通信上。Ping-pong 把 512 个 token 切成 4 个 micro-batch（各 128 个），交错执行：

```
Attention:  [mb1] [mb2] [mb3] [mb4]  ← 不停在算，没有空闲
A→E通信:       [mb1] [mb2] [mb3] [mb4]
Expert:           [mb1] [mb2] [mb3] [mb4]  ← 也不停在算
E→A通信:             [mb1] [mb2] [mb3] [mb4]
```

Attention 算完 mb1 就立刻发出去，不等 Expert，马上算 mb2。等 mb2 算完时，mb1 已经到了 Expert 开始算了。通信发生在 Attention 算 mb2 的同时 → **通信时间被计算遮盖了**。

多层 MoE 的完整时间线（m=4, L=2）：

```
Time →
Attention: [1₁][2₁][3₁][4₁] [1₂][2₂][3₂][4₂]
  A2E:        [1₁][2₁][3₁][4₁] [1₂][2₂][3₂][4₂]
Expert:          [1₁][2₁][3₁][4₁] [1₂][2₂][3₂][4₂]
  E2A:              [1₁][2₁][3₁][4₁] [1₂][2₂][3₂][4₂]
         （下标 = 第几层 MoE layer）
```

### 三个约束条件

| 约束 | 公式 | 含义 |
|------|------|------|
| ① 计算平衡 | T_a ≈ T_e | Attention 和 Expert 计算时间要接近，否则快的等慢的 |
| ② 通信可遮盖 | T_c < T_f | 通信时间 < 计算时间（T_f = max{T_a, T_e}）|
| ③ 流水线够深 | m × T_f ≥ 2×(T_f + T_c) | micro-batch 数量要足以填满流水线 |

→ 当 T_c < ½T_f 时，最少 m=3；否则 m=4。

### 延迟公式

```
单 micro-batch 经过 L 层：T_iter ≈ (T_a + T_e + 2T_c) + m·T_f·(L-1)
整个 global batch：T_total = (T_a + T_e + 2T_c) + T_f·(mL-1)
```

---

## Deployment Plan Search（§4.2, Algorithm 1）

### 搜索空间

- tp_a, tp_e：Attention / Expert 的 Tensor Parallelism 度
- n_a：Attention 副本数（Data Parallelism 度）
- m：micro-batch 数
- B：global batch size

### 搜索逻辑

1. 枚举 (tp_a, tp_e)，检查显存限制
2. 对每对，计算 n_a 使 T_a ≈ T_e → n_a = (k₁E)/(k₃K)
3. 枚举 m ∈ {3, 4, ...}
4. 用 SIMULATE 二分搜索满足 SLO 的最大 B
5. 选 throughput per unit cost 最高的方案

### 四个 GEMM（Table 2）

| GEMM | Input Shape | Param Shape | 属于 |
|------|-------------|-------------|------|
| QKV Project | (b_a, h) | (h, h(1+2/g)/tp_a) | Attention |
| Attn Output | (b_a, h/tp_a) | (h/tp_a, h) | Attention |
| FFN Input | (b_e, h) | (h, h'/tp_e) | Expert |
| FFN Output | (b_e, h'/tp_e) | (h'/tp_e, h) | Expert |

### 通信时间估计（公式 6）

```
T_c = max{ b_a·h·K / (tp_a · W_a · Util(b_a·h·K/tp_a)),
            b_e·h / (tp_e · W_e · Util(b_e·h/tp_e)) }
```

取发送和接收中较慢的那个。

---

## Heterogeneous Deployment（§4.3, Table 3）

| GPU | 价格(归一化) | GB/$ | TFLOPS/$ | 适合 |
|-----|-----------|------|----------|------|
| H20 | 1.85 | **51.9** | 80.0 | **Attention**（高带宽高容量） |
| L40S | 1.08 | 44.4 | **335.2** | **Expert/FFN**（高算力低价） |
| H800 | 5.28 | 15.2 | 187.3 | 传统方案（两项都不是最优） |

核心洞察：Attention 是 memory-intensive → 需要高 GB/$（H20 赢）；FFN disaggregate 后变成 compute-intensive → 需要高 TFLOPS/$（L40S 赢）。传统方案用一种 GPU 做两种负载，必有一端浪费。

---

## High-Performance M2N 通信库（§5, Figures 5-7）

### 为什么不用 NCCL？

传统 MoE 用 **All2All** 通信（所有 GPU 互相发数据），NCCL 就是干这个的。但 MegaScale-Infer 的通信模式变了：M 个 Attention 节点各自往 N 个 Expert 节点发不同量的数据（取决于 Gate 选路结果），这是 **M2N 点对点通信**，不是规整的集合通信。NCCL 用在这里是杀鸡用牛刀。

### NCCL 的三个问题（Figure 5）

单 sender 向 N 个 receiver 各发 128KB：
- Median latency：NCCL 远高于 perftest baseline
- P99 latency：NCCL 在 N=32 时暴增（GPU 同步 + group 操作不稳定）

三个根因：
1. **GPU→CPU 中间拷贝**：NCCL 要先把数据从 GPU 显存拷到 CPU 内存再发。对几百 KB 的小消息，拷贝开销比传输本身还大
2. **group 操作批处理**：peer-to-peer 操作以 8 个为一批处理，Expert 数量多时效率下降
3. **初始化和同步开销**：每次通信前要做 group 初始化 + GPU 同步，高频小消息场景下成为主要开销

### 自研 M2N 库设计（Figure 6 Sender / Figure 7 Receiver）

**核心思路：让 CPU 做通信的控制和搬运，GPU 只管计算。**

具体流程（以 Attention Node 0 发 96KB hidden state 给 Expert Node 2 为例）：

```
Sender:
  ① Wait on CUDA event（前序 Attention kernel 完成）
  ② Tensor 在 GPU registered memory 就位
  ③ Block GPU stream（GPU 暂停后续 kernel）
  ④ CPU Core Sender 用 RDMA write immediate 直接发送
    （不经过 CPU 内存，RDMA 硬件直接 GPU 显存→远端 GPU 显存）
  ⑤ Poll completion queue 确认对面收到
  ⑥ Unblock stream（GPU 继续执行后续 kernel）

Receiver:
  ① Wait on CUDA event
  ② 确认 buffer 可用
  ③ Block stream
  ④ Poll completion queue（确认数据到达）
  ⑤ GDRCopy flush（确保 GPU 能看到最新数据）
  ⑥ Unblock stream → Expert kernel 开始算
```

关键优化：**没有 GPU→CPU 拷贝**（RDMA 直接搬），**没有 group 初始化**（点对点直接发），**没有 GPU 同步**（CPU 用 event 和 flag 协调）。

### 流量优化
- **High-priority ACKs**：ACK 包放高优先级队列，避免被数据包阻塞导致双向通信延迟抖动
- **Congestion control fine-tuning**：针对不均衡通信调整拥塞控制

### 与 DeepEP 的对比
- DeepEP（DeepSeek）：GPU-to-GPU 直接通信，用 custom PTX 减少 L2 cache 冲突，GPU SM 资源消耗大
- MegaScale-Infer：CPU-to-CPU 通信（RDMA），不占 GPU SM，但单连接吞吐低于 GPU 方案
- 在 MoE serving 场景（每对 sender-receiver 数据量 ~数百 KB），CPU 单线程即可饱和带宽

### Ping-pong 和 M2N 的关系

**Ping-pong 定义了"通信应该在什么时候发生、要多快"，M2N 库保证"通信确实能这么快且稳定"。** 一个是架构设计，一个是工程实现。特别是 P99 尾延迟的改善很关键——ping-pong pipeline 要求每个 micro-batch 的通信时间稳定可预测，NCCL 偶尔的延迟尖峰会打破流水线节奏，M2N 库的稳定性保证了 ping-pong 能持续高效运转。

---

## Implementation（§6）

### Fused Kernels
1. TP 通信 + GEMM 融合（用 Flux 库：all-gather + GEMM → 一个 kernel）
2. 多个 memory-intensive 操作融合（gate 选路 + 中间计算 + token scatter → 一个 kernel）

### Load Balance
- **Expert 侧**：根据历史流量统计热度，热门 expert 冗余部署（贪心算法最小化 max 负载）
- **Attention 侧**：根据序列长度估算计算时间，组 batch 时平衡各节点工作量
- 生产观察（Figure 16）：decoding 阶段 expert 负载较稳定（可静态均衡），prefill 阶段波动大（需频繁调整）

---

## 实验结果

### 测试模型（Table 4）

| Model | Layers | Hidden | #Experts | top-k | Params |
|-------|--------|--------|----------|-------|--------|
| Mixtral 8x22B | 56 | 6144 | 8 | 2 | 141B |
| DBRX | 40 | 6144 | 16 | 4 | 132B |
| Scaled-MoE | 48 | 8192 | 32 | 4 | 317B |

### 同构集群 A800（Figure 8）

| 指标 | vs vLLM | vs TensorRT-LLM |
|------|---------|-----------------|
| 解码吞吐（Scaled-MoE）| **7.11×** | **1.90×** |
| 解码吞吐（Mixtral）| 2.56× | 1.28× |
| TBT 延迟 | 相当 | 相当 |
| 端到端含 prefill | ~1.18× | ~1.18× |

### 异构集群 H20+L40S（Figure 9）

| 指标 | vs vLLM(L40S) | vs TRT-LLM(H20) |
|------|---------------|-----------------|
| 每美元解码吞吐 | **3.24×** | **1.86×** |
| 每瓦吞吐（Figure 10）| 1.80× | — |

### M2N 通信微基准（Figures 11-12）

| 指标 | vs NCCL（256KB 典型场景）|
|------|------------------------|
| Median latency | **-68.2%** |
| P99 latency | **-92.9%** |
| Throughput | **4.2×** |
| 小数据 throughput（peak）| 9.9× |

### Ablation（Figures 13-15）

- **Disaggregation 单独贡献**：+4.66×（即使用 NCCL）
- **M2N 通信库额外贡献**：+1.53×
- **micro-batch m=1→2**：+1.9×（开始 ping-pong）
- **m=2→3**：+1.10-1.38×（通信被遮盖）
- **Attention DP 最佳值**：DP=8 时 T_a ≈ T_e，吞吐达峰

---

## 生产部署经验（§8）

- 近万张 GPU 集群，异构部署降本 1.5-2.0×
- Expert 负载不均匀：存在明显热门 expert（Figure 16a）
- Decoding 阶段负载跨 batch 稳定，可用静态均衡
- Prefill 阶段波动大，需要更频繁调整
- Attention 侧也有负载不均（序列长度差异），用预估计算时间组 batch

---

## Disaggregation 演进脉络

```
DistServe (OSDI 2024)
  → Prefill / Decode 解耦（请求级别）
  → 解决：P 是 compute-bound，D 是 memory-bound，混在一起互相干扰
  → 笔记：[[distserve]]

Mooncake (FAST 2025)
  → KVCache-centric 三资源池解耦
  → 解决：长上下文场景下 KV Cache 成为独立瓶颈资源
  → 笔记：[[mooncake]]

MegaScale-Infer (2025)
  → Attention / FFN 层内解耦（专为 MoE）
  → 解决：MoE 稀疏性让 FFN 退化为 memory-bound
  → 本篇

共同思路：识别不同计算模块的资源特征差异 → 拆开 → 独立扩展 + 独立选硬件
```

---

## 我的理解（学习笔记）

**完整因果链**：MoE 稀疏（top-k 选路）→ 每 expert 的有效 batch 小（156×2/8=39）→ arithmetic intensity 低于 roofline 平衡点 → FFN 退化为 memory-bound → 拆开 Attention 成多副本 → 多副本的 token 汇聚给 Expert → Expert 有效 batch 倍增 → FFN 回到 compute-bound → 但拆开引入空闲和通信开销 → ping-pong 遮盖空闲 → M2N 库保证通信够快且稳定。

**为什么不直接加大 batch 而要复制 Attention**：因为 KV Cache 占显存。512 个请求的 KV Cache 可能撑爆 80GB 显存，所以单 GPU 上 batch size 有上限。拆成 4 个 Attention 副本后每个只存 128 个请求的 KV Cache，显存压力降到 1/4。而 Expert 节点不存 KV Cache，只存 FFN weight，可以接收所有副本的 token 汇聚。**Attention 的瓶颈是显存容量（KV Cache），Expert 的瓶颈是 batch 太小，拆开后两边的瓶颈都解了。**

**和 DistServe 的本质区别**：
- DistServe 是**请求级别**的拆分（一个请求先在 Prefill 节点算完，整体移到 Decode 节点），拆分点在两个阶段之间
- MegaScale-Infer 是**层内**的拆分（每一层的 Attention 和 FFN 在不同节点上算），拆分点在每一层内部
- DistServe 的通信是请求维度的 KV Cache 传输，MegaScale-Infer 的通信是每层 token 级别的 M2N

**Ping-pong 的本质**：不是减少总工作量，而是通过**流水线重叠**把空闲时间填满——当一个操作在等待时，让另一个操作同时跑，等待时间就"看不见"了。和 FlashAttention-4 的 ping-pong（MMA 和 softmax 交替执行两个 tile）思路一样，但遮盖的对象不同：FA4 遮盖的是同一芯片内两个功能单元之间的串行等待，MegaScale-Infer 遮盖的是跨节点的网络通信延迟。

**M2N 通信库的工程价值**：NCCL 为 all-reduce / all-to-all 集合通信设计，用在 M2N 点对点场景是杀鸡用牛刀。自研库去掉所有不必要的抽象层（group init、GPU sync、GPU-CPU copy），在百 KB 级消息上优势巨大。P99 尾延迟的改善尤其关键——ping-pong 要求通信时间稳定可预测，M2N 库的稳定性保证了流水线节奏不被打破。

**异构部署是最大的商业价值**：技术上最有趣的是 ping-pong pipeline，但对字节跳动来说，真正省钱的是异构部署——H20 做 Attention（高带宽 → memory-bound 友好）+ L40S 做 Expert（高算力/$ → compute-bound 友好），每美元吞吐 3.24× 提升。传统方案用一种 GPU 做两种负载，必有一端浪费。

---

## 关联笔记

- [[distserve]] — P/D 解耦的开创性工作，MegaScale-Infer 在此基础上进一步做 Attention/FFN 解耦
- [[mooncake]] — KVCache-centric 解耦，另一种 disaggregation 方向
- [[vllm-pagedattention]] — MegaScale-Infer 的 Attention 节点使用 PagedAttention 管理 KV Cache
- [[sglang]] — 另一种推理优化方向（前端语言+后端运行时 co-design）
- [[flash-attention]] — MegaScale-Infer 的 Attention 计算使用 FlashAttention
- [[AI-Infra-Cheatsheet]] — §2.3 MoE 部分的 GPU 利用率公式可与本文对照

---

*学习方式：原论文精读（17 页 + 附录），全部 Figure 和 Table + 对话深入讨论*
*最后更新：2026-03-11*
