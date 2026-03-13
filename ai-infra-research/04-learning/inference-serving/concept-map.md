---
tags: [learning]
subfield: inference-serving
created: 2026-03-05
---

# Inference Serving 概念图谱

> 目标：理解"如何高效地把 LLM 部署给大量用户同时使用"

---

## 核心问题是什么？

LLM 推理有两个独特的性质，让它比传统 Web 服务难得多：

1. **输出是逐 token 生成的（autoregressive）**：不像图片分类一次性出结果，LLM 每次只生成一个 token，然后把这个 token 加入输入再生成下一个。这意味着一个请求要占用 GPU 很长时间。

2. **KV Cache 占显存且大小不确定**：每个生成步骤都需要缓存之前所有 token 的 Key-Value 向量（KV Cache），输出越长，占的显存越多，而且事先不知道用户会生成多长。

这两个性质导致了推理系统的核心挑战：**如何让 GPU 一直在干活（高利用率），同时满足用户的延迟需求（低 TTFT/TPOT）**。

---

## 核心概念

### 第一层：基础指标（必须先记住这几个词）

**Throughput vs Latency**
- Throughput（吞吐量）：单位时间内处理多少请求 / 生成多少 token，代表系统效率
- Latency（延迟）：用户等多久才能看到回复，代表用户体验
- 两者天然有张力：批量处理可以提高吞吐，但会增加单个请求的等待时间

**TTFT（Time to First Token）**：用户发送请求后多久看到第一个字。对于聊天场景非常敏感。

**TPOT（Time Per Output Token）**：生成每个后续 token 平均花多少时间，决定输出"流"的速度。

**Prefill vs Decode**：
- Prefill（预填充）：处理用户的输入 prompt，一次性并行计算，是 compute-bound
- Decode（解码）：逐 token 自回归生成，是 memory bandwidth-bound（每次只算一个 token，GPU 大量算力闲置）

### 第二层：核心技术

**Batching（批处理）**：把多个用户的请求合并成一批一起处理，提高 GPU 利用率。
- 传统 Static Batching：等一批请求凑齐再处理，延迟高
- **Continuous Batching**（Orca 论文提出）：请求一进来就加入正在处理的 batch，一个请求完成就立刻腾出位置，GPU 几乎不空转。这是现代推理系统的基础。

**KV Cache 管理**
- KV Cache 是每个请求的 attention key/value 向量的缓存，让 decode 阶段不用重新计算已经处理过的 token
- 问题：显存有限，KV Cache 会碎片化（就像内存碎片），导致实际可用显存比看起来少很多
- **PagedAttention**（vLLM 论文提出）：把 KV Cache 切成固定大小的"页"，像操作系统的虚拟内存一样管理，消除碎片。这是 vLLM 的核心贡献。

**FlashAttention**：重新实现 attention 计算，通过 IO 感知的方式（减少 HBM 读写次数）大幅提速。不是推理系统层面的优化，而是 kernel 层面的优化。你作为半导体分析师，可以理解为"更充分利用 GPU 的 SRAM 而不是反复访问 HBM"。

**Prefill-Decode Disaggregation（P/D 解耦）**：
- 发现 Prefill（compute-bound）和 Decode（memory-bound）的需求完全不同
- 解决方案：用不同的机器/实例专门跑 Prefill，另一批机器跑 Decode
- 代表论文：DistServe、Mooncake（DeepSeek 开源）

**Speculative Decoding（推测解码）**：
- Decode 太慢的根本原因是每次只生成 1 个 token，GPU 大量闲置
- 解决方法：用小模型先"猜"好几个 token，大模型一次性验证，正确的就接受。平均下来相当于每次生成多个 token。
- 关键洞察：验证比生成便宜（可以并行）

**Structured Generation / Constrained Decoding**：
- 让 LLM 只生成符合特定格式（JSON、正则表达式）的输出
- SGLang 的核心特色，对实际 LLM 应用非常重要

### 第三层：系统级优化

**Chunked Prefill**：把长 prompt 的 prefill 分成小块，和 decode 任务混在一起处理，避免 prefill 霸占 GPU 导致 decode 延迟激增。

**Prefix Caching / Prompt Caching**：如果多个请求有相同的 system prompt 前缀，可以共享这部分的 KV Cache，避免重复计算。

**Tensor Parallelism 在推理中的应用**：把一个模型的权重切分到多张 GPU 上，让单个请求的延迟降低（代价是要用 NVLink/InfiniBand 通信）。

---

## 概念关系图

```
用户请求
   ↓
[请求调度层]
├─ Continuous Batching ← Orca (2022)
├─ Chunked Prefill
└─ Prefill/Decode 分离 ← DistServe, Mooncake

   ↓
[显存管理层]
└─ PagedAttention / KV Cache 管理 ← vLLM (2023)
   └─ Prefix Caching

   ↓
[Kernel 执行层]
├─ FlashAttention 1/2/3
├─ Speculative Decoding
└─ Structured Generation ← SGLang

   ↓
[硬件层 - 你已有基础]
GPU (HBM bandwidth, SRAM, compute)
```

---

## 关键问题（读论文时带着这些问题）

1. 这个系统的主要 bottleneck 在哪？是 compute、memory bandwidth 还是 memory capacity？
2. 为了提高 throughput，牺牲了什么 latency？这个 trade-off 合理吗？
3. 这个方案在什么 workload 下最有效？什么情况下会失效？
4. 和现有方案相比，核心创新点是什么？这个想法是否可以进一步推广？