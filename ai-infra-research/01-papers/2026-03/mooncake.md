---
title: "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"
tags: [inference-serving, disaggregation, kv-cache, distributed-storage, prefix-caching]
subfield: inference-serving
venue: "FAST 2025 (Best Paper)"
date: 2026-03-09
authors: [Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu]
institution: [Moonshot AI, Tsinghua University]
url: "https://arxiv.org/abs/2407.00079"
status: 已读
rating: ⭐⭐⭐⭐⭐
---

# Mooncake：KVCache-centric 的 P/D 解耦架构

## 一句话总结

与其把 KVCache 当作 GPU 内存里的"副产品"，不如把它当作头等公民——整个集群的调度、存储、网络拓扑，都围绕 KVCache 的生命周期来设计。用廉价的 CPU DRAM 换昂贵的 GPU 计算，是 Mooncake 的核心哲学。

---

## 核心洞察

**KVCache 的调度是 LLM serving 调度的本质。**

Kimi 真实 workload：平均输入长度 7590 tokens，输入/输出比约 720:1，是典型的超长上下文场景。P/D 混跑的干扰问题在此场景下被放大数倍。

---

## 架构：三个物理分离的资源池

```
用户请求
    ↓
[Conductor 全局调度器]
    ↓              ↓               ↓
[Prefill Pool]  [KVCache Pool]  [Decoding Pool]
GPU 计算        CPU/DRAM/SSD     GPU 计算
(compute-bound)  (存储/传输)    (memory-bound)
```

- **Prefill Pool**：GPU 做 compute-intensive 的 prefill 计算，CPU/DRAM 承载本地 KVCache 缓存
- **KVCache Pool**：把集群所有节点的 CPU DRAM 和 SSD 组成全局共享的分布式存储，通过 RDMA 互联
- **Decoding Pool**：GPU VRAM 存放当前 batch 的 KVCache，做 memory-bound 的 decoding
- **Conductor**：全局调度器，管理三个池子的协同

每个阶段优化目标不同：
- Prefill：max Cache Reuse，约束 TTFT SLO 和最低 MFU
- Decoding：max Throughput，约束 TBT SLO 和 VRAM 上限

---

## 关键设计 1：链式哈希的前缀匹配

### 问题

KVCache 有严格的因果依赖：`token_i 的 K、V = f(token_0, token_1, ..., token_i)`。只有前缀完全相同，KVCache 才能安全复用。

### 链式哈希结构

每个 block（512 tokens）的 hash 包含前缀的累积信息：

```
Hash_A = H(block_1 的 tokens)
Hash_B = H(Hash_A + block_2 的 tokens)   ← 包含了 A 的信息
Hash_C = H(Hash_B + block_3 的 tokens)   ← 包含了 A+B 的信息
```

如果用独立 hash，来自不同 session、不同 system prompt 的相同内容块会被误认为可以复用。链式 hash 保证：两个 block 的 hash 相同 ⟺ 自身内容相同 AND 之前所有内容也完全相同。

### 前缀匹配流程

```
新请求 tokens 分成 blocks：
Block 1: Hash_A → Match ✅（命中，复用 KVCache）
Block 2: Hash_B → Match ✅
Block 3: Hash_C → Match ✅
Block 4: Hash_D → Mismatch ❌（从这里开始重新 prefill）
```

命中的 blocks 通过 RDMA 传到 Prefill Node 的 GPU，未命中的才需要计算。

---

## 关键设计 2：Layer-wise 流式传输与计算 Overlap

一次请求的四个步骤：

1. **KVCache Reuse**：命中的 prefix blocks 从 CPU DRAM 经 RDMA 传到 Prefill GPU
2. **Incremental Prefill**：只对未命中的 tokens 做 prefill 计算
3. **KVCache Transfer**：每计算完一层，立刻把该层 KVCache 异步传给 Decoding Node（与计算 overlap）
4. **Decoding**：Decoding Node 收到完整 KVCache 后加入 continuous batching

两个关键 overlap：
- Layer-wise prefill：边计算边写回 CPU DRAM，VRAM 占用始终只有"当前层"
- 异步 Load：Decoding Node 在已有请求做 decoding 的同时，并发加载新请求的 KVCache

实验表明 Layer-wise Prefill 与标准 Prefill 延迟几乎完全重合，写回操作被计算完全 hide 掉。

---

## 关键设计 3：KVCache 分层存储

```
GPU VRAM  → 热数据，当前正在用的 KVCache（最快，最贵，最小）
CPU DRAM  → 温数据，最近用过的 KVCache（主要存储层）
SSD       → 冷数据，历史 KVCache（最慢，最便宜，最大）
```

Conductor 根据访问频率动态管理 blocks 在三层之间的迁移。

缓存命中分布极度不均匀：超过 50% 的 blocks 从未被第二次命中，但少数 blocks（system prompt）被命中 10000+ 次。这驱动了热块复制（多节点副本）和冷块淘汰策略。LRU 在所有场景下最优，因为 Kimi workload 时间局部性很强（多轮对话短期内复用相同前缀）。

---

## 关键设计 4：预测性 Early Rejection

### 问题

系统过载时需要拒绝部分请求，但朴素 early rejection 导致负载振荡：

```
负载高 → 大量拒绝 → 负载骤降 → 接受大量请求 → 负载骤升 → 循环
```

本质是 bang-bang 控制器，依赖瞬时负载，没有考虑 in-flight 请求的剩余生命周期。

### 解法

预测每个请求的输出长度，估算 decoding queue 的未来状态，做前瞻性 accept/reject 决策。实验中负载曲线平稳，接近系统容量上限。

---

## 关键设计 5：Chunked Pipeline Parallelism（CPP）

超长 context（128k tokens）跨多节点 prefill 的并行策略选择：

- **TP**：每层都要 all-reduce，跨节点通信代价极高
- **SP**：每层都要 Ring/Striped Attention，通信频繁，且与 KVCache RDMA 传输抢网络
- **CPP（Mooncake 的选择）**：把 tokens 分成 chunks，不同 chunk 给不同节点处理，只在 pipeline stage 边界通信一次，可被计算 overlap

CPP 的优势：通信少、不抢 KVCache 传输的网络资源、自然适配长短 context。

---

## 实验结果

- 模拟场景：相比 baseline 吞吐提升最高 **525%**
- 真实负载：同等硬件下可多处理 **75%** 的请求
- TTFT SLO 满足率 ~99%，TBT SLO 满足率 ~100%（baseline 仅 57%）
- TBT 改善特别显著的原因：P/D 分离后 decoding 节点不受 prefill 干扰，token 生成节奏极为稳定

---

## 什么被大量复用？

用户看到的只是自己的消息，但实际发给模型的包含大量隐藏前缀：

```
[系统 prompt ~几百到几千 tokens，用户不可见]
[多轮对话历史]
[用户最新消息]
```

被复用的不只是 system prompt，而是整个 prompt 序列的公共前缀（包括历史对话）。这是 prefix caching 对长对话场景收益特别大的原因。

---

## 与前人工作的对比

```
维度              vLLM    SGLang          DistServe       Mooncake
P/D 分离          ❌      ❌              ✅              ✅
Prefix Cache      ❌      ✅（单节点）     ❌              ✅（全集群跨节点）
Cache 共享范围    —       单机 GPU VRAM   —               全集群 CPU DRAM
多节点 Prefill    ❌      ❌              —               ✅（CPP）
过载处理          假设充足 假设充足        假设充足        ✅（预测性早期拒绝）
```

---

## 我的理解

- Mooncake 的核心哲学是"用廉价存储换昂贵计算"：CPU DRAM 便宜且每台机器都有大量闲置，用来存 KVCache 比在 GPU 上重新计算划算得多
- 链式哈希是个非常优雅的设计：一个简单的递归 hash 就保证了前缀匹配的正确性，避免了误复用
- 和 SGLang RadixAttention 的区别：SGLang 在单机 GPU VRAM 内做前缀复用，Mooncake 把它扩展到了全集群的 CPU DRAM + SSD，规模完全不同
- 和 DistServe 的关系：DistServe 提供了 P/D 解耦的理论框架和 placement 搜索算法，Mooncake 在工业实践中把重心转向了 KVCache 的存储和传输——DistServe 回答"怎么分配 GPU"，Mooncake 回答"KVCache 怎么存、怎么传、怎么复用"
- Layer-wise 流式传输和计算 overlap 是工程实践的典型技巧：利用 GPU 计算远快于网络传输的事实，把传输隐藏在计算背后
- 预测性 early rejection 解决了一个很实际的工程问题：过载场景不能简单用阈值控制，需要前瞻性预测

---

## 关联笔记

- [[distserve]] — DistServe 提供 P/D 解耦的理论框架（goodput + placement 搜索），Mooncake 在此基础上以 KVCache 为中心重新设计了工业级架构
- [[sglang]] — SGLang RadixAttention 在单机做前缀复用，Mooncake 将其扩展到全集群分布式存储
- [[vllm-pagedattention]] — vLLM PagedAttention 解决单请求内 KV Cache 碎片化，Mooncake 进一步解决跨请求、跨节点的 KV Cache 复用和分层存储
- [[flash-attention]] — FlashAttention 优化单次 Attention 的 GPU 计算效率，Mooncake 优化的是 KVCache 在整个集群中的生命周期管理
- [[AI-Infra-Cheatsheet]] — KV Cache 公式可用于估算 Mooncake 中每个 block 的存储开销

---

*学习方式：Claude 对话精读 + 笔记整理*
*最后更新：2026-03-09*
