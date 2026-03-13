---
title: "Orca: A Distributed Serving System for Transformer-Based Generative Models"
aliases: [Orca, Continuous Batching, Iteration-level Scheduling]
tags: [paper]
subfield: inference-serving
venue: OSDI 2022
date: 2022-07-11
authors: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun
institution: Seoul National University / FriendliAI
url: https://www.usenix.org/conference/osdi22/presentation/yu
status: 已读
rating: ⭐⭐⭐
created: 2026-03-05
---

# Orca: A Distributed Serving System for Transformer-Based Generative Models

**来源**: OSDI 2022 | **日期**: 2022-07-11 | **机构**: Seoul National University / FriendliAI
**链接**: https://www.usenix.org/conference/osdi22/presentation/yu

---

## 一句话总结

提出 iteration-level scheduling（即 Continuous Batching）和 selective batching 两个技术，让 LLM 推理系统可以在每个 token 生成后动态调整 batch 组成，消除"等最慢请求"的浪费，在 GPT-3 175B 上实现同延迟下 36.9× 的吞吐提升。

## 摘要翻译

大规模 Transformer 生成模型（如 GPT-3）引起了巨大关注，对 serving 系统提出了新需求。由于这些模型以自回归方式逐 token 生成，处理一个推理请求需要运行模型多次迭代。然而，现有推理系统因其不灵活的调度机制——无法在处理过程中改变当前 batch 的请求组成——而表现不佳。本文提出 iteration-level scheduling，以迭代（而非请求）为粒度调度执行；同时提出 selective batching，仅对部分操作应用 batching。基于这两项技术实现了分布式推理系统 Orca，在 GPT-3 175B 上显著优于 NVIDIA FasterTransformer：同延迟下吞吐量提升 36.9×。

## 动机（Why）

- LLM 推理的核心特征：**多次迭代**。生成一个回复可能需要几百次 iteration，而不像图片分类只跑一次。
- 现有系统（Triton + FasterTransformer）使用 **request-level scheduling**：凑齐一个 batch → 整批跑完 → 才能接新请求。三个严重问题：
  1. **早结束的请求白占资源**：batch 里有的请求 10 token 就结束，有的要 500 token，短请求结束后 GPU 在做无用功（Figure 3）
  2. **已完成请求无法立刻返回**：必须等整个 batch 跑完，用户延迟增加
  3. **新请求排队时间长**：必须等当前 batch 整批结束才能进入
- 注意：这个问题在**训练**中不存在，因为训练用 teacher forcing，一个 iteration 就处理完整个 batch

## 方法（How）

### 核心技术 1：Iteration-level Scheduling

**改变调度粒度：从"整个请求"变成"每一次 iteration"。**

每生成一个 token 后，调度器都介入一次：
1. 检查哪些请求已结束 → 移出 batch，立即返回结果
2. 检查等待队列里有没有新请求 → 加入 batch
3. 组建新 batch，送去 engine 跑下一个 iteration

效果：GPU 几乎永远在做有用功，请求的排队时间大幅缩短。

### 核心技术 2：Selective Batching

**问题**：iteration-level scheduling 下，batch 里的请求处于不同阶段——有的在 prefill（处理多个输入 token），有的在 decode 第 3 步，有的在第 200 步。序列长度不同，无法简单拼成一个规整的 `[B, L, H]` 张量。

**解决**：把 Transformer 的操作分两类区别对待：

| 操作类型 | 代表操作 | 是否关心序列长度 | 处理方式 |
|---------|---------|-------------|--------|
| 可 batch | Linear、LayerNorm、GeLU、Add | 不关心 | 所有请求的 token 拍平成 `[总token数, H]`，一起算 |
| 不可 batch | Attention | 关心（每个请求只能 attend 自己的 KV） | **Split** → 每个请求独立做 Attention → **Merge** → 继续 batch |

**关键 insight**：Attention 不 batch 的性能损失很小——因为 Attention 不涉及模型参数读取（只是 QK^T 和 softmax·V），不 batch 也不会多读权重。而 Linear 层才是读权重的大头，它能 batch，吞吐就上来了。

### 系统架构

- **Request Pool**：管理所有请求的状态和 KV Cache
- **Scheduler**：每个 iteration 介入，执行 FCFS + 显存约束的调度算法（Algorithm 1）
- **Execution Engine**：跨多 GPU/多机的执行引擎
- **控制/数据平面分离**（Figure 7）：控制消息走 CPU（gRPC），tensor 数据走 GPU 直连（NCCL），两者并行，调度开销被隐藏
- **Attention K/V Manager**：为每个请求动态管理 KV Cache 的显存分配

## 图表解读

- **Figure 3**：request-level scheduling 的浪费——x₂ 在 iter 2 就结束了，但 iter 3-4 仍在给它做无意义计算（灰色 "-"），x₂ 的用户也要等 x₁ 跑完才能收到结果
- **Figure 4**：Orca 系统总览——调度器每个 iteration 和 engine 交互一次（虚线箭头），可以动态插入/移出请求
- **Figure 5**：Selective batching 的具体实现——非 Attention 操作用拍平的 `[7, H]` 张量一起算，Attention 前 Split、后 Merge
- **Figure 8**：Pipeline parallelism 对比——Orca 不需要切 microbatch（因为每个 iteration 调度器都介入），FasterTransformer 必须切，导致 batching 效率和 pipeline 效率的 trade-off
- **Figure 10**：端到端性能——Orca 在所有模型规模和负载水平下都显著优于 FasterTransformer

## 实验结果

- **环境**：Azure ND96asr A100 v4 VM（8×A100 40GB，NVLink），最多 4 台 VM
- **模型**：GPT-3 配置，13B / 101B / 175B / 341B
- **引擎性能**（Figure 9）：Selective batching vs 全 batch 的差异很小（验证了"Attention 不 batch 损失小"的 insight）
- **端到端性能**（Figure 10）：
  - 175B 模型，同延迟 190ms/token：Orca 吞吐 6.81 req/s vs FasterTransformer 0.185 req/s = **36.9× 提升**
  - 核心原因：FasterTransformer 在 (1) 请求长度不同 (2) 到达时间不同 (3) 生成长度不同 三种情况下都效率很低，Orca 全部解决
- **同质请求**（Figure 11）：即使所有请求完全相同（FasterTransformer 最优场景），Orca 仍然不输

## 局限与待改进

- **KV Cache 显存管理粗糙**：Orca 为每个请求预分配 max_tokens 大小的 KV Cache 空间，和 FasterTransformer 类似（vLLM 的 PagedAttention 后来专门解决了这个问题）
- **调度算法简单**：FCFS + max batch size 约束，没有优先级、SLO 感知等高级策略
- **未讨论 Prefill/Decode 的资源竞争**：长 prompt 的 prefill 会拖慢 decode（后续 Sarathi-Serve 的 chunked prefill 解决了这个）
- **未开源**：Orca 是 FriendliAI 的商业产品，没有公开代码（vLLM 后来以开源形式实现了类似思路）

## 我的理解

这篇论文的 idea 层面不难——"结束了就移出去，有新的就加进来"。真正的贡献在于把这个直觉变成了一个在 GPU 集群上高效运行的系统：

1. Selective batching 的 CUDA kernel 实现（Split/Merge + fused attention）
2. 控制/数据平面分离，让调度开销不成为瓶颈
3. 在多机 pipeline parallelism 下仍然保持 iteration-level 的调度粒度

这也是 AI Infra 论文的典型特征：**idea 清晰但 contribution 在工程实现**。发在 OSDI（系统顶会）而非 NeurIPS（算法顶会），正是因为核心挑战是系统设计而非算法创新。

这篇论文和 [[vllm-pagedattention|vLLM/PagedAttention]] 是互补关系：
- Orca 解决的是**调度层**的问题：什么时候处理哪些请求
- vLLM 解决的是**显存管理层**的问题：KV Cache 怎么高效存储
- 现代推理系统（vLLM、SGLang）两者都用了

## 关联笔记

- 前置知识：无（这是 inference serving 的奠基论文之一）
- 互补论文：[[vllm-pagedattention]]（显存管理层，解决了 Orca 没解决的 KV Cache 碎片化）
- 后续工作：[[Sarathi-Serve]]（chunked prefill，解决 prefill/decode 资源竞争）
- 相关概念：[[Continuous Batching]]、[[KV Cache]]、[[Selective Batching]]

---
*笔记日期: 2026-03-05*