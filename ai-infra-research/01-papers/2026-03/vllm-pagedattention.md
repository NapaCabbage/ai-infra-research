---
title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
aliases: [vLLM, PagedAttention]
tags: [paper]
subfield: inference-serving
venue: SOSP 2023
date: 2023-10-23
authors: Woosuk Kwon, Zhuohan Li, Sicheng Lin, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Danyang Zhuo, Clark Barrett, Ion Stoica
institution: UC Berkeley
url: https://arxiv.org/abs/2309.06180
status: 已读
rating: ⭐⭐⭐
created: 2026-03-05
---

# Efficient Memory Management for Large Language Model Serving with PagedAttention

**来源**: SOSP 2023 | **日期**: 2023-10-23 | **机构**: UC Berkeley
**链接**: https://arxiv.org/abs/2309.06180

---

## 一句话总结

把操作系统虚拟内存的分页（paging）思想应用到 LLM 推理的 KV Cache 管理中，通过非连续的 block 分配消除显存碎片化，将显存利用率从 ~50% 提升到 ~95%，吞吐量提升 2-4×。

## 动机（Why）

- LLM 推理中，每个请求需要维护 KV Cache，它随着生成 token 动态增长，且最终长度事先未知
- 传统做法是为每个请求预分配一块**连续的**显存空间（按最大可能长度），导致三种浪费：
  - **预留浪费（reservation waste）**：预分配了最大长度，但实际生成可能很短
  - **内部碎片（internal fragmentation）**：分配的连续空间里有很多用不到的区域
  - **外部碎片（external fragmentation）**：多个请求分配释放后，空闲显存被切成不连续的小块，虽然总量够但无法分配给新请求
- 实测中，现有系统的 KV Cache 显存利用率只有 20.4%~38.2%，大量显存被浪费

## 方法（How）

**核心思想：借用操作系统的虚拟内存分页机制。**

1. **Block 化存储**：将 KV Cache 切分为固定大小的 block（默认 16 tokens/block），每个 block 可以存放在显存中的任意位置，不需要连续
2. **Block Table（页表）**：维护一个映射表，记录每个请求的每个逻辑 block 对应的物理 block 位置（就像 OS 的页表把虚拟地址映射到物理地址）
3. **按需分配**：不再预分配最大长度，而是每生成一个 block 的 token 才分配新的物理 block
4. **Copy-on-Write**：当多个请求共享前缀（如相同的 system prompt）时，共享 KV Cache block，只在需要修改时才复制（prefix caching 的前身）

**Block size 的 trade-off**（16 tokens/block）：
- Block 太大 → 内部碎片增加（最后一个 block 可能半空）
- Block 太小 → block table 开销增大，非连续内存访问增多（GPU 偏好连续读取 coalesced access）
- 16 是经验甜点

**与 OS 虚拟内存的类比**：

| 操作系统 | vLLM |
|---------|------|
| 虚拟页（virtual page） | 逻辑 KV block |
| 物理页帧（physical frame） | GPU 显存中的物理 block |
| 页表（page table） | Block table |
| 按需调页（demand paging） | 生成时按需分配 block |
| Copy-on-Write | 共享前缀的 KV Cache |

## 图表解读（论文补充，博客未涉及）

- **Figure 2**（显存浪费量化）：Orca 的三种显存分配策略（Max/Pow2/Oracle）利用率仅 20.4%~38.2%，vLLM 达到 **96.3%**。这个数字是核心卖点。
- **Figure 8-9**（Copy-on-Write 机制）：多个请求共享同一 prompt 时，共享 KV Cache 的物理 block（引用计数 = 2），当某个请求需要写入不同 token 时才触发拷贝。Beam search（width=4）场景下内存节省高达 **37.6%~55.2%**。
- **Figure 13**（batch size 对比）：vLLM 平均可以同时处理 30.42 个请求（ShareGPT），Orca 仅 7.00 个。直接反映了显存利用率提升带来的 batch 能力飞跃。
- **Figure 18a**（核心代价）：PagedAttention kernel 相比 FasterTransformer 的注意力 kernel 延迟高 **20~26%**，因为需要额外查 block table 做间接寻址。但 Attention 不是整个 forward pass 的大头，端到端影响有限。
- **Figure 18b**（block size 选择）：block size 太小（GPU 并行度不足）或太大（碎片增加）都劣化。**block size = 16** 综合最优。
- **Figure 19**（Preemption 策略）：当显存不够时，两种驱逐方式——recomputation（丢弃 KV Cache 以后重算）vs swapping（换到 CPU 内存）。小 block 时 recompute 更优（PCIe 传输小数据效率低），大 block 时 swap 更优。

## 实验结果

- **显存利用率**：从 20.4%~38.2%（Orca）提升到 **96.3%**（vLLM）
- **吞吐量**：比 Orca 和 FasterTransformer 提升 **2~4×**；在 parallel sampling / beam search 下优势更大（因为 Copy-on-Write 共享）
- **Prefix sharing**：5-shot prompt 共享场景下吞吐提升 **3.58×**
- **代价**：PagedAttention kernel 本身延迟 +20~26%（仅影响 Attention 算子，整体可接受）

## 局限与待改进

- PagedAttention kernel 延迟 +20~26%，在极端延迟敏感场景下需要关注（后续 FlashAttention 与 PagedAttention 的融合版本缓解了这个问题）
- Block 粒度的内部碎片仍然存在（最后一个 block 的浪费），只是比整段连续分配小得多
- 论文主要关注单机场景，没有深入讨论多机推理的 KV Cache 管理（后续 Mooncake 等工作解决）
- Preemption 策略（recompute vs swap）较简单，没有做到 SLO-aware

## 关于 GPU vs CPU 负担

PagedAttention 主要增加 **GPU** 端的负担，而非 CPU：
- **GPU 端代价**：每次 Attention 计算都要先查 block table 做间接寻址，且非连续内存访问降低了 coalesced access 效率。这就是 20~26% kernel 延迟增加的来源。
- **CPU 端开销极小**：block table 的维护（分配/释放/映射）属于控制平面逻辑，计算量极小，每个 iteration 只需广播一次控制消息给 GPU workers。
- **为什么值得**：GPU kernel 慢了 20%，但 batch size 可以扩大 2~4×（因为显存利用率从 30% 到 96%），整体吞吐量净增 2~4×。

## 我的理解

这个 idea 本质上很简单——用了操作系统几十年前就有的分页思想。但它之所以是好工作，是因为精准地识别了 LLM 推理的核心瓶颈（KV Cache 显存碎片化），然后用一个成熟的、被验证过的抽象来解决它。

论文比博客多了两个重要维度：
1. **代价是量化的**：不是"免费午餐"，PagedAttention kernel 有 20~26% 的延迟代价，但被 batch size 扩大抵消了
2. **Copy-on-Write 不是噱头**：在 beam search 下内存节省 37~55%，prefix sharing 场景吞吐提升 3.58×，这些在实际部署中非常实用

从半导体视角看：这个工作说明推理系统的瓶颈往往不是 GPU 算力不够，而是显存管理不够聪明。但优化显存管理本身也不是免费的——间接寻址会损失 GPU 内存访问的 coalesced 效率，这是一个经典的"灵活性 vs 局部性"的 trade-off。

## 关联笔记

- 基础论文：[[orca-continuous-batching|Orca]]（Continuous Batching，vLLM 在其调度层基础上做显存管理优化）
- 相关概念：[[KV Cache]]、[[PagedAttention]]、[[Continuous Batching]]、[[Copy-on-Write]]
- 后续工作：[[DistServe]]（P/D 解耦）、[[Mooncake]]（跨机器 KV Cache 管理）
- 下一篇：[[FlashAttention]]（kernel 层面的 Attention 优化，和 PagedAttention 是不同层次的优化）

---
*笔记日期: 2026-03-05（博客初版） → 2026-03-05（论文补充更新）*