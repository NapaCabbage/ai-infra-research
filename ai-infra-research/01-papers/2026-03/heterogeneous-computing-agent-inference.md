---
title: "Heterogeneous Computing: The Key to Powering the Future of AI Agent Inference"
tags: [inference-serving, roofline, hardware, memory-capacity, agent, MoE, MLA]
subfield: inference-serving
venue: "arXiv preprint (2026.01)"
date: 2026-03-09
authors: [Aaron Zhao, Junyi Liu]
institution: [Imperial College London, Microsoft Research]
url: "https://arxiv.org/abs/2601.22001"
status: 已读
rating: ⭐⭐⭐⭐
---

# Heterogeneous Computing：OI + CF 分析框架

## 一句话总结

传统 Roofline 模型只能判断 compute-bound 或 bandwidth-bound，漏掉了第三个关键瓶颈——**内存容量**。论文提出 OI（Operational Intensity）+ CF（Capacity Footprint）两个指标，构成四象限分析框架，统一解释了为什么需要 P/D 解耦、为什么需要 MLA、为什么需要分层存储、为什么需要异构硬件。

---

## 背景：Roofline 模型是什么？

GPU 做任何计算都需要两步：**从 HBM 读数据** → **用 ALU 算**。谁更慢，谁就是瓶颈。

```
GPU 的两个能力：
  算力上限：H100 = 990 TFLOPS（FP16）
  带宽上限：H100 = 3.35 TB/s

平衡点 = 算力 / 带宽 = 990T / 3.35T ≈ 295 FLOPs/Byte
```

**Arithmetic Intensity（OI）= 每搬 1 byte 做几次运算**

```
你的 OI > 295  → 算力先用完，带宽有余 → Compute-bound
你的 OI < 295  → 带宽先用完，算力闲着 → Memory-bandwidth-bound
```

画成图：

```
性能
(FLOPS)
  │            ╱ ← 算力上限（水平线，"屋顶"）
  │          ╱
  │        ╱  ← 带宽上限（斜线，斜率 = 带宽）
  │      ╱
  │    ╱
  │  ╱
  │╱
  └──────────────────── OI (FLOPs/Byte)
        ↑                    ↑
   Decode 在这里          Prefill 在这里
   (bandwidth-bound)      (compute-bound)
```

### 用 LLaMA 7B 的具体例子

**Prefill（输入 4 个 token）：**
- 计算量：4 个 token × 所有层的矩阵乘法 = 大量运算
- 读取量：读一遍模型权重 ≈ 14 GB
- OI = 大量运算 / 14 GB → **OI 高 → Compute-bound**

**Decode（生成 1 个 token）：**
- 计算量：1 个 token × 所有层的矩阵乘法 = 很少运算
- 读取量：还是读一遍模型权重 ≈ 14 GB
- OI = 很少运算 / 14 GB → **OI 极低 → Memory-bandwidth-bound**

同样读 14GB 权重，Prefill 做了 4+ 个 token 的活，Decode 只做了 1 个 token 的活。

### Batching 的本质

```
Batch=1:   读 14GB 权重，算 1 个 token   → OI 极低
Batch=128: 读 14GB 权重，算 128 个 token  → OI 提升 128 倍！
```

Batch 让每次搬运做更多"活"，OI 往右移，逼近 compute-bound。

---

## 论文核心贡献：Roofline 的第三个维度——CF

### 传统 Roofline 只看了两个象限

传统 Roofline 只有 OI 一个维度，只能判断 compute-bound 或 bandwidth-bound。但实际中 LLM Decode 时，MBU（带宽利用率）和 MFU（算力利用率）可能**同时很低**——不是系统效率差，而是被**内存容量**卡住了。

### CF（Capacity Footprint）= 每个请求需要多少 DRAM 容量

```
OI = FLOPs / Bytes_moved（每搬 1 byte 做多少运算）→ 判断带宽够不够
CF = Bytes_needed / Request（每个请求需要多少显存）→ 判断容量够不够
```

### 四象限模型（Figure 1）

```
              OI 高                    OI 低
         ┌──────────────────┬──────────────────┐
CF 低    │  Compute-Bound   │  Memory BW-Bound │  ← 传统 Roofline 只看这一行
         │  (Prefill-Attn,  │  (Decode-FFN      │
         │   Prefill-FFN)   │   高 Batch)       │
         ├──────────────────┼──────────────────┤
CF 高    │  Memory Capacity │  Bound by Both   │  ← Roofline 看不到这一行！
         │  Bound           │  Capacity + BW   │
         │                  │  (Decode-Attn     │
         │                  │   长上下文)        │
         └──────────────────┴──────────────────┘
```

右下角是最糟糕的：OI 低（带宽受限）+ CF 高（容量也不够）。Agent 场景的长上下文 Decode 正好落在这里。加更多 GPU 也没用——因为问题不是算力不够，而是显存容量装不下 KV Cache。

---

## Figure 2：不同 Agent 类型的工作负载差异

四种 Agent（LLaMA-70B）的 token 使用模式：

```
                Chatbot    Coding     Web-use    Computer-use
Prompt tokens   ~几百      ~几千      ~几万       ~几万
Decode tokens   ~几百      ~几百      ~几百       ~几百
输入/输出比      ~1:1       ~10:1      ~100:1      ~100:1
```

关键发现：
- Chatbot 的 CF 和 OI 还算合理，传统架构勉强够用
- Coding/Web-use/Computer-use 的 CF 爆炸（工具定义 + 环境上下文 + 多轮交互），OI 极低
- **雪球效应**：Agent 多轮交互让 context 不断增长，Coding Agent 可达 300K-1M tokens
- Figure 2 中间图：大多数 Agent 场景的 CF 已**超过单张 B200（192GB）**的容量上限
- Figure 2 右图：Decode 阶段 OI 远低于 H100/B200 的 Roofline 屋顶 → 算力被严重浪费

---

## Figure 3：Attention 机制对 CF 的影响

```
横轴：Context length（10³ 到 10⁵，对数坐标）
纵轴：Capacity Footprint（GB/Request）
```

- **MHA**：CF 增长最快，长上下文时远超其他
- **GQA**：好很多（KV 头数少）
- **MLA**（latent_dim=64）：几乎是平线，CF 极低

与 Cheatsheet §八 的 MHA→GQA→MLA 演进完全对应。MLA 的低 CF 不仅省显存，更重要的是让系统**不会掉进 capacity-bound 象限**。

---

## Figure 4：Dense vs MoE 的 CF 和 OI

- MoE（DeepSeek-R1-671B）的 CF 远超 Dense 模型 → 所有专家权重要加载
- MoE 的 OI 远低于 Dense → 每 token 只激活 5% 参数，但权重全读一遍 → 更加 memory-bound
- Batch=16 时 MoE 的 CF 可能接近 100+ GB/request，轻松超出单卡容量

---

## 系统级启示（Section 2.4）

1. **P/D 解耦应该是默认架构**：Prefill 和 Decode 的 OI 差异巨大 → [[distserve]]
2. **Agent 需要更大内存容量**：长 context 的 CF 超出单卡 → [[mooncake]] 的 VRAM/DRAM/SSD 分层存储
3. **内存容量和带宽是双重瓶颈**：不只是带宽，容量本身也卡 → 需要异构硬件 + 光互联

---

## 五个前瞻假设（Section 4）

1. **专用 Prefill/Decode 加速器**：不止两种，不同 Agent、不同模态可能需要不同芯片（NVIDIA Rubin CPX 已经开始）
2. **Agent-硬件 co-design**：模型训练后针对目标硬件蒸馏适配，没有相同硬件就无法复现推理效率
3. **新范式可能颠覆硬件格局**：扩散模型、状态空间模型等可能重新定义推理需求
4. **光互联（Optical I/O）扩大 scale-up 域**：支持更多异构芯片互联，D2D 级别带宽 <1pJ/bit
5. **内存解耦（Memory Disaggregation）**：Agent 需要长短期记忆（KV Cache + 知识库 + 文件系统），需要系统级分层存储

---

## 这篇论文和已学内容的完整映射

```
论文观点                          已学对应内容
─────────────────────────────────────────────────────────
OI（Arithmetic Intensity）       Cheatsheet §十 Roofline 分析
CF（Capacity Footprint）         Cheatsheet §五 显存全景 + §八 KV Cache
P/D 解耦是必然                   DistServe + Mooncake + Dynamo
MoE 更 memory-bound             Cheatsheet §四 MoE 架构
MLA 降低 CF                      Cheatsheet §八 MHA→GQA→MLA
分层存储解决 capacity            Mooncake 的 VRAM/DRAM/SSD
Agent 雪球效应                   Mooncake 的 Kimi 场景（avg 7590 tokens）
光互联 + 异构硬件                Hao Zhang talk 前瞻部分
拆分粒度越来越细                 Hao Zhang talk 的 AFD 趋势
```

---

## 我的理解

- 这篇论文不提新系统，而是提供了 **OI + CF 两个维度的分析框架**，统一解释了推理领域所有主要优化方向的动机
- CF 这个概念之前隐含在各种论文里（Mooncake 讲分层存储、DistServe 讲显存约束），但从没有人显式定义并画成四象限。这是论文最大的贡献
- 对我的投研视角特别有用：评估推理硬件/系统时，可以用 OI/CF 框架快速判断它在优化哪个象限的问题
- 右下角（高 CF + 低 OI）是最难的象限，也是未来投入最大的方向——Agent 场景的长上下文 Decode 正好落在这里

---

## 关联笔记

- [[distserve]] — 论文用 OI 差异证明 P/D 解耦的必要性，DistServe 是这个方向的奠基论文
- [[mooncake]] — 论文的 CF 分析直接指向 Mooncake 的分层存储方案（VRAM/DRAM/SSD）
- [[sglang]] — SGLang 的 RadixAttention 通过前缀复用降低 CF（减少重复 KV Cache）
- [[flash-attention]] — FlashAttention 的 Tiling 在 SRAM 层面优化了带宽利用，是提升 OI 的底层技术
- [[AI-Infra-Cheatsheet]] — §十 Roofline 分析是本论文 OI 维度的基础，§八 KV Cache 公式是 CF 的计算依据
- Hao Zhang talk — 张灏的 AFD 趋势和本文的异构硬件假设一脉相承

---

*学习方式：阅读论文 PDF + Claude 对话讲解*
*最后更新：2026-03-09*
