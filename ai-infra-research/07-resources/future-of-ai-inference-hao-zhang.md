---
title: "The Future of AI Inference — Hao Zhang (UCSD)"
tags: [inference-serving, survey, disaggregation, AFD, video-diffusion, retrospective]
subfield: inference-serving
source: "Dynamo Day 演讲 + Disaggregated Inference: 18 Months Later 博文"
date: 2026-03-09
speaker: Hao Zhang (UCSD)
---

# AI 推理的未来：2025 复盘 + 2026 展望

> Hao Zhang 是 DistServe（论文中称 D-SERV）的核心作者之一，UCSD 教授，长期深耕 AI 推理系统。
> 本笔记综合两个来源：Dynamo Day 压轴演讲 + [Disaggregated Inference: 18 Months Later](https://haoailab.com/blogs/distserve-retro/) 博文（2025.11，作者团队的官方回顾）。

---

## Part 1：2025 复盘——P/D 解耦全面落地

### 我们已经学过的内容在这里得到了完整验证

Hao Zhang 用一张图（即 DistServe Figure 1 的数据）展示了 P/D 解耦的收益：

```
混跑（continuous batching）：1.6 req/s/GPU
拆分后（2 Prefill + 1 Decode = 3 GPU）：10 req/s 总量 → 3.3 req/s/GPU
提升：~2×
```

这和我们在 [[distserve]] 笔记中记录的实验结果完全一致。

### 落地时间线

```
2023 末  UCSD 开源 D-SERV（= DistServe 原型）+ 微软 Split-Wise 同期发布
2024     字节跳动、Google 率先在生产环境替换 continuous batching
2025 初  DeepSync with Ray 公开支持拆分架构
2025 GTC NVIDIA 重点推介 Dynamo
2025 末  几乎所有生产级推理系统均采用 P/D 解耦
```

### 核心待解问题（仍在研究中）

1. **资源分配**：根据 workload 动态决定 Prefill/Decode 的 GPU 数量与部署位置
   - 对应 [[distserve]] 的 Simulator + placement 算法
   - 对应 NVIDIA Dynamo 的 AI Configurator + Planner

2. **KV Cache 传输**：异构硬件集群下最小化传输延迟
   - 对应 [[mooncake]] 的 RDMA Transfer Engine + Layer-wise 流式传输
   - 对应 NVIDIA Dynamo 的 NIXL

---

## 补充：Disaggregated Inference 18 Months Later（博文回顾）

> 来源：[haoailab.com/blogs/distserve-retro](https://haoailab.com/blogs/distserve-retro/)（2025.11，Junda Chen, Yonghao Zhuang, Hao Zhang）

### 为什么 2024 年没人用？

DistServe 论文 2023 末发布后，开源社区**并没有立即采用**。原因不是 idea 不好，而是：
- 现有推理系统（vLLM 等）的代码架构需要**深度重构**才能支持 P/D 分离
- 工程改造量大，社区观望心态

### 什么驱动了 2025 年的全面落地？

两个转折点：
1. **业务规模化**：越来越多企业把 LLM 作为核心产品组件，跑满负载后发现 throughput 不是唯一指标——**尾延迟直接影响业务增长甚至存亡**，goodput 的重要性被验证
2. **NVIDIA GTC 2025**：NVIDIA 发布 Dynamo，官方背书 P/D 解耦架构，成为行业转折点

到 2025 年末，几乎所有生产级推理框架（Dynamo、llm-d、Ray Serve、SGLang、vLLM、LMCache、Mooncake）都支持 disaggregation。

### 聚合 vs 解离：不是非黑即白

博文中一个重要观点：**Chunked Prefill（聚合方案）和 Disaggregation 各有适用场景**。

Chunked Prefill 是把长 Prefill 切小块，和 Decode 混着跑——减轻干扰但不消除。在小规模、短输入场景可能更简单高效。但在大规模集群（数百到数千 GPU）、长输入、严格 SLO 场景下，disaggregation 优势明显。

### KV Cache 传输开销：实测数据

博文给出了具体数字：OPT-175B 模型，2048 tokens 的请求，KV Cache 传输延迟 **17.6ms**——比单步 Decode（30-50ms on A100）还短。

前提是需要 NVLink 级别的带宽。如果集群只有 PCIe 或低速跨节点网络，且 prompt 很短，解耦的收益可能不明显。这和 [[distserve]] 的 Algorithm 1/2 选择逻辑一致。

### 推理成本下降速度

博文提到 LLM 推理成本的下降速度**远超摩尔定律的 2× 节奏**——disaggregation 是其中的重要驱动力之一。

---

## Part 2 趋势一：AFD（Attention-FFN Disaggregation）

### 核心 idea：拆分从"阶段级"下沉到"层内级"

P/D 解耦是把 Prefill 和 Decode 两个**阶段**拆开。AFD 更进一步，把 Transformer **每一层内部**的 Attention 模块和 FFN/MoE 模块也拆开，部署在不同 GPU 上：

```
P/D 解耦（2023-2025）：
  Prefill GPU ──KV Cache──→ Decode GPU
  （阶段级拆分）

AFD（2026+）：
  Attention GPU ──activation──→ FFN/MoE GPU ──activation──→ Attention GPU
  （层内级拆分，每层都要通信）
```

### 为什么要拆？

Attention 和 FFN/MoE 的计算特征不同（和 P/D 的逻辑一样）：

```
              Attention              FFN/MoE
适合的并行    数据并行、大 batch      专家并行、小 batch
瓶颈         KV Cache 读写           参数量大、稀疏激活
```

串行放在一起 = 两者都无法用最优配置。

### 关键洞察：MoE 场景下通信"免费"

初看 AFD 每层都要通信，开销巨大。但 MoE 模型本身每层就有 all-to-all 通信（专家路由）。AFD 的层间 activation 传输可以和 MoE 的 all-to-all **合并**完成：

```
原本 MoE 每层：计算 → all-to-all（专家路由）→ 计算
AFD + MoE：   计算 → all-to-all（专家路由 + AFD 传输合并）→ 计算
                              ↑ 只要合并后不比原来慢，AFD 就是零额外开销
```

### 实验结果

字节跳动 Mega-Scale Infer 原型验证：MoE 模型下 AFD 较串行实现吞吐提升 **1.9×**，几乎无额外延迟。

### 与我们已学内容的联系

这是 DistServe "瓶颈不同就拆开" 思想的自然延伸：
- 2023：P/D 解耦（阶段级）→ DistServe
- 2025：全面落地 → Dynamo
- 2026：AFD（层内级）→ 下一代架构

---

## Part 2 趋势二：视频扩散模型推理

### 现状

- 生成 1 分钟视频平均成本 **$10**，比 LLM 推理贵数百倍
- 13B 参数 DIT 生成 5 秒 720p 视频：单 A100 需 **16 分钟**
- **80%** 算力消耗在长序列 3D Attention（视频隐序列长度达 115K tokens）

### 架构

```
输入视频/文本 → VAE 编码器 → [DIT Block × 50-100 次迭代] → VAE 解码器 → 输出视频
                              ↑ 主要开销在这里
```

DIT（Diffusion Transformer）的扩散过程需要重复运行 50-100 次，每次都要做完整的 Attention + FFN，成本被放大数十倍。

### 优化方向

团队的 Fast Video 系统：从 Attention 内核、内存布局、系统架构全栈优化，已实现 1.3B 模型实时 480p 生成，目标 2026 年实现 1080p/4K 实时生成。

### 与已学内容的联系

- FlashAttention 对视频推理尤其重要：115K 序列长度的 Attention 没有 Tiling + Online Softmax 根本算不动
- P/D 解耦的思路可能也会应用到 DIT：VAE 编解码器和 DIT Block 的计算特征完全不同

---

## 拆分趋势的演进脉络（核心总结）

```
2022  Orca：Continuous Batching（请求级调度）
           ↓
2023  DistServe/Splitwise：P/D Disaggregation（阶段级拆分）
           ↓
2024  Mooncake：KVCache-centric Architecture（存储级拆分）
           ↓
2025  Dynamo：生产级全链路落地
           ↓
2026  AFD：Attention-FFN Disaggregation（层内级拆分）
           ↓
未来？ 算子级拆分？模型设计与系统设计完全协同演进
```

Hao Zhang的判断：系统设计的拆分粒度会越来越细，模型设计阶段就需要考虑推理部署效率。这和 SGLang 的 "前端+后端 co-design" 思想一脉相承。

---

## 关联笔记

- [[distserve]] — 张灏是 DistServe（D-SERV）的核心作者，本 talk 的 2025 复盘部分就是 DistServe 落地的总结
- [[mooncake]] — Mooncake 解决了 P/D 解耦后的 KV Cache 存储和传输问题，是张灏提到的"核心待解问题"的工业实践
- [[sglang]] — SGLang 的 co-design 思想与张灏预判的"模型设计与系统设计协同演进"一致
- [[flash-attention]] — FlashAttention 对视频推理（115K 序列）至关重要
- [[orca-continuous-batching]] — Orca 的 continuous batching 是拆分前的基线，张灏复盘中的"2025 年之前主流架构"

---

*来源：Dynamo Day 演讲（Hao Zhang，UCSD）*
*最后更新：2026-03-09*
