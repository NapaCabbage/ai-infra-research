---
tags: [briefing, research-direction]
date: 2026-03-09
purpose: 内部会议汇报
---

# AI 推理基础设施：2025 复盘与 2026 重点方向

---

## 一、2025 复盘

**P/D 解耦已成为行业标准。** Prefill（理解阶段，compute-bound）和 Decode（生成阶段，memory-bandwidth-bound）的资源需求截然不同。DistServe（OSDI 2024）提出将两者拆分到不同 GPU 独立优化，实验显示 goodput 提升 2-7×。2025 年，NVIDIA Dynamo、SGLang、vLLM、Mooncake 等所有主流框架均已支持 P/D 解耦，从学术原型走向生产标准。

**KV Cache 管理成为系统设计核心。** Mooncake（Kimi，FAST 2025 Best Paper）将 KV Cache 提升为一等公民，构建 VRAM/DRAM/SSD 三级分层存储 + 全集群 RDMA 共享，实现 75% 请求量提升。SGLang 的 RadixAttention 实现跨请求前缀复用。这标志着推理系统设计从"GPU 算力调度"转向"KV Cache 生命周期管理"。

---

## 二、2026 重点研究方向

### 方向 1：Agent 推理的内存容量瓶颈（全新问题）

> 来源：Aaron Zhao, Junyi Liu — "Heterogeneous Computing" (arXiv 2601.22001)

传统 Roofline 模型只关注 compute-bound 或 bandwidth-bound，**漏掉了第三个瓶颈——内存容量**。论文提出 OI（Operational Intensity）+ CF（Capacity Footprint）四象限框架：

- Agent 场景（Coding、Web-use、Computer-use）输入长度可达 300K-1M tokens，KV Cache 远超单卡 192GB（B200）容量
- 此时 MFU 和 MBU 同时很低——不是效率差，是**显存装不下**
- MoE 模型进一步放大此问题：DeepSeek-R1 (671B) 的 CF 在 batch=16 时可达 100+ GB/request

**投资/研究启示：** 下一代推理硬件的核心竞争力不再只是算力或带宽，而是**内存容量和内存解耦能力**。

### 方向 2：层内拆分 AFD（Attention-FFN Disaggregation）

> 来源：Hao Zhang (UCSD) — Dynamo Day 2025/2026 演讲

拆分粒度从"阶段级"（P/D 解耦）下沉到"层内级"：将 Transformer 每一层的 Attention 和 FFN/MoE 模块拆开，部署在不同 GPU 上。

- 关键洞察：MoE 模型本身每层已有 all-to-all 通信（专家路由），AFD 的层间传输可与之合并，**零额外延迟开销**
- 字节跳动 Mega-Scale Infer 验证：MoE 场景下吞吐提升 **1.9×**
- 预计 2026 下半年头部厂商推出支持 AFD 的推理引擎

**拆分演进脉络：**

```
2022 Orca（请求级）→ 2023 DistServe（阶段级）→ 2024 Mooncake（存储级）
→ 2025 Dynamo（生产落地）→ 2026 AFD（层内级）→ 未来 算子级？
```

### 方向 3：异构硬件与光互联

> 来源：Aaron Zhao 论文 + Hao Zhang 演讲

- NVIDIA 路线图已包含专用 Prefill 芯片（Rubin CPX），未来可能出现**多种推理加速器**共存于同一系统
- 光互联（Optical I/O）将提供 D2D 级别带宽（<1pJ/bit），使计算与内存的物理解耦成为可能
- Agent-硬件 co-design：模型训练后需针对目标推理硬件做蒸馏适配，硬件差异将成为推理效率的壁垒

### 方向 4：视频扩散模型推理

> 来源：Hao Zhang (UCSD) — Fast Video 项目

- 当前 1 分钟视频生成成本 ~$10，是 LLM 推理的数百倍
- 13B DIT 模型生成 5 秒 720p 需 16 分钟（单 H100），80% 算力在长序列 3D Attention（115K tokens）
- 目标：2026 年实现 1080p/4K 实时视频生成
- LLM 推理优化技术（FlashAttention、P/D 解耦）将迁移至视频推理

---

## 三、核心判断

1. **推理 > 训练**：数据中心工作负载正从训练主导转向推理主导，Agent 场景加速此趋势
2. **内存 > 算力**：下一代推理系统的瓶颈从算力转向内存容量和带宽，MLA 和分层存储是关键缓解手段
3. **拆分持续深化**：从 P/D 阶段级拆分到 Attention/FFN 层内级拆分，系统设计与模型设计的协同演进越来越紧密
4. **异构是必然**：同构 GPU 集群无法适配多样化的 Agent 推理需求，专用加速器 + 光互联 + 内存解耦将成为标配



**排排优先级、业务-2聊一聊信息收集**
1. 聚焦agent的问题、学术界工作
2. 不太确定多模态
3. robotics稍微靠后
4. ai应用研究让bd做