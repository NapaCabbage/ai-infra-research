---
tags: [learning]
subfield: scheduling
created: 2026-03-05
---

# Distributed Scheduling / Orchestration 概念图谱

> 目标：理解"如何在 GPU 集群层面统筹分配资源，让训练和推理任务高效运行"
> 建议在学完 Inference Serving 和 Training Infra 之后再看

---

## 核心问题是什么？

如果 Inference Serving 解决的是"单次推理怎么做快"，Training Infra 解决的是"一次训练怎么并行"，那 Scheduling 解决的是**更高一层的问题**：

- 一个公司有 10,000 张 H100，同时有 50 个训练任务和无数推理请求，**怎么分配这些 GPU？**
- 某个训练任务需要 1000 张 GPU，但现在只有 800 张空闲，**等还是用更少的 GPU 先跑？**
- 推理请求有轻有重（prompt 长短不同），**怎么安排处理顺序才能最大化吞吐/最小化 P99 延迟？**

---

## 核心概念

### 第一层：推理调度

**Request Scheduling（请求级调度）**
- 最基本：先来先服务（FCFS）
- 优化：优先处理短请求（减少队头阻塞）
- 更进一步：Sarathi-Serve 的 chunked prefill + 统一调度

**Preemption（抢占）**：
- 当一个新的高优先级请求进来，是否中断当前正在处理的请求？
- KV Cache 需要保存（swap 到 CPU 内存）或丢弃重算，都有代价
- vLLM 支持基于 PagedAttention 的高效抢占

**Multi-instance Routing（多实例路由）**：
- 真实部署中有多个推理实例（多台机器），需要把请求路由到合适的实例
- 朴素做法：Round-Robin
- 优化做法：根据每个实例的当前负载、KV Cache 状态（prefix cache 命中率）来路由

**SLO-aware Scheduling（SLO 感知调度）**：
- SLO = Service Level Objective，例如"P99 TTFT < 2 秒"
- 系统要在满足 SLO 的前提下最大化吞吐

### 第二层：训练集群调度

**Gang Scheduling（组调度）**：
- 分布式训练要求所有参与的 GPU 同时启动，否则等待的 GPU 白白浪费
- 传统调度器（如 Kubernetes）不支持 gang，需要特殊处理

**Preemption in Training**：
- 训练任务周期长（几天到几个月），能否在中途暂停、换给更紧急的任务？
- 需要 checkpoint 机制支持

**Placement Policy（放置策略）**：
- 一个 8-GPU 的训练任务，应该放在同一台机器上（用 NVLink）还是跨机器（用 IB）？
- 一般同机器优先，但资源不一定够

**Resource Fragmentation（资源碎片化）**：
- 某任务需要 12 张 GPU，但现在有 3 台机器各空 4 张，能不能用？
- 和内存碎片类似，是集群调度的核心挑战

### 第三层：计算框架

**Ray**：
- UC Berkeley 开发的分布式计算框架，Python 友好
- Ray Serve：基于 Ray 的 LLM serving 框架
- 大量 LLM 公司的推理/训练基础设施基于 Ray 构建

**Kubernetes + GPU 调度**：
- K8s 是容器编排的工业标准，但原生不支持 GPU gang scheduling
- 通常搭配 Volcano 或 Yunikorn 调度器

**Slurm**：
- HPC 传统调度器，很多学术 GPU 集群在用
- 支持 gang scheduling，但不够灵活

---

## 与推理和训练的关系

```
集群层（Scheduling）
├── 把 GPU 分给哪些任务？多少 GPU？何时分？
├── 推理实例之间如何路由请求？
└── 训练集群的资源如何动态分配？

推理实例层（Inference Serving）
├── Continuous Batching
├── KV Cache 管理
└── Request 内部调度

训练任务层（Training Infra）
├── 3D Parallelism
├── ZeRO
└── 通信优化
```