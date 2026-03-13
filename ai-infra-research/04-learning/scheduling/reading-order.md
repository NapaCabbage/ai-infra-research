---
tags: [learning]
subfield: scheduling
created: 2026-03-05
---

# Scheduling 精选阅读路径

> 建议学完 Inference Serving 和 Training Infra 基础后再看，约 2-3 周

---

## 推理调度

### 必读 ①：Sarathi-Serve（Chunked Prefill 的完整方案）
- **论文**：*Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*
- **发表**：OSDI 2024
- **为什么读**：把 chunked prefill 和 continuous batching 结合，解决了 prefill 霸占 GPU 导致 decode 延迟抖动的问题，是推理调度层的重要工作
- **链接**：https://arxiv.org/abs/2403.02310

### 选读 ②：Llumnix（动态迁移）
- **论文**：*Llumnix: Dynamic Scheduling for Large Language Model Serving*
- **发表**：OSDI 2024
- **核心想法**：推理实例之间可以在线迁移请求（连同 KV Cache），实现更动态的负载均衡
- **链接**：https://arxiv.org/abs/2406.03243

---

## 训练集群调度

### 必读 ③：Gandiva（GPU 集群调度先驱）
- **论文**：*Gandiva: Introspective Cluster Scheduling for Deep Learning*
- **发表**：OSDI 2018
- **为什么读**：虽然是比较早期的工作，但把 DL 训练的调度问题清晰地定义出来了，是理解后续工作的基础
- **链接**：https://www.usenix.org/conference/osdi18/presentation/xiao

### 选读 ④：Alpa（自动并行化 + 调度）
- **论文**：*Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning*
- **发表**：OSDI 2022
- **核心想法**：自动搜索最优并行策略，不需要人手动决定用多少 TP/PP/DP
- **链接**：https://arxiv.org/abs/2201.12023

---

## 框架层

### 选读 ⑤：Ray（分布式 Python 的基础）
- 不是论文，是官方文档：docs.ray.io
- 重点看 Ray Core 的 actor 模型 + Ray Serve 的 deployment 概念
- 能帮你理解为什么 vLLM/SGLang 等推理框架选择 Ray 作为底层