---
tags: [learning]
subfield: compiler-runtime
created: 2026-03-05
---

# Compiler / Runtime 精选阅读路径

> 这个子领域最技术性，建议边学边关注，不用集中突破

---

## 最低限度（必须了解）

### ①：FlashAttention-2（已在 Inference Serving 路径中）
- 如果已经读过，从 compiler 视角重新看一遍
- 核心：为什么 IO-aware 的 kernel 设计能大幅提速？和 HBM/SRAM 的关系是什么？

### ②：Triton 官方 Tutorial（实践入门）
- https://triton-lang.org/main/getting-started/tutorials/
- 看前 3 个 tutorial（Vector Add、Fused Softmax、Matrix Multiplication）
- 不需要自己运行，读代码理解思想即可
- 读完你能理解：为什么 Triton 让研究者能写高效 kernel

---

## 进阶阅读

### ③：量化（对投研最实用）
- **论文**：*AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*（MIT，MLSys 2024）
- **为什么读**：量化是当前降低推理成本最实用的技术，AWQ 是学术界最干净的工作之一
- **链接**：https://arxiv.org/abs/2306.00978

### ④：torch.compile 概念
- 不是论文，是官方文档：pytorch.org/docs/stable/torch.compiler.html
- 了解 TorchDynamo（捕获计算图）+ Inductor（代码生成）的大致原理即可

### ⑤：TVM（了解 AI 编译器的全貌）
- **论文**：*TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*（OSDI 2018）
- **链接**：https://arxiv.org/abs/1802.04799
- 经典论文，理解 AI 编译器的基本架构（IR、schedule、codegen）
- 不需要深入，读 Introduction + Overview 即可

---

## 持续关注（不需要专门学，看到就看）

- **CUDA Graph**：每次 vLLM/SGLang 的 release note 里都会提到，了解它做什么就行
- **FP8 训练**：DeepSeek-V3 用了，H100 原生支持，未来会越来越重要
- **MLA（Multi-head Latent Attention）**：DeepSeek 提出，减少 KV Cache 的新思路，算是 attention 机制和 compiler 的交叉