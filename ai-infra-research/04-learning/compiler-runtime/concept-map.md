---
tags: [learning]
subfield: compiler-runtime
created: 2026-03-05
---

# Compiler / Runtime 概念图谱

> 目标：理解"AI 计算如何从 Python 代码变成 GPU 上高效运行的指令"
> 这是最底层、最技术性的子领域，适合长期保持关注，不需要集中突破

---

## 核心问题是什么？

你写了 `model(input)` 这行 Python，GPU 是怎么知道要做什么的？中间经过了什么？

```
Python (PyTorch/JAX)
    ↓ 算子拆解 + 图优化
计算图 (IR)
    ↓ 代码生成 + 调度
CUDA Kernel（.ptx / SASS）
    ↓
GPU 硬件执行
```

每一步都有优化的空间，这就是 Compiler/Runtime 在做的事。

---

## 核心概念

### 第一层：Kernel（最底层的执行单元）

**CUDA Kernel**：GPU 上执行的函数，由成千上万个线程并行运行。每个 kernel 对应一个算子（如 matmul、softmax）。

**Kernel Fusion（融合）**：把多个 kernel 合并成一个，避免中间结果写回 HBM 再读出来的开销。FlashAttention 本质上就是把 attention 里多个算子 fuse 成一个 kernel。

**Occupancy（占用率）**：GPU SM（流多处理器）上实际跑的 warp 数 / 最大可能 warp 数。高 occupancy 不一定等于高性能，但 occupancy 太低通常意味着 GPU 没被充分利用。

### 第二层：算子库 vs 编译器

**cuBLAS / cuDNN**：NVIDIA 提供的手工优化算子库，matmul 用 cuBLAS，convolution 用 cuDNN，性能极好但不灵活。

**Triton**：OpenAI 开发的 GPU 编程语言，比 CUDA 高级，让研究者能用 Python-like 语法写出接近手工优化性能的 kernel。FlashAttention 的 Triton 实现让它得以广泛普及。

**TVM / Apache TVM**：陈天奇（CMU）主导，AI 编译器框架，自动生成和优化 kernel，支持多种硬件后端（不只是 NVIDIA GPU）。

**torch.compile / torch Inductor**：PyTorch 2.0 引入，把 Python 代码编译成优化后的 kernel，不需要用户手写 CUDA。对大多数模型有 1.5-2x 的提升。

### 第三层：计算图优化

**Operator Fusion**：在图级别把多个算子融合，减少内存访问。

**Graph Partitioning**：在分布式训练中，决定计算图的哪些部分在哪张 GPU 上运行（和 Tensor Parallelism 相关）。

**Quantization（量化）**：
- 把权重/激活从 fp16 变成 int8/int4/fp8，减少显存占用和带宽需求
- 精度损失需要用 calibration 或 QAT（量化感知训练）来控制
- Song Han（MIT）的 AWQ、SmoothQuant 是代表工作

### 第四层：Runtime 优化

**CUDA Graph**：把一系列 CUDA kernel 调用打包成一个"图"，一次提交，减少 CPU-GPU 调度开销。推理中特别有用（decode 阶段每步形状相同，非常适合）。

**Paged KV Cache 的底层**：vLLM 的 PagedAttention 需要定制 CUDA kernel 支持非连续内存上的 attention，这就是 compiler/runtime 层和 serving 层的交叉点。

---

## 你需要深入到什么程度？

作为投研/研究跟踪背景，你不需要：
- 自己写 CUDA kernel
- 理解 PTX 指令集
- 实现 TVM 的 pass

你需要能：
- 看到"我们用 Triton 重写了这个 kernel"，知道这意味着什么
- 理解"kernel fusion 减少了 HBM 访问"和你的半导体知识（HBM bandwidth 是瓶颈）挂钩
- 判断量化方案（int4/fp8）的精度-性能 trade-off
- 在会议上听懂 compiler/runtime 相关报告的 motivation 部分