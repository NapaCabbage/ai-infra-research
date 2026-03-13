---
tags: [learning, gpu]
created: 2026-03-12
updated: 2026-03-12
---

# GPU 架构学习路径

> 信息来源：Modal Labs [GPU Glossary](https://modal.com/gpu-glossary)（79 篇词条）
> 目标：理解 GPU 硬件和编程模型，为理解 DeepSeek-V3 FP8 训练、FlashAttention 内核优化等打下基础

---

## 为什么要学 GPU 架构

我们在 training infra 学习中遇到了多处"需要 GPU 背景"的知识缺口：

```
DeepSeek-V3:
  - SM 为什么能被分配做通信？（20/132 个 SM 做 All-to-All）
  - Warpgroup MMA 是什么？为什么两个 warpgroup 可以交替执行？
  - Tensor Core FP8 累加精度为什么只有 14 bit？
  - 自定义 PTX 指令是什么意思？

FlashAttention:
  - Tiling 为什么能加速？SRAM vs HBM 的带宽差距
  - Shared Memory bank conflict 是什么？
  - Occupancy 怎么影响性能？

Roofline 分析:
  - Compute-bound vs Memory-bound 的硬件根源是什么？
  - Arithmetic Intensity 的物理含义
```

---

## 学习结构（4 个模块，建议 3-5 天）

### 模块 1：硬件物理层 → [[gpu-hardware]]
> 从芯片到核心，理解 GPU 的物理结构

核心词条（按阅读顺序）：
1. **cuda-device-architecture** — GPU 统一架构演进
2. **streaming-multiprocessor (SM)** — GPU 的核心计算单元（对标 CPU 核心）
3. **streaming-multiprocessor-architecture** — SM 版本与架构代际
4. **cuda-core** — 标量计算单元
5. **tensor-core** — 矩阵计算单元（★ 最重要，理解 FP8/BF16 加速的硬件基础）
6. **warp-scheduler** — 调度器如何在纳秒级切换线程组
7. **register-file** — SM 内最快的存储
8. **l1-data-cache** — SM 内的 SRAM 缓存
9. **gpu-ram** — HBM（全局显存）
10. **tensor-memory / tensor-memory-accelerator** — Hopper/Blackwell 新增的专用存储

辅助词条：core, graphics-processing-cluster, texture-processing-cluster, load-store-unit, special-function-unit

### 模块 2：编程模型层 → [[gpu-programming-model]]
> 从线程到 kernel，理解 CUDA 的软件抽象

核心词条（按阅读顺序）：
1. **cuda-programming-model** — 三大抽象：线程层级、内存层级、同步
2. **thread → warp → thread-block → thread-block-grid** — 线程层级（由小到大）
3. **warpgroup** — Hopper 引入的 128 线程组（理解 DeepSeek-V3 WGMMA 的关键）
4. **kernel** — CUDA 程序的基本单元（含矩阵乘法示例）
5. **memory-hierarchy** — 内存层级总览
6. **registers → shared-memory → global-memory** — 三级存储（由快到慢）
7. **parallel-thread-execution (PTX)** — GPU 的"汇编语言"（理解自定义 PTX 指令）
8. **streaming-assembler (SASS)** — GPU 机器码

辅助词条：cooperative-thread-array, compute-capability, thread-hierarchy

### 模块 3：性能分析层 → [[gpu-performance]]
> 从 Roofline 到 Occupancy，理解性能瓶颈与优化

核心词条（按阅读顺序）：
1. **roofline-model** — 性能分析的核心框架
2. **arithmetic-intensity** — 计算密度（FLOPs/Byte）
3. **compute-bound / memory-bound** — 两种瓶颈类型
4. **memory-bandwidth / arithmetic-bandwidth** — 两个"屋顶"
5. **occupancy** — SM 利用率（active warps / max warps）
6. **latency-hiding** — GPU 最核心的性能机制（通过大量线程遮盖延迟）
7. **memory-coalescing** — 合并内存访问（性能关键）
8. **bank-conflict** — Shared Memory 访问冲突
9. **warp-divergence** — 分支导致的性能损失
10. **register-pressure** — 寄存器不够用时的性能退化

辅助词条：littles-law, peak-rate, pipe-utilization, warp-execution-state, scoreboard-stall, issue-efficiency, branch-efficiency, active-cycle, overhead, streaming-multiprocessor-utilization, performance-bottleneck

### 模块 4：软件工具层（选读）
> Host 端工具链，按需查阅

- **cuda-software-platform** — CUDA 平台总览
- **cuda-c** — CUDA C++ 语法扩展
- **nvcc** — CUDA 编译器
- **cublas / cudnn** — 高性能数学库
- **nsight-systems** — 性能分析工具
- 其他：nvidia-smi, nvml, cupti, cuda-driver-api, cuda-runtime-api 等

---

## 与已学内容的关联

```
模块 1 (硬件) → 解答 DeepSeek-V3 的 SM 分配问题、Tensor Core 精度问题
模块 2 (编程) → 解答 PTX 自定义指令、Warpgroup MMA
模块 3 (性能) → 加深 Roofline 理解、解答 FlashAttention 的 tiling/coalescing 优化
模块 4 (工具) → 已在 Cheatsheet §十二 有覆盖，选读补充
```
