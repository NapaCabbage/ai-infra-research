---
title: GPU 编程模型：从线程到 Kernel
tags: [learning, gpu, programming-model]
created: 2026-03-12
---

# GPU 编程模型：从线程到 Kernel

## 概述

CUDA（Compute Unified Device Architecture）编程模型是 NVIDIA GPU 编程的基础，其核心在于**三个关键抽象**：
1. **线程层次结构**（Thread Hierarchy）
2. **内存层次结构**（Memory Hierarchy）
3. **栅栏同步**（Barrier Synchronization）

这个模型让程序能透明地扩展到更大的 GPU 设备上，无需重写代码。

---

## 第一部分：线程层次结构（自下而上）

### 基础：线程（Thread）

**线程**是 GPU 编程的最小单位。特点：
- 每个线程有自己的 PC（程序计数器）和寄存器
- 单个 CUDA Core 执行单个线程的指令
- GPU 线程不用于系统调用（与 POSIX 线程不同）
- 线程有私有栈在全局内存中（用于寄存器溢出和函数调用栈）

### 第二层：Warp（经线）

**Warp** 是 32 个线程的分组，是 GPU 执行的基本单位。

关键特性：
- **SIMT 执行**：Warp 中所有线程执行相同指令但处理不同数据
  ```
  同步执行：Thread 0, 1, 2, ... 31 同时执行相同指令
  ```
- **典型延迟隐藏**：当一个 Warp 等待内存时，Warp 调度器选择另一个 Warp 运行
- **Warp 发散**：当线程执行不同指令时性能大幅下降

```
┌─ Warp（32 threads，SIMT）
│  └─ Thread 0, 1, 2, ... 31（并行执行相同指令）
└─ 多个 Warp 在同一 SM 上交错执行（隐藏延迟）
```

### 第三层：Thread Block（线程块 = Cooperative Thread Array）

**Thread Block** 是多个 Warp 的集合，是程序员可以控制的最小同步单位。

特点：
- 一个 Block 最多 1024 个线程（通常是 32 的倍数）
- Block 内所有线程在同一 Streaming Multiprocessor (SM) 上运行
- Block 内线程可通过 `__syncthreads()` 栅栏同步
- 可访问共享内存（Shared Memory），实现线程间通信

```
┌─ Thread Block（e.g., 256 threads = 8 warps）
│  ├─ Warp 0（threads 0-31）
│  ├─ Warp 1（threads 32-63）
│  └─ ...
└─ Block 内线程可同步，访问共享内存
```

### 第四层：Warpgroup（128 线程 = 4 Warp）

**Warpgroup** 是 Hopper 及更新架构引入的概念：4 个连续 Warp = 128 个线程。

为什么重要（与 DeepSeek-V3 相关）：
- 支持 Warpgroup 级别的矩阵乘法指令 `wgmma.mma_async`
- **消除显式 Inter-warp 同步需求**
- 更高效地利用 Tensor Cores 的算术带宽
- DeepSeek-V3 的 KV Cache 通信机制就使用 Warpgroup 级指令

```
┌─ Warpgroup（128 threads）
│  ├─ Warp 0, 1, 2, 3（4 个连续 Warp）
│  └─ 对应 SM 内一个子单元（有自己的调度器和 Tensor Core）
```

### 最高层：Grid（网格）

**Grid** 是所有 Thread Block 的集合，跨越整个 GPU。

特点：
- 可 1D/2D/3D 组织
- Block 之间**无保证执行顺序**（任意交错都有效）
- Block 间不能通过栅栏同步，只能通过全局内存 + 原子操作协调
- 对应内存层次的最高级：全局内存

```
Grid 示意：
┌─────────────────────────────────────┐
│      Thread Block Grid (GPU)        │
├─────────────────────────────────────┤
│ Block(0,0)  Block(1,0)  Block(2,0)  │
│ Block(0,1)  Block(1,1)  Block(2,1)  │
└─────────────────────────────────────┘
 ↓ 分配到多个 SM（Streaming Multiprocessor）
```

### 硬件映射

```
编程模型                →    硬件映射
Thread              →    CUDA Core（单个核心）
Warp (32 threads)   →    同一 SM 内的 32 个 Core（SIMT 执行）
Warpgroup           →    SM 内一个子单元（4个 Warp 的调度单元）
Thread Block        →    Streaming Multiprocessor (SM)
Grid                →    整个 GPU（多个 SM）
```

---

## 第二部分：内存层次结构

GPU 的内存层次与线程层次对应，自下而上：

### 1. 寄存器（Registers）- 单个线程

**存储**：SM 的寄存器文件
**访问延迟**：~0 个周期（1 个周期内可得）
**大小**：通常每个线程数百个寄存器（64-255）
**作用域**：私有，仅该线程可访问

```
寄存器冲突：
- 每个线程用寄存器数越少 → 单个 SM 能调度更多线程
- 更多线程 → 更好的延迟隐藏
- 因此编译器会平衡：生成较少寄存器的代码
```

### 2. 共享内存（Shared Memory）- Thread Block 级

**存储**：SM 的 L1 数据缓存
**访问延迟**：~30 个周期
**大小**：每个 Block 可配置（通常 48-96 KB）
**作用域**：Block 内所有线程可访问

**关键用途**：
- 线程间通信和同步
- 重用全局内存加载的数据
- 减少全局内存访问次数

**Flash Attention 的例子**：
```
Flash Attention 使用 Tiling + Shared Memory：
1. 将注意力计算分块处理
2. 将 Q, K, V 的部分加载到共享内存
3. Block 内线程通过共享内存高效通信
4. 减少全局内存带宽消耗
→ 结果：显著加速注意力计算
```

### 3. L1/L2 缓存

**L1 缓存**：每个 SM 的数据缓存（与共享内存复用 L1 资源）
**L2 缓存**：GPU 级缓存（所有 SM 共享）

自动管理，程序员不直接控制，但理解其存在有助于优化。

### 4. 全局内存（Global Memory）- Grid 级

**存储**：GPU 的 HBM（High Bandwidth Memory，如 HBM2e）
**访问延迟**：~400 个周期（需要内存控制器）
**大小**：GB 级（40GB-192GB 取决于 GPU）
**作用域**：所有线程可访问（跨 Block）

**协调**：通过原子操作和栅栏（需要 Grid-level 同步）

### 内存层次图示

```
访问速度 ↑
   1 cycle  │ ┌─────────────────┐
            │ │   寄存器(Reg)   │ Per-thread
   ~30 cy   │ ├─────────────────┤
            │ │  共享内存(SMEM) │ Per-block
            │ ├─────────────────┤
            │ │    L1/L2 缓存   │ 自动管理
~400 cycles │ ├─────────────────┤
            │ │   全局内存(HBM) │ Per-grid
            └─────────────────────
                  ↓ 容量增加
```

**策略**：最大化共享内存使用，减少全局内存往返

---

## 第三部分：Kernel 和矩阵乘法示例

### 什么是 Kernel？

**Kernel** 是在 GPU 上执行的函数，被主机（CPU）启动一次，但并行执行多次（每个线程一次）。

```cpp
// 从 CPU 调用
mm_kernel<<<gridDim, blockDim>>>(A, B, C, N);
//  ↑          ↑        ↑
// 启动一次   Grid 大小  Block 大小
// 但在 GPU 上执行了成千上万次（每个线程一次）
```

### 示例 1：朴素矩阵乘法（无优化）

```cpp
__global__ void mm_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];  // 全局内存读取
        }
        C[row * N + col] = sum;
    }
}
```

**问题**：
- 每个线程对每对 A[row, k] 和 B[k, col] 都需要从全局内存读取
- **算术强度** = 1 FLOP / 2 reads = 0.5（极低）
- 内存带宽成为瓶颈，无法充分利用 GPU 算术能力

### 示例 2：Tiling 矩阵乘法（使用共享内存）

```cpp
#define TILE_WIDTH 16

__global__ void mm_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];  // 共享内存
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float c_output = 0;

    for (int m = 0; m < N/TILE_WIDTH; ++m) {
        // 1. 将 A 和 B 的 tile 加载到共享内存
        As[threadIdx.y][threadIdx.x] = A[row * N + (m * TILE_WIDTH + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * N + col];

        // 2. 等待所有线程完成加载
        __syncthreads();

        // 3. 使用共享内存的数据进行计算（无全局内存访问）
        for (int k = 0; k < TILE_WIDTH; ++k) {
            c_output += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // 4. 等待所有线程完成计算，再加载下一个 tile
        __syncthreads();
    }
    C[row * N + col] = c_output;
}
```

**优化关键点**：

1. **Tiling（分块）**：
   - 将 C 矩阵分成小块（16×16）
   - 每个 Thread Block 计算一个块

2. **共享内存重用**：
   - 加载 A 的 tile 到 SMEM：A[row, m*16:(m+1)*16]
   - 加载 B 的 tile 到 SMEM：B[m*16:(m+1)*16, col]
   - Block 内所有线程共享这两个 tile，实现数据重用

3. **算术强度提升**：
   ```
   朴素版：  1 FLOP 需要 2 次全局内存读取 → 0.5 FLOP/read
   Tiling：  16 FLOP 需要 2 次全局内存读取 → 8 FLOP/read
   ```

4. **栅栏同步**：
   - `__syncthreads()` 确保：
     - 所有线程完成加载后再计算
     - 所有线程完成计算后再加载下一个 tile
   - 防止竞态条件

---

## 第四部分：PTX 和 SASS（编译栈）

### Parallel Thread Execution（PTX）

**PTX** 是 NVIDIA 的中间表示 (Intermediate Representation)，类似 LLVM-IR。

**特点**：
- 源代码级：相对可读的"虚拟 GPU 汇编"
- 向前兼容：旧的 PTX 代码能在新 GPU 上通过 JIT 编译运行
- 线程模型清晰：PTX 显式表示线程、Warp、Block 的概念

**为什么写 PTX？**

DeepSeek-V3 的例子：
- KV Cache 通信需要特殊的 warpgroup 级指令 `wgmma.mma_async`
- 高级语言（CUDA C++）在 Hopper 之前还不支持这些指令
- 因此需要**内联 PTX**（手写 PTX 片段嵌入 C++ 代码中）

```ptx
// 示例：寄存器声明
.reg .f32 %f<7>;  // 声明 7 个 32 位浮点寄存器

// 示例：融合乘加
fma.rn.f32 %f5, %f4, %f3, 0f3FC00000;
//          ↑    ↑    ↑    ↑
//         结果  操作数1 操作数2 常数

// 示例：获取线程索引（用于计算 Warp Rank）
mov.u32 %r1, %ctaid.x;    // 获取 Cooperative Thread Array ID
mov.u32 %r2, %ntid.x;     // 获取 CTA 中线程数
mov.u32 %r3, %tid.x;      // 获取线程索引
```

### Streaming Assembler（SASS）

**SASS** 是 GPU 汇编，最接近硬件。

**特点**：
- 架构相关：SM90a (Hopper) 的 SASS 与 SM89 (Lovelace) 的不同
- 最低级可读代码：通常由编译器生成，很少手写
- 用于性能调试：通过 NVIDIA Nsys/Nsight 查看生成的汇编来优化

---

## 与已学内容的关联

### 1. DeepSeek-V3 的 PTX 优化

**背景**：DeepSeek-V3 在推理时使用自定义 PTX 进行 KV Cache 通信。

**为什么需要？**
- 标准 CUDA C++ 不支持 Hopper 新指令（如 `wgmma` 等）
- **Warpgroup 级同步** 比 Block 级同步更高效
  - Block 级：`__syncthreads()` 需要 256 个线程同步 → 开销大
  - Warpgroup 级：仅 128 个线程，同步开销 1/2

**实现**：
```
通过内联 PTX：
1. 编写 warpgroup 级的 mma_async 指令
2. 消除显式 inter-warp 同步
3. 减少通信延迟
→ 更高的吞吐和延迟隐藏
```

### 2. Flash Attention 的内存优化

**核心想法**：使用 Tiling + Shared Memory 减少 HBM 访问。

**对标我们的理解**：
```
朴素注意力：
  Q, K, V ← HBM（大量读取）
  计算注意力
  结果 → HBM
  算术强度低

Flash Attention（Tiling）：
  将 Seq Len 分块
  ┌─ Block 0
  │  Q_tile, K_tile, V_tile ← SMEM
  │  计算（Block 内所有线程共享数据）
  │  结果 → SMEM
  │  结果 → HBM
  └─ Block 1, 2, ...

  效果：
  - SMEM 带宽 (1-2 TB/s) vs HBM (141 GB/s)
  - SMEM 访问快 ~10-100 倍
  - 算术强度大幅提升
```

### 3. 计算能力（Compute Capability）与新指令

**概念映射**：
- **CC 9.0a (Hopper)**：引入 `wgmma`, `tma` 指令
  - Warpgroup 矩阵乘法异步加载
  - Tensor Memory Accelerator（用于数据搬运）
  - DeepSeek-V3 充分利用这些新指令

- **CC 10.0f (Blackwell)**：进一步优化
  - 更高的 Tensor Core 吞吐
  - 更灵活的 Warpgroup 编程模型

**启示**：理解 CC 版本 → 理解可用指令 → 理解性能上限

---

## 关键概念总结

| 概念 | 层级 | 大小 | 内存 | 同步方式 |
|------|------|------|------|---------|
| Thread | 最小 | 1 | 寄存器 | N/A |
| Warp | 执行单位 | 32 | 寄存器 | SIMT（隐式） |
| Warpgroup | 新抽象 | 128 | 寄存器 | 支持异步操作 |
| Block | 同步单位 | ~256 | 共享内存 | `__syncthreads()` |
| Grid | 全 GPU | ~1000 | 全局内存 | 原子操作 |

---

## 编译栈概览

```
CUDA C/C++ 代码
    ↓
NVCC 编译器
    ├→ PTX 中间表示（可读，向前兼容）
    │   ↓ JIT 编译
    └→ SASS 汇编（架构相关，执行）
        ↓
    GPU 硬件执行
```

**何时手写 PTX**：
- 需要特定硬件指令（Hopper 的 `wgmma`）
- 自动编译器无法生成所需的指令组合
- 需要极致性能优化

---

## 学习资源

- NVIDIA CUDA C++ Programming Guide
- Lindholm et al., 2008: Tesla GPU 原始论文
- Modal Labs GPU Glossary（本笔记的来源）
- Flash Attention 论文系列（注意力优化实例）
- Colfax CUTLASS Tutorial（Warpgroup 矩阵乘法）
