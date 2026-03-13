---
title: GPU 性能分析：从 Roofline 到优化实战
tags: [learning, gpu, performance]
created: 2026-03-12
---

# GPU 性能分析：从 Roofline 到优化实战

## 核心概念导航

GPU 性能优化的本质就是在三个维度的竞争中找到制约因素：**算力**、**带宽**、**开销**。

---

## Part 1: Roofline 框架 - 性能天花板

### 1.1 Roofline 模型概述

Roofline 是一个二维图表，用来快速判断 kernel 受何种资源限制：
- **横轴**：Arithmetic Intensity（AI，操作数/字节数）
- **纵轴**：性能（FLOPs/s）
- **两道天花板**（roofs）：形成一个带角的"屋顶"

```
                    ↑ 性能 (FLOPs/s)
                    |         __________  Compute Roof
                    |        /
                    |       /   Memory-Bound  |  Compute-Bound
                    |      /__________________|
                    |   Memory Roof (slope = mem_bw)
                    └─────────────────────────→ Arithmetic Intensity
```

**Ridge Point**（脊点）：两道屋顶的交点，是从内存限制跃升到算力限制的临界点。

### 1.2 两道屋顶的含义

**Compute Roof（算力屋顶）**
- 高度 = Peak Arithmetic Bandwidth（操作数/秒）
- 对于 H100：Tensor Core FP32/TF32 约 989 TFLOPS
- 对于 B200：BF16 约 2250 TFLOPS，FP8 可达 4500 TFLOPS

**Memory Roof（内存屋顶）**
- 斜率 = Memory Bandwidth（字节/秒）
- H100 HBM3：3.35 TB/s
- B200 HBM3e：8 TB/s
- Shared Memory：数倍于 HBM 带宽
- Register：最快，但容量有限（H100 单 SM 约 65K × 32bit）

### 1.3 Ridge Point 的计算

```
Ridge Point = Compute Bandwidth / Memory Bandwidth
           = 算力(TFLOPS) / 内存(TB/s)

H100 示例：989 TFLOPS / 3.35 TB/s = 295 FLOPs/byte
B200 示例：2250 TFLOPS / 8 TB/s = 281 FLOPs/byte
```

**启示**：Ridge Point 随着 GPU 代际而提高，这正是"内存墙"的体现——算力增长快于内存带宽增长。

---

## Part 2: 算力与内存限制

### 2.1 Compute-Bound 内核（高 AI）

**定义**：AI > Ridge Point，核心受算力限制。

**特征**：
- 大矩阵乘法（GEMM）AI = O(N)，趋势优势明显
- 每字节做很多运算
- 瓶颈是 Tensor Core/CUDA Core 的吞吐

**LLM 推理关键案例：Prefill 阶段**
- Prefill（处理 prompt）是 compute-bound
- 原因：权重 W（1 TB for 500B params）只需加载一次，却被 prompt 中所有 token 使用
- 多 token 场景中 AI 很高，数值上：
  - 权重 1 TB，prompt 1K token
  - FLOPs = 1K × 2 × 500B ≈ 1e18
  - Bytes = 1 TB ≈ 1e12
  - AI = 1e6 >> Ridge Point

### 2.2 Memory-Bound 内核（低 AI）

**定义**：AI < Ridge Point，核心受内存带宽限制。

**特征**：
- 向量化操作：SAXPY (y = ax+y) AI = O(1)
- 每字节几乎不做运算
- 瓶颈是从 HBM 取数据

**LLM 推理关键案例：Decode 阶段**
- Decode（生成每个 token）是 memory-bound
- 原因：权重必须逐次全部加载，但每次只处理一个 token
- 数值上：
  - 权重 1 TB 每次全部加载
  - FLOPs/token = 2 × 500B = 1e12
  - Bytes = 1 TB = 1e12
  - AI = 1 << Ridge Point

---

## Part 3: 延迟隐藏与 Occupancy

### 3.1 Little's Law - 并发的黄金法则

```
required_concurrency = latency (cycles) × throughput (ops/cycle)
```

**应用**：GPU 通过大量并发 warp 来隐藏长延迟。

| 场景 | 内存延迟 | 吞吐 | 需要 Warp 数 |
|------|--------|------|-----------|
| 内存访问 | 400 cycles | 1 op/cycle | 400 |
| Tensor Core MMA | 12 cycles | 10 ops/cycle | 120 |

**关键洞察**：内存延迟高，但带宽也很低，实际需要的 warp 数与算术延迟接近。这是吞吐导向 GPU 的美妙设计。

### 3.2 Occupancy - 活跃 Warp 比例

**定义**：Active Warps / Max Warps per SM

H100 H100 规格：
- Max 64 warps/SM（64 × 32 = 2048 threads）
- Max 32 blocks/SM
- Shared Memory：228 KB
- Registers：65536 × 32-bit

**示例计算**：
- 32 threads/block, 8 regs/thread, 12 KB shared mem/block
- Shared mem limit：228KB / 12KB = 19 blocks
- 实际可运行：19 blocks = 19 warps（远低于 64）

**关键认知**：
- Occupancy 不是目标本身，它只是**隐藏延迟的必要条件**
- 一旦足够隐藏延迟（通常 50%+ occupancy），增加 occupancy 可能反而降性能
  - 原因：Register Pressure 加重，减少每 thread 可用资源
  - 高性能 GEMM kernel 常常运行在**个位数百分比 occupancy**

### 3.3 Warp Execution States

四种状态（非互斥）：

1. **Active**：kernel 开始到结束的生命周期
2. **Eligible**：ready to issue instruction（无依赖阻塞）
3. **Stalled**：等待某个依赖（记录在 Scoreboard）
4. **Selected**：本周期被 warp scheduler 选中执行

**Scoreboard Stalls 分类**：
- **Short Stalls**（同 SM 内）：shared memory、Tensor Core MMA、特殊函数
- **Long Stalls**（跨 SM 的内存）：global memory load/store
  - Memory-bound kernel 的主要开销

---

## Part 4: 内存优化 - 共享内存与全局内存

### 4.1 Memory Coalescing（全局内存对齐访问）

**问题**：DRAM 物理一次访问会抓取**128 字节**的连续数据（cache line），但如果逻辑请求离散，就会浪费。

**最优情况**：
- 32 threads in warp，每个 thread 读 1 × float (4 bytes)
- 连续地址：`float v = data[tid]`
- 完美对齐：所有 32 个逻辑请求正好填满 1 × 128B 物理事务

**访问步长的影响**（Tesla T4 实测）：

| Stride | 吞吐 (GB/s) | 效率 |
|--------|-----------|------|
| 1      | 206.0     | 100% |
| 2      | 130.5     | 63%  |
| 4      | 68.8      | 33%  |
| 16     | 16.8      | 8%   |

**关键**：步长为 2 时吞吐减半，因为 DRAM burst 数翻倍。

### 4.2 Bank Conflicts（共享内存串行化）

**共享内存组织**：
- 32 个 bank，每 bank 4 bytes
- 连续的 32-bit word 映射到连续的 bank
- Addresses 差 128 bytes = 32 × 4 映射到同一 bank

**冲突案例**：列访问
```cpp
float value = data[tid * 32];  // 所有 32 threads 都访问 Bank 0！
// 导致 32x 延迟增加：10 cycles → 几百 cycles
```

**解决**：
- **Padding**：行数改为 33 而非 32
- **Transpose**：在 shared memory 中倒置数据
- **Bank format selection**（Hopper+ 支持多种 bank 宽度）

---

## Part 5: 执行效率 - Warp 动力学

### 5.1 Warp Divergence（分支分散）

**问题**：同一 warp 中的 32 threads 必须同步执行同一指令，但有条件分支时怎么办？

**编译器解法**：Predication（谓词执行）
```nasm
FSETP.GT.AND P0, PT, R4, 0.5   // 设置 P0 = (R4 > 0.5)
FADD R0, R4, 2                  // 所有 threads 执行（存储 R0+2）
@P0 FMUL R0, R4, 4              // 只有 P0=true 的 threads 执行（覆写 R0）
```

**成本**：被屏蔽的 threads 不执行真实计算，但指令被"发射"，浪费吞吐。

**Branch Efficiency**：度量 uniform control flow 的占比。`if (idx < n)` bounds check 通常高效，因为大多数 warp 内所有 threads 同时越界或不越界。

### 5.2 Register Pressure（寄存器压力）

**问题**：Registers 有限（H100: 65536 per SM），当一个 thread block 用太多 registers，能加载的 blocks 数减少，occupancy 下降。

**量化**：
- 如果 kernel 用 128 regs/thread
- Block size 128 threads
- Registers needed = 128 × 128 = 16384
- Max blocks = 65536 / 16384 = 4（远低于 32 limit）
- 实际 occupancy = 4 blocks × 128 threads / 2048 = 25%

**权衡**：
- 更多 registers → 更好的算术效率（减少内存访问）
- 但 occupancy 下降 → 难以隐藏延迟
- **Hopper+ 新特性**：Asynchronous Copies、TMA 通过绕过 registers 缓解此压力

### 5.3 Issue Efficiency（发射效率）

**定义**：每个周期发射指令的比例（最高 100%）。

**低 Issue Efficiency 的原因**：
- Eligible warps 不足（occupancy 太低）
- 所有 warps 都 stalled（等依赖、内存）

**诊断**：
- Issue Efficiency < 50%：大量 stall，检查 scoreboard stalls
- Issue Efficiency = 100% with low pipe util：可能是内存 bottleneck 但已隐藏延迟

---

## Part 6: 性能诊断决策树

```
┌─ kernel 性能不达预期
│
├─ 第一层：确定制约
│  │
│  ├─ Roofline 分析
│  │  ├─ AI < Ridge Point？→ Memory-Bound
│  │  │   ├─ Coalescing 是否完美？
│  │  │   │   ├─ No → 调整访问模式
│  │  │   │   └─ Yes → 下一层
│  │  │   └─ Memory Pressure
│  │  │       ├─ Long Scoreboard Stalls 高？→ 掩盖（增加 warp）
│  │  │       └─ L2 miss 率高？→ 数据局部性优化
│  │  │
│  │  └─ AI > Ridge Point？→ Compute-Bound
│  │      ├─ Pipe Utilization 低？→ 下一层
│  │      └─ Issue Efficiency 低？→ 检查 warp divergence / register pressure
│  │
│  └─ 都不是？→ Overhead-Bound
│      ├─ Kernel launch 开销（CUDA API ~10μs）
│      └─ Host-device 同步
│
└─ 第二层：瓶颈升级
   │
   ├─ Memory Bottleneck Lifting
   │  ├─ 增加 AI：gradient checkpointing、数据压缩
   │  └─ 改进带宽使用：Shared Memory tiling、Coalescing
   │
   ├─ Compute Bottleneck Lifting
   │  ├─ 增加 operand reuse（tile 更大）
   │  ├─ 减少 warp divergence
   │  └─ 降低 register pressure
   │
   └─ 重复直到 occupancy/latency hiding 充足
```

---

## Part 7: 与已学内容的关联

### 7.1 Cheatsheet §十：Roofline 基础

你的笔记可能已覆盖 Roofline 的概念图和 Ridge Point 的计算。本笔记扩展了：
- **硬件原理**：为什么会出现两道屋顶（Moore's Law vs. Dennard Scaling vs. Memory Wall）
- **多层级内存**：不同 subsystem 的 roofline（Registers > L1 > L2 > Shared Mem > HBM）
- **数值示例**：H100/B200 的真实参数和 AI 计算

### 7.2 LLM 推理：Prefill vs. Decode

**Prefill（Compute-Bound）的硬件根源**：
- 权重加载一次，被多个 token 摊销
- AI = (# tokens × 2 × params) / params_bytes = 2 × # tokens >> Ridge Point
- 在 batch prefill 中，AI 可轻易达到数百，远超 H100 Ridge Point 295

**Decode（Memory-Bound）的硬件根源**：
- 每生成一个 token，权重必须全部加载
- 即使 batch decode，每个样本仍需加载完整权重
- 要转变为 compute-bound，需要 speculative decoding（预生成多个 token）或 multi-token prediction

### 7.3 DeepSeek-V3 的 52% MFU：剩余 48% 去哪了？

Model FLOPs Utilization (MFU) = Achieved TFLOPS / Peak TFLOPS

**52% 意味着什么**：
- 理论 Peak：H100 Tensor Core 989 TFLOPS （FP32 精度）
- 实际：989 × 0.52 ≈ 514 TFLOPS

**48% 的损失来源**（根据 roofline + warp dynamics）：
1. **不是算力计算本身的损失**（Tensor Core 完全饱和不现实）
2. **占主导**：
   - Stall Cycles：long scoreboard stalls（内存等待），约 15-20%
   - Pipeline Underutilization：warp divergence/control flow，约 5-10%
   - Register Pressure & Synchronization：约 10-15%
3. **其他**：
   - Host overhead（kernel launch）
   - Load imbalance（最后 warp 提前完成）
   - 内存带宽有限导致的 compute starvation（虽然 prefill 应该不是）

**优化方向**（涉及 §十 以上的概念）：
- 增加 occupancy 以隐藏 long stalls
- 使用 async copy / TMA 减少 register pressure
- 在 SRAM（shared mem）中 tile 计算以增加数据复用

### 7.4 FlashAttention：内存优化的典范

**核心思路**：将 Attention 从 "HBM-limited" 转变为 "compute-bound"。

**Tiling into SRAM**：
```
标准 Attention：
  Q, K, V ∈ HBM （大矩阵，内存坏）
  HBM reads: seq_len² × d × 3 （平方复杂度！）

FlashAttention：
  分块 tile Q, K, V 到 Shared Memory（例如 4KB per tile）
  每个 SM 本地计算 tile 内的 attention
  HBM reads：线性，只需读一遍数据
```

**与 Coalescing 和 Bank Conflicts 的关系**：
- **Memory Coalescing**：Flash Attention 的 SRAM 加载/存储设计完全对齐到 warp，每个 thread 读连续的 float
- **Bank Conflicts**：shared memory 中 Q·K^T 的矩阵乘法，若按行主序存放 Q，可能出现列访问冲突
  - 解决：转置或 padding 列数到 33（而非 32）
  - 或使用 bank format swapping（Hopper+ 功能）

**AI 的改变**：
```
标准：AI = FLOPs / (3 × seq_len × d × 2) [HBM bytes]
      = O(seq_len² × d) / (6 × seq_len × d) = O(seq_len/6) [低]

FlashAttention：AI 增加因为 SRAM 读写廉价
      = O(seq_len² × d) / (小 SRAM traffic + HBM O(seq_len × d))
      = O(seq_len) [高得多]
```

---

## 总结：性能优化的三环论

```
     ┌─────────────────┐
     │  Architecture   │
     │  (Roofline)     │
     └────────┬────────┘
              │
              ↓
     ┌─────────────────┐
     │   Scheduling    │
     │  (Occupancy,    │
     │   Latency Hide) │
     └────────┬────────┘
              │
              ↓
     ┌─────────────────┐
     │  Optimization   │
     │  (Memory, Compute)
     └─────────────────┘
```

1. **Architecture**：理解 roofline、两道屋顶、ridge point
2. **Scheduling**：通过 occupancy、Little's Law、warp states 隐藏延迟
3. **Optimization**：精准定位 bottleneck（compute/memory），逐步升级，不断迭代

GPU 性能优化的最高境界是**用尽可能少的硬件资源，达到 roofline 天花板**。Hopper 及以后的新特性（async copy、TMA、independent thread scheduling）都在朝这个目标进化。
