---
tags: [learning, gpu, hardware]
created: 2026-03-06
source: "GPU Mode Lecture 4 - Compute and Memory Basics (Thomas Viehmann), PMPP Book Ch4+5"
---

# GPU 架构基础：从硬件理解 AI Infra 的性能瓶颈

> 你不需要会写 CUDA kernel，但需要理解 GPU 的执行模型和内存层次。这决定了你能不能看懂 FlashAttention 为什么快、vLLM 为什么要做 PagedAttention、以及为什么 Decode 是 memory-bound。

---

## 一、CPU vs GPU：设计哲学的根本区别

先理解一个核心问题：**为什么 GPU 比 CPU 快那么多？**

答案不是"GPU 更强"，而是**设计取舍不同**：

```
CPU 的设计目标：让单个线程跑得尽可能快
├── 大缓存（Cache）→ 减少等待数据的时间
├── 乱序执行（Out-of-order）→ 指令不用排队
├── 分支预测（Branch predictor）→ 猜下一步干什么
└── 结果：单线程极强，但核心数少（几十个）

GPU 的设计目标：让成千上万个线程同时跑
├── 小缓存，把面积给了更多 ALU（计算单元）
├── 没有乱序执行、没有分支预测
├── 靠"线程多"来掩盖等待时间（latency hiding）
└── 结果：单线程很弱，但总吞吐量巨大
```

**类比**：CPU 像一个超级大厨（一个人做满汉全席），GPU 像1000个实习生（每人只会切一种菜，但同时切）。矩阵乘法、Attention 计算这类任务，恰好是"每人切一种菜"类型的——所以 GPU 碾压 CPU。

---

## 二、GPU 硬件架构：从芯片到线程

### 2.1 整体结构：SM 是 GPU 的基本"工厂"

一块 GPU 芯片由多个 **SM（Streaming Multiprocessor，流式多处理器）** 组成，所有 SM 共享一块 **Global Memory（全局显存，即 HBM）**。

```
GPU 芯片
├── SM 0  ─┐
├── SM 1   │  每个 SM 是独立的计算单元
├── SM 2   │  有自己的 ALU、寄存器、共享内存
├── ...    │
├── SM N  ─┘
├── L2 Cache（所有 SM 共享）
└── Global Memory / HBM（80GB on H100）
    ↑ 带宽瓶颈就在这里：3.35 TB/s (H100)
```

**不同 GPU 的 SM 数量**：

| GPU      | SM 数量 | 架构代号           |
| -------- | ----- | -------------- |
| RTX 3090 | 82    | Ampere (GA102) |
| A100     | 108   | Ampere (GA100) |
| H100 SXM | 132   | Hopper (GH100) |
| B200     | 192   | Blackwell      |

> SM 数量越多 → 能同时处理的 thread block 越多 → 总吞吐越大。这就是为什么 H100 比 A100 快——不仅是频率提升，更是 SM 数量从 108 增加到 132。

### 2.2 SM 内部结构：寄存器、ALU、共享内存

每个 SM 内部有这些关键组件（以 Ampere GA10x 为例）：

```
一个 SM（实际分成 4 个 Processing Block）
├── 4 × Warp Scheduler（每个调度一个 warp）
├── 4 × 32 个 FP32 ALU = 共 128 个 FP32 计算单元
│   （其中一半也支持 INT32）
├── 4 × Tensor Core（专门做矩阵乘法，BF16/FP8 极快）
├── 4 × 16,384 个 32-bit 寄存器 = 共 65,536 个寄存器
│   （寄存器是线程私有的，速度最快）
├── 128 KB 可配置为 L1 Cache + Shared Memory
│   （比如 shared memory 用 64KB，L1 用 64KB）
└── Load/Store 单元（负责读写 Global Memory）
```

**关键数字（每个 SM）**：

| 资源 | 数量 | 说明 |
|------|------|------|
| FP32 ALU | 128 个 | 每个时钟周期可做 128 次浮点运算 |
| Tensor Core | 4 个 | 矩阵乘法加速器，H100 支持 FP8 |
| 寄存器 | 65,536 × 32bit | 所有线程共享这个寄存器池 |
| 最大线程数 | 1,536（有些GPU是2,048） | 同时驻留在 SM 上的线程 |
| Shared Memory | 最大 100KB（可配置） | SM 内所有线程可共享 |
| L1 Cache | ≥ 28KB（和 shmem 共享 128KB） | 自动缓存 |

### 2.3 FP64 陷阱：消费级 GPU 的致命慢速区

消费级 GPU（RTX 系列）的 FP64 算力只有 FP32 的 **1/64**——每个 SM 只有 2 个 FP64 单元（vs 128 个 FP32）。这些单元纯粹是为了"让 FP64 代码能跑"，而不是为了跑得快。

**实际影响**：如果代码里不小心用了 64 位浮点常量（比如 Python 的 `float` 默认是 FP64），会触发 FP64 计算路径，性能暴跌 64 倍。Thomas Viehmann 在讲座中提到，他曾因把 PyTorch 二项式 kernel 的索引从 INT32 改成 INT64，导致性能大幅下降。

> **数据中心 GPU（A100/H100）** 的 FP64 算力正常（A100 FP64 = 9.7 TFLOPS，约为 FP32 的 1/2），但 LLM 场景基本用不到 FP64。

### 2.4 为什么寄存器和 Shared Memory 如此重要？

这是理解 GPU 性能优化的核心：

```
内存层次（速度从快到慢）：

寄存器（Register）    → 线程私有，0 延迟，~20 TB/s
  ↓
Shared Memory (SRAM) → 同一 block 内共享，~19 TB/s，128KB/SM
  ↓
L1/L2 Cache          → 自动缓存，L2 全芯片共享（H100: 50MB）
  ↓
Global Memory (HBM)  → 所有线程可见，3.35 TB/s，80GB
  ↓
CPU 主存 (DRAM)      → 需要过 PCIe，12.8 GB/s，>1TB
```

**速度差距惊人**：SRAM（Shared Memory）的带宽是 HBM 的 **~6倍**（19 vs 3.35 TB/s），但容量只有 HBM 的 **~六万分之一**（128KB vs 80GB）。

> **这就是 FlashAttention 的核心洞察**：标准 Attention 把中间结果（N×N 的 attention matrix）写到 HBM 再读回来，FlashAttention 把计算拆成小块（tiling），让中间结果始终留在 SRAM 里，避免 HBM 往返。省的就是这 6 倍带宽差距中的无谓搬运。

---

## 三、CUDA 执行模型：Grid → Block → Warp → Thread

你不需要写 CUDA 代码，但需要理解这套执行层次，因为所有 AI Infra 论文都会提到。

### 3.1 四层结构

```
Kernel（一次 GPU 函数调用）
└── Grid（一个 kernel 启动产生的所有线程）
    └── Block（线程块，被分配到一个 SM 上执行）
        └── Warp（32 个线程，GPU 实际调度的最小单位）
            └── Thread（单个线程）
```

- **Kernel**：这次任务

- **Grid**：任务的全部工人

- **Block**：分给一个车间的一组工人

- **Warp**：车间里每次一起干活的小组

- **Thread**：单个工人

**具体数字举例**：假设你要对一个 1M 元素的向量做 ReLU：

```
1M 个元素 → 需要 1M 个线程
每个 Block 有 256 个线程（你选的 block size）
→ 需要 1M / 256 = 4,096 个 Block
→ 这 4,096 个 Block 组成一个 Grid
→ GPU 把 Block 分配给 SM（H100 有 132 个 SM）
→ 每个 SM 同时处理若干 Block
```

### 3.2 核心规则

**规则 1：Block 被整体分配给 SM，不可拆分**
- 一个 Block 内的所有线程在同一个 SM 上执行
- Block 内的线程可以通过 Shared Memory 通信和同步
- 不同 Block 之间**没有执行顺序保证**，也不能直接通信

**规则 2：Warp（32 线程）是实际执行单位**
- GPU 不是一个线程一个线程执行的，而是 **32 个一组（warp）同时执行同一条指令**
- 这叫 **SIMT（Single Instruction, Multiple Threads）**
- 类比：32 个士兵听一个口令齐步走

**规则 3：Block 的执行顺序不确定**
- GPU 硬件自行调度哪些 Block 先执行
- 同一个 kernel 的不同 Block 可能在不同 SM 上并行，也可能排队
- 这使得 CUDA 程序能**自动适配**不同 SM 数量的 GPU（透明扩展性）

### 3.3 Thread 的 ID 和线性化

CUDA 中线程可以用 1D、2D 或 3D 方式组织（方便映射到矩阵等数据结构），但硬件内部会把它们**线性化**后分成 warp：

```
2D Block (4×4) 的线性化顺序：
T(0,0) T(0,1) T(0,2) T(0,3) | T(1,0) T(1,1) ...
        行优先展开 → 线性 ID: 0, 1, 2, 3, 4, 5, ...

Warp 划分：线性 ID 0-31 = Warp 0，32-63 = Warp 1，...
```

> 这个线性化规则决定了**哪些线程在同一个 warp 里**——这直接影响内存访问效率（coalescing）和分支性能（divergence）。

---

## 四、Warp Divergence：为什么 if-else 在 GPU 上很贵

### 4.1 问题

同一个 warp 的 32 个线程必须执行同一条指令。如果代码里有 `if-else`，部分线程走 if、部分走 else，会发生什么？

```c
if (threadIdx.x < 4) {
    A;  // 线程 0-3 执行
    B;
} else {
    X;  // 线程 4-31 执行
    Y;
}
Z;  // 所有线程执行
```

### 4.2 旧架构（≤ Pascal）：串行执行两个分支

```
时间线：
[全部线程] → [分歧] → [只有 0-3 执行 A,B] → [只有 4-31 执行 X,Y] → [合流] → [全部执行 Z]
```

- 两个分支**串行**执行，走 if 时 else 的线程空闲（浪费）
- 总时间 = if 分支时间 + else 分支时间

### 4.3 新架构（≥ Volta/H100）：更灵活但仍有代价

Volta 起每个线程有独立的 Program Counter，分支后可以**交错执行**（比如先执行 A 和 X，再执行 B 和 Y），延迟隐藏更好。但需要手动调用 `__syncwarp()` 来重新同步 warp 内线程（不再自动合流）。

**最坏情况**：如果一个 warp 内 32 个线程走了 32 条不同的分支，性能损失高达 **31/32**——几乎只剩一个线程在干活。

### 4.4 规避分歧的技巧

**条件 load/store 不会触发分歧**：GPU 支持 predicated load/store 指令，类似 `cond ? x[i] : 0` 的模式，硬件会用 mask 控制哪些线程真正执行读写，不需要走分支路径。这是优化分支代码的首选方案。

**循环也会产生分歧**：如果 warp 内不同线程的循环次数不同（例如 `for(i=0; i<a[threadIdx.x]; i++)`），早结束的线程会空等最慢的那个。整个 warp 的执行时间取决于最慢线程。

> **对 AI Infra 的意义**：这就是为什么高效 kernel（如 FlashAttention、Triton 生成的代码）要尽量让同一个 warp 的线程走相同路径。Attention mask 的处理、padding 的处理，都需要考虑 divergence 开销。

---

## 五、Occupancy（占用率）：GPU 利用率的关键指标

### 5.1 什么是 Occupancy

```
Occupancy = SM 上实际驻留的 warp 数 / SM 最大能驻留的 warp 数
```

Occupancy 高 → 当一个 warp 在等待内存数据时，SM 可以切换到另一个 warp 执行 → **隐藏延迟（latency hiding）**。

### 5.2 什么限制了 Occupancy

三个资源会限制一个 SM 上能塞多少 Block/Warp：

**① 寄存器数量**
- SM 总共 65,536 个寄存器
- 如果你的 kernel 每个线程用 64 个寄存器，一个 256 线程的 Block 就需要 256 × 64 = 16,384 个寄存器
- 一个 SM 只能放 4 个这样的 Block → 1,024 个线程（最大1,536）
- 如果每个线程用 128 个寄存器？只能放 2 个 Block → 512 线程 → occupancy 骤降
- 更糟：如果寄存器不够分，会 **register spill**（溢出到 HBM） → 极慢

**② Shared Memory**
- 128KB 总量，kernel 申请越多 shared memory → 能同时驻留的 Block 越少
- FlashAttention 需要在 shared memory 里放 Q、K、V 的 tile → 需要仔细调整 tile 大小

**③ Block Size**
- 太小（比如 32 线程）→ 浪费调度开销
- 太大（比如 1024 线程）→ 一个 SM 只能放 1-2 个 Block → 灵活性差
- 经验值：**256 或 512 线程/Block**，且应该能**整除 SM 最大线程数**（1536），避免碎片
- 例：1536 / 256 = 6 个 Block/SM ✓，1536 / 1024 = 1.5 → 只能放 1 个 Block，浪费 33%

### 5.3 Occupancy 不是越高越好

高 occupancy 有助于隐藏延迟，但有时候低 occupancy + 充分利用寄存器和 shared memory 反而更快（比如 FlashAttention 故意用更多 shared memory 换取更少的 HBM 访问）。这是一个 **trade-off**。

### 5.4 实用工具

**查询硬件参数**（避免硬编码）：
```python
# PyTorch 查询 GPU 参数
props = torch.cuda.get_device_properties(0)
props.name                        # 'NVIDIA H100 80GB HBM3'
props.total_memory                # 显存大小
props.multi_processor_count       # SM 数量
props.max_threads_per_multi_processor  # 每 SM 最大线程数
props.regs_per_multiprocessor     # 每 SM 寄存器总数
```

**编译器提示**：CUDA 中可用 `__launch_bounds__(maxThreadsPerBlock)` 告诉编译器 kernel 的线程数量，辅助编译器优化寄存器分配，避免 register spill。

**分析工具**：NVIDIA **Nsight Compute** 可直接分析 occupancy、内存访问模式等，取代了早期的 Excel 计算表。

---

## 六、Memory Coalescing：内存访问模式决定性能

### 6.1 GPU 读 HBM 的方式

GPU 访问 Global Memory 不是按单个字节读的，而是 **按 32/128 字节的 cache line 整块读**。

```
好的访问模式（Coalesced，合并访问）：
Thread 0 读 addr[0], Thread 1 读 addr[1], Thread 2 读 addr[2]...
→ 32 个线程的请求落在连续地址 → 一次内存事务搞定

坏的访问模式（Strided，跳跃访问）：
Thread 0 读 addr[0], Thread 1 读 addr[128], Thread 2 读 addr[256]...
→ 32 个线程的请求分散在不同 cache line → 需要多次内存事务
→ 大量带宽浪费
```

### 6.2 对 AI Infra 的实际影响

**矩阵存储顺序（Row-major vs Column-major）** 直接决定了访问模式：

```
矩阵 A[M][N] 按行存储（Row-major，C/PyTorch 默认）：
A[0][0], A[0][1], A[0][2], ..., A[1][0], A[1][1], ...

如果 32 个线程分别处理一行的 32 个元素 → Coalesced ✓
如果 32 个线程分别处理一列的 32 个元素 → Strided ✗ → 慢很多
```

> 这就是为什么 Tensor Core 要求特定的数据布局（比如把矩阵切成 16×16 的 tile），以及为什么 DeepSeek-V3 的 MLA 要把 K、V 压缩成连续的低维 latent vector——除了省空间，内存访问模式也更友好。

### 6.3 各级内存的声明方式（CUDA 中）

不需要记住语法，但理解对应关系有助于读懂论文中的优化描述：

| 内存类型 | CUDA 声明 | 生命周期 | 典型用途 |
|---------|----------|---------|---------|
| 寄存器 | 局部变量（自动分配） | 线程 | 高频访问的临时变量 |
| Local Memory | 数组变量（编译器自动溢出） | 线程 | 寄存器放不下的数据，实际存在 HBM，很慢 |
| Shared Memory | `__shared__` 修饰符 | Block | Tiling 的核心，FlashAttention 的主战场 |
| Global Memory | 指针参数传入 kernel | 应用 | 模型权重、KV Cache、输入输出 |
| Constant Memory | `__constant__` 修饰符 | 应用 | kernel 启动参数，有专门缓存 |

> **关键认知**：变量一旦不在寄存器中，访问延迟会大幅上升。所以 kernel 优化的核心是尽量让高频数据留在寄存器和 Shared Memory 里。

---

## 七、Kernel Fusion：为什么 PyTorch Eager 模式慢

### 7.1 问题：每个操作都要访问一次 HBM

PyTorch 默认的 eager 执行方式，每个算子（op）都是独立的 kernel：

```
操作：y = relu(x + bias)

Eager 方式（两次 HBM 往返）：
1. kernel_add:  从 HBM 读 x, bias → 计算 x+bias → 写回 HBM
2. kernel_relu: 从 HBM 读 (x+bias) → 计算 relu → 写回 HBM

Fused 方式（一次 HBM 往返）：
1. kernel_fused: 从 HBM 读 x, bias → 计算 x+bias → 计算 relu → 写回 HBM
```

Fused 版本减少了一次 HBM 读写。如果是 10 个连续操作，差距可达 **数倍**。

### 7.2 Kernel Fusion 的演进

```
PyTorch JIT → NVFuser → torch.compile + Inductor/Triton
```

现在 `torch.compile()` 会自动做 kernel fusion，把连续的 pointwise 操作合并成一个 Triton kernel，大幅减少 HBM 访问。

### 7.3 FlashAttention 是 Kernel Fusion 的极致

标准 Attention 的计算流程：

```
标准方式：
Q×K^T → 写到 HBM（N×N 的矩阵！）→ 读回来 softmax → 写到 HBM → 读回来 × V → 写结果

FlashAttention：
把 Q、K、V 分块（tile）加载到 SRAM
在 SRAM 里完成 Q×K^T → softmax → ×V 的完整计算
只把最终结果写回 HBM
→ 省掉了 N×N attention matrix 的 HBM 读写
```

> **SRAM 带宽 19 TB/s vs HBM 带宽 3.35 TB/s**——FlashAttention 把计算搬到了"高速公路"上，所以对长序列能快 2-4 倍。

### 7.4 Fusion 带来的精度差异

Kernel fusion 可能导致微小的数值差异——这是**正常**的，不是 bug。原因是浮点加法**不满足结合律**：

```
(a + b) + c  ≠  a + (b + c)    （浮点数下可能不相等）
```

FP32 的相对精度约 1e-7，fusion 改变了运算顺序，最终结果会有这个量级的差异。如果对精度要求极高，需使用 FP64 或调整计算顺序。

### 7.5 实际收益：近似 Gelu 融合案例

Thomas Viehmann 在讲座中演示了近似 Gelu（多个 pointwise 操作组合）的融合效果：手写融合 kernel 比 PyTorch eager 逐算子实现快 **7-8 倍**，甚至略快于 PyTorch 内置实现。

---

## 八、PyTorch 程序的时间花在哪里

在优化之前，先理解瓶颈在哪（引自 Thomas Viehmann 的经验法则）：

```
PyTorch 程序耗时分解：
├── Python 开销（解释器、对象创建等）
├── 数据管理开销（Tensor 分配、元数据处理）
├── 数据 I/O（磁盘→CPU→GPU 的数据搬运）
└── GPU 计算
    ├── Kernel Launch 固定开销（~3μs/次）
    ├── 内存访问（读输入/写输出到 HBM）  ← 通常是瓶颈
    └── 实际计算（FLOPs）
```

**实践法则**：
1. 如果 `nvidia-smi` 显示 GPU 利用率远低于 100% → 瓶颈在数据搬运或 Python 开销，先别优化 kernel
2. 只要 Tensor 有几百个元素以上，"Python 慢"的影响只是个位数百分比
3. 对于大模型推理（我们关心的场景），**内存访问几乎总是瓶颈**——这就是为什么 Cheatsheet 里的公式 3（Decode 吞吐 ≈ 带宽 / 权重大小）如此重要

### 8.2 "光速"测算法（Speed of Light）：你的 kernel 最快能多快？

任何 kernel 的理论最大速度由三部分决定，取最慢的那个：

```
理论耗时 = max(内存传输时间, 计算时间) + kernel 启动开销

其中：
  内存传输时间 = 总数据量 / HBM 带宽
  计算时间     = 总 FLOPs / GPU 峰值算力
  启动开销     ≈ 3μs（固定，无法优化）
```

**具体案例**（Thomas Viehmann 的 rgb2gray 示例）：

```
任务：将 2048×2048 RGB 图像转灰度
  数据量：读 3 bytes/pixel × 4M pixels = 12MB，写 1 byte/pixel × 4M = 4MB，共 16MB
  计算量：5 ops/pixel × 4M pixels = 20M FLOPs

RTX 3090 上的理论耗时：
  内存传输：16 MB / 900 GB/s  ≈ 18μs  ← 瓶颈！
  计算：    20M FLOPs / 35.6 TFLOPS ≈ 0.6μs
  启动开销：~3μs
  理论极限 ≈ 18 + 3 = 21μs

实测：~27μs → 达到理论上限的 ~78%，优化空间极小
```

> 这个方法（业界称为"Speed of Light"分析）是判断一个 kernel 是否还有优化空间的标准做法。如果实测已经接近理论上限，就不值得继续优化这个 kernel 了。

---

## 九、Tiling（分块优化）：FlashAttention 的直接前置知识

### 9.1 问题：朴素矩阵乘法的访存灾难

矩阵乘法 C = A × B（n×n），朴素实现中每个输出元素需要读取 A 的一整行和 B 的一整列：

```
for (每个输出元素 C[i][j]) {
    for (k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];  // 读 A 和 B 各 n 次
    }
}
```

**每个输入元素被从 Global Memory 读了 n 次**。n = 4096 时，这意味着同一个数据被反复从 HBM 搬了 4096 遍——极度浪费带宽。

### 9.2 解决方案：把数据分块搬到 Shared Memory 复用

核心思路：既然同一个数据会被多个线程反复使用，不如**先一起搬到 Shared Memory（快 6 倍），再从 Shared Memory 反复读取**。

```
分块大小 T = 16 的矩阵乘法：

for (每个 T×T 的分块 phase) {
    ① 所有线程协作，把 A 的一个 T×T 块加载到 Shared Memory
    ② 所有线程协作，把 B 的一个 T×T 块加载到 Shared Memory
    ③ __syncthreads()  ← 同步！确保所有数据加载完毕
    ④ 每个线程从 Shared Memory 读数据做计算
    ⑤ __syncthreads()  ← 再次同步！确保计算完毕再加载下一块
}
```

**效果**：每个输入元素的 Global Memory 读取次数从 n 次降低到 **n/T** 次，减少 T 倍。

```
T = 16 时：访存量减少 16 倍
T = 32 时：访存量减少 32 倍（但需要更多 Shared Memory）
```

### 9.3 为什么需要 `__syncthreads()`

Shared Memory 是 Block 内所有线程**共享**的。如果线程 A 还没把数据加载完，线程 B 就开始读，会读到垃圾数据。所以每次加载和计算之间必须设置**同步屏障**：

```
加载 → __syncthreads() → 计算 → __syncthreads() → 加载下一块 → ...
        所有人加载完了            所有人计算完了
        才能开始计算              才能加载下一块
```

两次同步都不能省：第一次确保读到正确数据，第二次确保没有人还在用上一轮的数据。

### 9.4 非对齐矩阵的处理

如果矩阵尺寸不是 T 的整数倍，超出边界的线程写 0（padding），计算逻辑不变。

### 9.5 性能收益

Thomas Viehmann 的测试结果：16×16 分块矩阵乘法比朴素实现快 **~25%**（900μs → 700μs）。通过进一步增大 tile、每个线程计算多个元素（thread coarsening），还能继续提升。

> **这就是 FlashAttention 的直接前身**：FlashAttention 把 Q、K、V 矩阵分成小 tile，加载到 Shared Memory 中完成 attention 计算，本质上就是矩阵乘法 tiling 在 attention 场景的推广。理解了 tiling，你就理解了 FlashAttention 为什么要分块、为什么能快。

---

## 十一、Tensor Core：矩阵乘法加速器

### 9.1 为什么需要 Tensor Core

普通 FP32 ALU 每个时钟做 1 次乘加（FMA）。Tensor Core 每个时钟做一个**小矩阵乘法**（例如 4×4×4 或 16×8×16），吞吐量高几十倍。

| GPU | FP32 (普通ALU) | BF16 Tensor Core | FP8 Tensor Core |
|-----|---------------|-------------------|-----------------|
| A100 | 19.5 TFLOPS | 312 TFLOPS | — |
| H100 | 67 TFLOPS | 990 TFLOPS | 1,979 TFLOPS |

> H100 的 BF16 Tensor Core 算力是 FP32 的 **15 倍**。这就是为什么 LLM 训练和推理都用 BF16/FP8——不只是省显存，更是因为 Tensor Core 才是真正的算力来源。

### 9.2 使用 Tensor Core 的条件

Tensor Core 需要特定的数据布局和对齐方式：
- 矩阵维度必须是 8 或 16 的倍数
- 数据类型必须是 BF16/FP16/FP8/INT8
- 需要通过特定 API（cuBLAS、CUTLASS、Triton）调用

> 这就是为什么 LLM 架构设计时，hidden dimension、head dimension 都选 128 的倍数——为了对齐 Tensor Core 的要求，榨取最大算力。

---

## 十二、总结：GPU 知识和 AI Infra 论文的连接

| GPU 概念 | 对应的 AI Infra 优化 | 论文/系统 |
|---------|-------------------|---------|
| SM 并行 → 需要足够多的 Block | Continuous Batching 增加并行度 | Orca |
| HBM 带宽瓶颈 | PagedAttention 减少 KV Cache 碎片 | vLLM |
| SRAM vs HBM 速度差 | Tiling + 在 SRAM 里完成 Attention | FlashAttention |
| Tensor Core 需要对齐 | Hidden dim = 128 的倍数 | 所有主流 LLM |
| Kernel Launch 开销 | CUDA Graph 批量提交 kernel | vLLM, SGLang |
| 内存访问模式（Coalescing） | MLA 压缩 KV 为连续 latent vector | DeepSeek-V3 |
| Kernel Fusion | torch.compile / Triton 编译优化 | Inductor, Triton |
| Tiling（分块到 SRAM） | 矩阵乘法分块 → Attention 分块 | FlashAttention |
| Occupancy / Latency Hiding | 调整 block size 和 shared memory | FlashAttention-2 |
| FP64/INT64 陷阱 | 确保用 BF16/FP32，避免意外降速 | 所有 kernel 开发 |

---

## 附录：名词对照表

| 英文 | 中文 | 一句话解释 |
|------|------|---------|
| SM (Streaming Multiprocessor) | 流式多处理器 | GPU 的基本计算单元，类比 CPU 核心 |
| Warp | 线程束 | 32 个线程，GPU 调度的最小单位（AMD 称 Wavefront，默认 64 线程） |
| Thread Block | 线程块 | 被分配到一个 SM 上的一组线程 |
| Grid | 网格 | 一次 kernel 调用产生的所有 Block |
| Register | 寄存器 | 线程私有，速度最快的存储 |
| Shared Memory (SHMEM) | 共享内存 | SM 内 Block 共享的 SRAM |
| Global Memory (HBM) | 全局显存 | 所有线程可访问，就是"80GB 显存" |
| Coalescing | 合并访问 | 相邻线程访问相邻地址，一次内存事务 |
| Occupancy | 占用率 | SM 上活跃 warp 的比例 |
| Divergence | 分歧 | warp 内线程走不同分支，性能损失 |
| Tensor Core | 张量核心 | 矩阵乘法硬件加速器 |
| Kernel | 核函数 | 一次 GPU 函数调用 |
| Kernel Fusion | 算子融合 | 合并多个操作减少 HBM 访问 |
| Latency Hiding | 延迟隐藏 | 一个 warp 等待时切换到另一个 |
| Register Spill | 寄存器溢出 | 寄存器不够，数据溢出到慢内存 |
| SIMT | 单指令多线程 | GPU 的执行模型 |
| Tiling | 分块 | 将大数据拆成小块放入 SRAM 复用 |
| `__syncthreads()` | 线程块同步 | 确保 Block 内所有线程到达同一点 |
| `__syncwarp()` | Warp 同步 | Volta+ 架构中重新同步分歧的 warp |
| Speed of Light | 理论性能上限 | kernel 的理论最大速度，用于评估优化空间 |
| `__launch_bounds__` | 启动约束 | 告诉编译器线程数，优化寄存器分配 |
| Nsight Compute | 性能分析工具 | NVIDIA 官方 kernel 级性能分析器 |
| Local Memory | 本地内存 | 寄存器溢出的去处，虽名为"本地"但实际在 HBM，很慢 |
| Constant Memory | 常量内存 | 存放 kernel 启动参数，有专门缓存加速 |

---

*来源：GPU Mode Lecture 4 (Thomas Viehmann, 2024-02-03) + PMPP Book Ch4-5 + NVIDIA Ampere/Hopper Architecture Whitepaper*
*最后更新：2026-03-06（v2: 根据讲座语音转写补充 Tiling、FP64陷阱、Speed of Light、内存声明方式等内容）*
