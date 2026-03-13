---
title: "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"
tags: [inference-serving, attention, kernel-optimization, blackwell, gpu-architecture]
subfield: inference-serving
venue: "arXiv 2603.05451"
date: 2026-03-10
authors: [Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, Tri Dao]
institution: [Princeton University, Meta, Colfax Research, NVIDIA, Georgia Tech, Together AI]
url: "https://arxiv.org/abs/2603.05451"
status: 已读
rating: ⭐⭐⭐⭐⭐
---

# FlashAttention-4：算法与内核流水线协同设计，应对非对称硬件扩展

## 一句话总结

Blackwell GPU 的张量核心吞吐量翻倍但 SFU（指数运算）和 SMEM 带宽不变，瓶颈从 MMA 转移到了 softmax 和内存流量。FA4 通过重设计流水线 + 软件模拟 exp + 条件 softmax 重缩放（前向）和 TMEM + 2-CTA MMA（反向）来应对这种**非对称硬件扩展**，在 B200 上达到 1613 TFLOPs/s（71% 利用率）。

---

## 核心问题：非对称硬件扩展（Asymmetric Hardware Scaling）

从 Hopper H100 到 Blackwell B200，硬件资源的增长是**不均匀的**：

| 硬件资源 | H100 (Hopper) | B200 (Blackwell) | 提升 |
|---|---|---|---|
| BF16 Tensor Core | 4096 ops/clock/SM (1 PFLOP) | 8192 ops/clock/SM (2.25 PFLOP) | **2×** |
| MMA Tile 尺寸 | 64 × N | 128 × N | **2× 面积** |
| Exponential Unit (MUFU) | 16 ops/clock/SM | 16 ops/clock/SM | **不变** |
| SMEM 读带宽 | 128 bytes/clock/SM | 128 bytes/clock/SM | **不变** |
| TMEM（张量内存） | 无 | 256 KB/SM | **全新** |
| MMA 异步性 | 写入寄存器（阻塞） | 写入 TMEM（全异步） | **质变** |

**结果**：在 FA3（Hopper）上 softmax 只是"两个 GEMM 之间的小操作"，在 Blackwell 上它成了**关键瓶颈**。FA3 无法在 B200 上运行（Hopper MMA 指令不兼容）。

---

## GPU 片上硬件术语串讲

> 以下内容整理自学习过程中的对话，帮助理解论文中大量出现的 GPU 术语。

### 功能单元（每个 SM 内部）

**Tensor Core（张量核心）**：做矩阵乘法的专用硬件。一次吃进去一整块矩阵（Blackwell 上 128×128×16），一条指令算完整块的乘累加。这个操作叫 **MMA（Matrix Multiply-Accumulate，矩阵乘累加）**——既是硬件操作也是 GPU 指令的名字。B200 每 SM 每 clock 做 8192 次 BF16 MMA。

**MUFU / SFU（Multi-Function Unit / Special Function Unit）**：做 exp、sin、cos 等特殊数学函数的硬件。Softmax 里的 e^x 就靠它。每 SM 每 clock 只能做 **16 次** exp——比张量核心慢 512 倍。这就是 FA4 论文里说的"指数运算瓶颈"。

**FMA（Fused Multiply-Add）**：做普通浮点乘加 a×b+c 的通用算术单元。FA4 把 exp 拆成多项式，用 FMA 来算，绕过 MUFU 瓶颈。

**TMA（Tensor Memory Accelerator）**：专门搬数据的 DMA 引擎（HBM→SMEM→寄存器），不占计算单元的时间。Hopper 引入，Blackwell 继承。

### 存储层级（从慢到快，从大到小）

```
HBM（显存）         80-192 GB     ~3 TB/s      所有 SM 共享
    ↓
L2 Cache           ~60 MB        更快          所有 SM 共享，自动管理
    ↓
SMEM（共享内存）    ~228 KB/SM   128 B/clk/SM  同一 CTA 内线程共享，程序员手动管理
    ↓
TMEM（张量内存）    256 KB/SM    直连张量核心    Blackwell 新增！
    ↓
Register File      256 KB/SM     最快           每个线程私有
```

DSMEM（Distributed Shared Memory）：同一 cluster 内不同 CTA 的 SMEM 可以互相访问。FA4 的 2-CTA 模式用它在两个 CTA 间交换数据。

### 线程组织层级

```
Thread（线程）                    1 个
    ↓ ×32
Warp（线程束）                    32 个线程，同步执行同一指令
    ↓ ×4
Warpgroup（线程束组）             128 个线程，Blackwell MMA 最小执行单位
    ↓
CTA（Cooperative Thread Array）  一组 warpgroup，运行在同一 SM，共享 SMEM
    ↓ ×2
Threadblock Cluster              多个 CTA 跨 SM 协作（同一 GPC）
    ↓
Grid                             所有 CTA 的集合
```

### 编程工具链（从底到顶）

```
PTX（GPU 汇编语言）     直接对应硬件指令
    ↑ 编译
CUDA C++               传统方式
    ↑ 抽象
CUTLASS                 NVIDIA C++ 模板库，封装 MMA/TMA 等操作
    ↑ 进一步抽象
CuTe-DSL               CUTLASS 的 Python 版 DSL（FA4 用的）
    ↑ 更高层
Triton                  OpenAI 的 Python DSL，更简单但控制力弱
```

**CuTe-DSL** = NVIDIA CUTLASS 团队的 Python 领域特定语言。用 Python 语法写 GPU kernel，编译后生成和手写 CUDA C++ 一样底层的 PTX。编译链：Python → CuTe-DSL 编译器 → PTX → ptxas → SASS（GPU 机器码）。FA4 用它实现，编译速度比 FA3 的 C++ 快 20-30 倍。

---

## Hopper vs Blackwell 硬件详解（论文 §2.2）

### 内存层级的关键变化

Hopper 没有 TMEM，MMA 结果写入寄存器——两个问题：(1) 寄存器压力极大（accumulator 占大量寄存器）；(2) MMA 写寄存器会阻塞后续操作。

Blackwell 新增 TMEM（256 KB/SM），直连张量核心，warp-synchronous。MMA 异步写入 TMEM，不阻塞。TMEM 按 32 列 = 16 KB 粒度分配，需要手动管理。**同时解决了寄存器压力和异步重叠两个问题。**

### 张量核心的关键变化

Blackwell MMA tile = 128×N（Hopper 64×N），面积翻倍。更重要的是 Blackwell MMA **全异步**——写 TMEM 不阻塞，MMA 发射后不用等结果写完就能继续做其他事（softmax、数据搬运），这是 FA4 ping-pong 流水线的硬件基础。

### 2-CTA 张量核心（Blackwell 新特性）

同一 cluster 内一对 CTA 协同执行一个 MMA：
- 一个线程发起 MMA，另一个 CTA 必须在场
- 两个 CTA 共享彼此的 TMEM
- Output accumulator 沿 M 维分到两个 CTA → 支持 M = 256（单 CTA 限制 M ≤ 128）
- Operand B 沿 N 维分到两个 CTA 的 SMEM → **每个 CTA 只存一半 B，SMEM 流量减半**
- 代价：必须全程固定配对，不能中途切换

### 瓶颈转移的精确数字

论文给出三个关键吞吐量：
- Tensor Core = 8192 ops/clock/SM（来自 2.25 PFLOP / 1850 MHz / 148 SMs）
- MUFU = 16 ops/clock/SM（与 Hopper 相同）
- SMEM = 128 bytes/clock/SM（与 Hopper 相同）

Tensor Core 比 MUFU 的速度差：Hopper 上 4096/16 = 256 倍，Blackwell 上 8192/16 = **512 倍**。差距扩大了一倍。

> 注：论文提到 B300/GB300 已把 MUFU 翻倍到 32 ops/clock，说明 NVIDIA 也意识到了这个非对称问题。

---

## 前向传播优化

### Table 1 详细推导：Roofline 分析

前向传播做两个 GEMM + 一个 softmax：S = QK^T → P = softmax(S) → O = PV

对于一个 tile（Q 有 M 行，K/V 有 N 行，head dim = d），逐项算三种资源的 cycle 消耗：

**① MMA 需要多少 cycle？**

两个 GEMM 各需要 2MNd FLOPs（乘和加各一次），总计 4MNd FLOPs。B200 张量核心 = 8192 ops/clock：
- T_MMA = 4MNd / 8192
- 代入 M=N=d=128：T_MMA = 4×128³/8192 = **1024 cycles**

**② SMEM 读取需要多少 cycle？**

关键：两个 GEMM 的数据来源不同。
- QK^T 是 **SS（Shared-Shared）**：Q 和 K^T 都从 SMEM 读（此时 TMEM 里还没有 accumulator）
- PV 是 **TS（Tensor-Shared）**：P 在 TMEM 里（softmax 结果直接写进了 TMEM），V 从 SMEM 读

如果矩阵比 128×128 大，需要多条 MMA 指令，每条都**重新读** SMEM 操作数。BF16 = 2 bytes/element, SMEM 带宽 = 128 bytes/clock。最终：
- T_smem = 3MNd / 8192
- 代入 M=N=d=128：T_smem = 3×128³/8192 = **768 cycles**

**③ Exp 需要多少 cycle？**

Softmax 对 M×N 个元素做 exp。MUFU = 16 ops/clock：
- T_exp = MN / 16
- 代入 M=N=128：T_exp = 128²/16 = **1024 cycles**

**Table 1 结果：**

| Resource | 128³ | 256 × 128² |
|---|---|---|
| MMA compute | **1024** | **2048** |
| Shared memory | 768 | 1536 |
| Exponential unit | **1024** | **2048** |

### Table 1 的关键洞察：为什么 Exp 在 Hopper 上不是问题

**同样的 128³ 配置，如果在 Hopper 上算：**
- T_MMA = 4×128³ / **4096** = **2048 cycles**（张量核心吞吐只有一半）
- T_smem = 3×128³ / 8192 = **768 cycles**（SMEM 带宽没变）
- T_exp = 128² / 16 = **1024 cycles**（MUFU 没变）

| | Hopper | Blackwell |
|---|---|---|
| MMA | **2048** ← 瓶颈 | **1024** |
| SMEM | 768 | 768 |
| Exp | 1024 | **1024** ← 追平 MMA |

Hopper：MMA = 2048 远大于 Exp = 1024，MMA 是绝对瓶颈，Exp 轻松被隐藏。FA3 不需要关心 Exp。

Blackwell：MMA = 1024 = Exp = 1024，两者并列。**Exp 追平了 MMA，没有任何余量了。**

### 但 Table 1 是理想化的——实际更严峻

Table 1 假设三种资源完美并行（整体耗时 = max），但 softmax 和 MMA 之间有**串行数据依赖**：

```
S = Q · K^T     ← MMA 算完 S
P = exp(S - m)  ← Exp 必须等 S 算完才能开始
O = P · V       ← MMA 必须等 P 算完才能开始
```

不做优化，实际接近 MMA + Exp = 1024 + 1024 = 2048（串行），而不是 max = 1024（并行）。所以必须精心设计流水线把它们重叠起来，而且 Exp 不能比 MMA 多出哪怕一点点 cycle，否则那部分就是纯浪费。

**这就是为什么 FA4 必须同时做三件事：**
1. **Ping-pong 流水线** — 两个 Q tile 交替，MMA 和 Exp 在时间上重叠
2. **软件模拟 exp** — 把一部分 exp 分流到 FMA，降低 MUFU 的实际 cycle
3. **条件 rescaling** — 跳过 ~90% 不必要的 rescale，减少 Exp 的总工作量

三者缺一不可。只做 ping-pong 不做后两个，Exp 稍微超过 MMA 就会暴露。

### 优化 ①：Ping-Pong 流水线（Figure 1）

**Figure 1 左半部分（逻辑视图）**：Q 矩阵分成 8 个 tile（Q₀-Q₇），每两个分给一个 CTA。K 矩阵按列分 tile 在 inner loop 遍历。V 同样。

**Figure 1 右半部分（时间线）**：4 个 SM 的执行调度，每 SM 三行：
- **Tensor Cores**（黄/橙）：S = Q·K 和 O' = P·V 的 MMA
- **MUFU.EX2**（绿色）：softmax 的 exp 运算
- **Tensor Cores SP/TS MMA**（红色）：rescale/correction MMA

上标 H = "高" Q tile，L = "低" Q tile。当 Q^H 做 MMA 时，Q^L 做 softmax；反之。

关键时间关系：一个 GEMM = 512 cycles，一个 MUFU.EX2 = 1024 cycles。两个 GEMM 的时间（1024 cycles）正好覆盖一个 MUFU → ping-pong 完美重叠。

### 优化 ②：软件模拟 2^x 指数运算

MUFU 只有 16 ops/clock，是瓶颈。解决方案：在 FMA 单元上用多项式逼近，与 MUFU 并行执行。

核心公式：2^x = 2^⌊x⌋ · 2^(x-⌊x⌋)

- 整数部分 2^⌊x⌋ → IEEE 754 exponent 字段位移，几乎免费
- 小数部分 2^(x_frac) → 多项式逼近（Cody-Waite range reduction + Horner 方法 FMA 评估）

**Table 2：多项式模拟精度**

| Method | FP32 Max rel err | BF16 Max rel err |
|---|---|---|
| Hardware MUFU.EX2 | 1.41×10⁻⁷ | 3.89×10⁻³ |
| Degree 3 polynomial | 8.77×10⁻⁵ | 3.90×10⁻³ |
| Degree 5 polynomial | 1.44×10⁻⁷ | 3.89×10⁻³ |

FP32 层面 3 阶多项式比硬件差 600 倍，但转 BF16 后几乎一样——BF16 量化误差（~3.9×10⁻³）远大于多项式误差。99% 输入匹配硬件到 1 BF16 ULP 以内。

**Partial emulation**：只对 10-25% 的 softmax 行元素用软件模拟，其余仍用硬件 MUFU，平衡吞吐量和寄存器压力。

### 优化 ③：条件 Softmax 重缩放（Conditional Rescaling）

标准 Online Softmax 每个新 K block 都做 rescale。FA4 观察到：

1. 只有 m_j > m_{j-1}（发现更大值）时才需要 rescale
2. 可以容忍松弛：只在 m_j - m_{j-1} > τ 时 rescale（τ = log₂(256) = 8.0）

最终用 m_final 和 ℓ_final 做一次归一化即可保证正确性。实测**跳过 ~90% 的 rescale 操作**。

---

## 反向传播优化

### Table 3：反向 Roofline 分析（M=N=d=128）

反向有 5 个 MMA（2.5 倍于前向），其中 3 个 SS、2 个 TS。SMEM 流量是绝对瓶颈：

| Resource | 1-CTA (M=128) | 2-CTA (M=256) |
|---|---|---|
| MMA compute | 2560 | 2560 |
| **Total shared memory** | **3328** | **2688** |
| Exponential unit | 1024 | 1024 |

1-CTA：SMEM 比 MMA 多 30%（3328 vs 2560），完全 SMEM-bound。
2-CTA：SMEM 降到 2688（仅比 MMA 多 5%），接近平衡。主要来源：MMA operands 从 2048→1536（每 CTA 只读一半 B），dQ write+read 从 1024→512（原子归约减半）。代价是多 384 cycles 的 DSMEM 通信。

### Figure 2：反向传播计算图（1-CTA 模式）

分三段：
- **Prologue**：加载 Q₀, dO₀, K, V，计算 S₀ 和 dP₀
- **Main Loop**：5 个 MMA + 2 个逐元素操作，精心排序使 MMA 与 softmax 重叠
- **Tail**：收尾 dK 和 dQ 累加

TMEM 分配：S 和 P 共享一个 region，dP/dS 和 dQ 共享另一个（5 个 accumulator 装不下，只能放 4 个）。

### Figure 3：2-CTA 反向 dQ 分解（6 步图解）

- (a) 逻辑视图 dS·K
- (b) CTA tile 分布在两个 Threadblock Cluster
- (c) 两个 CTA 各持有 N 行 dS
- (d) 用 DSMEM 交换 dS 的一半 → 每 CTA 形成 (M/2 × 2N) 视图
- (e) 执行 (M/2 × 2N)(2N × d) 的 UMMA
- (f) 输出 dQ 的 (M/2 × d)

**关键好处**：每 CTA 只写 M/2 行 dQ，全局原子归约次数减半。

### 确定性反向传播

用 semaphore lock 序列化 dQ 归约，保证可复现性（RL 训练需要）。通过 CTA swizzling 减少 stall。达到非确定性 1-CTA 速度的 **75%**。

---

## 调度策略（Scheduling）

### LPT（Longest-Processing-Time First）

Causal masking 下不同 worktile 的 mainloop 长度不同。朴素递增顺序导致尾部浪费。LPT 优先调度最长 tile：batch 最外层，head 在 batch 内 swizzle，mblock 按逆序。

效果：BF16 head-dim-128，MHA **4-8% TFLOPS** 增益，MQA-8 **7-14%** 增益。

### LPT for Variable Sequence Length

预处理 kernel 按 per-worktile 最大执行时间排序 batch，写出 virtual→actual batch index mapping。Metadata 可缓存，无额外开销。

---

## CuTe-DSL 实现（Section 4）

FA4 完全用 CuTe-DSL 编写，零 CUDA C++ 组件。编译链：Python → PTX → SASS。

**Table 4：编译速度对比**

| | Forward | Backward |
|---|---|---|
| FA3 (C++) | 55s | 45s |
| FA4 (CuTe-DSL) | 2.5s | 1.4s |
| 加速 | **22×** | **32×** |

FA3 需要预编译数百个 kernel 变体。FA4 JIT 按需编译。FlexAttention 和 block-sparse attention 已在 FA4 框架上成功构建。

---

## 实验结果

测试平台：B100 180GB SXM6, CUDA 13.1, PyTorch 2.10.0

### Figure 4：前向 TFLOPS（head dim 128）

- 左图（non-causal）：FA4 在所有序列长度领先，长序列（32K）达 ~1613 TFLOPS
- 右图（causal）：LPT 使 causal 场景优势更大，长序列 ~1400+ TFLOPS
- vs cuDNN 9.13.0 加速 1.1-1.3×，vs Triton 加速 2.1-2.7×
- 注：cuDNN 9.19.1.2 已吸收 FA4 部分技术，差距缩小

### Figure 5：前向 TFLOPS（head dim 192,128 = DeepSeek V3 配置）

- 16 heads，192 query dim / 128 KV dim
- FA4 长序列 ~1600 TFLOPS，一致优于两个 cuDNN 版本

### Figure 6：反向 TFLOPS（head dim 128）

- 左（non-causal）长序列 ~1450 TFLOPS
- 右（causal）长序列 ~1400+ TFLOPS
- 2-CTA 模式显著优于 1-CTA

### Figure 7：确定性反向传播消融（causal, head dim 128）

4 种调度策略排名：CAUSAL SPT > REVERSE MBLOCK LPT > NAIVE LPT > NAIVE
- SPT 最优，达 **950 TFLOPS**（非确定性模式的 75%）
- NAIVE 最差，长序列仅 ~540 TFLOPS

### Figure 8：确定性反向传播消融（non-causal + causal 并列）

- 左（non-causal）：swizzle vs naive，swizzle 在所有长度优于 naive 约 10-15%
- 右（causal）：同 Figure 7 的 4 种策略，确认 SPT + swizzle 最优

---

## FlashAttention 演进全景

```
FA1 (2022)    Tiling + Online Softmax → 消灭 HBM 瓶颈
    ↓
FA2 (2023)    改循环顺序（外Q内KV）→ 零跨SM通信，利用率翻倍
    ↓
FA3 (2024)    Hopper 硬件优化 → warp specialization + async + FP8
    ↓
FA4 (2026)    Blackwell 非对称扩展 → 算法-内核协同设计
              ・软件模拟 exp 缓解 SFU 瓶颈
              ・条件 rescaling 减少非 matmul 操作
              ・TMEM + 2-CTA 减少 SMEM 流量
              ・CuTe-DSL Python 实现（编译快 20-30×）
```

每一代都是同一个模式：**硬件变了 → 找到新瓶颈 → 针对性地改算法**。不是简单移植，而是深刻理解硬件后重新设计算法。

---

## 我的理解

1. **这是对 GPU 的针对性优化——因为硬件变化带来的人工算法设计优化。** 每一代 FlashAttention 的核心不是算法本身多巧妙，而是**对硬件瓶颈的诊断能力**决定了优化方向。Table 1 的 roofline 分析就是诊断工具——算出各资源的 cycle，一眼看出谁是瓶颈。

2. **非对称扩展是普遍趋势**：不止 Hopper→Blackwell，未来每代 GPU 都会面临"某些单元涨得快、某些不涨"的问题。FA4 的方法论（Roofline 分析 → 找瓶颈 → 针对性优化）是通用的。这和之前读过的异构计算论文（OI+CF）是同一方法论，FA4 只是做到了最极致的粒度。

3. **Table 1 的核心洞察**：在 Hopper 上 MMA = 2048 而 Exp = 1024，MMA 是绝对瓶颈，Exp 轻松隐藏。在 Blackwell 上 MMA = 1024 = Exp = 1024，两者并列，没有任何余量。而且 MMA 和 Exp 之间有串行数据依赖（S 算完才能做 exp，exp 完才能做下一个 MMA），不做 ping-pong 流水线就是 1024+1024 的串行。

4. **具体的算法设计（ping-pong、软件模拟 exp、2-CTA 等）需要极深的 GPU kernel 工程能力**，对于我的学习阶段不需要完全掌握。重要的是理解"硬件变 → 瓶颈移 → 算法跟"这条主线。

5. **CuTe-DSL 降低门槛**：从 C++ 模板元编程（55s 编译）到 Python JIT（2.5s 编译），意味着更多研究者可以参与 attention 优化。

---

## 关联笔记

- [[flash-attention]] — FA1/FA2/FA3 三部曲，FA4 是这条线的延续
- [[gpu-architecture-basics]] — TMEM 是 Blackwell 新增的内存层级；2-CTA MMA 是新计算模式
- [[heterogeneous-computing-agent-inference]] — OI+CF Roofline 框架，FA4 的 Table 1/3 是同一方法论的应用
- [[AI-Infra-Cheatsheet]] — §十 Roofline 模型可结合 FA4 的非对称扩展案例更新

---

*学习方式：原论文精读（全部 20 页 + 附录）+ 深入对话讨论*
*最后更新：2026-03-10*
