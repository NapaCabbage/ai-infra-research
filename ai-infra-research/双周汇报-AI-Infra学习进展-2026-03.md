# AI Infra 学习进展汇报
**周期：2026年3月上旬双周**

---

## 一、研究方向定位

### 为什么学 AI Infra

AI Infra（AI 基础设施）是连接模型算法和硬件的中间层，是当前 AI 产业链中技术迭代最快、投资价值最高的细分方向之一。随着大模型训练和推理成本成为竞争核心，Infra 的效率直接决定企业的经济性和竞争力。作为半导体分析师，已有 GPU 硬件生态的理解，需要系统补齐从硬件到系统软件的完整技术栈知识，以具备独立评判技术方案价值的能力。

### 为什么从推理 Infra 开始，而不是训练

学习路径设计考虑了三个因素：

**一、门槛和回报比更优。** 推理 Serving 是近两年最活跃的子领域，论文密度极高，vLLM、SGLang 等开源系统影响力大、讨论广泛，入门后可以立刻跟上行业讨论。训练 Infra 涉及更多底层并行策略，需要更强的基础。

**二、产业链关联性强。** 大模型从训练结束到服务用户，推理端的规模和成本决定商业模式是否跑通。推理优化（KV Cache 管理、调度效率、Attention 加速）是决定 inference cost per token 的核心，这直接关系到云厂商 GPU 使用效率和客户端 API 定价能力。

**三、技术渗透性高。** 推理 Infra 的核心概念（Prefill/Decode 分离、KV Cache、带宽瓶颈、算子融合）在理解训练 Infra 时会反复出现，是很好的学习起点。

### 整体路线规划

```
阶段 1：推理 Serving（已完成）
  → Continuous Batching → KV Cache 管理 → Attention 算子优化
  → 推理系统解耦 → 异构计算与长上下文

阶段 2：训练 Infra（进行中）
  → Tensor/Pipeline/Data 并行 → ZeRO → 激活显存优化
  → DeepSeek-V3 工业实践 → GPU 硬件架构补齐
  → RL Training Infra（下一步）

阶段 3：调度（待开始）
阶段 4：编译器/Runtime（持续跟踪）
```

---

## 二、阶段一：推理 Serving（已完成）

### 2.1 领域认知建立

推理 Serving 的核心挑战在于 LLM 的**自回归生成特性**：每个 token 的生成依赖前序全部 token，导致两阶段不同瓶颈：

- **Prefill**（处理输入 prompt）：大量并行矩阵乘法，计算密集（Compute-bound）
- **Decode**（逐 token 生成）：每步仅 1 个 token，但要读取全部模型权重，带宽密集（Memory bandwidth-bound）

理解这个分裂是理解所有推理优化工作的基础。

---

### 2.2 论文一：Orca（Continuous Batching 奠基）
**OSDI 2022 | 首尔大学 / FriendliAI**

**解决的问题：** 传统 request-level scheduling 把一批请求打包处理，整批跑完才换下一批。但不同请求的生成长度差异巨大（有的 10 token，有的 500 token），短请求结束后 GPU 仍在为它做无意义计算，同时新请求在队列里等待，GPU 利用率极低。

**两个核心技术：**

第一，**Iteration-level Scheduling（Continuous Batching）**。将调度粒度从"整个请求"细化到"每个 token 生成步"。每生成一个 token 后，调度器介入一次：将已结束的请求移出、将等待队列中的新请求加入。效果是 GPU 几乎永远在做有用功，请求延迟大幅缩短。

第二，**Selective Batching**。Iteration-level scheduling 带来新问题：同一 batch 内请求处于不同生成阶段，序列长度不同，无法统一处理。解决方法是将 Transformer 操作分两类：Linear/LayerNorm/GeLU 等操作不依赖序列长度，把所有请求的 token 拍平成 `[总token数, H]` 的张量一起算；Attention 操作必须按请求独立计算（每个请求只能 attend 自己的 KV Cache）。这个切分的关键 insight 是：Attention 不参与模型权重的读取（只做 QK^T 和 softmax·V），不 batch 损失很小；而 Linear 层才是权重读取的大头，必须 batch。

**结果：** GPT-3 175B，同延迟下吞吐量比 FasterTransformer 提升 **36.9×**。

---

### 2.3 论文二：vLLM / PagedAttention（KV Cache 显存管理）
**SOSP 2023 | UC Berkeley**

**解决的问题：** Orca 等系统为每个请求预分配连续的最大长度 KV Cache 空间，导致三种显存浪费：预分配未用完的空间（reservation waste）、连续块内部碎片（internal fragmentation）、多次分配释放后的外部碎片（external fragmentation）。实测显存利用率仅 20-38%，大量 GPU 显存被浪费。

**核心技术：** 将操作系统虚拟内存的**分页（Paging）机制**应用到 KV Cache 管理。

- KV Cache 切分为固定大小的 block（默认 16 tokens/block），每个 block 可以存放在显存的任意位置，不需要连续
- 每个请求维护一个 Block Table（类似 OS 页表），记录逻辑 block 到物理 block 的映射
- **按需分配**：每生成满一个 block 才分配新的物理 block，不再预分配最大长度
- **Copy-on-Write**：多个请求共享相同前缀（如 system prompt）时，共享 KV Cache block，仅在需要独立写入时才触发复制

**代价与权衡：** PagedAttention kernel 延迟比标准 kernel **高 20-26%**，因为需要查 block table 做间接寻址，破坏了 GPU 偏好的 coalesced memory access。但 Attention 不是整个 forward pass 的计算大头，端到端影响有限，被 batch size 扩大的收益远超。

**结果：** 显存利用率从 ~30% 提升到 **96.3%**，吞吐量提升 **2-4×**。Beam search 场景因共享前缀，内存节省 37-55%，吞吐提升 3.58×。

---

### 2.4 论文三：FlashAttention 三部曲（Attention 算子优化）
**NeurIPS 2022 / ICML 2024 | Stanford / Princeton**

**解决的问题：** 标准 Attention 的计算流程需要把 N×N 的 score 矩阵（N = 序列长度）反复写入和读取 HBM（显存）。N=4096 时矩阵有 16M 元素，要在 HBM 里经历两写两读。问题不是计算量，而是**内存搬运量**——Attention 是 memory bandwidth-bound。

**两个关键技术：**

第一，**Tiling（分块）**。将 Q、K、V 切成小 tile，每次只把一小块加载到 SRAM（L1 缓存/共享内存，带宽 ~10-20 TB/s，比 HBM 快 3-6×）。在 SRAM 里完成 QK^T → softmax → ×V 的完整计算，N×N 的 score 矩阵**从不完整出现在 HBM 中**。

第二，**Online Softmax（增量式 softmax）**。Tiling 带来难题：softmax 需要整行的全局最大值和求和，但分块后每次只看到一小块。解决方法是维护三个 running 统计量（当前最大值 m、归一化因子 l、输出累积 O），每来一个新 block 时用指数运算性质 `e^(a+b) = e^a · e^b` 动态修正已有结果，保证最终等价于全局 softmax。

**FA1 → FA2 的核心改进：** 循环顺序改变（从"外层 K/V，内层 Q"改为"外层 Q，内层 K/V"）。外层 Q 时，每个 SM 独立负责一个 Q block 的完整计算，输出 O 始终在 SRAM 中累加，消除了跨 SM reduction 通信。GPU 利用率从 **35% 提升到 73%**，速度翻倍。

**FA3 的改进方向：** 针对 Hopper 架构（H100）的专属优化，利用 WGMMA（Warpgroup MMA 指令）、TMA（Tensor Memory Accelerator 异步搬数据）实现计算与内存传输的流水线并行。

---

### 2.5 论文四：SGLang（推理系统效率优化）
**主要贡献：RadixAttention + 批量优化**

**核心创新 RadixAttention**：用 LRU radix tree 管理 KV Cache，支持跨请求的前缀自动共享（multi-turn 对话、few-shot prompt、相同 system prompt 的批量请求）。对比 vLLM 的 PagedAttention，RadixAttention 在前缀命中率高的场景下 TTFT 快 **5倍**，吞吐量提升数倍。

此外提出 **Compressed Finite State Machine** 加速 constrained decoding（输出格式约束），和 **Continuous Batching 的多种工程优化**（overlap CPU/GPU 调度、tensor parallelism 等）。

---

### 2.6 论文五：DistServe（Prefill/Decode 解耦）
**核心问题：** Prefill 和 Decode 对资源的需求截然不同（Prefill 是 compute-bound，Decode 是 memory bandwidth-bound），放在同一组 GPU 上互相干扰，时延和吞吐都难以最优。

**方案：** 将 Prefill 和 Decode 物理分离到不同 GPU 组，各自独立调度和扩缩容，通过网络传输 KV Cache。每组 GPU 可以针对自身工作负载做最优的并行策略（TP/PP 组合）。

**权衡：** KV Cache 的网络传输开销；Prefill 完成后需要等待 Decode 侧有空闲 GPU 接收。

---

### 2.7 论文六：Mooncake（分布式 KV Cache 管理）
**来自月之暗面（Kimi）的工程实践**

**核心问题：** 超长上下文（128K-1M token）使单机 KV Cache 放不下，且 Prefill 耗时极长。

**方案：** KV Cache 分层存储（GPU VRAM → CPU DRAM → SSD），用 RDMA/NVLink 实现跨机器高速传输。将 Prefill 计算与 KV Cache 管理完全解耦——用一批 GPU 专门做 Prefill 计算，计算完成后把 KV Cache 通过网络存储，Decode 阶段从网络取用。

---

### 2.8 论文七：异构计算与 Agent 推理
**2025-2026 最新方向**

对传统 Roofline 模型（只看 FLOP/Byte 比值）的扩展：引入 **Capacity Footprint（CF）= 每个请求需要多少显存**，构成四象限分析框架。Agent 场景下长上下文 Decode（KV Cache 几百 GB）落在"低 OI + 高 CF"的右下角，既带宽受限又显存容量受限，单纯增加算力无效，需要 Mooncake 式的分层存储 + MLA 的低 CF 设计共同解决。

---

## 三、阶段二：训练 Infra（进行中）

### 3.1 论文八：Megatron-LM（Tensor Parallelism 奠基）
**arXiv 2019 | NVIDIA**

**解决的问题：** 单张 GPU 放不下的大模型如何高效训练。Data Parallelism 要求每张卡有完整模型副本，模型大到一定程度就失效。

**Tensor Parallelism 的核心切法：**

MLP（FFN）层有两个矩阵：第一个按**列切**，第二个按**行切**。关键在于 GeLU 激活函数是逐元素操作——列切第一个矩阵后，各 GPU 可以独立完成 GeLU 而不需要通信；行切第二个矩阵后，各 GPU 各自得到部分和，最后一次 All-Reduce 求和即可。整个 MLP 只需 **1 次 All-Reduce**。

Self-Attention 天然按 head 切分：Q、K、V 投影矩阵按列切（每 GPU 负责部分 head），独立算 attention，输出投影按行切，同样 1 次 All-Reduce。

**每层通信量：** 共 4 次 All-Reduce（Attention forward + backward 各 1 次，MLP forward + backward 各 1 次）。正因每层都要通信，TP 必须限制在节点内 NVLink（900 GB/s），无法跨节点使用 InfiniBand（25-50 GB/s）。实际 TP 上限 = 单节点 GPU 数量（通常 8）。

**附加发现：** Pre-LayerNorm（残差前归一化）比 Post-LayerNorm（残差后归一化）显著提升大模型训练稳定性，后来 LLaMA 全系列均采用（RMSNorm 为 Pre-LN 变体）。

**结果：** 512 张 V100，8.3B GPT-2，扩展效率 76%，WikiText-103 perplexity SOTA。

---

### 3.2 论文九：Megatron-LM 扩展版（3D Parallelism）
**SC 2021 | NVIDIA**

**贡献：** 在 TP 基础上加入 Pipeline Parallelism，形成 TP + PP + DP 的三维并行框架（PTD-P）。

**Pipeline Parallelism 的核心问题是 Bubble**：将模型按层分配到不同 GPU 后，GPU 需要等待上游算完才能开始，空闲时间（bubble）= `(p-1)/(m+p-1)`，p = pipeline stages，m = micro-batches。通过切 micro-batch 来填充流水线气泡。

**三种 pipeline 调度方式的演进：**

- GPipe：所有 micro-batch 先全部 forward，再全部 backward。简单但 bubble 大，且需要同时存储所有 micro-batch 的激活值（显存峰值高）。
- 1F1B（PipeDream）：warm-up 后交替一次 forward + 一次 backward，激活值同时在飞的量从 m 个减到 p 个，显存更稳定，bubble 大小与 GPipe 相同但实践中更稳定。
- Interleaved 1F1B：每个 GPU 负责多个不连续的层段（每个 GPU 负责 v 段），bubble 缩小为 `(1/v) × (p-1)/m`，代价是每个 pipeline 步需要额外 2(v-1) 次点对点通信。

**并行策略组合原则：**

- TP（每层切）：通信量最大且在关键路径上 → 节点内 NVLink
- PP（按层切）：通信量中等，点对点发送激活值 → 跨节点 InfiniBand
- DP（扩大 batch）：通信可与 backward 完全重叠 → InfiniBand

PTD-P 比纯 ZeRO-3 高 **70% 吞吐**：ZeRO-3 的高频 All-Gather 全走 InfiniBand，PTD-P 把高频通信（TP）留在 NVLink 内。

---

### 3.3 论文十：Reducing Activation Recomputation（激活显存优化）
**MLSys 2023 | NVIDIA Megatron 团队**

**问题：** 训练时 backward 需要 forward 保存的中间激活值，但 Transformer 每层的激活值显存占用极大（与 batch × seq_len 成正比），往往超过权重本身。朴素解法是 Activation Checkpointing（完全重算），但多花 33% 计算量。

**Sequence Parallelism（SP）**：TP 将 Attention 和 FFN 按隐藏维度切分，但 LayerNorm 和 Dropout 操作不依赖隐藏维度，无法 TP。SP 将这些操作按序列维度切分：每张 GPU 只处理 1/T 的序列长度（T = TP 度数）。

**SP 不增加通信量的数学原因：** TP 中已有的 All-Reduce = Reduce-Scatter + All-Gather。SP 将这个 All-Reduce 拆开：TP 区域前用 All-Gather 拼回完整序列，TP 区域后用 Reduce-Scatter 切回序列片段。每个 GPU 的激活值从 `[B, S, H]` 降低到 `[B, S/T, H]`，激活显存降低 T 倍，通信量不变。

**Selective Recomputation**：观察到 Transformer 层的激活值中，attention score 矩阵（大小为 `[B × heads × S × S]`）占用显存极大（长序列时比 hidden state 大很多），但重算代价相对小（只需重算 Attention 内部，不需要重算整层）。选择性地只重算这部分 → 节省 **70% 激活显存**，仅多 **2-3% 计算量**（相比完全重算的 33%）。

**组合效果：** SP + Selective Recomputation → 激活显存降低 5×，吞吐提升 30%，MFU 从 43% 提升到 56%。

---

### 3.4 论文十一：DeepSeek-V3 Technical Report（工业极致优化）
**2024 | DeepSeek**

**背景：** 671B 参数 MoE 模型，2048 张 H800 GPU 训练，总成本 $5.576M。体现了当前中国 AI 公司在训练 Infra 工程上的高水平。

**并行策略：** PP=16 + EP=64 + ZeRO-1，**无 TP**。不用 TP 的原因：DeepSeek-V3 采用无 TP 的 Expert Parallelism（64 张 GPU 各存 4 个 expert），All-to-All 通信走 IB，节点内 NVLink 专门留给 All-to-All 的 combine 阶段。

**DualPipe（双向流水线调度）：** 针对 MoE 模型中 All-to-All 通信耗时长（计算通信比接近 1:1）的问题，将 pipeline 调度改为双向——两端同时向中间推进 micro-batch，配合将每个 MoE block 细粒度拆分为 ATTN → DISPATCH → MLP → COMBINE 四个子步骤，使计算与通信在时间维度重叠。代价：需同时维护两个方向的 pipeline 状态，参数显存 2×。

**MoE All-to-All 通信两次的原因：** 每个 MoE block 需要 dispatch（将 token 发给 expert 所在 GPU）和 combine（将 expert 结果收回原 GPU），共 2 次 All-to-All。这是因为每个 token 必须回到原来的 GPU 做残差连接（H_mid 存在原 GPU 上）。58 个 MoE 层 × 2 = forward 约 116 次 All-to-All。

**FP8 混合精度训练：** 使用 E4M3 格式，但不是简单全精度降低。细粒度量化（activation 按 1×128 tile，weight 按 128×128 block 分别计算 scale factor），解决 tensor 内异常值导致的精度损失。关键设计：每 128 个 FP8 乘加累积后提升为 FP32 做一次加法（在 CUDA Core 上执行），防止 INT8/FP8 累加器精度不足导致的误差积累。

**内存极致优化：** RMSNorm 的激活值不存（backward 重算）、FP32 EMA 权重放 CPU 更新、embedding 和 output head 共享参数（词表矩阵不需要存两份）。预留 20 个 SM 专门运行通信 kernel，其余 SM 同时做计算，实现计算/通信 overlap。

---

### 3.5 GPU 硬件架构补习

在阅读 DeepSeek-V3 和 FlashAttention 论文过程中，发现需要理解 GPU 底层架构才能真正理解某些技术决策（如"为什么 20 个 SM 做通信"、"FP8 每 128 元素提升为 FP32"），因此系统补习了 GPU 硬件架构知识，构建了 `gpu-hardware.md`、`gpu-programming-model.md`、`gpu-performance.md` 三份结构化笔记。

核心收获：
- **SM 内部结构**：CUDA Core（标量 FMA，128个/SM）+ Tensor Core（矩阵块 CISC 指令，4个/SM，一条指令 ~8192 FLOPs）+ SFU + LSU + Register File（256 KB）+ L1/SMEM（228 KB）
- **延迟隐藏机制**：H100 每 SM 驻留 2048 个 Thread（64 个 Warp），但只有 128 个 CUDA Core，比例 16:1。Warp Scheduler 在 1 个 cycle 内切换 warp，利用海量线程切换填满内存延迟（~400 cycle）
- **内存层级与 Tiling**：Register（1 cycle）→ L1/SMEM（~30 cycle）→ L2（~200 cycle）→ HBM（~400 cycle）。Tiling 的本质是把大矩阵分块搬到 SRAM，提升 FLOPs-per-HBM-read 比值
- **CUDA 编译链**：CUDA C++ → PTX（架构无关汇编）→ SASS（机器码）。Triton/CuTe-DSL 是构建在此之上的更高层 DSL

---

## 四、副线：SageAttention 论文（偶发阅读）
**ICLR 2025 | 清华大学**

在推理 Infra 学习过程中阅读了这篇关于 Attention 量化的工作。

**解决的问题：** 将 Attention 计算中 Q、K、P、V 的矩阵乘法量化到 INT8 以加速推理，同时保持精度。直接量化 Q、K 会因 K 矩阵的 channel-wise outlier（通道级异常值）导致精度崩溃（图像生成结果模糊，语言模型准确率大幅下降）。

**K 矩阵 outlier 的根本原因：** 训练中 W_k 的某些列学习到大范数权重，使得所有 token 在这些 channel 上都有一个大的"共享偏置"（体现在热力图上是明显的竖条纹）。这是 K 作为"被搜索目标"的训练压力决定的，Q 和 V 没有这个问题。

**三个核心技术：**

**Smooth K**：对 K 矩阵减去列均值（`K_smooth = K - mean(K)`）。数学上可证明不改变 Attention 输出（减去常数后 softmax 不变），但消除了 channel outlier，量化误差大幅降低。速度开销仅 0.2%。

**混合精度策略**：Q、K 量化到 INT8（RTX4090 上 INT8 比 FP16 快 4×），P（softmax 结果）和 V 保持 FP16 并使用 FP16 accumulator（速度比 FP16+FP32 accumulator 快 2×，且精度远好于 INT8）。两个 Matmul 各自走最优路径。

**Adaptive Quantization**：对每一层分别测试不同量化方案，选择最快且精度达标（cosine similarity > 99.8%）的方案。整体再提速 12%。

**结果：** 比 FlashAttention2 快 **2.1×**，比 xformers 快 **2.7×**，端到端模型指标几乎无损失。

---

## 五、知识体系建设

### Cheatsheet（速查手册）

建立并持续维护 `AI-Infra-Cheatsheet.md`，共 14 个章节，涵盖：数据类型速查、Transformer 推理全流程、MoE 架构与训练、KV Cache 公式、GPU 带宽速查、GPU 硬件架构、Roofline 分析、训练显存公式、分布式训练并行策略详解（TP/PP/DP/EP/ZeRO）、编译器全栈。

### 论文笔记库

共建立 14 篇结构化论文笔记，统一格式：一句话总结、动机、核心方法（含图表解读）、实验结果、局限性、我的理解、关联笔记。

### 学习路径文件

为推理 Infra 和训练 Infra 各建立 reading-order.md，记录已读论文、掌握程度、待深入问题，以及下一步学习计划。

---

## 六、下一步计划（RL Training Infra，4 天）

**背景判断：** DeepSeek-R1（2025.01）证明 RL post-training 可以让模型涌现长链推理能力后，RL Training Infra 成为 2026 年最热方向。RL 训练的 Infra 挑战与 pre-training 根本不同——训练每步都需要先跑推理（rollout），推理耗时占 80-85%，且推理和训练系统需要深度耦合。

已通过对话建立的基础概念：PPO vs GRPO vs R1 的算法差异、Advantage 计算机制、Reward Model 和 Critic Model 的来历与架构、RL loss 的 backward 机制、分离式 vs 混合式 RL 训练框架（OpenRLHF vs veRL）、长尾 rollout 问题。

**计划阅读：**

- Day 1：DeepSeek-R1（GRPO 算法 + 多阶段训练流程）
- Day 2：DAPO + REINFORCE++（GRPO 的迭代改进）
- Day 3：OpenRLHF + veRL（系统架构核心）
- Day 4：RollPacker + 长尾优化方案

---

*汇报日期：2026-03-13*
