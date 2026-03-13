---
tags: [learning]
created: 2026-03-05
---

# AI Infra 速查手册

> 反复看、随时更新。数学不要背，要理解推导过程，下次自己能算出来。

---

## 一、Bit / Byte 基础

```
1 Byte = 8 bits
1 KB   = 1,024 Bytes   ≈ 10³ Bytes
1 MB   = 1,024 KB      ≈ 10⁶ Bytes
1 GB   = 1,024 MB      ≈ 10⁹ Bytes
1 TB   = 1,024 GB      ≈ 10¹² Bytes
```

**记忆口诀**：存储/带宽单位里，大写 B = Byte，小写 b = bit。网络带宽通常用 Gbps（bit），GPU 带宽用 GB/s（Byte）。

---

## 二、数据类型速查

| 格式 | 位数 | 字节数 | 数值范围 | 典型用途 |
|------|------|--------|---------|---------|
| FP64 | 64 bit | 8 B | ±1.8×10³⁰⁸ | 科学计算，LLM 基本不用 |
| **FP32** | **32 bit** | **4 B** | ±3.4×10³⁸ | 训练时的主精度副本（optimizer states）|
| **BF16** | **16 bit** | **2 B** | ±3.4×10³⁸（同FP32范围！）| 训练 forward/backward，H100首选 |
| **FP16** | **16 bit** | **2 B** | ±65504（范围小，易溢出）| 推理，训练中也用，需注意溢出 |
| **FP8** | **8 bit** | **1 B** | 两种：E4M3/E5M2 | DeepSeek-V3训练，H100原生支持 |
| **INT8** | **8 bit** | **1 B** | -128 ~ 127 | 推理量化，LLM.int8() |
| **INT4** | **4 bit** | **0.5 B** | -8 ~ 7 | 激进量化，AWQ/GPTQ |

**BF16 vs FP16 的关键区别**：BF16 截掉了尾数位（精度低）但保留了指数范围（和FP32一样），所以训练时不容易溢出。FP16 精度高但范围小，遇到大梯度会溢出（NaN）。H100 训练推荐 BF16。

---

## 三、Transformer 推理：一次 Forward Pass 的完整流程

> 以 LLaMA 7B 为例：vocab=32000, d=4096, heads=32, head_dim=128, layers=32, d_ffn=11008

### 完整流程（Prefill 阶段，输入 "我爱北京" = 4 个 token）

```
输入 [我,爱,北,京] → tokenizer → 4 个 token_id
  ↓
① Embedding 查表：E[32000×4096]，每个 token_id 查一行
  → X: [4 × 4096]

  ╔══════════════════════════════════════════════════════╗
  ║ 以下重复 32 层（层间串行，层内 Attention 多头并行）       ║
  ╠══════════════════════════════════════════════════════╣
  ║                                                      ║
  ║ ② LayerNorm → Attention                             ║
  ║   投影：Q = X×W_q, K = X×W_k, V = X×W_v            ║
  ║         [4×4096]×[4096×4096] = [4×4096]              ║
  ║   拆头：reshape → [4×32×128]（32头并行）               ║
  ║   每头：S = q×k^T = [4×128]×[128×4] = [4×4]         ║
  ║         causal mask → softmax → ×v → [4×128]        ║
  ║   拼回：[4×32×128] → [4×4096]                        ║
  ║   输出投影：×W_o → [4×4096]                           ║
  ║   残差连接：+ X                                       ║
  ║   ★ 存 K, V 到 KV Cache                             ║
  ║                                                      ║
  ║ ③ LayerNorm → FFN                                   ║
  ║   升维：×W_up   [4096→11008] → [4×11008]             ║
  ║   门控：×W_gate [4096→11008] → SiLU 激活              ║
  ║   逐元素相乘 → 降维：×W_down [11008→4096]             ║
  ║   残差连接：+ 上一步输出                                ║
  ║   → [4×4096]  送入下一层                               ║
  ║                                                      ║
  ╚══════════════════════════════════════════════════════╝

④ 取最后 token 的输出 h₄: [1×4096]
  × LM_Head [4096×32000] → logits [1×32000]
  → softmax → 采样 → 输出第一个新 token "天"
```

### 各模块的角色

```
Embedding     → 把 token ID 变成向量（查表，不是计算）
Attention     → token 之间互相看，交换信息（"社交"）
  Q = "我在找什么"（提问）
  K = "我能提供什么"（名片）
  V = "我的实际内容"（正文）
  Q×K^T = 相关性打分 → softmax归一化 → ×V = 加权提取内容
FFN           → 每个 token 独立消化信息（"独立思考"）
  升维(4096→11008) → 非线性激活 → 降维(11008→4096)
  为什么升维：高维空间里非线性变换表达力更强，降回来传给下一层
残差连接      → 每层追加新信息而非替换（像画画逐层加细节）
32 层叠加     → 浅层学词性/语法，中层学语义/指代，深层学推理/抽象
```

### Prefill vs Decode：同一个模型，不同的瓶颈

```
                    Prefill（理解阶段）           Decode（表达阶段）
输入                所有 prompt token（已知）      1 个新 token（刚生成的）
Q 的形状            [N × d]                      [1 × d]
Q×K^T              [N×d]×[d×N] = [N×N]          [1×d]×[d×(N+t)] = [1×(N+t)]
FFN                [N×d]×[d×4d] 大矩阵乘法       [1×d]×[d×4d] 一行乘大矩阵
瓶颈                Compute-bound（计算量大）      Memory-bound（搬权重多算的少）
KV Cache           一次性生成并存储               每步追加 1 组 K,V
类比                老师一次批改全班作业            学生一个个进来考试
```

**为什么 Decode 是 memory-bound？** 每生成 1 个 token，所有权重矩阵（W_q, W_k, W_v, W_o, W_up, W_gate, W_down × 32层）都要从 HBM 读一遍，但只做 1 行的矩阵乘法。读的多、算的少 → 带宽是瓶颈。

**Multi-Head Attention 补充**：32 个 head 在同一层内**并行**计算（彼此独立），32 层之间**串行**（下一层依赖上一层输出）。GQA（LLaMA 用的）让 128 个 Q head 共享 8 个 KV head → KV Cache 缩小 16×。

---

## 四、MoE 架构：Dense 模型的 FFN 层拆分

> MoE 是在 Dense Transformer 基础上，**只改了 FFN 层**，其他部分（Embedding、Attention、LM Head）完全不变。

### Dense vs MoE：每层的区别

```
Dense（LLaMA 7B）每层：
  x → Attention → residual → FFN（1个） → residual → 下一层
                               ↑
                        W_up:   4096 → 11008
                        SiLU 激活
                        W_down: 11008 → 4096
                        所有 token 都走同一个 FFN

MoE（DeepSeek-V3）每层：
  x → Attention → residual → Router → Expert 1 (FFN)  → 加权求和 → residual
                               ↓      Expert 2 (FFN)
                               ↓      ...
                               ↓      Expert 256 (FFN)
                               ↑
                          Router 选出 Top-8 个专家
                          每个 token 只走其中 8 个 FFN
```

### Router 是什么？

Router = 一个**可学习的线性层** + Softmax，作用是"选路"：

```
Router: x (4096) → W_router (4096 × 256) → Softmax → scores (256)
取 Top-K（如 K=8）→ 选出得分最高的 8 个专家
对这 8 个专家的输出按 score 加权求和
```

W_router 是训练时和模型一起学出来的参数，学到的是"什么样的 token 应该交给哪个专家"（比如数学 token → 某几个专家，代码 token → 另几个专家）。注意 Router 不是激活函数（ReLU/SiLU 是固定数学变换，没有可学习参数）。

### 为什么用 MoE？参数量与计算量解耦

```
DeepSeek-V3：
  总参数量：671B（256 个专家，每个都有完整 FFN 权重）
  每 token 激活参数：37B（只用 8 个专家 + 共享层）

  → 用 37B 的计算成本，获得接近 671B 的模型能力
```

核心 trade-off：模型越大能力越强，但计算量也越大。MoE 让参数量暴增（256× FFN 权重），但每个 token 的计算量只略增。

### MoE 对推理系统的影响

```
显存：所有 256 个专家权重都要存在显存里（不同 token 路由到不同专家）
      → 671B × 2 bytes = 1342 GB，比同等计算量的 Dense 37B 大得多

KV Cache：Attention 层没变 → KV Cache 大小和 Dense 模型一样
          （取决于 hidden size 和层数，和专家数无关）

通信：不同 token 被路由到不同专家，专家可能分布在不同 GPU 上
      → 需要 all-to-all 通信把 token 发给对应 GPU
      → 这就是 AFD 能"免费"搭便车的那个 all-to-all
```

### MoE 训练：一个 Token 在一个 MoE Block 中的完整旅程

> Dense 模型中，同一组 GPU 既算 Attention 又算 FFN（TP 切分后每张 GPU 各算一部分，依次执行）。MoE 模型也一样——同一张 GPU 在 Attention 阶段做"计算者"，在 FFN 阶段做"expert 服务者"，**角色交替进行，不存在"只做 attention"或"只做 FFN"的 GPU**。

以 DeepSeek-V3 为例（EP=64，每张 GPU 存 4 个 expert，无 TP）：

```
步骤 1: Attention（本地计算，无通信）
  每张 GPU 独立计算自己持有的 token 的 attention（MLA 压缩了 KV，单卡可算）
  输出：每个 token 得到一个 hidden state
  ★ 第一个残差连接：H_mid = H + Attention(LayerNorm(H))

步骤 2: Router / Gating（本地计算，无通信）
  每个 token 的 H_mid 过一个小线性层 → softmax → 选出 Top-8 expert
  此时每张 GPU 知道："我的 token A 要去 expert 37（GPU 9）和 expert 182（GPU 45）..."

步骤 3: All-to-All DISPATCH（第一次通信）
  每张 GPU 把自己的 token 发给对应 expert 所在的 GPU
  64 张 GPU 之间做 All-to-All → 数据从"按来源 GPU 组织"变为"按目标 expert 组织"
  通信之后：每张 GPU 收到"所有要来找我这 4 个 expert 的 token"

步骤 4: Expert FFN 计算（本地计算，无通信）
  每张 GPU 用自己的 4 个 expert 处理收到的 token
  注意：收到的 token 数量 B' ≠ 本地 token 数量 B（取决于路由结果）

步骤 5: All-to-All COMBINE（第二次通信）
  计算结果要送回原来的 GPU（因为原 GPU 需要做残差连接 + 加权求和）
  再做一次 All-to-All，方向相反 → 数据从"按 expert 组织"变回"按 token 归属组织"

步骤 6: 加权求和 + 残差（本地计算）
  每个 token 的 8 个 expert 结果按 router 权重加权求和
  ★ 第二个残差连接：H_out = H_mid + weighted_sum(expert_results)
  H_out 传给下一个 Block，重复步骤 1-6
```

**每个 MoE block = 2 次 All-to-All**。DeepSeek-V3 有约 58 个 MoE 层，即 forward 约 116 次 All-to-All。这就是通信量巨大、计算通信比接近 1:1 的原因。

### 残差连接：Transformer 的"高速公路"

残差连接常被略过，但对理解显存和通信至关重要。每个 Transformer block 有**两个**残差连接：

```
H（block 输入）
├────────────────┐
│                │  ← H 原封不动绕过
▼                │
LayerNorm → Attn │
▼                │
+ ◄──────────────┘  ← 第一个残差：H_mid = H + Attn(LN(H))
│
├────────────────┐
│                │  ← H_mid 原封不动绕过
▼                │
LayerNorm → FFN  │
▼                │
+ ◄──────────────┘  ← 第二个残差：H_out = H_mid + FFN(LN(H_mid))
```

用公式表示：`H_mid = H + Attention(LayerNorm(H))`, `H_out = H_mid + FFN(LayerNorm(H_mid))`

残差的作用：每层只学"修正量"而非完整表示。梯度可以沿残差"高速公路"直接回传（∂H_out/∂H_mid 中包含常数 1），防止深层网络梯度消失。

**对 MoE 训练的影响**：H_mid 从 dispatch 开始到 combine 结束期间必须一直保留在显存中（等着做第二个残差连接），这是 MoE 训练显存压力的重要来源。

### MoE 训练显存时间线（单个 Block 内）

```
显存占用
  ▲
  │      峰值1                    峰值2
  │     ┌──┐                    ┌──┐
  │    ╱    ╲      ╱╲          ╱    ╲      ╱╲
  │   ╱      ╲    ╱  ╲        ╱      ╲    ╱  ╲
  │  ╱        ╲  ╱    ╲      ╱        ╲  ╱    ╲
  │ ╱          ╲╱      ╲    ╱          ╲╱      ╲
  ├──────────────────────────────────────────────► 时间
  │ Attn    残差1  Dispatch  Expert   Combine  残差2
  │              ←─通信─→             ←─通信─→
  │
  └── 常驻显存（权重 + optimizer states）

峰值 1（Attention）：H + QKV + attention scores [B × heads × seq × seq]
  → 通过 selective recomputation 缓解（不存 attention scores，backward 重算）

峰值 2（Expert FFN）：H_mid（等着做残差）+ 收到的 B' 个 token + up_proj [B' × 4d]
  → B' 不可控！expert 热门时 B' >> B → 显存爆炸风险
  → DeepSeek-V3 用 FP8 缓存激活 + load balancing 缓解
```

关键显存流转：

```
阶段 0: 持有 H [B×d]
阶段 1: H + attention 中间激活 → 峰值 1
阶段 1.5: H_mid = H + attn_out → 释放 H 和 attn 中间物，只持有 H_mid
阶段 3: H_mid + dispatch 收发缓冲（B' 个 token 涌入）
阶段 4: H_mid + expert 中间激活 [B'×4d] → 峰值 2
阶段 5: H_mid + combine 收发缓冲
阶段 6: H_out = H_mid + expert 结果 → 释放 H_mid → 只持有 H_out → 传给下一层
```

---

## 五、推理时 GPU 显存全景

推理时 GPU 显存里装的不只是"模型参数"，而是四类数据：

```
GPU 总显存 = ① 模型权重（固定）
           + ② KV Cache（随 token 动态增长）
           + ③ 激活值缓冲（临时，每层复用）
           + ④ 工作空间（临时，框架/CUDA开销）
```

| 部分 | 特征 | 大小量级 | 谁管它 |
|------|------|---------|-------|
| ① 权重 | 固定不变 | 最大（BF16下占 60-70%） | 模型加载时一次性分配 |
| ② KV Cache | 随 token 增长，每个请求独立 | 长上下文下可超过权重 | vLLM PagedAttention |
| ③ 激活值 | 临时，每层计算完就释放 | 很小（< 5%） | 框架自动管理 |
| ④ 工作空间 | CUDA kernel 临时内存 | 很小 | CUDA runtime |

> **核心认知**：推理优化的本质是管理 ② KV Cache——它是唯一随请求动态变化且可能爆炸增长的部分。这就是为什么 Orca（调度）和 vLLM（显存管理）都围绕 KV Cache 做文章。

---

## 五、三个核心公式

### 公式 1：模型权重显存（固定成本）

```
显存占用（GB）= 参数量（B）× 每参数字节数
```

| 精度 | 每参数字节 | 快速心算 |
|------|-----------|---------|
| FP32 | 4 B | 参数量(B) × 4 |
| BF16/FP16 | 2 B | 参数量(B) × 2 |
| FP8/INT8 | 1 B | 参数量(B) × 1 |
| INT4 | 0.5 B | 参数量(B) × 0.5 |

### Dense vs MoE 模型对比（BF16）

| 模型 | 架构 | 总参数 | 每次推理激活参数 | 权重显存(BF16) | 需要的H100 |
|------|------|--------|-------------|-------------|-----------|
| LLaMA 3.1 8B | Dense | 8B | 8B（全部） | **16 GB** | 1张 |
| LLaMA 3.1 70B | Dense | 70B | 70B（全部） | **140 GB** | 2张 |
| LLaMA 3.1 405B | Dense | 405B | 405B（全部） | **810 GB** | ≥11张 |
| Qwen-2.5 72B | Dense | 72B | 72B（全部） | **144 GB** | 2张 |
| DeepSeek-V3 | **MoE** | 671B | **37B**（每token） | 1342 GB总 | 总参~17张 |
| Mixtral 8×7B | **MoE** | 47B | **13B**（每token） | 94 GB总 | 2张 |

> **Dense vs MoE 的关键区别**：
> - **Dense**（LLaMA）：每个 token 经过所有参数，权重全部要加载到显存
> - **MoE**（DeepSeek-V3）：每个 token 只激活一小部分专家（37B / 671B ≈ 5.5%），但**所有参数仍需存在显存中**（因为不同 token 可能路由到不同专家）
> - MoE 的优势：同等计算量下参数更多（模型更强），推理 FLOP 和 Dense 37B 相当但效果接近 Dense 400B+
> - MoE 的劣势：权重显存需求大（671B 全部要加载），Expert Parallelism 通信开销

---

### 公式 2：KV Cache 显存（动态成本）

> 详见下方第八节，这里给出直觉：KV Cache 就是模型的"历史记忆仓库"，每生成一个 token 就往仓库里存一份 K 和 V，序列越长仓库越大。

### 公式 3：Decode 吞吐量（搬运速度决定生成速度）

Decode 阶段是 memory bandwidth-bound——每生成一个 token，GPU 必须把**全部模型权重**从 HBM 读一遍。所以：

```
单请求 Decode 速度上限 ≈ HBM带宽 / 模型权重大小

例：H100 + LLaMA 70B (BF16)
  = 3.35 TB/s / 140 GB
  = ~24 tokens/s（理论上限）
```

**Batching 的魔法**：读一次权重可以同时给 B 个请求算，所以：

```
Batch Decode 吞吐 ≈ HBM带宽 / (权重大小 / B)  [简化，忽略KV读取]
                   = 单请求速度 × B

但实际受限于显存容量（要装得下 B 个请求的 KV Cache）
```

| 场景 | 模型 | GPU | 理论单请求速度 | Batch=32 吞吐 |
|------|------|-----|-------------|-------------|
| LLaMA 70B BF16 | 140 GB | H100 (3.35 TB/s) | ~24 tok/s | ~768 tok/s |
| LLaMA 70B INT4 | 35 GB | H100 (3.35 TB/s) | ~96 tok/s | ~3072 tok/s |
| LLaMA 405B BF16 | 810 GB | 11×H100 | ~4 tok/s/GPU | 取决于并行策略 |

> **三个公式的直觉**：
> - 公式 1（权重）= **固定成本**：开店要多大的房子
> - 公式 2（KV Cache）= **历史库存**：服务越多客人、聊越久，库存越大
> - 公式 3（Decode 速度）= **搬运速度**：房子里的东西要搬一遍才能服务一个客人，带宽就是搬运工的速度

---

## 七、训练显存公式

训练比推理占用多得多，因为需要存储梯度和优化器状态：

### 混合精度训练（标准配置）

```
总显存 ≈ 参数显存 × 训练倍率

常见倍率（使用Adam优化器，混合精度）：
- 参数（BF16）：  1×  →  2 bytes/param
- 梯度（BF16）：  1×  →  2 bytes/param
- FP32参数副本：  2×  →  4 bytes/param  ← optimizer需要
- Adam m/v：     4×  →  8 bytes/param  ← 各4 bytes
────────────────────────────────────────
合计：               16 bytes/param
```

| 模型 | 参数量 | 推理(BF16) | 训练(Adam混精) |
|------|--------|-----------|--------------|
| 7B | 7B | 14 GB | 112 GB |
| 70B | 70B | 140 GB | 1,120 GB |
| 405B | 405B | 810 GB | 6,480 GB |

> 这就是为什么大模型训练必须用 ZeRO 分散这些状态，或者用 FP8 训练（DeepSeek-V3 做法）把训练开销大幅降低。

---

## 八、KV Cache 公式

### 核心公式

```
KV Cache = 2（K和V）
         × num_layers（层数）
         × num_kv_heads（KV头数，GQA下远小于query头数）
         × head_dim（每个头的维度）
         × seq_len（序列长度）
         × bytes_per_element（精度，通常2 bytes=BF16）
```

### 常见模型每token KV Cache大小（BF16，来源：DeepSeek-V3论文 Table 1）

| 模型 | 机制 | 每token KV(BF16) | 相对DS-V3 |
|------|------|-----------------|---------|
| **DeepSeek-V3** | **MLA** | **70.272 KB** | **1x（基准）** |
| Qwen-2.5 72B | GQA | 327.680 KB | 4.66x |
| LLaMA-3.1 405B | GQA | **516.096 KB** | 7.28x |

> 推导验证（LLaMA 3.1 405B）：2 × 126层 × 8个KV头 × 128维 × 2字节 = **516,096 bytes = 516.096 KB** ✓
>
> LLaMA 3.1 全系列用 GQA：128个 query head，但只有 **8个 KV head**，KV Cache 相比标准MHA缩小16×。

### 不同上下文长度的KV Cache（256K = 262,144 tokens）

| 上下文长度 | LLaMA 3.1 70B (GQA) | LLaMA 3.1 405B (GQA) | DeepSeek-V3 (MLA) |
|-----------|--------------------|--------------------|-----------------|
| 8K | 2.5 GB | 4.0 GB | 0.54 GB |
| 32K | 10 GB | 16 GB | 2.1 GB |
| 128K | 40 GB | 63 GB | 8.6 GB |
| **256K** | **80 GB** | **125 GB** | **17 GB** |

> 💡 **"80GB"记忆的真正来源**：**LLaMA 3.1 70B + 256K context 的 KV Cache 恰好约 80 GB**，这才是"80GB"的出处，而不是405B模型的权重大小。
>
> **405B + 256K 上下文**：权重 810 GB + KV Cache 125 GB ≈ **935 GB 总计**，约需 12 张 H100。
>
> **DeepSeek-V3 MLA 的优势**：同样256K上下文，KV Cache 只需 17 GB，是 405B GQA 的 1/7。这是 MLA 最核心的工程价值。

### Attention 机制演进：MHA → GQA → MLA

**MHA（Multi-Head Attention）**：Q、K、V头数相同，KV Cache 最大
- 每层存储：n_heads × head_dim × 2（K和V）× 2字节

**GQA（Grouped Query Attention）**：K、V头数大幅减少，Q头不变
- LLaMA 3.1：128个Q头，只有8个KV头，KV Cache缩小16×
- 模型效果基本不损失

**MLA（Multi-head Latent Attention，DeepSeek提出）**：把K、V压缩成低维潜变量（latent vector）存储，推理时再还原
- DeepSeek-V3：每层只存 576维 的潜变量（512维KV潜变量 + 64维解耦RoPE），而不是完整的K、V
- 推导：576维 × 61层 × 2字节 = **70,272 bytes = 70.272 KB** ✓（与论文完全吻合）
- 效果：比同量级GQA模型KV Cache再缩小 4-7×，同时模型效果更好（因为参数利用率更高）

---

## 九、GPU 内存带宽速查

| GPU | HBM容量 | HBM带宽 | 互联 |
|-----|---------|---------|------|
| A100 80GB | 80 GB | 2 TB/s | NVLink 3.0 (600 GB/s) |
| H100 80GB SXM | 80 GB | **3.35 TB/s** | NVLink 4.0 (900 GB/s) |
| H100 NVL | 94 GB | 3.9 TB/s | NVLink 4.0 |
| H200 | 141 GB | **4.8 TB/s** | NVLink 4.0 |
| B200 | 192 GB | ~8 TB/s | NVLink 5.0 |

> **为什么带宽比算力更重要**：Decode 阶段（逐token生成）是 memory bandwidth-bound。GPU 的 FP16 算力提升了很多代，但带宽提升相对慢，所以 decode 始终是瓶颈。H200 的 4.8 TB/s 带宽比 H100 的 3.35 TB/s 快 43%，decode 速度近似线性提升。

---

## 九½、GPU 硬件架构：从芯片到指令

> GPU 和 CPU 的核心设计哲学不同：CPU 把大量晶体管花在"聪明"上（分支预测、乱序执行、大 Cache），用复杂逻辑隐藏延迟；GPU 把晶体管花在"计算单元"上，用**海量线程切换**隐藏延迟。

### SM（Streaming Multiprocessor）内部结构

SM 是 GPU 的基本计算单元，相当于 GPU 的"核心"（H100 有 132 个 SM）。每个 SM 内部：

```
┌─────────────────────────────────────────────────┐
│  SM（Streaming Multiprocessor）                   │
│                                                   │
│  ┌─────────────┐  控制与调度                       │
│  │ Warp        │  4 个 Warp Scheduler              │
│  │ Schedulers  │  每 cycle 选一个就绪 warp 发射指令    │
│  └──────┬──────┘                                   │
│         ↓ 分配给不同执行单元                          │
│  ┌──────────────────────────────────────────┐      │
│  │  执行单元                                  │      │
│  │  CUDA Core ×128   整数/浮点标量运算          │      │
│  │  Tensor Core ×4    矩阵块运算（CISC 风格）   │      │
│  │  SFU（特殊函数）    sin/cos/exp/sqrt         │      │
│  │  LSU（读写单元）    发起 load/store 请求      │      │
│  └──────────────────────────────────────────┘      │
│                                                   │
│  ┌──────────────────────────────────────────┐      │
│  │  存储层级（SM 内部）                        │      │
│  │  Register File   256 KB  最快，每 thread 私有 │      │
│  │  L1 / Shared Mem 228 KB  可配置，block 内共享  │      │
│  └──────────────────────────────────────────┘      │
└─────────────────────────────────────────────────┘

SM 外部（芯片级共享）：
  L2 Cache         50 MB    所有 SM 共享
  Memory Controller         管理 HBM 读写
  NVLink Controller         管理 GPU 间通信
  Copy Engine / DMA         异步数据搬运（CPU↔GPU）
```

### 线程层级：Thread → Warp → Block → Grid

```
Thread（线程）    最小执行单位，对应一个数据点的计算
  ↓ 32 个 thread
Warp（线程束）    GPU 实际调度的最小单位，32 个 thread 锁步执行同一条指令
  ↓ 多个 warp
Block（线程块）   共享 Shared Memory，可通过 __syncthreads() 同步
  ↓ 多个 block
Grid（网格）      一次 kernel launch 的全部 block，分配到所有 SM 上执行
```

**关键概念——Warp**：GPU 不单独调度 thread，而是以 32 个 thread 为一组（warp）一起调度。同一个 warp 里的 32 个 thread 在同一 cycle 执行同一条指令，只是操作不同数据（SIMT）。"Warp"来自纺织术语（经线），32 根线并排织布。

### 延迟隐藏：为什么 Thread 数远多于 CUDA Core 数

```
H100 每个 SM：128 个 CUDA Core，但可驻留 2048 个 Thread（64 个 Warp）

比例 = 16:1（thread : core）

工作原理：
  Warp A 执行 → 遇到内存读取（~400 cycles 延迟）
  Warp Scheduler 立刻切到 Warp B（1 cycle 切换，零开销）
  Warp B 也等了 → 切到 Warp C → ... → 切到 Warp N
  ~400 cycles 后 Warp A 数据回来 → 切回继续
  → 只要驻留 warp 够多，CUDA Core 几乎永远不闲

类比：128 个收银台，2048 个顾客。顾客刷卡等待时收银员先服务下一位。
```

这就是 GPU 的核心策略：**不靠大 Cache 隐藏延迟（CPU 的做法），而是靠海量线程轮流执行来填满等待时间。** 所以 GPU 的 Cache 很小但寄存器文件很大（256KB/SM），要给每个驻留 thread 保存执行上下文。

### Tensor Core vs CUDA Core

```
CUDA Core：标量处理器，一条指令算一个 FMA（a×b+c）
  → 一个 cycle 一个 CUDA Core 做 1 次乘加 = 2 FLOPs

Tensor Core：矩阵块处理器（CISC/数据流风格），一条指令算整个矩阵块
  → H100: 一条 WGMMA 指令完成 16×16×16 矩阵乘加
  → = 8192 FLOPs/指令（相当于 ~4000 条 CUDA Core FMA 指令）
```

Tensor Core 不是"很多 CUDA Core 打包"，而是**完全不同的电路设计**——固定功能的矩阵乘法加速器，类似数据流（dataflow）芯片的思路：数据进去，矩阵乘法结果直接出来，中间不需要取指-译码-执行的循环。这就是为什么 DeepSeek-V3 训练和 FlashAttention 都尽量把计算映射到 Tensor Core 上。

### 内存层级与 Tiling 的核心思想

```
存储层级         容量        带宽            延迟
Register        256 KB/SM   ~20 TB/s       1 cycle
L1/Shared Mem   228 KB/SM   ~10-20 TB/s    ~30 cycles
L2 Cache        50 MB       ~5 TB/s        ~200 cycles
HBM (显存)      80 GB       3.35 TB/s      ~400 cycles
                ↑ 越大          ↑ 越慢           ↑ 越慢
```

**Tiling（分块搬运）**：SRAM（L1/Shared Memory）容量远小于 HBM，但带宽高 3-6×、延迟低 10×+。核心优化思想：把大矩阵分成小 tile，搬进 SRAM 反复计算，算完换下一块。每次 HBM 读取被复用多次。

```
朴素做法：每次计算都从 HBM 读 → 1 FLOP / 1 HBM read
Tiled 做法：搬一个 16×16 tile 到 SMEM → 16 FLOPs / 1 HBM read

FlashAttention = Tiling 应用于 Attention（难点在 softmax 需要全局 max，
                 用 online softmax 边搬边修正解决）
DeepSeek-V3 FP8 "1×128 tile 量化" = 在 tile 粒度上做精度管理
```

### 与训练论文的关联

```
DeepSeek-V3 "预留 20 个 SM 做通信"
  → SM 是独立调度单位，可以把一部分 SM 专门用于 NCCL 通信 kernel
  → 其余 SM 同时做计算 → 计算/通信 overlap

DeepSeek-V3 FP8 训练 "每 128 个元素累加到 FP32"
  → Tensor Core 原生支持 FP8 输入，但累加精度不够
  → 每 128 次 FP8 乘加后，用 CUDA Core 做一次 FP32 提升

FlashAttention Tiling
  → 把 Q×K^T 和 softmax 和 ×V 全部融合在 SRAM 里完成
  → 不把中间的 attention score 矩阵 [seq×seq] 写回 HBM
```

---

## 十、Roofline 分析（判断瓶颈）

### 核心直觉

GPU 做任何计算都需要两步：**从 HBM 读数据 → 用 ALU 算**。谁更慢，谁就是瓶颈。

```
H100 的两个能力：
  算力上限：990 TFLOPS（FP16）
  带宽上限：3.35 TB/s

平衡点 = 算力 / 带宽 = 990T / 3.35T ≈ 295 FLOPs/Byte
```

### OI（Operational Intensity）= 每搬 1 byte 做几次运算

```
你的 OI > 295  → 算力先用完，带宽有余 → Compute-bound
你的 OI < 295  → 带宽先用完，算力闲着 → Memory-bandwidth-bound
```

### 具体例子（LLaMA 7B，权重 14GB）

```
Prefill（输入 4 个 token）：读 14GB 权重，做 4 个 token 的计算 → OI 高 → Compute-bound
Decode（生成 1 个 token）：读 14GB 权重，做 1 个 token 的计算 → OI 极低 → Bandwidth-bound
Batch=128 Decode：         读 14GB 权重，做 128 个 token 的计算 → OI 提升 128× → 逼近 Compute-bound
```

同样读 14GB 权重，Prefill 做了 4+ 个 token 的活，Decode 只做 1 个。搬的一样多，算的少得多 → Decode 被带宽卡住。Batch 让每次搬运做更多"活"，OI 往右移。

### Roofline 图

```
性能
(FLOPS)
  │            ╱ ← 算力上限（水平线，"屋顶"）
  │          ╱
  │        ╱  ← 带宽上限（斜线，斜率 = 带宽）
  │      ╱
  │    ╱
  │  ╱
  │╱
  └──────────────────── OI (FLOPs/Byte)
        ↑                    ↑
   Decode 在这里          Prefill 在这里
   (bandwidth-bound)      (compute-bound)
```

| 操作 | OI | Bound类型 |
|------|---------|---------|
| Prefill（处理长prompt）| 高 | Compute-bound |
| Decode（逐token生成）| 极低 | Memory bandwidth-bound |
| 大 batch Decode | 中等 | 逼近 Compute-bound |
| 大batch训练 | 高 | Compute-bound |

### Roofline 的局限：漏掉了第三个瓶颈

> 来源：[[heterogeneous-computing-agent-inference]]（Zhao & Liu, 2026）

传统 Roofline 只看 OI（带宽够不够），漏掉了**内存容量**。论文引入 CF（Capacity Footprint）= 每个请求需要多少显存（Bytes/Request），构成四象限：

```
              OI 高                    OI 低
         ┌──────────────────┬──────────────────┐
CF 低    │  Compute-Bound   │  Memory BW-Bound │  ← Roofline 只看这一行
         │  (Prefill)       │  (Decode 高Batch) │
         ├──────────────────┼──────────────────┤
CF 高    │  Capacity-Bound  │  BW + Capacity   │  ← Roofline 看不到！
         │                  │  (Agent 长上下文   │
         │                  │   Decode)         │
         └──────────────────┴──────────────────┘
```

Agent 场景的长上下文 Decode（KV Cache 几百 GB）落在右下角：带宽不够 + 容量也不够。加更多 GPU 也没用——问题不是算力，是显存装不下。这解释了为什么需要 Mooncake 的分层存储（VRAM/DRAM/SSD）和 MLA 的低 CF 设计。

---

## 十一、常用换算速查

```
模型权重显存（BF16）：  参数量(B) × 2 GB
模型权重显存（INT4）：  参数量(B) × 0.5 GB
训练总显存（Adam）：    参数量(B) × 16 GB

KV Cache 每token（BF16，精确值来源：DeepSeek-V3论文）：
  DeepSeek-V3 (MLA, 61层)：  0.069 MB = 70.272 KB/token
  LLaMA 70B  (GQA, 80层)：  0.320 MB = 327.680 KB/token
  LLaMA 405B (GQA, 126层)： 0.516 MB = 516.096 KB/token

常见上下文：
  1K  tokens ≈ 750 个英文单词（或 ~500 中文字符）
  8K  tokens ≈ 6,000 英文单词（≈一篇长文）
  128K tokens ≈ 一本书
  256K tokens ≈ 两本书（LLaMA 3.1 最大上下文）
```

---

## 十二、GPU 编程框架与编译器全景

> 从硬件指令到 Python 一行调用，中间隔了多少层？这一节理清 GPU 编程的完整栈，以及为什么 FlashAttention-4 选择用 CuTe-DSL 而不是 CUDA C++。

### 全栈总览（从底到顶）

```
┌──────────────────────────────────────────────────────────┐
│  你的 Python 代码（PyTorch / JAX / TensorFlow）            │  ← 开发者接触的层
├──────────────────────────────────────────────────────────┤
│  编译器前端 / DSL                                          │
│  ┌─────────┐  ┌─────────┐  ┌───────────┐                │
│  │  Triton  │  │CuTe-DSL │  │ Torch     │                │
│  │(OpenAI)  │  │(NVIDIA)  │  │ Compile   │                │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘                │
│       │              │              │                      │
├───────┼──────────────┼──────────────┼──────────────────────┤
│  底层 GPU 库                                               │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐                  │
│  │ CUTLASS  │  │  cuDNN   │  │ cuBLAS  │                  │
│  │(模板库)   │  │(DNN库)   │  │(BLAS库) │                  │
│  └────┬─────┘  └────┬─────┘  └────┬────┘                  │
│       │              │             │                       │
├───────┼──────────────┼─────────────┼───────────────────────┤
│  CUDA C++ / CUDA Runtime                                   │  ← 传统 GPU 编程
├──────────────────────────────────────────────────────────┤
│  PTX（Parallel Thread Execution）                          │  ← GPU "汇编语言"
├──────────────────────────────────────────────────────────┤
│  SASS（GPU 机器码）                                        │  ← 硬件直接执行
├──────────────────────────────────────────────────────────┤
│  GPU 硬件（Tensor Core, MUFU, FMA, SMEM, TMEM...）        │
└──────────────────────────────────────────────────────────┘
```

### 每一层是什么

**SASS（Shader Assembly）**：GPU 实际执行的机器码，对应具体架构（sm_90 = Hopper, sm_100 = Blackwell）。人类不直接写。

**PTX（Parallel Thread Execution）**：NVIDIA 定义的 GPU "汇编语言"，是架构无关的中间表示。由 ptxas 编译器翻译成特定架构的 SASS。写 PTX 相当于写汇编——最大控制力，但极其繁琐。

**CUDA C++**：最经典的 GPU 编程方式。用 C++ 写 kernel 函数，用 `<<<blocks, threads>>>` 语法启动。开发者手动管理线程、共享内存、同步。FlashAttention-1/2/3 都用 CUDA C++ 写。

**CUTLASS（CUDA Templates for Linear Algebra Subroutines）**：NVIDIA 官方 C++ 模板库，封装了高性能 MMA、TMA、warp 调度等操作。写 CUTLASS 比裸写 CUDA 高效很多，但 C++ 模板元编程非常难读、编译慢（一个 kernel 编译 55 秒）。FA3 就基于 CUTLASS。

**cuBLAS / cuDNN**：NVIDIA 闭源高性能库。cuBLAS 做矩阵乘法，cuDNN 做卷积/attention 等神经网络操作。用户调用一行函数，内部是 NVIDIA 工程师手调的 kernel。FA4 的 benchmark 对比对象之一就是 cuDNN 9.13。优点是开箱即用，缺点是黑盒、不灵活。

### 新一代编译器 / DSL

**Triton（OpenAI）**：用 Python 写 GPU kernel 的 DSL。开发者用 `@triton.jit` 装饰器标记 kernel，用 Python 语法描述 tile 级别的计算（`tl.load`, `tl.dot`, `tl.store`），Triton 编译器自动处理线程映射、共享内存分配、指令调度。

```
优点：门槛极低，Python 程序员几天就能上手
缺点：抽象层级高，底层控制力弱（无法精确控制 TMEM 分配、warp 调度、寄存器使用）
适合：快速原型、中等性能需求的 kernel
```

**CuTe-DSL（NVIDIA CUTLASS 团队）**：用 Python 写 kernel，但编程模型与 CUTLASS C++ **同构**——同样的底层控制力，只是语法从 C++ 模板换成了 Python 元编程。编译链：Python → PTX → SASS。

```
优点：和 CUTLASS C++ 完全等价的底层控制力 + Python 的快速迭代（JIT 编译 2.5s vs C++ 55s）
缺点：仍然需要深刻理解 GPU 架构（TMEM 分配、tile 大小、流水线设计）
适合：追求极致性能的 kernel（FlashAttention-4 就用它）
```

**torch.compile（PyTorch）**：自动将 PyTorch 模型编译优化。用户不写 kernel，编译器自动做算子融合、内存优化。底层可以调用 Triton 生成 kernel。

```
优点：零门槛，加一行 model = torch.compile(model) 就行
缺点：自动优化有上限，无法达到手写 kernel 的性能
适合：通用模型加速，不需要极致优化
```

### 对比总结

| 工具 | 语言 | 控制力 | 开发效率 | 性能上限 | 典型用户 |
|------|------|--------|---------|---------|---------|
| PTX / SASS | 汇编 | 最高 | 极低 | 理论最高 | 几乎没人直接写 |
| CUDA C++ | C++ | 很高 | 低 | 很高 | 传统 GPU 工程师 |
| CUTLASS | C++ 模板 | 很高 | 低（编译慢） | 很高 | FA3、cuDNN 内部 |
| **CuTe-DSL** | **Python** | **很高** | **中高（JIT）** | **很高** | **FA4** |
| **Triton** | **Python** | **中等** | **高** | **中高** | **研究者、快速原型** |
| cuDNN / cuBLAS | API 调用 | 无 | 最高 | 高（但不灵活） | 普通开发者 |
| torch.compile | Python | 无 | 最高 | 中等 | 所有 PyTorch 用户 |

### 为什么 FA4 选择 CuTe-DSL 而不是其他？

```
需求：精确控制 TMEM 分配、2-CTA MMA 模式、ping-pong warpgroup 调度、
     寄存器预算、SMEM bank 分配——这些 Triton 做不到

但 CUTLASS C++ 编译太慢（55s/kernel × 数百变体），开发迭代痛苦

CuTe-DSL = CUTLASS 的底层控制力 + Python 的 JIT 编译速度
         → 编译快 22-32×，表达力不损失
         → FlexAttention 等变体已在 FA4 框架上成功构建
```

### 上层框架怎么调用这些 kernel？

```
用户代码（PyTorch）
  model(input)
      ↓
  F.scaled_dot_product_attention(Q, K, V)     ← PyTorch API
      ↓ 自动选择后端
  ┌─────────────────────────────────────┐
  │ Backend 选择（按优先级）：             │
  │   1. FlashAttention（如果安装了）     │
  │   2. cuDNN attention                  │
  │   3. Math（朴素实现，最慢）            │
  └─────────────────────────────────────┘
```

PyTorch 2.0+ 的 `scaled_dot_product_attention` 会自动选最快的后端。安装了 flash-attn 包就走 FlashAttention kernel，否则 fallback 到 cuDNN 或朴素实现。开发者不需要了解底层 kernel 细节。

### 与已学内容的关联

FlashAttention 演进映射到编程工具链：
- **FA1/FA2**（2022-2023）：CUDA C++，手动管理 tiling 和 SMEM
- **FA3**（2024）：CUTLASS C++ 模板，利用 Hopper 的 WGMMA 和 TMA
- **FA4**（2026）：CuTe-DSL（Python），同等控制力但编译快 20-30×

趋势：**底层控制力不变，开发效率大幅提升**。从 C++ 模板元编程到 Python JIT，门槛降低意味着更多研究者能参与 kernel 优化。

---

## 十三、训练基础：梯度下降与反向传播

### 训练一步的完整流程

```
Forward:  输入 → 逐层矩阵乘 → 得到 loss（衡量输出和目标的差距）
Backward: 从 loss 出发 → 逐层用链式法则求每个 weight 的梯度（∂loss/∂W）
Update:   W_new = W_old - lr × ∂loss/∂W（沿梯度下方走一步）
```

### 每一层 backward 要算两个东西

以线性层 Y = X × W 为例：
```
Forward:  1 次矩阵乘   Y = X × W
Backward: 2 次矩阵乘
  ∂L/∂W = X^T × ∂L/∂Y   ← 这个 weight 的梯度（用来更新 W）
  ∂L/∂X = ∂L/∂Y × W^T   ← 传给上一层（继续反向传播）
```

→ **Backward 计算量 ≈ 2× Forward**
→ 整个训练一步：forward 占 1/3，backward 占 2/3
→ Backward 还需要 forward 时存的中间激活值（X），这就是训练比推理吃显存的原因

### 链式法则（Chain Rule）与反向传播

越靠前的层，"链"越长。第 k 层的 weight 梯度 = 从 loss 一路乘回来的偏导数之积：
```
∂loss/∂W_k = ∂loss/∂out_L × ∂out_L/∂out_{L-1} × ... × ∂out_{k+1}/∂out_k × ∂out_k/∂W_k
```

**具体例子（3 层网络）**：x=2, W1=0.5, W2=3.0, W3=-1.0, target=1.0

```
Forward:  y=W1×x=1.0 → z=W2×y=3.0 → out=W3×z=-3.0 → L=½(-3-1)²=8.0

Backward（从 loss 出发，逐层往回算）：
  起点：∂L/∂out = out - target = -4.0

  Layer 3: ∂L/∂W3 = ∂L/∂out × z = -4.0 × 3.0 = -12.0    ← 用了存的 z
           传给 L2: ∂L/∂z = ∂L/∂out × W3 = -4.0 × (-1.0) = 4.0

  Layer 2: ∂L/∂W2 = ∂L/∂z × y = 4.0 × 1.0 = 4.0          ← 用了存的 y
           传给 L1: ∂L/∂y = ∂L/∂z × W2 = 4.0 × 3.0 = 12.0

  Layer 1: ∂L/∂W1 = ∂L/∂y × x = 12.0 × 2 = 24.0          ← 用了存的 x

Update:  W1: 0.5 - 0.01×24.0 = 0.26
         W2: 3.0 - 0.01×4.0  = 2.96
         W3: -1.0 - 0.01×(-12.0) = -0.88   → Loss 从 8.0 降到 2.77 ✓
```

**反向传播的聪明之处**：每层 backward 只需"上游传来的梯度 × 本层 forward 时的输入"，不用从头重算整条链。复杂度从 O(L²) 降到 O(L)。

**核心记忆**：∂L/∂W = 上游梯度 × **本层 forward 时的输入**。所以 forward 必须存每层的输入（激活值），backward 用完即可释放。这就是训练比推理吃显存的根本原因。

**Activation Checkpointing**：故意不存激活值，backward 时从最近的 checkpoint 重新 forward 算出来。省显存，多花 ~33% 计算。几乎所有大模型训练标配。

---

## 十四、分布式训练：并行策略速查

### Data Parallelism（DP）

```
4 张卡上 weights 完全相同（W_t）
  → 各自用不同数据算 forward + backward → 得到不同的梯度 g0,g1,g2,g3
  → All-Reduce 求平均：g_avg = (g0+g1+g2+g3)/4
  → 每张卡各自更新：W_{t+1} = W_t - lr × g_avg
  → 因为起点相同 + 更新量相同 → 终点也相同
```

**注意：同步的是梯度，不是 weights。Weights 从头到尾没有分叉过。**

等价于：一张卡用 4 倍大 batch 的结果。DP 本质 = 用多卡模拟大 batch。

### Tensor Parallelism（TP）

把单层的 weight 矩阵切分到同节点内多张 GPU：
```
W (h, 4h) 切成 W1 (h, 2h) + W2 (h, 2h)
GPU 0: Y0 = X × W1     GPU 1: Y1 = X × W2
拼接/归约得到完整结果
```
- **每一层**都要通信（All-Reduce / All-Gather）→ 必须用 NVLink（400-900 GB/s）
- 通常只在节点内做，不跨机器

### Pipeline Parallelism（PP）

把模型不同层分给不同 GPU：
```
GPU 0: Layer 1-10     GPU 1: Layer 11-20
Forward 数据从前往后流，Backward 梯度从后往前流
```
- 核心问题：**Bubble**（GPU 等待上游/下游算完时的空闲）
- GPipe：切 micro-batch 填 bubble
- PipeDream 1F1B：warm-up 后交替一次 forward、一次 backward → bubble 更小 + 内存更稳定

### Expert Parallelism（EP）

MoE 专用：256 个 expert 分散到多张 GPU，每张只存一部分 expert 权重。

```
DeepSeek-V3: EP=64, 每张 GPU 存 4 个 expert
  → token 在当前 GPU 算完 attention 后，被 router 指定去某个 expert
  → 该 expert 大概率不在当前 GPU 上 → 必须把 token 发过去

通信模式：每张 GPU 都要给不同 GPU 发 token，同时接收其他 GPU 发来的 token
  → 这就是 All-to-All（多对多 reshuffle）
  → 每个 MoE block 2 次 All-to-All：dispatch（发出去）+ combine（收回来）
  → dispatch 把数据从"按来源 GPU"转为"按目标 expert"
  → combine 是反向操作：从"按 expert"转回"按 token 归属"

为什么 combine 不可省略：
  → 8 个 expert 结果要加权求和，且需要与 H_mid 做残差连接
  → 只有 token 原来的 GPU 持有 H_mid → 结果必须回传
```

### 3D Parallelism 组合

```
节点内（NVLink 高带宽）：Tensor Parallelism
节点间（IB/Ethernet）：  Pipeline Parallelism
多副本（扩大 batch）：   Data Parallelism
```

### 集合通信操作详解

**命名规则**：来自 MPI（消息传递接口，1990s 标准）。"Reduce"不是"减少"，是函数式编程的"归约"——把多个值通过某种操作（sum/max/min）合成一个值。"All"= 所有节点都拿到结果。

**All-Reduce**：归约 + 所有人都拿到结果
```
4 张卡各有一个值：[3] [5] [7] [1]
All-Reduce (sum) → 每张卡都得到 [16]
```
用途：DP 梯度同步（sum 后每卡自己除以 d 得平均值）、TP 激活值合并。
实现：All-Reduce = Reduce-Scatter + All-Gather（环形算法）。

**All-Gather**：每人一块碎片 → 拼成完整数据给所有人
```
4 张卡各有碎片：[A] [B] [C] [D]
All-Gather → 每张卡都得到 [A,B,C,D]
```
用途：ZeRO-3 每层 forward 前拼回完整参数。不是求和，是**拼接**。

**Reduce-Scatter**：先归约（求和），再每人只拿结果的一块
```
卡0: [a1, a2, a3, a4]
卡1: [b1, b2, b3, b4]
卡2: [c1, c2, c3, c4]
卡3: [d1, d2, d3, d4]

Reduce-Scatter →
  卡0 拿到 [a1+b1+c1+d1]       ← 第1块的和
  卡1 拿到 [a2+b2+c2+d2]       ← 第2块的和
  卡2 拿到 [a3+b3+c3+d3]
  卡3 拿到 [a4+b4+c4+d4]
```
用途：ZeRO-2 梯度切分（每卡只保留自己负责的 1/d 梯度总和）。

**All-to-All**：每人给每人发不同的数据（重新洗牌/矩阵转置）
```
卡0: [a0, a1, a2, a3]    ← 分别要给卡0,1,2,3 的数据
卡1: [b0, b1, b2, b3]
卡2: [c0, c1, c2, c3]
卡3: [d0, d1, d2, d3]

All-to-All →
  卡0 收到: [a0, b0, c0, d0]    ← 所有卡给卡0的
  卡1 收到: [a1, b1, c1, d1]
  卡2 收到: [a2, b2, c2, d2]
  卡3 收到: [a3, b3, c3, d3]
```
用途：MoE Expert Parallelism，token 路由到不同 expert 所在的 GPU。不是求和也不是拼接，是**重新分配**。

**Broadcast**：一份 → 广播给所有人
```
卡0 有 [X]
Broadcast → 所有卡都有 [X]
```
用途：参数初始化。

### 三种并行的通信成本排序

| | TP | PP | DP |
|---|---|---|---|
| 通信什么 | 激活值（All-Reduce） | 激活值（点对点） | 梯度（All-Reduce） |
| 频率 | 每层每 micro-batch | 每 micro-batch 每 stage pair | 每 batch 一次 |
| 在计算关键路径上？ | ✅ 不可重叠 | 部分可重叠 | 大部分可与 backward 重叠 |
| 带宽需求 | **最高** → NVLink | 中等 → InfiniBand | **最低**（可遮盖）→ InfiniBand |

→ 这就是为什么 TP 必须在节点内（NVLink），PP 和 DP 可以跨节点（InfiniBand）。

### ZeRO 三阶段

DP 的问题：每卡存完整的 weights + gradients + optimizer states（Adam 的 momentum + variance），冗余存储。

以 1.5B 模型（FP16）、d=8 为例：

| | 参数(FP16) | 梯度(FP16) | Optimizer(FP32) | 每卡总显存 | 节省 |
|---|---|---|---|---|---|
| 纯 DP | 3 GB | 3 GB | 12 GB | **18 GB** | — |
| ZeRO-1 | 3 GB | 3 GB | **1.5 GB**（1/d） | **7.5 GB** | 58% |
| ZeRO-2 | 3 GB | **0.375 GB** | **1.5 GB** | **4.875 GB** | 73% |
| ZeRO-3 | **0.375 GB** | **0.375 GB** | **1.5 GB** | **2.25 GB** | 87% |

ZeRO 的本质：每张卡只**存**一部分，用的时候临时 All-Gather 拼回来，算完丢弃。**用通信换显存**。代价是 ZeRO-3 每层都要跨节点 All-Gather，通信量 = 1.5× 纯 DP。

和 TP 的区别：ZeRO 切的是**状态（存储）**，每张卡仍算完整的 forward/backward。TP 切的是**计算和状态**，每张卡只算一部分矩阵乘法。

PTD-P（Megatron 3D）比 ZeRO-3 高 70% 吞吐的原因：ZeRO-3 的高频 All-Gather 全走 InfiniBand，PTD-P 把高频通信（TP）留在 NVLink 内。

PyTorch FSDP = ZeRO Stage 3 的官方实现。

### Activation Checkpointing

Forward 时不存中间激活值 → backward 时重算 → 省显存但多 ~33% 计算量。几乎所有大模型训练标配。

---

*最后更新：2026-03-12（§九½ 新增 GPU 硬件架构：SM 内部结构、线程层级、延迟隐藏、Tensor Core 原理、内存层级与 Tiling、与训练论文关联）*