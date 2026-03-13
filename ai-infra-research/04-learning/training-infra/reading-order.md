---
tags: [learning]
subfield: training-infra
created: 2026-03-05
updated: 2026-03-12
---

# Training Infra 精选阅读路径

> 你有 LLaMA 3.1 论文基础 + inference serving 系统阅读经验，估计 3-5 周完成
> 已通过对话校准：DP/TP/PP/EP 概念、梯度下降与反向传播、All-Reduce 含义

---

## 阶段 0：训练基础回顾 ✅ 已通过对话完成

通过对话校准了以下概念：
- **Data Parallelism**：同步的是梯度不是 weights，每张卡起点相同 → 算不同数据 → 梯度不同 → All-Reduce 平均梯度 → 各自用相同梯度更新 → 终点也相同
- **Forward vs Backward**：forward 每层 1 次矩阵乘，backward 每层 2 次（∂L/∂W 和 ∂L/∂X）；backward 需要 forward 存的激活值；backward 计算量 ≈ 2× forward
- **Pipeline Parallelism**：把层分给不同 GPU，bubble 用 micro-batch 填充（GPipe → 1F1B）
- **Tensor Parallelism**：切 weight 矩阵分到同节点内多卡，每层都要通信
- **All-Reduce**：Reduce（归约）+ All（广播给所有人），DP 里用来平均梯度
- **ZeRO**：三个 stage 分别切分 optimizer states / gradients / weights，省显存但增通信

---

## 阶段 1：并行化基础（第 1-2 周）

### 必读 ①：Megatron-LM 原始论文（TP 的奠基） ✅ 已完成
- **论文**：*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*
- **发表**：arXiv 2019 (NVIDIA)
- **笔记**：[[megatron-lm]]
- **掌握程度**：理解 MLP 切分（Figure 3a：列切 A + 行切 B = 1 次 All-Reduce）、Self-Attention 切分（Figure 3b：按 head 切）、每层 4 次 All-Reduce（Figure 4）、TP=8 上限的物理原因、pre-LN 对训练稳定性的重要性（Figure 7）、TP+DP 混合并行
- **链接**：https://arxiv.org/abs/1909.08053

### 必读 ②：Megatron-LM 扩展版（TP + PP + DP 组合） ✅ 已完成
- **论文**：*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*
- **发表**：SC 2021
- **笔记**：[[megatron-lm-3d]]
- **掌握程度**：理解 PTD-P 组合策略（TP 节点内 + PP 跨节点 + DP 扩规模）、三种 pipeline 调度（GPipe → 1F1B → Interleaved 1F1B）、bubble 公式 (1/v)×(p-1)/m、scatter/gather 通信优化、三条 Takeaway 指导原则、1T 模型 3072 GPU 52% 利用率、PTD-P 比 ZeRO-3 高 70%
- **链接**：https://arxiv.org/abs/2104.04473

### 选读 ③：ZeRO（DeepSpeed）
- **论文**：*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*
- **发表**：SC 2020
- **为什么读**：和 Megatron 的 TP 是解决"显存不够"的两条路线，理解区别很重要
- **关键概念**：三个 stage 各切什么、通信量如何变化、和 TP 的 trade-off
- **读法**：重点 §3-5（三个 stage 的推导），可以跳过实现细节
- **链接**：https://arxiv.org/abs/1910.02054

---

## 阶段 2：通信与内存优化（第 2-3 周）

### 必读 ④：通信原语 ✅ 已通过对话完成
- 已掌握：All-Reduce（归约+广播）、All-Gather（拼接）、Reduce-Scatter（归约+切分）、All-to-All（重新洗牌）
- 关键理解：All-Reduce = Reduce-Scatter + All-Gather（这是 SP 零额外通信的数学基础）
- 已更新到 Cheatsheet §十四

### 必读 ⑤：Activation Checkpointing + Sequence Parallelism ✅ 已完成
- **论文**：*Reducing Activation Recomputation in Large Transformer Models*
- **发表**：MLSys 2023 (Megatron 团队)
- **笔记**：[[reducing-activation-recomputation]]
- **掌握程度**：理解 SP 核心（TP 管 h 维度，SP 管 s 维度，LayerNorm/Dropout 按序列切分，全层激活被 t 除）、零额外通信原理（All-Reduce 拆成 RS+AG）、Selective Recomputation（只丢 attention 内部大 tensor，省 70% 显存多 2-3% 计算）、组合后激活降 5× 吞吐提升 30% MFU 56%
- **链接**：https://arxiv.org/abs/2205.05198

---

## 阶段 3：工业实践（第 3-4 周）

### 必读 ⑥：DeepSeek-V3 Technical Report（训练部分） ✅ 已完成（Section 3）
- **论文**：*DeepSeek-V3 Technical Report*（2024）
- **笔记**：[[deepseek-v3-training]]
- **掌握程度**：理解 PP=16 + EP=64 + ZeRO-1 无 TP 架构、DualPipe 双向流水线（细粒度拆分 + 计算通信重叠 + 2× 参数代价）、MoE 每 block 2 次 All-to-All（dispatch+combine）及 IB+NVLink 分层利用、FP8 混合精度（fine-grained quantization + accumulation precision promotion）、极致内存优化（重算 RMSNorm、EMA 放 CPU、共享 embedding+output head）
- **待深入**：DualPipe 完整调度时序、FP8 per-group scaling 数学、Warpgroup MMA 硬件机制（需 GPU 架构背景）
- **链接**：https://arxiv.org/abs/2412.19437

### ~~必读 ⑦：LLaMA 3.1 vs DeepSeek-V3 对比阅读~~ → 跳过
- **跳过原因**：LLaMA 3.1 训练部分是 Megatron 3D 的标准实践（TP+PP+DP），并行策略无创新，bubble 优化不显著。DeepSeek-V3 在 DualPipe、EP+All-to-All 优化、FP8 训练等方面远超 LLaMA 3.1 的工程深度。对比价值有限。
- **结论**：LLaMA 3.1 训练 infra ≈ Megatron 标准配置 + 长上下文退火 + GQA 省 KV Cache，已通过 Megatron 论文覆盖。

---

## 阶段 4：RL Training Infra ← 2026 年最热方向

> 背景：DeepSeek-R1（2025.01）证明 RL 可以让模型"学会思考"后，RL post-training 成为行业标配。但 RL 训练 infra 和 pre-training 完全不同——每步训练都需要先做推理（rollout），推理占总时间 80-85%，长尾问题严重。算法层面（GRPO/DAPO/REINFORCE++）也没有收敛，框架需要同时灵活和高效。

### 前置知识：已通过对话完成 ✅

通过对话校准了以下 RL 核心概念：

- **RL vs Pre-training 的区别**：pre-training 数据是现成的，RL 的数据（rollout）要自己生成，占 80-85% 时间
- **PPO**：4 个模型（Policy + Reference + Reward + Critic），用 Critic 预测 baseline 来算 advantage，每个 token 位置都有 advantage。Proximal = clipping 约束每次更新幅度
- **GRPO**：3 个模型（砍掉 Critic），同一 prompt 生成 16 个回答做组内相对比较，advantage = (reward - 组均值) / 组标准差。整条回答共享一个 advantage
- **DeepSeek-R1 的极简做法**：2 个模型（连 Reward Model 也砍了），纯 rule-based reward（答案对=1 错=0），证明纯 RL + 简单 reward 就能涌现长链推理
- **Advantage 的作用**：RL 版的"标签"，乘在 loss 上控制梯度方向和大小。正数=加强该回答概率，负数=削弱，绝对值=调整幅度。没有 advantage 则所有回答概率都被提高（包括错的）
- **Reward Model 来历**：pre-trained LLM + 把 LM Head 换成标量输出头 + 用人类偏好比较对训练（Bradley-Terry loss）
- **Critic Model 来历**：pre-trained LLM + value head，RL 过程中在线训练（自监督），每个位置预测"从这里到结尾的期望 reward"
- **RL 训练的 backward**：和 pre-training 完全一样的反向传播机制，区别只在 loss = -advantage × log P(回答)
- **分离式 vs 混合式架构**：OpenRLHF 用两组 GPU 分别做推理和训练（空间换时间），veRL 同一组 GPU 交替切换推理/训练模式（时间换空间）
- **长尾问题**：一个 batch 里最长的 rollout 拖住所有 GPU

### 待深入理解（诚实记录）

- [ ] advantage 的直觉理解还不够深（"为什么乘 advantage"能复述但还没内化）
- [ ] 分离式 vs 混合式的具体实现细节不清楚
- [ ] RL 算法迭代（DAPO/REINFORCE++ 对 GRPO 的改进）未接触
- [ ] Rollout 长尾优化的具体方案未接触

---

### Day 1：读 DeepSeek-R1 §2-3
- **论文**：*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*（2025.01）
- **链接**：https://arxiv.org/abs/2501.12948
- **重点**：
  - §2: GRPO 算法的完整公式（对照我们对话中的理解）
  - §3: 训练流程（冷启动 SFT → 纯 RL → 拒绝采样 → 二次 SFT → 二次 RL）
  - 关注 reward 设计（rule-based 怎么具体实现的）
- **带着问题读**：R1 的多阶段训练，每个阶段解决什么问题？为什么不能直接一步到位？

### Day 2：读 DAPO + 扫 REINFORCE++
- **DAPO**：*DAPO: An Open-Source LLM Reinforcement Learning System at Scale*（ByteDance, 2025.03）
  - 对 GRPO 的四项改进：decoupled clipping、去 KL、dynamic sampling、token-level loss
  - 链接：https://arxiv.org/abs/2503.14476
- **REINFORCE++**：嵌在 OpenRLHF 论文中
  - 单样本（k=1）就能工作，比 GRPO 更 token-efficient
  - 用 global baseline 而非 group-local normalization
- **对比分析**（选读）：*Comparative Analysis of PPO, GRPO, and DAPO*
  - 链接：https://arxiv.org/abs/2512.07611
- **核心问题**：DAPO 的 dynamic sampling 解决什么问题？为什么 REINFORCE++ 只需 1 个样本？

### Day 3：读 OpenRLHF + veRL（RL Infra 核心）
- **OpenRLHF**：*OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework*
  - Ray + vLLM 分离式架构
  - 链接：https://arxiv.org/abs/2501.03262
- **veRL (HybridFlow)**：*HybridFlow: A Flexible and Efficient RLHF Framework*（字节, EuroSys 2025）
  - 3D-HybridEngine：训练和推理间 zero-redundancy 参数迁移
  - 链接：https://arxiv.org/abs/2409.19256
- **重点**：画出两种架构的系统图，对比 GPU 利用率、权重同步方式、适用场景

### Day 4：Rollout 长尾优化 + 总结
- **RollPacker**：长尾问题最直观的解法（tail batching）
  - 链接：https://arxiv.org/abs/2509.21009
- **ROLL Flash**（扫读）：异步 RL 方案
  - 链接：https://arxiv.org/abs/2510.11345
- **RollArt**（扫读）：Agent RL 场景
  - 链接：https://arxiv.org/abs/2512.22560
- **总结**：更新 cheatsheet，写 RL Infra 全景笔记

### 选读（有余力时）

- **选读 ⑪：故障恢复与可靠性**
  - LLaMA 3.1 提到 ~1-2% GPU 故障率
  - 关注：checkpoint 策略、快速恢复、silent data corruption 检测
- **选读 ⑫：长上下文训练**
  - Ring Attention、Sequence Parallelism 的演进

---

## 读完之后你能做什么

- 理解 LLaMA 3.1 论文 Section 3 的每个技术选择背后的原因
- 看 DeepSeek-V3 技术报告时，能判断他们的训练创新点是否真实有价值
- 对比 training infra 和 inference infra 的不同挑战（已有基础）
- 理解 RL training 为什么需要训练+推理耦合的全新 infra 架构
- 判断 RL 框架（OpenRLHF/veRL/AReaL）的技术选择和 trade-off
- 从投研角度：理解 NVLink/InfiniBand 为什么是 NVIDIA 的护城河，集群调度软件的价值
