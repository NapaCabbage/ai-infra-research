---
title: "SGLang: Efficient Execution of Structured Language Model Programs"
tags: [inference-serving, framework, structured-generation, kv-cache]
subfield: inference-serving
venue: "NeurIPS 2024"
date: 2026-03-09
authors: [Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Mosharaf Chowdhury, Ion Stoica]
institution: [UC Berkeley, Stanford]
url: "https://arxiv.org/abs/2312.07104"
status: 已读
rating: ⭐⭐⭐⭐
---

# SGLang：前端语言 + 后端运行时 Co-design 的推理框架

## 一句话总结

SGLang 的核心主张：推理框架不应该把 LLM 当无状态黑盒（`generate(prompt) → text`），而应该**理解应用层的程序结构**，用前端语言暴露结构信息，后端运行时针对性优化。RadixAttention 和 FSM Constrained Decoding 是这个思想的两个具体实例。

---

## 核心 Thesis：前端 + 后端 Co-design

```
之前的框架（vLLM / TGI）：
  接口：generate(prompt) → text
  问题：框架不知道应用长什么样，无法做跨请求优化

SGLang 的思路：
  第一层（Frontend）：设计编程语言，让用户声明程序结构
    → "这些请求共享前缀"、"输出要符合这个 JSON schema"

  第二层（Backend - RadixAttention）：
    → 利用前缀结构，自动复用 KV Cache

  第三层（Backend - FSM Constrained Decoding）：
    → 利用输出格式约束，编译成高效的 token mask
```

类比：就像编译器论文同时讲"循环展开"和"常量折叠"——两个优化完全独立，但统一在"编译器理解程序结构来优化执行"的 thesis 下。

---

## 贡献 1：RadixAttention（KV Cache 跨请求复用）

### 问题

实际 LLM 应用中，大量请求共享前缀：

```
请求 1：[System Prompt] + "翻译这句话：Hello"
请求 2：[System Prompt] + "翻译这句话：World"
请求 3：[System Prompt] + "翻译这句话：Goodbye"
                ↑
         完全相同的前缀，但 vLLM 每次都重新算 KV Cache
```

vLLM 的 PagedAttention 解决了**单请求内**的显存碎片化，但没有解决**跨请求**的 KV Cache 重复计算。

### Radix Tree（基数树）数据结构

**什么是 Radix Tree？**

Radix Tree = 压缩版 Trie（前缀树）。普通 Trie 每个节点存一个字符，Radix Tree 把只有一个子节点的链压缩成一条边：

```
普通 Trie：                    Radix Tree：
    (root)                      (root)
     |                          /    \
     t                      "test"  "team"
     |                        |
     e                      "ing"
    / \
   s   a
   |   |
   t   m
   |
   i
   n
   g
```

**为什么选 Radix Tree 而不是 Hash Map？**

```
Hash Map：只能精确匹配完整 key
  → "System Prompt + 用户问题A" 和 "System Prompt + 用户问题B"
    是两个不同的 key，无法发现共享前缀

Radix Tree：天然支持最长前缀匹配
  → 自动发现 "System Prompt" 是共享前缀
  → 对应的 KV Cache 只算一次，后续请求直接复用
```

### RadixAttention 工作流

```
请求 1 到达："[SYS] 翻译：Hello"
  → Radix Tree 查找 → 没有匹配
  → 完整计算 KV Cache
  → 存入 Radix Tree：[SYS] → [SYS]翻译：Hello

请求 2 到达："[SYS] 翻译：World"
  → Radix Tree 查找 → 最长匹配 = "[SYS]"
  → 复用 [SYS] 的 KV Cache，只算 "翻译：World" 的部分
  → 省掉了 System Prompt 的重复 Prefill！

请求 3 到达："[SYS] 总结这段话..."
  → Radix Tree 查找 → 最长匹配 = "[SYS]"
  → 同样复用 [SYS] 的 KV Cache
```

LRU 淘汰策略管理显存：当显存不够时，淘汰最久没用的 KV Cache 节点。

---

## 贡献 2：FSM Constrained Decoding（结构化输出约束）

### 问题

很多应用要求 LLM 输出特定格式（JSON、SQL、正则匹配等），之前的做法低效：

```
方法 1：Prompt 里说 "请输出 JSON" → 不可靠，经常格式错误
方法 2：生成完再 validate，错了就 retry → 浪费算力
方法 3：每步 decode 时检查约束 → 正确但慢（每步都要解析语法）
```

### FSM（有限状态机）方案

SGLang 把输出格式约束**预编译成 FSM**，decode 时只需要查表：

```
JSON Schema: {"name": string, "age": number}

预编译成 FSM：
  状态 0: 期待 '{'           → 只允许 '{' 的 token
  状态 1: 期待 '"name"'      → 只允许 '"name"' 的 token
  状态 2: 期待 ':'           → 只允许 ':' 的 token
  状态 3: 期待 string value  → 允许所有 string token
  状态 4: 期待 ','           → 只允许 ',' 的 token
  ...

每个状态预编译一个 token mask（哪些 token 合法）
```

### 关键优化：预编译 Token Mask

```
传统方法：每步 decode 都要
  1. 解析当前状态
  2. 查语法规则
  3. 遍历整个词表（32000+ tokens）判断合法性
  → 每步都有 CPU 开销

SGLang：离线预编译
  1. 对每个 FSM 状态，提前算好合法 token 的 bitmap
  2. Decode 时直接用 bitmap mask logits → O(1) 查表
  → 几乎零额外开销
```

---

## 为什么 RadixAttention 和 FSM 放在一篇论文里？

这是我最初的困惑：两个技术解决完全不同的问题，为什么合在一起？

答案是它们背后的 **thesis 统一**：

```
                        RadixAttention              FSM Constrained Decoding
利用什么结构？          请求之间的前缀共享           输出的语法格式约束
结构从哪来？            Frontend 语言声明            Frontend 语言声明
后端怎么优化？          KV Cache 跨请求复用          预编译 token mask
框架"看到了"什么？      不同请求的前缀关系           输出必须满足的语法规则
```

共同点：都是框架"理解了应用的结构"之后才能做的优化。如果框架只有 `generate(prompt) → text` 的接口，这两个优化都做不了。

---

## SGLang 前端语言示例

```python
@function
def multi_turn_qa(s, questions):
    s += system("You are a helpful assistant.")  # 共享前缀
    for q in questions:
        s += user(q)
        s += assistant(gen("answer", max_tokens=256))
    # SGLang 自动识别：所有 questions 共享 system prompt → RadixAttention

@function
def json_extract(s, text):
    s += user(f"Extract info from: {text}")
    s += assistant(gen("result", regex=r'\{"name": "\w+", "age": \d+\}'))
    # SGLang 自动把 regex 编译成 FSM → Constrained Decoding
```

Frontend 语言让用户**声明**结构，Backend 自动利用这些结构做优化。

---

## 我的理解

- SGLang 的 idea 层次很清晰：不是两个独立 trick 的拼凑，而是有统一的 thesis（前端暴露结构 + 后端利用结构）
- RadixAttention 从数据结构角度看很优雅：Radix Tree 天然适合最长前缀匹配，比 hash map 更适合 KV Cache 复用场景
- FSM Constrained Decoding 的核心是"预编译"——把运行时的语法检查提前到编译时，这个思路在 CS 里非常经典（JIT → AOT）
- 工程影响：SGLang 现在是使用最广泛的推理框架之一，说明这个 co-design 思想确实有用
- 和之前学的联系：RadixAttention 建立在 PagedAttention（[[vllm-pagedattention]]）之上，解决了 vLLM 没解决的跨请求复用问题

---

## 关联笔记

- [[vllm-pagedattention]] — PagedAttention 解决单请求内显存碎片化，RadixAttention 进一步解决跨请求 KV Cache 复用
- [[orca-continuous-batching]] — Orca 的 continuous batching + SGLang 的 RadixAttention = 调度层 + 缓存层的完整优化
- [[flash-attention]] — FlashAttention 优化单次 Attention 计算，SGLang 优化多次请求间的 KV Cache 管理，层次不同
- [[speculative-decoding]] — Speculative Decoding 加速单请求的 Decode，SGLang 加速多请求的 Prefill（前缀复用），互补
- [[AI-Infra-Cheatsheet]] — KV Cache 公式可以算出前缀复用节省了多少显存

---

*学习方式：扫读论文 + Claude 对话理解核心思想*
*最后更新：2026-03-09*
