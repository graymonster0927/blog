 ---
title: transformer
date: 2025-07-28
categories: [笔记, LLM]
tags: [LLM]
---


# **Transformer 从 Encoder 到 Decoder 的完整过程**

> **重点前提：**
> 你输入的是一个句子（10个 token，每个是 512 维向量），采用 8 个注意力头。

---

## 🌍 整体结构（Encoder → Decoder）

```text
          [Input Tokens]                [Target Tokens]
               │                             │
          Embedding                     Embedding (shifted)
               │                             │
          Positional Encoding          Positional Encoding
               │                             │
           Encoder Blocks ─────┐     Decoder Blocks
               ↓              │         │
     Encoder Output (memory)  └──► Encoder-Decoder Attention
                                         ↓
                                Linear + Softmax → 下一个 token
```

---

## 🔧 我们假设输入：

* 源句子（英文）长度 = 10，维度 = 512 → `X_src ∈ [10, 512]`
* 目标句子（法语）长度 = 8（逐步生成） → `X_tgt ∈ [8, 512]`
* 多头数 = 8，每头维度 = 64

---

## 🧱 第一步：Encoder

### 输入：

```text
X_src: [10, 512]
```

### 编码器结构（每层）：

#### ① Self-Attention（Q=K=V=X\_src）

* 投影：

  $$
  Q = X W^Q,\quad K = X W^K,\quad V = X W^V \quad \text{→ shape: } [10, 512]
  $$
* 拆成 8 个头：

  $$
  Q_h, K_h, V_h ∈ [8, 10, 64]
  $$
* 注意力计算：

  $$
  \text{Attention}_h = \text{softmax}(Q_h K_h^T / \sqrt{64}) V_h \quad → [8, 10, 64]
  $$
* 合并：

  $$
  \text{Concat}_{h=1}^8 → [10, 512]
  $$
* 输出线性变换：

  $$
  O = \text{Concat}(...) W^O ∈ [10, 512]
  $$

#### ② Add + Norm + FeedForward + Norm

输出保持在 `[10, 512]`，送入下一层（通常堆叠 6 层）

最终：

```text
Encoder output E ∈ [10, 512] （语义表示）
```

---

## 🧱 第二步：Decoder

### 输入：

```text
X_tgt: [1, 512] （目标最后一个token, 如果是第一次用<BOS>/<s>）
E: [10, 512] （encoder 的输出）
```

### 解码器结构（每层）：

#### ① Masked Self-Attention（Q=K=V=X\_tgt）

* 和 Encoder 的 attention 一样，但加了 causal mask 只看自己和前面：
* 这里比如生成了8个token, 前7个已经缓存了K V

```text
输入：
X_tgt: [1, 512]
↓ Linear → Q, K, V ∈ [1, 512]
↓ reshape → 8 个头 → [1, 8, 64]
↓ QK^T + mask → softmax → attention → 乘 V
↓ 拼接 → [1, 512]
```

---

#### ② 编码器-解码器注意力（Q=X\_tgt, K=V=E）

这是重点！

* **Q** 来自 decoder 当前生成的 token 向量 `X_tgt`
* **K, V** 来自 encoder 输出的上下文表示 `E`

```text
Q = X_tgt W^Q → [1, 512]
K = E W^K     → [10, 512]
V = E W^V     → [10, 512]
↓ reshape 成多头
Q ∈ [1, 8, 64], K/V ∈ [1, 10, 64]
↓ 计算 QK^T → softmax → 乘上 V
↓ 输出 [1, 512]
```

这一步是 Decoder “读取” Encoder 编码好的上下文。

---

#### ③ FFN + Add + Norm

维度仍然是 `[1, 512]`

---

### 最终：

送入 Linear + Softmax：

```text
output = Linear([1, 512]) → softmax([1, vocab_size]) → 下一个 token 概率分布
```

---



## 🔄 每步生成一个 token：

Decoder 是自回归的，每次预测一个 token，再喂回去继续预测下一个。

---

## ✅ 总结 QKV 的来源对比

| 模块                   | Query 来源      | Key/Value 来源  | 是否 Mask |
| -------------------- | ------------- | ------------- | ------- |
| Encoder Self-Attn    | Encoder 输入    | Encoder 输入    | ❌ No    |
| Decoder Self-Attn    | Decoder 已生成部分 | Decoder 已生成部分 | ✅ Yes   |
| Encoder-Decoder Attn | Decoder       | Encoder 输出    | ❌ No    |

---

## ✅ 输入维度变化总览

| 阶段                 | 维度                |
| ------------------ | ----------------- |
| 输入 token embedding | `[10, 512]`       |
| Encoder 输出         | `[10, 512]`       |
| Decoder 输入         | `[1, 512]`        |
| Decoder 输出         | `[1, 512]`        |
| Linear + Softmax   | `[1, vocab_size]` |

---
