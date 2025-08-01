---
title: transformer
date: 2025-07-28
categories: [笔记, LLM]
tags: [LLM]
---

当然可以！我帮你把内容排版得更整齐清晰，方便阅读和展示：

---

# QKV

> Query（Q）查询向量：要找的信息，比如你在搜索框输入的问题
> Key（K）键向量：索引的信息，比如网页的关键词
> Value（V）值向量：实际内容，比如网页的正文内容
> 通过计算 Query 和所有 Key 的匹配度（相关性），选出最相关的 Value，并加权组合，得到最终的结果。

### 💡 QKᵀ × V 的简洁说明（Self-Attention 的核心步骤）

* `QKᵀ`：大小为 **3 × 3**，第 `i` 行表示第 `i` 个词对其他词的关注程度（由 Query 和 Key 点积得到）
* `V`：大小为 **3 × 512**，每个词的表示向量

➡️ 执行矩阵乘法：

```text
(Q × Kᵀ) × V   →   (3×3) × (3×512) = 3×512
```

每一行是一个词的最终表示，融合了它对所有词（包括自己）的注意力权重与那些词的向量：

```text
QKᵀ[0] = [1, 2, 3]   # 第一个词对第1/2/3个词的注意力权重
Output[0] = 1 * V[0] + 2 * V[1] + 3 * V[2]   →   长度为 512 的向量
```

> 实际中会在 `QKᵀ` 上先除以 √d，再做 softmax。

---

# **Transformer 从 Encoder 到 Decoder 的完整过程**

> **重点前提：**
> 你输入的是一个句子（10 个 token，每个是 512 维向量），采用 8 个注意力头。

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

---

#### ② Add + Norm + FeedForward + Norm（详解）

在 Transformer 编码器的每一层中，**自注意力模块之后**，还会经过如下模块：

👉 **第一步：Add & Norm（残差连接 + 层归一化）**

```text
输入：X ∈ [10, 512]
自注意力输出：A ∈ [10, 512]
```

操作：

```text
Z1 = LayerNorm(X + A)
```

* **Add**：把注意力输出 `A` 与原始输入 `X` 相加（残差连接，防止梯度消失）。
* **Norm**：对结果进行 Layer Normalization，稳定训练。

> ✅ 结果仍是 `[10, 512]`，形状不变，只是数据分布被归一化了。

---

👉 **第二步：前馈网络（FeedForward）**

对每个时间步（词）单独进行两个线性层的变换：

```text
FF(Z1) = max(0, Z1 · W1 + b1) · W2 + b2
```

* `W1` 通常把维度升到更高，如 `[512 → 2048]`
* 然后 ReLU 激活
* 再用 `W2` 降回来 `[2048 → 512]`

所以这其实是：

```text
Z2 = FFN(Z1) ∈ [10, 512]
```

---

#### 前馈层示例与 ReLU 非线性激活

* 假设输入是 `[5, 512]`，表示 5 个词，每个词是 512 维向量。

* 前馈层首先用权重矩阵 `W1`（大小 `[512, 2048]`）将每个词向量升维：
  输入 `[5, 512]` × `W1` → 输出 `[5, 2048]`

* 然后对 `[5, 2048]` 的结果逐元素应用 **ReLU 激活**：

  $$
  \text{ReLU}(x) = \max(0, x)
  $$

  它是非线性函数，把负数变为 0，正数保持不变，赋予模型学习复杂模式的能力。

* 接着用权重矩阵 `W2`（大小 `[2048, 512]`）降维回 `[5, 512]`：
  `[5, 2048]` × `W2` → `[5, 512]`

* 最终，前馈层将每个词的向量从 512 维“扩展-非线性-压缩”，得到更丰富的表示，同时保持形状不变。

---

👉 **第三步：再一次 Add & Norm**

```text
Z3 = LayerNorm(Z1 + Z2)
```

* 又一次残差连接，防止信息丢失
* 再次归一化，保持分布稳定

---

✅ 输出保持在 `[10, 512]`，送入下一层

因为每一层都保持输入输出维度一致（通过残差连接 + 前馈的升降维组合），所以你可以**堆叠多层 Transformer**（通常是 6 层或 12 层）而不会改变 shape。

---

### 📌 最终输出：

```text
Encoder output E ∈ [10, 512]
```

表示输入句子中每个词（共 10 个 token）在上下文语境下的深层语义表示（512 维向量）。

这个输出可以被送给 Decoder 或下游任务（分类、翻译、生成等）。

---

## ✅ 总结多头和多层

| 设计       | 多头注意力         | 多层堆叠             |
| ---------- | ------------------ | -------------------- |
| 并行性     | 并行（多个头）     | 顺序（逐层）         |
| 表达多样性 | 多个头看不同关系   | 每层提取不同抽象层次 |
| 功能       | 丰富信息维度       | 加深理解深度         |
| 类比       | 多个视角看一个问题 | 一步步深入分析问题   |

## ✅ 总结自注意力层和前馈层
| 模块           | 功能说明                   |
| -------------- | -------------------------- |
| 自注意力层     | 让每个词看全局，获取上下文 |
| 前馈神经网络层 | 让每个词自己思考，重新编码 |

---

## 🧱 第二步：Decoder（生成第 8 个 token 的详细流程）

### 输入：

```text
X_tgt ∈ [7, 512]    # 前 7 个已生成目标 token 的表示
E ∈ [10, 512]       # Encoder 输出（10 个源 token 的语义表示）
```

---

### 每一层包含 3 个主要模块：

---

#### ① Masked Self-Attention（Q = K = V = X\_tgt）

> 💡 当前 decoder 使用所有已生成的目标 token 来“自我关注”，只允许看见自己和之前的（用 mask 限制未来）

##### 步骤详解：

1. **输入：**
   `X_tgt ∈ [7, 512]`，前 7 个生成的 token 向量

2. **线性变换生成 Q, K, V：**

   ```text
   Q = X_tgt · Wq ∈ [7, 512]
   K = X_tgt · Wk ∈ [7, 512]
   V = X_tgt · Wv ∈ [7, 512]
   ```

3. **切分为多头：**

   ```text
   Q, K, V → reshape → [7, 8, 64]
   ```

4. **计算注意力：**

   ```text
   attention_scores = Q · Kᵀ / √64 → [7, 7]
   ```

   * 注意力权重矩阵是 `[7, 7]`，表示第 i 个 token 对前 1～i 个 token 的关注程度
   * **下三角 mask** 会将未来位置的得分设为 `-∞`，防止泄漏未来信息
  
  ```text
  1 -∞ -∞ -∞ -∞ -∞ -∞
  1  2 -∞ -∞ -∞ -∞ -∞
  1  2  3 -∞ -∞ -∞ -∞ 
  1  2  3  4 -∞ -∞ -∞
  1  2  3  4  5 -∞ -∞
  1  2  3  4  5  6 -∞
  1  2  3  4  5  6  7
  ```
1. **softmax + 加权求和 V：**

   ```text
   attention = softmax(masked_scores) · V → [7, 64]
   ```

2. **合并 8 个头 → 拼接输出：**

   ```text
   合并所有 head → [7, 512]
   ```

---

#### ② Encoder-Decoder Attention（Q = X\_tgt, K = V = E）

> 💡 让 decoder 的每个生成 token 学会**从源句子中找线索**
> 恩其实这里只拿最后一个token去做Encoder-Decoder Attention生成新词就行
##### 步骤详解：

1. **输入：**

   * `X_tgt ∈ [7, 512]` 当前目标 token 序列
   * `E ∈ [10, 512]` Encoder 输出

2. **生成 Q、K、V：**

   ```text
   Q = X_tgt · Wq ∈ [7, 512]
   K = E · Wk ∈ [10, 512]
   V = E · Wv ∈ [10, 512]
   ```

3. **切分为多头：**

   ```text
   Q ∈ [7, 8, 64], K ∈ [10, 8, 64], V ∈ [10, 8, 64]
   ```

4. **跨序列注意力计算：**

   ```text
   attention_scores = Q · Kᵀ / √64 → [7, 10]
   attention_weights = softmax(scores)
   context = attention_weights · V → [7, 64]
   ```

5. **合并头 → 输出 `[7, 512]`**

> ✅ 每个目标 token 都对源句中 10 个 token 做 attention，决定该参考哪些词翻译当前词。

---

#### ③ FeedForward + Add + Norm

> 💡 每个位置单独进行两层线性+ReLU变换，提升表示复杂度

##### 步骤详解：

1. **前馈网络（对每个 token 向量独立处理）：**

   ```text
   FF(x) = ReLU(x · W1 + b1) · W2 + b2
   W1 ∈ [512, 2048], W2 ∈ [2048, 512]
   ```

2. **输入 `[7, 512]` → 输出 `[7, 512]`**，形状保持不变

3. **加残差 + LayerNorm 两次：**

   ```text
   Z1 = LayerNorm(X_tgt + self_attention_output)
   Z2 = LayerNorm(Z1 + FFN(Z1))
   ```

---

### 🔚 输出层（预测第 8 个 token）

Decoder 最后一步会取出**当前最后一个 token 的向量**，映射为词表概率：

```text
input: Z2[-1] ∈ [1, 512]    # 第 7 个 token 的输出
↓ Linear → [1, vocab_size]
↓ softmax → 概率分布 → 选出第 8 个 token
```

---

## 🔄 自回归生成逻辑

```text
X_tgt = [token1, token2, ..., token7]
↓ Decoder 输出第 8 个 token
↓ 将 token8 拼到 X_tgt
↓ 继续下一轮生成
```

Decoder 每次都用**累积的目标序列**作为输入，每次预测一个 token。

---

## ❓ 那纯 Decoder 模型还“关心问题”吗？

> ✅ 关心，只是方式不同 —— **问题是 prompt 的一部分，直接喂给了 Decoder**

在 GPT 等纯 Decoder 结构中，不再使用 Encoder 来单独编码问题或上下文。相反，它会把“问题 + 回答起始标记”拼成一个完整的序列送入 Decoder：

```text
输入 token（Prompt）：
Q: What is the capital of France? A:
```

Decoder 会依次生成下一个 token，比如：

```text
→ Paris
```

**注意力计算时：**

* 当前 token（比如 `Paris`）产生一个 Query
* 所有之前 token（包括问题部分）都作为 Key 和 Value
* 于是，模型在预测时自然就会“参考”问题文本

这意味着模型**从 attention 的角度，确实在“关注问题”**：

```text
Q = 当前 token 的表示（如 A:）
K = 所有之前 token 的表示（包括 Q: What...）
V = 同上
→ QKᵀ → softmax → V → 输出 Paris
```

✅ 所以，虽然不显式区分“输入”和“输出”，但生成过程确实通过上下文记住并利用了问题信息。

---

## 📌 总结一轮生成的关键操作（第 8 个 token）

| 步骤 | 模块                      | 输入维度                 | 输出维度     | 作用说明                     |
| ---- | ------------------------- | ------------------------ | ------------ | ---------------------------- |
| 1    | Masked Self-Attention     | `[7, 512]`               | `[7, 512]`   | 关注历史 token，不看未来     |
| 2    | Encoder-Decoder Attention | `[7, 512]` + `[10, 512]` | `[7, 512]`   | 参考源句上下文，对齐翻译目标 |
| 3    | FeedForward + Norm        | `[7, 512]`               | `[7, 512]`   | 增强非线性表达能力           |
| 4    | 输出预测第 8 个 token     | `Z2[-1] ∈ [1, 512]`      | `[1, vocab]` | 得到下一个词的概率分布       |

---

## ✅ 总结 QKV 的来源对比

| 模块                 | Query 来源         | Key/Value 来源     | 是否 Mask |
| -------------------- | ------------------ | ------------------ | --------- |
| Encoder Self-Attn    | Encoder 输入       | Encoder 输入       | ❌ No     |
| Decoder Self-Attn    | Decoder 已生成部分 | Decoder 已生成部分 | ✅ Yes    |
| Encoder-Decoder Attn | Decoder            | Encoder 输出       | ❌ No     |

---

## ✅ 输入维度变化总览

| 阶段                 | 维度              |
| -------------------- | ----------------- |
| 输入 token embedding | `[10, 512]`       |
| Encoder 输出         | `[10, 512]`       |
| Decoder 输入         | `[1, 512]`        |
| Decoder 输出         | `[1, 512]`        |
| Linear + Softmax     | `[1, vocab_size]` |

---


## 🚀 加速技巧：缓存 K/V 以支持高效自回归生成

> 每次生成一个 token 时，不必重复计算前面 token 的 Key/Value，可直接复用！

---

#### 🧠 为什么要缓存？

在自回归生成中，decoder 每一步都使用之前生成的所有 token：

* 第 8 步需要用到前 7 个 token 的 **Self-Attention**
* 如果每次都重新算 1\~7 的 `K` 和 `V`，会非常低效（重复计算）

✅ **缓存法**：每次生成时，只计算**当前 token 的 Q/K/V**，然后把 K/V 加入缓存，下次直接用！

---

#### ⚙️ 如何缓存？

在 Decoder 的 **Masked Self-Attention** 和 **Encoder-Decoder Attention** 中都可以缓存：

| 模块                      | 可缓存内容               | 特点说明                               |
| ------------------------- | ------------------------ | -------------------------------------- |
| Masked Self-Attention     | K, V 来自 `X_tgt`        | 前面生成过的 token 不变，可缓存其 K/V  |
| Encoder-Decoder Attention | K, V 来自 encoder 输出 E | encoder 输出固定，一开始就可一次性缓存 |

---

#### 🧩 第 8 步时的缓存使用流程（示例）

```text
# 假设第 7 步已缓存：
K_prev, V_prev ∈ [7, 8, 64]

# 当前第 8 个 token：
x₈ ∈ [1, 512]
↓
Q₈, K₈, V₈ = Linear(x₈) → reshape → [1, 8, 64]

# 因为是masked, 旧的不需要和新加行(因为看不到未来词)做计算 所以只需要对新加的一行和整体K计算
# 因此需要缓存之前的K(K_prev)
# 之后再和所有V计算
# 拼接缓存：
K_all = concat(K_prev, K₈) ∈ [8, 8, 64]
V_all = concat(V_prev, V₈) ∈ [8, 8, 64]

# 用 Q₈ 做 masked attention：
attention(Q₈, K_all, V_all) → 输出当前 token 的向量
```

* 下一轮生成时，只需再加一行，就能扩展到 `[9, 8, 64]`，递增即可。
* 这样可以把复杂度从 **O(n²)** 降为 **O(n)**（n 为当前生成位置）

---

#### ✅ 总结：缓存机制带来的优势

| 优点         | 描述                                         |
| ------------ | -------------------------------------------- |
| 减少重复计算 | 只计算当前 token 的 K/V，历史的直接复用      |
| 提升推理效率 | 每步生成计算量从 O(n²) 降到 O(n)             |
| 支持高效部署 | 尤其在长序列生成时（如翻译、文本生成）更高效 |

---

