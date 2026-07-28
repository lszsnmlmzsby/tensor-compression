# Dense Cross-Attention 版本架构图中文说明

本文对应 `tensor_to_llm_dense_cross_attention_architecture.svg`。下面不按模块定义展开，而是顺着图中的箭头，说明数据从输入到答案依次经过什么。

整张图分为三部分：

1. **Stage 1**：让数值场表示与数值文本表示相互对齐。
2. **Stage 2**：把数值场做成可供 Qwen 反复读取的共享记忆，并训练三处 dense cross-attention。
3. **Inference**：固定所有已训练模块，使用与 Stage 2 相同的路径回答问题。

三个面板分别表示不同过程，面板之间没有直接的数据连线。

## 1. Stage 1：先对齐数值场与数值文本

Stage 1 从同一个二维数值场 `X` 出发，但将它送入上下两条不同路径。

### 1.1 上方 Field branch

上方路径为：

```text
Field X
→ Value-preserving encoder
→ Z
→ Spatial adapter
→ mn field tokens
→ 与 Probe 合并
→ Frozen Qwen 第 ℓ 层
→ h^f
```

各步含义如下：

- **Value-preserving encoder** 把数值场转换成仍保持网格结构的中间表示 `Z`。
- `Z` 的形状记为 `m × n × d_E`。原来的 `m × n` 个位置仍然一一保留，每个位置变成一个 `d_E` 维向量。
- **Spatial adapter** 在每个位置加入二维位置信息，并让不同位置交换全场信息。
- 输出共有 `mn` 个 field tokens，每个网格位置对应一个 token。
- 同一个简短 **Probe** 接在这些 field tokens 后面，再送入冻结的 Qwen。
- 图中的 `h^f` 是最后一个 Probe token 在 Qwen 第 `ℓ` 层得到的向量。它概括了当前数值场的信息。

这里 Qwen 不更新参数，主要训练 encoder 和 spatial adapter。

### 1.2 下方 Text branch

下方路径为：

```text
Field X
→ 写成数值文本
→ Qwen tokenizer
→ Numeric text tokens
→ 与同一个 Probe 合并
→ Frozen Qwen 第 ℓ 层
→ h^t
```

这条路径不直接处理二维网格，而是先把场中的数值按固定格式写成文本，再由 Qwen tokenizer 转成 tokens。

上下两条路径使用完全相同的 Probe，也在相同的 Probe 位置读取向量。因此，`h^f` 和 `h^t` 表示的是同一个数值场，只是输入形式不同。

### 1.3 对比对齐

右侧矩阵比较一批样本中所有 `h^f` 和 `h^t`：

- 同一个数值场产生的 field/text 向量位于矩阵对角线，应当更加接近。
- 不同数值场的向量位于非对角线，应当更容易区分。

Stage 1 训练结束后，场分支能产生与数值文本含义一致的表示。后续模型会从这些已经学好的场表示和相关检查点出发，而不是重新从头学习数值场。

## 2. Stage 2：建立可反复读取的场记忆

Stage 2 有三股主要数据流：

```text
场的内容信息 ─┐
              ├→ Shared tensor memory → 三处 cross-attention → Qwen → Answer
场的精确数值 ─┘

自然语言问题 ───────────────────────────────→ Qwen
```

数值场先形成共享记忆；自然语言问题走普通 Qwen 文本路径。二者不是在输入端直接拼接，而是在 Qwen 的第 8、20、32 层之后逐次交互。

### 2.1 从 Field X 得到 Z

左侧第一段为：

```text
Field X → Trained encoder → Z
```

这个 encoder 已在 Stage 1 训练完成，因此图中写作 **Trained encoder**。Stage 2 不再更新它，灰色框也表示参数保持固定。实际训练时，`Z` 可以预先计算并保存在缓存中；图里的 encoder 主要表示 `Z` 的来源。

得到 `Z` 后，数据分成内容路径和数值路径。

### 2.2 上方内容路径

内容路径为：

```text
Z
→ Trained adapter
→ Content LayerNorm
→ Content memory C(X)
```

- **Trained adapter** 就是 Stage 1 已经训练好的 spatial adapter，并在 Stage 2 保持固定。它让每个网格位置既保留本地信息，也能知道全场的情况。
- **Content LayerNorm** 对这些向量做统一整理，使后面的模块更容易稳定地读取。它含有可学习的缩放和偏置参数，并在 Stage 2 中更新，因此图中使用红色可训练样式。
- 输出记为 `C(X)`。它主要描述每个位置的空间背景、局部形态和全场上下文。

`C(X)` 仍有 `mn` 个位置，位置顺序与原数值场一致。

### 2.3 下方精确数值路径

数值路径为：

```text
Z 的第一个通道 z
→ Trainable per-cell value encoder
→ Value embeddings E_z(X)
```

`Z` 的第一个通道 `z` 是直接保留的数值通道。value encoder 对每个位置分别处理 `z`，并加入若干不同形式的数值特征，帮助模型区分非常接近的数值。

输出 `E_z(X)` 也是 `mn` 个向量，每个向量重点保存对应位置的精确数值信息。

因此，两条路径的分工可以简单理解为：

- `C(X)` 回答“这个位置处在怎样的空间环境中”；
- `E_z(X)` 回答“这个位置的具体数值是什么”。

### 2.4 组成 Shared tensor memory

对于每一个网格位置，模型将该位置的 `C(X)` 和 `E_z(X)` 放在一起，形成一对信息。全部位置共同组成共享记忆：

```text
Shared tensor memory
= mn 个 [content state, value embedding] 对
```

这个记忆有两个重要特点：

1. **保持网格位置**：仍然有 `mn` 个 cell，不会把整个数值场压成一个向量。
2. **与问题无关**：同一个数值场只需建立一次记忆，不同问题可以重复读取它。

图中的共享记忆只保存 `C(X)` 和 `E_z(X)`。真正用于注意力计算的 `K` 和 `V`，由第 8、20、32 层后的三个 bridge 分别生成，并不是提前存进共享记忆的一套公共 `K/V`。

### 2.5 自然语言问题进入 Qwen

右上方的问题路径为：

```text
Natural-language question q
→ Frozen Qwen（tokenizer + embedding、decoder、LM head）
```

图中将 tokenizer、embedding、decoder 和 LM head 收进同一个 **Frozen Qwen** 框，表示它们属于同一条连续的文本处理路径。问题仍然先变成普通文本 token 和对应向量，再按正常顺序进入冻结的 Qwen；合并画法并没有改变实际计算顺序。这里不会把 `mn` 个场向量直接塞到问题 token 前面。

### 2.6 在第 8、20、32 层后读取场记忆

Qwen 的文本向量会在三个位置读取共享记忆：

图中用三个圆形符号 `CA₈`、`CA₂₀`、`CA₃₂` 表示这三处 cross-attention，分别接在 Qwen 第 8、20、32 层之后。

```text
Qwen blocks 1–8
→ dense cross-attention
→ Qwen blocks 9–20
→ dense cross-attention
→ Qwen blocks 21–32
→ dense cross-attention
→ Qwen blocks 33+ + LM head
```

这里的 **dense** 表示：每个文本 token 都可以查看全部 `mn` 个场位置，而不是先挑少量位置，也没有单独的行列路由。

每个 cross-attention bridge（桥接模块）都有自己的一套 `K/V` 转换：

- `K` 主要由内容表示 `C(X)` 产生，用来判断当前文本更需要关注哪些位置；
- `V` 同时使用内容表示和数值表示，把被读取位置的信息送回 Qwen；
- 三个 bridge 参数互不共享，因此不同深度可以采用不同的读取方式。

bridge 还带有一个可学习的门，用来控制场信息加入当前 Qwen 向量的强度。这样 Qwen 的原始文本能力可以保留，同时逐层加入数值场信息。

### 2.7 从 LM head 得到答案与训练损失

最后的数据流为：

```text
Frozen blocks 33+ + LM head
→ Answer logits
├→ Answer
└→ QA + matched losses
```

`Answer logits` 是选择答案之前的分数。训练损失直接使用这些分数，而不是等选出离散答案后再计算。

QA 相关损失包括：

- `L_choice`：让正确候选答案的分数最高；
- `0.02 L_answer`：让答案 token 的语言模型预测正确；
- `0.1 L_matched`：在同一个数值场的一组相关问题中，进一步拉开正确与错误答案的分数。

### 2.8 数值重建支路

value embeddings 还有一条只在训练时使用的短支路：

```text
E_z(X)
→ Trainable linear z readout
→ z-hat
→ Reconstruction loss
```

线性读出尝试从 `E_z(X)` 恢复原来的 `z`。如果能恢复，说明 value embeddings 没有丢失关键数值。代码内部使用 `Smooth L1(z-hat, z)` 计算这项 reconstruction loss；图中只保留更直观的损失名称。这个损失权重较小，作用是辅助保持数值精度，不负责直接生成答案。

Stage 2 的总目标可以写成：

```text
L = L_choice
  + 0.02 L_answer
  + 0.1 L_matched
  + 0.01 L_reconstruction
```

训练时主要更新图中红色模块：Content LayerNorm、value encoder、linear z readout，以及第 8、20、32 层后的三个 dense cross-attention bridge。Qwen 主体、Trained encoder 和 Trained adapter 保持冻结。

## 3. Inference：固定结构后回答问题

推理时的数据流与 Stage 2 基本相同：

```text
Field X
→ Trained encoder
→ Z
├→ Trained adapter → Fixed Content LayerNorm → C(X)
└→ Fixed value encoder → E_z(X)
→ Shared tensor memory

Question q
→ Frozen Qwen（tokenizer + embedding、decoder、LM head）
→ 在 L8、L20、L32 后读取 Shared tensor memory
→ LM head
→ Restricted-choice answer
```

与训练阶段相比，推理阶段有三点变化：

1. 所有模块都固定，不再更新参数。
2. 不再计算 QA、matched 或 z reconstruction loss。
3. 最终在允许的答案候选中比较分数，选择分数最高的答案，因此图中写作 **Restricted-choice answer**。

同一个数值场面对多个问题时，`C(X)`、`E_z(X)` 和 Shared tensor memory 可以重复使用，只需替换右侧的自然语言问题。

## 4. 各模块在三个阶段中的状态

| 模块 | Stage 1 | Stage 2 | Inference |
|---|---|---|---|
| Value-preserving encoder | 训练 | 冻结，可读取缓存 `Z` | 固定 |
| Spatial adapter | 训练 | 冻结 | 固定 |
| Content LayerNorm | 不使用 | 训练 | 固定 |
| Per-cell value encoder | 不使用 | 训练 | 固定 |
| Linear z readout | 不使用 | 训练，仅用于辅助损失 | 不使用 |
| Dense cross-attention bridges | 不使用 | 训练 | 固定 |
| Qwen 主体与 LM head | 冻结 | 冻结 | 冻结 |

## 5. 一句话概括

Stage 1 先让“数值场向量”和“数值文本向量”表达相同含义；Stage 2 再把每个场位置拆成内容信息与精确数值信息，组成共享记忆，让自然语言问题在 Qwen 的三个深度逐次读取这份记忆，最后得到答案。
