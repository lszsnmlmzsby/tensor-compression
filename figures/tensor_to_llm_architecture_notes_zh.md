# Field-to-LLM 架构图中文说明

本文方法把二维单变量数值场转换为可直接输入语言模型的连续 embedding，并通过两个训练阶段建立数值场、空间位置和自然语言问题之间的联系。图中应区分训练阶段与推理路径，也应区分全局 field tokens 和问题相关的局部 evidence tokens。

## 1. 通用场表示

输入为规则网格上的数值场

```text
X in R^(m x n).
```

Value-preserving encoder 不做空间下采样。每个位置保留原始标量通路，并附加局部卷积特征：

```text
z_(r,c) = [X_(r,c); local_features_(r,c)] in R^d_E.
```

Spatial adapter 在每个位置加入二维行列位置编码，通过空间 Transformer 交换全局信息，并保留位置对应的局部残差。其输出为

```text
P(X) = [p_1, ..., p_(mn)] in R^(mn x d_L).
```

一个 grid cell 对应一个输出槽位，但不意味着该 token 只包含一个标量。空间 Transformer 使每个 token 可以包含全场上下文，局部残差则保留该位置的直接数值通路。

## 2. Stage 1：Field--text alignment

Stage 1 使用同一个数值场构造两条路径：

```text
Field path: [P(X); shared probe]
Text path:  [Embed(Serialize(X)); shared probe]
```

两条路径使用同一个简短 probe，且 probe 后不附加答案。它们经过共享且冻结的 Qwen，在同一语义位置，即最后一个 probe token，读取第 ell 个 Transformer block 的 hidden state：

```text
h_i^f = Readout_ell([P(X_i); probe])
h_i^t = Readout_ell([Embed(Serialize(X_i)); probe]).
```

训练使用双向对比目标，使配对的 field/text 表示相互接近，并区分 batch 内的有效负样本。固定 whitening 仅用于改善 Stage 1 损失的数值条件和检索诊断；相同的固定变换应用于两条路径，它不是可学习的双分支 projection，也不会进入 Stage 2 或推理路径。Stage 1 同时在原生 LLM hidden space 中施加约束，避免只在辅助空间中获得较好的检索结果。

正式 Stage 1 不依赖 AE decoder。Encoder 中的原始数值通路由结构直接保证，卷积特征和 spatial adapter 由 alignment objective 联合训练。

## 3. Stage 2：自然语言条件下的数值场推理

Stage 1 的 field tokens 本身与问题无关。Stage 2 先进行一次 direct QA warm start，使同一套 grid-aligned interface 产生可供问答使用的全局表示：

```text
G(X) = [g_1, ..., g_(mn)].
```

随后冻结生成 G(X) 的 global branch，并训练一个 question-conditioned local evidence reader。Local reader 不替换 G(X)，而是在其前面增加少量与问题相关的 evidence tokens。

### 3.1 问题表示

自然语言问题先单独经过冻结 Qwen。当前实现使用第 2 层和第 6 层的 token-level hidden states，经可训练投影和加权融合得到问题上下文：

```text
T(q) = sum_l softmax(zeta)_l A_l(Q_l(q)).
```

两个 learned role queries 通过 cross-attention 读取 T(q)，分别得到 role states。Role 是通用的证据槽位，不是任务标签：点值问题通常启用一个 role，两点或两区域比较可启用两个 role，全局极值定位可以关闭局部 role。

### 3.2 二维路由和局部证据

每个 role state 分别产生 row query 和 column query。所有行和列具有由固定轴位置编码及可训练残差构造的 row/column keys。行列相似度相加后在整个网格上 softmax：

```text
omega_(j,r,c) = softmax_(r,c)(row_score_(j,r) + col_score_(j,c))
v_j = sum_(r,c) omega_(j,r,c) g_(r,c).
```

v_j 经过一个小型 residual bottleneck 得到 refined evidence。Learned gate 决定该 role 是否有效；无效 role 不仅被置零，也会从 LLM attention mask 中移除。

### 3.3 最终 LLM 输入

最终回答阶段的 embedding 顺序为

```text
[local evidence tokens; global field tokens; question embeddings].
```

其中：

- local evidence tokens 随自然语言问题改变；
- global field tokens 保留完整场信息和每个 cell 的槽位；
- question embeddings 是普通 tokenizer 和 LLM embedding 的输出。

完整冻结 Qwen 在这条拼接序列上预测答案。Whitening、AE decoder、训练目标和监督坐标都不进入推理输入。

## 4. Stage 2 训练目标

Stage 2 使用以下六类监督：

```text
L_stage2 = lambda_choice L_choice
         + lambda_LM     L_LM
         + lambda_route  L_route
         + lambda_gate   L_gate
         + lambda_group  L_group
         + lambda_swap   L_swap.
```

- `L_choice`：在合法答案集合上的 restricted-choice cross entropy；
- `L_LM`：答案 token 与 EOS 的低权重 token-level NLL；
- `L_route`：路由分布在目标 cell 或目标区域上的 cross entropy；
- `L_gate`：每个 role 是否应启用的 binary cross entropy；
- `L_group`：同一 field 上不同坐标问题的 matched-group margin loss；
- `L_swap`：交换两个问题的 local evidence 后，要求正确答案 NLL 变差。

坐标和任务元数据只用于构造 routing/gate 监督，不作为模型输入。Swap 只交换 local evidence 及其 gate mask，保留 owner question 和原 field 的 global tokens。

## 5. 训练日程和参数状态

| 模块 | Stage 1 | Stage 2 local-reader training | Inference |
|---|---|---|---|
| Value-preserving encoder | trainable | frozen/cached | fixed |
| Spatial/global interface | trainable | direct QA warm start 后冻结 | fixed |
| Qwen | frozen | frozen | frozen |
| Question layer projections | absent | trainable | fixed |
| Role cross-attention | absent | trainable | fixed |
| Row/column router | absent | trainable | fixed |
| Evidence bottleneck and gates | absent | trainable | fixed |
| Whitening | loss/diagnostic only | absent | absent |
| AE decoder | absent | absent | absent |

正式训练顺序为：global interface 先进行 direct QA warm start；随后冻结 global branch，先训练 routing 与 gate，再联合训练答案和局部证据目标；最终进行一次较小规模的 continuation。低学习率完整数据续训没有通过预先规定的验证准入，因此最终 checkpoint 保留其 parent 参数。

## 6. 架构图应表达的关键关系

1. Stage 1 的 field/text 两条路径使用同一个 shared probe，并在相同 probe 位置读取 hidden state。
2. 图中使用 `m`、`n`、`d_E` 和 `mn` 等符号，具体网格大小与 token 数放在实验设置中。
3. Stage 2 中自然语言问题同时产生 question hidden states 和普通 question embeddings。
4. Local reader 同时读取问题上下文与冻结的 global field tokens，并产生少量 local evidence tokens。
5. 最终顺序明确画成 `[local; global; question]`，而不是只画 field tokens 与问题文本的直接拼接。
6. 推理图中不出现 whitening、decoder、routing target、监督坐标或训练 loss。
