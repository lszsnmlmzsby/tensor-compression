+.
# 大型数值张量压缩-重建-LLM 适配完整训练流程

本文档给出一套可工程落地的完整流程，用于训练如下系统：

- 输入：2D/3D/4D 数值张量
- 中间表示：可压缩的连续 latent tokens
- 输出 1：高保真重建张量
- 输出 2：可通过小型 adapter 映射为 LLM soft prompt 的 embedding
- 下游：分类、检索、异常检测、摘要生成、问答、报告生成等

目标不是一次性追求“最先进”，而是先建立一条稳定、可复现、可迭代的生产级训练链路。

---

## 1. 总体目标与推荐策略

推荐采用两阶段主线、四阶段完整流程：

1. 先训练张量压缩器，优先拿到高保真重建能力。
2. 再冻结或半冻结压缩器，训练 latent 到 LLM embedding 的 adapter。
3. 如果需要真实压缩率，再加入量化、超先验或残差误差控制分支。
4. 最后再针对具体下游任务做轻量微调。

推荐架构：

```text
tensor X
-> preprocessing
-> N-D encoder
-> continuous latent tokens Z
-> optional quantizer / hyperprior / residual tail
-> decoder
-> reconstructed tensor X_hat

Z + metadata
-> adapter (MLP or Q-Former/Perceiver-style)
-> soft prompt embeddings P
-> frozen or lightly tuned LLM
-> downstream task head / generation
```

原则：

- 压缩重建和语言对齐解耦训练
- 连续 latent 优先，离散 latent 作为后续增强选项
- 统一 2D/3D/4D 的方式是“统一数据接口 + 轴感知主干”，不是暴力使用统一 4D 卷积
- 每个阶段都必须输出可验证产物，避免只看最终下游指标

---

## 2. 建议的工程目录结构

建议最少采用如下目录组织：

```text
project/
  configs/
    data/
    model/
    train/
    task/
  data/
    raw/
    interim/
    processed/
    manifests/
  scripts/
    prepare_data.py
    build_manifest.py
    compute_stats.py
    train_compressor.py
    train_rd.py
    train_adapter.py
    train_downstream.py
    evaluate_reconstruction.py
    evaluate_downstream.py
    export_artifacts.py
  src/
    data/
    models/
    losses/
    trainers/
    metrics/
    tasks/
    utils/
  outputs/
    runs/
    checkpoints/
    reports/
    exports/
  docs/
```

最低要求：

- 所有实验必须由配置文件驱动
- 所有训练阶段都保存 `config.yaml`、`git commit id`、`seed`、`metrics.json`
- 所有数据分割都保存固定 manifest，禁止训练时随机重新划分

---

## 3. 数据规范

### 3.1 样本最小单元

每个样本建议包含以下字段：

| 字段 | 含义 |
|---|---|
| `sample_id` | 唯一 ID |
| `tensor_path` | 张量文件路径 |
| `shape` | 原始形状 |
| `dtype` | 数据类型 |
| `axes` | 轴语义，如 `["T","H","W","C"]` |
| `variables` | 变量名列表 |
| `units` | 单位列表 |
| `source` | 数据来源 |
| `split` | train/val/test |
| `task_labels` | 分类或回归标签 |
| `task_text` | 可选文本描述、报告、问答对 |
| `mask_path` | 可选，无效值掩码 |
| `norm_group` | 归一化统计所属组 |

推荐将元信息放在 `jsonl` 或 `parquet` manifest 中，而张量本体放在二进制文件中。

### 3.2 张量存储格式

推荐优先级：

1. `zarr`
2. `hdf5`
3. `npy/npz`

选择原则：

- 数据大且需要随机切块时，优先 `zarr`
- 单样本较大时，必须支持 chunked read
- 尽量避免训练时从文本格式恢复张量

### 3.3 数据切分原则

切分必须以“真实泛化边界”为准，不能只随机打散。

常见切分方式：

- 按样本来源切分
- 按时间段切分
- 按设备或实验批次切分
- 按场景类型切分

禁止：

- 同一个大张量切块后，部分块在训练集、部分块在测试集，但测试任务又依赖全局相关性
- 训练、验证、测试共享归一化统计时未记录来源

---

## 4. 数据准备流程

### 4.1 原始数据接入

输入要求：

- 明确原始 shape
- 明确每个维度含义
- 明确数值范围、无效值定义、缺失值定义
- 明确是否等间隔采样

产物：

- `data/manifests/raw_manifest.jsonl`

### 4.2 数据体检

在开始训练前，先做离线体检并生成报告。

至少检查：

- shape 分布
- 每个变量的均值、方差、极值、分位数
- NaN/Inf 比例
- 全零切片比例
- 异常尖峰比例
- 不同 split 的分布偏移

产物：

- `outputs/reports/data_profile_train.json`
- `outputs/reports/data_profile_val.json`
- `outputs/reports/data_profile_test.json`

硬门槛：

- 如果 NaN/Inf 比例高于业务可接受范围，必须先做数据清洗策略再训练
- 如果 train/val/test 分布明显失衡，先修正切分再训练

### 4.3 统一轴顺序

必须将原始张量映射到统一轴语义顺序。

推荐内部顺序：

- 2D：`[H, W, C]`
- 3D：`[D, H, W, C]`
- 4D：`[T, D, H, W, C]` 或 `[T, H, W, C]`

注意：

- 对于不同数据源，不能假设第一个维度永远是时间
- 必须把原始轴定义写入 manifest

### 4.4 归一化

推荐采用“每变量、每归一化组”的稳健归一化策略。

优先顺序：

1. `z-score`
2. `robust z-score`
3. `min-max`
4. `log + z-score`

推荐保存：

- `mean`
- `std`
- `min`
- `max`
- `p01`
- `p99`

这些统计量既用于训练，也用于后续 LLM metadata token。

产物：

- `data/processed/norm_stats.json`

### 4.5 掩码与缺失值处理

如果存在无效区域或缺失值：

- 保留显式 mask
- 不要直接把缺失值当真实 0
- 损失函数中对无效区域做 masking

### 4.6 分块与 patch 策略

对大型张量推荐离线或在线切块。

默认建议：

- 2D：`128x128` 或 `256x256`
- 3D：`64x64x64` 或 `96x96x96`
- 4D：时间短窗加空间块，如 `T=4~16`，空间 `64~128`

切块时保存：

- 原张量 ID
- patch 起点坐标
- patch shape
- patch 所属 split

产物：

- `data/manifests/patch_manifest_train.parquet`
- `data/manifests/patch_manifest_val.parquet`
- `data/manifests/patch_manifest_test.parquet`

### 4.7 数据增强

仅使用不会破坏物理语义的增强。

可选增强：

- 加性微噪声
- 随机裁剪
- 随机时间窗
- 小幅幅值缩放

谨慎使用：

- 任意旋转
- 任意翻转
- 任意重采样

如果轴带有明确物理含义，增强必须经过业务确认。

---

## 5. 模型定义

## 5.1 压缩器

压缩器由以下模块组成：

- `patch_embed`
- `axis-aware encoder`
- `latent bottleneck`
- `decoder`
- 可选 `quantizer`
- 可选 `hyperprior`
- 可选 `residual tail`

### 5.2 推荐压缩器参数起点

第一版可从以下配置起步：

| 参数 | 建议值 |
|---|---|
| latent token 数 `K` | 32 或 64 |
| latent 维度 `d_z` | 128 或 256 |
| 编码器层数 | 6 到 12 |
| 下采样层级 | 3 到 4 |
| prompt token 数 `L` | 16 或 32 |
| adapter 类型 | baseline 用 MLP，正式版用 Q-Former |

### 5.3 轴感知编码器

推荐模块组成：

- 局部卷积块提取邻域结构
- 沿单轴的 axial attention 或长卷积提取长程依赖
- 通道混合 MLP 做非线性融合
- 层次式下采样控制计算量

4D 输入不要默认使用纯 4D self-attention，优先采用：

- 时间轴单独建模
- 空间或体轴单独建模
- 交替堆叠

### 5.4 latent 设计

推荐采用连续 latent tokens，而非一开始就使用离散 codebook。

原因：

- 更利于高保真数值重建
- 更容易作为 soft prompt 的前驱表示
- 训练更稳定

可以将 latent 表示为：

```text
Z.shape = [K, d_z]
```

其中 `K` 为固定 token 数。

### 5.5 Decoder

要求：

- 能从 `Z` 恢复 patch 或整块张量
- 不允许原始高分辨率信息跨 bottleneck 直通
- 可以使用层次上采样和残差细化块

### 5.6 Adapter

推荐两条线并行维护：

1. `MLP projector`
2. `Q-Former/Perceiver-style adapter`

baseline：

```text
Z -> Linear -> GELU -> Linear -> P
```

正式版：

- 初始化 `L` 个 learnable query tokens
- queries 对 `Z` 做 cross-attention
- 输出固定长度 prompt embedding `P`

### 5.7 Metadata token

除了 `Z`，还建议将以下信息编码为 metadata token：

- 原始 shape
- 轴语义
- 变量名
- 单位
- 归一化统计
- 样本来源
- 时间戳或工况标签

这些 token 与 `P` 一起送入 LLM，可显著减少歧义。

---

## 6. 训练阶段总览

完整训练分为 6 个阶段。

1. 阶段 A：数据统计与 manifest 固化
2. 阶段 B：连续 latent 压缩器预训练
3. 阶段 C：率失真训练与量化增强
4. 阶段 D：残差误差控制分支训练
5. 阶段 E：adapter 对齐 LLM embedding
6. 阶段 F：下游任务训练与验收

并不是所有项目都必须跑完 6 个阶段。

最小可行路径：

- A -> B -> E -> F

如果只关注重建：

- A -> B -> C -> D

---

## 7. 阶段 A：数据统计与 manifest 固化

### 7.1 目标

- 建立稳定数据接口
- 固化 split
- 计算归一化统计
- 构建 patch manifest

### 7.2 输入

- 原始张量数据
- 原始元信息

### 7.3 输出

- `raw_manifest.jsonl`
- `patch_manifest_{train,val,test}.parquet`
- `norm_stats.json`
- `data_profile_*.json`

### 7.4 执行流程

1. 扫描原始数据目录并收集样本
2. 解析 shape、dtype、axes、variable info
3. 生成原始 manifest
4. 按规则切分 train/val/test
5. 计算每变量统计量
6. 生成 patch manifest
7. 抽样可视化并人工检查

### 7.5 验收条件

- 每个 patch 能追溯回原始样本
- split 无泄漏
- 归一化统计已固定且已落盘
- 至少完成一次人工 spot check

---

## 8. 阶段 B：连续 latent 压缩器预训练

### 8.1 目标

训练 `Encoder + Decoder`，优先优化重建质量，不强求真实码率最优。

### 8.2 输入

- 归一化后的 patch
- mask
- metadata

### 8.3 输出

- `compressor_stage_b_best.pt`
- `compressor_stage_b_last.pt`
- 重建评估报告

### 8.4 推荐损失函数

总损失建议：

```text
L_total =
  λ_rec * L_rec
  + λ_rel * L_rel
  + λ_grad * L_grad
  + λ_spec * L_spec
  + λ_reg * L_reg
```

各项含义：

- `L_rec`：MSE/MAE/Charbonnier
- `L_rel`：相对误差，适合数值范围变化大的变量
- `L_grad`：梯度损失，约束局部结构
- `L_spec`：频域损失，约束谱结构
- `L_reg`：轻量正则项

建议起点：

- `L_rec`: Charbonnier 或 MSE
- `L_rel`: masked relative L1
- `L_grad`: Sobel 或一阶差分

### 8.5 训练配置建议

| 项目 | 建议 |
|---|---|
| optimizer | AdamW |
| lr | `1e-4` 到 `3e-4` |
| weight decay | `1e-2` 或 `5e-2` |
| scheduler | cosine decay + warmup |
| precision | bf16 或 fp16 |
| grad clip | 1.0 |
| epochs | 50 到 200 |
| early stopping | 监控 val reconstruction |

### 8.6 监控指标

每个 epoch 至少记录：

- train/val reconstruction loss
- train/val relative error
- PSNR 或 SNR
- SSIM 或结构相似指标
- max absolute error
- max relative error
- 推理吞吐
- GPU 显存

如果是科学任务，建议增加：

- 导出量误差
- 守恒量误差
- 关键区域误差

### 8.7 验收条件

压缩器进入下一阶段前，至少满足以下之一：

- 在核心变量上达到预设误差门槛
- 对业务关键导出量误差可接受
- 重建质量优于 Tucker/TT 等 baseline

### 8.8 常见失败点

- 只优化 MSE，边界和尖峰被抹平
- latent 太小，导致结构信息丢失
- 数据切块过小，模型看不到全局上下文
- 归一化不稳定，训练震荡

---

## 9. 阶段 C：率失真训练与量化增强

### 9.1 目标

在尽量保住重建质量的前提下，提高压缩率，使 latent 具备更强的可编码性。

### 9.2 适用场景

以下情况建议执行本阶段：

- 需要真实压缩比
- 需要落盘存储或传输
- 需要评估比特率与精度的平衡

如果当前只关心 soft prompt，可暂时跳过。

### 9.3 训练内容

加入：

- 量化近似
- 超先验 `hyperprior`
- 率失真损失

损失写法可采用：

```text
L_rd = λ_dist * L_dist + β_rate * L_rate
```

其中：

- `L_dist` 继承阶段 B 的重建损失
- `L_rate` 为估计比特率或码长上界

### 9.4 实践建议

- 不要从头开始训练，基于阶段 B 权重继续训练
- 先小 `β_rate`，逐步增大
- 保留多个不同码率版本的 checkpoint

### 9.5 输出

- `compressor_rd_lowrate.pt`
- `compressor_rd_midrate.pt`
- `compressor_rd_hifidelity.pt`
- 率失真曲线报告

### 9.6 验收条件

至少输出一条可解释的率失真曲线：

- x 轴：estimated bitrate / latent size
- y 轴：重建误差或业务指标

---

## 10. 阶段 D：残差误差控制分支

### 10.1 目标

进一步降低局部误差，尤其是尖峰、边界、关键区域误差。

### 10.2 实现方式

推荐两种可落地做法：

1. 神经 residual head
2. 传统 error-bounded compressor 压残差

组合方式：

```text
X -> main compressor -> X_main
R = X - X_main
R -> residual branch -> R_hat
X_hat = X_main + R_hat
```

### 10.3 什么时候需要本阶段

- max error 很高
- 某些关键区域的误差不可接受
- 业务方更关心最坏情况而非平均误差

### 10.4 输出

- `compressor_residual_best.pt`
- 残差误差分析报告

---

## 11. 阶段 E：adapter 对齐 LLM

### 11.1 目标

将 `Z` 变成 LLM 可用的 soft prompt embedding，而不破坏压缩器的重建能力。

### 11.2 输入

- 已训练好的压缩器
- latent `Z`
- metadata tokens
- 文本标签、文本报告、问答对或其他下游文本监督

### 11.3 默认冻结策略

推荐：

- 冻结 Encoder
- 冻结 Decoder
- 冻结 LLM 主体
- 仅训练 adapter

可选：

- 在 adapter 收敛后，轻微解冻 Encoder 末端 1 到 2 层

不推荐：

- 一开始就联训整个压缩器和 LLM

### 11.4 训练目标

常见训练方式：

1. 条件文本生成
2. 问答
3. 检索对齐
4. 分类/回归
5. 多任务联合

推荐先做两类任务：

- 结构化问答
- 自动摘要/报告生成

原因：

- 最容易验证 soft prompt 是否真的携带张量信息
- 最容易构建监督数据

### 11.5 监督数据构造

优先级：

1. 人工标注报告
2. 业务规则生成摘要
3. 自动统计模板生成文本
4. 下游标签转自然语言模板

示例：

```text
输入：latent + metadata + prompt "请总结该张量的主要结构特征"
目标：对应文本摘要
```

### 11.6 Adapter 训练损失

生成任务：

- 语言建模交叉熵

检索或对齐任务：

- 对比学习损失

分类任务：

- 交叉熵或 BCE

推荐先单任务训练，再做多任务联合。

### 11.7 推荐训练顺序

1. 先用 `MLP projector` 做 baseline
2. 若效果有限，再切换到 `Q-Former`
3. 仅当 adapter 确认有效后，再考虑末端联合微调

### 11.8 输出

- `adapter_mlp_best.pt`
- `adapter_qformer_best.pt`
- `soft_prompt_eval.json`

### 11.9 验收条件

至少满足以下一条：

- 文本生成质量达到业务可用水平
- 检索召回优于直接用原始统计量做基线
- 分类/回归优于直接用压缩器均值池化后的浅层 MLP 基线

---

## 12. 阶段 F：下游任务训练

### 12.1 下游任务类型

推荐从以下四类任务中选择 1 到 2 个作为正式验收任务：

1. 分类
2. 回归
3. 检索
4. 文本生成/问答

### 12.2 输入形式

有三种标准输入形式：

1. 仅 `Z`
2. `Z + metadata`
3. `P + text instruction`

建议：

- 非语言任务优先使用 `Z + metadata`
- 语言任务使用 `P + text instruction`

### 12.3 推荐下游头

| 任务 | 输入 | 头部 |
|---|---|---|
| 分类 | `Z` | mean pool + MLP |
| 回归 | `Z` | attentive pool + MLP |
| 检索 | `Z` 或 `P` | projection + contrastive |
| 问答/报告 | `P + text` | LLM decoder |

### 12.4 冻结策略

默认：

- 冻结压缩器
- 训练 adapter 或小任务头

增强版：

- 仅解冻压缩器最后一层
- 或解冻 adapter + 压缩器末端若干块

### 12.5 下游指标

分类：

- accuracy
- macro F1
- AUROC

回归：

- MAE
- RMSE
- R2

检索：

- recall@k
- mAP

生成：

- task success rate
- factual consistency
- 业务人工评分

### 12.6 验收逻辑

必须同时看三类指标：

1. 重建指标
2. 下游指标
3. 延迟与资源指标

只有下游指标高但重建彻底崩掉，不算成功。

---

## 13. 训练编排建议

建议使用统一编排方式执行各阶段。

### 13.1 实验命名

推荐命名规则：

```text
{date}_{stage}_{dataset}_{model}_{seed}
```

示例：

```text
2026-04-13_stageB_datasetA_ndae_s42
```

### 13.2 日志系统

至少记录：

- loss 曲线
- 验证集关键指标
- 学习率
- 梯度范数
- GPU 占用
- 每轮样本吞吐

建议同时保存：

- 重建可视化切片
- 误差热图
- 频谱对比图

### 13.3 Checkpoint 策略

每个阶段至少保存：

- `last`
- `best_by_val_loss`
- `best_by_business_metric`

并保存：

- 训练配置
- 数据 manifest 哈希
- 归一化统计哈希

### 13.4 随机种子

每个关键实验至少跑 3 个种子：

- 42
- 123
- 3407

最终报告给出均值和标准差。

---

## 14. 评估与对照实验

### 14.1 必做 baseline

至少保留以下 baseline：

1. PCA 或 SVD 类压缩
2. Tucker 或 TT 分解
3. 不带 adapter 的 `Z -> mean pool -> MLP`
4. 文本任务中的纯统计量模板 baseline

### 14.2 核心评估矩阵

每个模型版本都需要填如下矩阵：

| 版本 | latent 大小 | 重建误差 | max error | 下游指标 | 推理延迟 | 显存 |
|---|---:|---:|---:|---:|---:|---:|

### 14.3 消融实验

推荐至少做 6 组消融：

1. 不使用 metadata token
2. 不使用 `L_rel`
3. 不使用 `L_grad`
4. latent token 数减半
5. MLP adapter 对比 Q-Former
6. 冻结压缩器对比末端联合微调

---

## 15. 部署与导出

### 15.1 需要导出的组件

最少导出：

- 归一化统计
- 压缩器权重
- adapter 权重
- LLM 版本信息
- 模型配置

### 15.2 在线推理链路

在线服务流程：

```text
raw tensor
-> axis normalization
-> tensor normalization
-> compressor encoder
-> latent Z
-> adapter
-> soft prompt P
-> LLM
-> task output
```

### 15.3 离线缓存

如果下游任务大量重复读取同一数据，建议缓存：

- `Z`
- `metadata tokens`
- `P`

这样可显著降低在线成本。

---

## 16. 推荐的阶段性交付物

### 16.1 第一阶段交付

目标：证明压缩器可用。

包含：

- 固化后的数据 manifest
- 阶段 B 最优 checkpoint
- 重建报告
- 与传统 baseline 的比较表

### 16.2 第二阶段交付

目标：证明 latent 可用于 LLM。

包含：

- adapter checkpoint
- soft prompt 任务结果
- LLM 对齐实验报告

### 16.3 第三阶段交付

目标：证明系统具备业务价值。

包含：

- 下游任务正式指标
- 资源消耗与延迟报告
- 失败案例分析

---

## 17. 一套可直接执行的最小实现路线

如果当前资源有限，建议按下面顺序推进。

### Sprint 1：打通数据与压缩器

- 完成 manifest 和 stats 生成
- 完成 patch dataloader
- 训练连续 latent 压缩器
- 做重建评估和 baseline 对照

验收：

- 压缩器收敛
- 验证集误差稳定
- 能导出 `Z`

### Sprint 2：打通 adapter 与一个简单任务

- 冻结压缩器
- 用 MLP projector 对接 LLM
- 先做分类或模板化摘要任务

验收：

- soft prompt 明显优于随机 prompt
- 下游优于简单统计量 baseline

### Sprint 3：增强与正式化

- 加入 Q-Former
- 加入率失真训练
- 加入残差分支
- 做完整消融

验收：

- 形成稳定版本矩阵
- 明确推荐配置

---

## 18. 风险与对应处理

### 18.1 重建好但下游差

可能原因：

- latent 虽保真，但不利于语言对齐
- metadata 不足
- adapter 太弱

处理：

- 增加 metadata token
- 升级 MLP 到 Q-Former
- 增加对比学习或问答监督

### 18.2 下游好但重建差

可能原因：

- 联合训练过早
- adapter 目标反向污染压缩器

处理：

- 回退到冻结压缩器训练
- 将重建和下游解耦

### 18.3 训练不稳定

可能原因：

- 归一化策略不对
- 数据分布跨度太大
- patch 过大或 batch 过小

处理：

- 重做统计
- 分变量训练或分组训练
- 引入 gradient clipping 和 warmup

### 18.4 泛化差

可能原因：

- split 泄漏
- 只学到局部纹理
- metadata 缺失

处理：

- 重审切分策略
- 增大上下文窗口
- 加入来源和轴信息

---

## 19. 最终推荐配置

如果需要一套默认起步配置，可直接采用：

### 压缩器

- 输入：标准化 patch
- patch：2D `128x128`，3D `64x64x64`，4D `T=8 + 空间 64~128`
- latent token 数：64
- latent dim：256
- 编码器：局部卷积 + axial attention + MLP
- 解码器：层次上采样 + 残差细化
- 损失：`L_rec + L_rel + L_grad`

### 率失真

- 基于阶段 B checkpoint 微调
- 从小权重 `β_rate` 开始

### LLM 对齐

- 先用 MLP projector 跑 baseline
- 再切换 Q-Former
- 冻结 LLM 主体
- prompt token 数：16

### 下游

- 先做一个非语言任务
- 再做一个语言任务

推荐组合：

- 非语言：分类或检索
- 语言：模板摘要或问答

---

## 20. 交付前检查清单

- 数据 split 是否固定并可追溯
- 归一化统计是否固定并落盘
- 阶段 B 压缩器是否达到重建门槛
- 是否完成至少一个传统 baseline 对照
- 是否证明 `Z` 比简单统计量更有用
- 是否证明 adapter 确实带来提升
- 是否同时报告重建、下游、资源三类指标
- 是否保存全部关键配置和 checkpoint

---

## 21. 总结

这条工程路线的关键不是把所有目标一次性绑死，而是分阶段建立能力：

1. 先拿到稳定、高保真的连续 latent 压缩器。
2. 再把 latent 通过小 adapter 变成 LLM 可用的 soft prompt。
3. 最后围绕具体下游任务做针对性微调和评估。

这样做的好处是：

- 每一步都有独立价值
- 每一步都能做验收
- 出问题时容易定位
- 便于后续替换压缩器、adapter 或 LLM，而不必推倒重来

如果后续要进入实现阶段，建议下一步先补两份文档：

- `experiment_plan.md`：列出首批 baseline、超参数网格和验收门槛
- `data_contract.md`：严格定义 manifest 字段、张量轴语义、归一化规则、下游标签格式
