# Tensor Compression 2.0

本仓库当前按三个功能块组织：

1. **压缩**：训练 tensor autoencoder，检查 HDF5 数据结构，并用 PDEBench 下游算子验证重建质量。
2. **Tensor Editor**：基于冻结 AE，在 latent 空间训练一个文本条件编辑器。这是实验性功能。
3. **Adapter**：导出 AE latent cache，训练 tensor/LLM 对齐模块，并用 soft prompt adapter 评估 LLM 的 tensor readout QA 能力。

基础安装：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

参数表统一格式：

| 列 | 含义 |
|---|---|
| 参数 | YAML key 或命令行参数名。 |
| 说明 | 这个参数控制什么。 |
| 可选值 | 允许填写的值或值类型。 |
| 可选值说明 | 只解释不同可选值的含义；不需要解释时写 `-`。 |

## 1. 压缩

### 1.1 探明 HDF5 文件 Key

用途：确认 PDEBench HDF5 里有哪些 dataset、shape 和 dtype，决定压缩配置中应该写 `hdf5_dataset_key` 还是 `hdf5_dataset_keys`。

命令：

```bash
export PDEBENCH_HDF5_PATH=/data/PiERN/PDEbench/data/2d-ns/xxx.hdf5
python -m unittest discover -s tests -p "test_inspect_pdebench_hdf5.py" -v
```

PowerShell：

```powershell
$env:PDEBENCH_HDF5_PATH="E:\path\to\xxx.hdf5"
python -m unittest discover -s tests -p "test_inspect_pdebench_hdf5.py" -v
```

如果没有设置 `PDEBENCH_HDF5_PATH`，测试会读取 `configs/compressor_2d.yaml` 中的 `data.source_roots.all_primary`。

常见输出：

```text
density: shape=(N, T, H, W), dtype=float32
pressure: shape=(N, T, H, W), dtype=float32
Vx: shape=(N, T, H, W), dtype=float32
Vy: shape=(N, T, H, W), dtype=float32
```

### 1.2 下载或定位 PDEBench 数据

用途：从 PDEBench 官方 CSV 中列出可下载文件，或下载匹配文件。

只列出匹配项：

```bash
python scripts/pdebench_download_helper.py \
  --pdebench-root ./PDEBench_code/PDEBench-main \
  --pde-name 2d_cfd \
  --filename-contains 2D_CFD_Turb_M0.1 \
  --root-folder /data/wyx/pdebench
```

实际下载：

```bash
python scripts/pdebench_download_helper.py \
  --pdebench-root ./PDEBench_code/PDEBench-main \
  --pde-name 2d_cfd \
  --filename-contains 2D_CFD_Turb_M0.1 \
  --root-folder /data/wyx/pdebench \
  --download \
  --skip-existing
```

命令行参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--pdebench-root` | PDEBench 仓库根目录。 | 路径 | - |
| `--pde-name` | 按 PDE 类型筛选下载项。 | 字符串；可重复传入 | 例如 `2d_cfd`。 |
| `--filename-contains` | 按文件名子串筛选下载项。 | 字符串；不传 | 不传表示不过滤文件名。 |
| `--root-folder` | 下载目标根目录。 | 路径 | - |
| `--download` | 是否实际下载匹配文件。 | 开关 | 不加：只打印匹配项和下载命令；加上：执行下载。 |
| `--skip-existing` | 目标文件已存在时是否跳过。 | 开关 | 不加：重新下载或覆盖；加上：跳过已有文件。 |

### 1.3 训练压缩模型

用途：训练 2D/3D/4D tensor autoencoder。当前主实验是 PDEBench 2D HDF5。

检查配置：

```bash
python scripts/train_compressor.py \
  --config configs/compressor_2d.yaml \
  --dry-run
```

训练：

```bash
python scripts/train_compressor.py \
  --config configs/compressor_2d.yaml
```

命令行参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--config` | 压缩模型 YAML 配置路径。 | 路径，必填 | - |
| `--dry-run` | 只构建数据集、模型、loss 等对象并检查配置。 | 开关 | 不加：正式训练；加上：只检查，不训练。 |

### 1.4 压缩配置文件

配置示例：`configs/compressor_2d.yaml`。

#### `experiment`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `name` | 实验名称，用于 run 目录命名。 | 字符串 | - |
| `output_root` | 训练输出根目录。 | 路径 | - |
| `seed` | 随机种子。 | 整数 | - |
| `device` | 训练设备。 | `auto`、`cpu`、`cuda`、`cuda:N` | `auto`：有 CUDA 则用 CUDA，否则 CPU；`cuda:N`：指定 GPU。 |
| `save_top_k` | 保留最佳 checkpoint 数。 | 整数 | 当前训练器主要保存 `best.pt` 和 `last.pt`。 |

#### `data.source_roots`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `all_primary` | 自动切分模式下的总数据入口。 | 文件路径或目录路径 | 单个 HDF5 实验直接指向 `.hdf5` 文件。 |
| `all_extra` | 自动切分模式下的额外数据来源。 | 路径列表 | 空列表表示没有额外来源。 |
| `train_primary` | 预定义切分模式下的训练目录。 | 路径 | - |
| `train_extra` | 训练集额外来源。 | 路径列表 | 空列表表示没有额外来源。 |
| `val_primary` | 预定义切分模式下的验证目录。 | 路径 | - |
| `val_extra` | 验证集额外来源。 | 路径列表 | 空列表表示没有额外来源。 |
| `test_primary` | 预定义切分模式下的测试目录。 | 路径 | - |
| `test_extra` | 测试集额外来源。 | 路径列表 | 空列表表示没有额外来源。 |
| `external_reference_roots` | 外部参考数据目录。 | 路径列表 | 当前压缩训练主流程通常为空。 |

#### `data.split`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `mode` | 数据划分方式。 | `auto`、`predefined` | `auto`：从 `all_primary` 自动切分；`predefined`：使用 train/val/test 目录。 |
| `seed` | split 随机种子。 | 整数 | - |
| `shuffle` | 自动切分前是否打乱样本。 | `true`、`false` | `true`：随机打乱；`false`：保持原顺序。 |
| `train_ratio` | 训练集比例。 | 0 到 1 | 仅 `mode: auto` 使用。 |
| `val_ratio` | 验证集比例。 | 0 到 1 | 仅 `mode: auto` 使用。 |
| `test_ratio` | 测试集比例。 | 0 到 1 | 仅 `mode: auto` 使用。 |

#### `data.dataset`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `recursive` | 目录输入时是否递归扫描。 | `true`、`false` | `true`：扫描子目录；`false`：只扫描当前目录。 |
| `allow_empty` | 是否允许数据集为空。 | `true`、`false` | 训练时建议 `false`，否则容易掩盖路径错误。 |
| `extensions` | 允许读取的文件扩展名。 | 扩展名列表 | 例如 `.npy`、`.npz`、`.h5`、`.hdf5`。 |
| `npz_key` | `.npz` 文件中要读取的数组 key。 | 字符串、`null` | `null` 表示不指定。 |
| `hdf5_dataset_key` | 单字段 HDF5 dataset key。 | 字符串、`null` | 如 `Vx`；使用时输入通道数为 1。 |
| `hdf5_dataset_keys` | 多字段 HDF5 dataset key 列表。 | 字符串列表 | 如 `[density, pressure, Vx, Vy]`；按顺序堆成通道。 |
| `hdf5_key_candidates` | 未指定 HDF5 key 时的候选 key。 | 字符串列表 | 空列表表示不使用候选。 |
| `detect_hdf5_by_signature` | 是否通过文件签名识别 HDF5。 | `true`、`false` | `true`：扩展名不标准时也可识别 HDF5。 |
| `hdf5_index_mode` | HDF5 内部索引方式。 | `auto`、`sample` | `sample`：按样本轴读取；PDEBench 常用。 |
| `hdf5_sample_axes` | 作为样本展开的 HDF5 轴。 | 整数列表、`null` | PDEBench `[sample,time,H,W]` 常用 `[0,1]`。 |
| `hdf5_sample_axis` | 单一样本轴。 | 整数、`null` | 向后兼容单轴 HDF5 配置。 |
| `allow_images` | 是否允许图片文件输入。 | `true`、`false` | 科学张量实验建议 `false`。 |
| `input_size` | 模型输入空间尺寸。 | `[H,W]` | 2D 模型使用。 |
| `strict_size` | 是否要求输入尺寸严格等于 `input_size`。 | `true`、`false` | `false`：允许 resize；`true`：尺寸不一致时报错。 |
| `resize_mode` | resize 插值方式。 | `bilinear` 等 | 连续物理场常用 `bilinear`。 |

#### `data.dataset.normalization`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `mode` | 归一化方式。 | `none`、`zscore`、`minmax` | `none`：不归一化；`zscore`：减均值除标准差；`minmax`：映射到 0 到 1。 |
| `scope` | 统计量计算范围。 | `global`、`channel` | `global`：全 tensor 共用统计量；`channel`：每个通道单独计算。 |
| `stats_path` | 外部统计量文件路径。 | 路径、`null` | 当前主流程通常为 `null`。 |
| `clip_min` | 归一化前最小裁剪值。 | 数值、`null` | `null` 表示不裁剪下界。 |
| `clip_max` | 归一化前最大裁剪值。 | 数值、`null` | `null` 表示不裁剪上界。 |

#### `data.loader`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `batch_size` | DataLoader batch 大小。 | 正整数 | - |
| `num_workers` | DataLoader worker 数。 | 非负整数 | `0`：主进程读取；大于 0：多进程读取。 |
| `shuffle_train` | 训练集是否打乱。 | `true`、`false` | `true`：每轮打乱。 |
| `pin_memory` | 是否启用 pinned memory。 | `true`、`false` | CUDA 训练通常设 `true`。 |
| `drop_last` | 是否丢弃最后一个不完整 batch。 | `true`、`false` | 小数据通常设 `false`。 |
| `persistent_workers` | worker 是否跨 epoch 保持常驻。 | `true`、`false` | 仅 `num_workers > 0` 时有效。 |

#### `model`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `name` | 模型注册名。 | `conv_token_autoencoder_2d`、`conv_token_autoencoder_3d`、`factorized_autoencoder_4d` | 分别对应 2D、3D、4D 压缩模型。 |
| `input_size` | 模型输入尺寸。 | `[H,W]` | 2D AE 使用。 |
| `base_channels` | encoder/decoder 的基础通道数。 | 正整数 | - |
| `channel_multipliers` | 每个下采样层的通道倍率。 | 正整数列表 | 列表长度决定 2D AE 下采样次数。 |
| `num_res_blocks` | 每个尺度的残差块数。 | 非负整数 | - |
| `latent_dim` | latent map 通道数。 | 正整数 | - |
| `latent_dim_base` | 按通道缩放 latent 时的基准维度。 | 正整数 | 仅 `latent_dim_scale_with_channels: true` 时使用。 |
| `latent_dim_scale_with_channels` | 是否按输入通道数缩放 `latent_dim`。 | `true`、`false` | `true`：多通道自动增大 latent；`false`：固定 latent 维度。 |
| `latent_dim_reference_channels` | latent 缩放的参考输入通道数。 | 正整数 | - |
| `latent_dim_round_to` | 缩放后的 latent 维度对齐粒度。 | 正整数 | 例如 32。 |
| `latent_grid` | latent map 空间网格。 | `[H_lat,W_lat]` | token 数为 `H_lat * W_lat`。 |
| `dropout` | dropout 概率。 | 0 到 1 | - |
| `norm` | 归一化层类型。 | `group`、`batch`、`identity` | `group`：小 batch 更稳；`batch`：依赖 batch 统计；`identity`：不用 norm。 |
| `activation` | 激活函数。 | `relu`、`gelu`、`silu` | - |
| `output_activation` | 输出激活函数。 | `identity`、`sigmoid`、`tanh` | 标准化连续场通常用 `identity`。 |

压缩率估算：

```text
input scalars = C_in * H * W
latent scalars = latent_dim * H_lat * W_lat
float compression ratio = input scalars / latent scalars
```

例如单字段 `Vx`，输入 `[1,512,512]`，latent `[512,16,16]`，约为 `262144 / 131072 = 2x`。

#### `loss`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `name` | loss 注册名。 | `composite_reconstruction_loss` | 当前实现的组合重建 loss。 |
| `weights.mse` | MSE loss 权重。 | 非负数 | - |
| `weights.l1` | L1 loss 权重。 | 非负数 | - |
| `weights.relative_l1` | relative L1 loss 权重。 | 非负数 | - |
| `weights.gradient` | 空间梯度误差权重。 | 非负数 | - |
| `eps` | 数值稳定项。 | 正数 | - |

#### `optimizer`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `name` | 优化器类型。 | `adamw`、`adam` | `adamw`：Adam + decoupled weight decay；`adam`：标准 Adam。 |
| `lr` | 学习率。 | 正数 | - |
| `weight_decay` | 权重衰减。 | 非负数 | - |

#### `scheduler`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `name` | 学习率调度器类型。 | `cosine`、`none` | `cosine`：余弦退火；`none`：不使用调度器。 |
| `t_max` | cosine 调度周期。 | 正整数 | 仅 `name: cosine` 使用。 |
| `min_lr` | cosine 最小学习率。 | 非负数 | 仅 `name: cosine` 使用。 |

#### `training`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `epochs` | 训练轮数。 | 正整数 | - |
| `mixed_precision` | 是否启用混合精度。 | `true`、`false` | CUDA 上通常设 `true`。 |
| `grad_clip_norm` | 梯度裁剪范数。 | 非负数 | `0` 表示不裁剪。 |
| `log_interval` | step 日志间隔。 | 正整数 | - |
| `val_interval` | 验证间隔。 | 正整数 | 当前训练器按 epoch 验证。 |
| `checkpoint_interval` | checkpoint 间隔。 | 正整数 | 当前训练器保存 best/last。 |

#### `visualization`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `enabled` | 是否保存重建可视化。 | `true`、`false` | - |
| `num_samples` | 每次可视化的样本数。 | 正整数 | - |
| `every_n_epochs` | 可视化间隔。 | 正整数 | - |
| `field_cmap` | 原场和重建场 colormap。 | Matplotlib colormap 名 | 例如 `turbo`。 |
| `error_cmap` | 误差图 colormap。 | Matplotlib colormap 名 | 例如 `inferno`。 |
| `robust_percentile` | 显示时忽略极端值的百分位。 | 0 到 50 | - |
| `display_channel` | 可视化的通道索引。 | 非负整数 | - |
| `add_colorbar` | 是否添加 colorbar。 | `true`、`false` | - |
| `save_dirname` | 可视化保存目录名。 | 字符串 | 位于 run 目录下。 |

#### `wandb`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `enabled` | 是否启用 W&B。 | `true`、`false` | - |
| `api_key` | W&B API key。 | 字符串、`null` | 建议用环境变量，不写入配置。 |
| `project` | W&B project 名。 | 字符串 | - |
| `entity` | W&B entity。 | 字符串、`null` | `null` 使用默认账号。 |
| `group` | W&B group 名。 | 字符串 | - |
| `tags` | W&B tags。 | 字符串列表 | - |
| `mode` | W&B 运行模式。 | `online`、`offline`、`disabled` | `online`：联网记录；`offline`：本地离线记录；`disabled`：禁用。 |
| `log_model` | 是否上传模型。 | `true`、`false` | 大 checkpoint 通常设 `false`。 |

#### `future`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `adapters.enabled` | 预留 adapter 开关。 | `true`、`false` | 当前压缩训练不使用。 |
| `adapters.module_name` | 预留 adapter 模块名。 | 字符串、`null` | - |
| `llm.enabled` | 预留 LLM 开关。 | `true`、`false` | 当前压缩训练不使用。 |
| `llm.model_name` | 预留 LLM 名称。 | 字符串、`null` | - |
| `llm.prompt_token_count` | 预留 prompt token 数。 | 正整数 | - |
| `tensor_3d.model_name` | 3D 压缩模型注册名。 | 字符串 | - |
| `tensor_3d.dataset_name` | 3D 数据集注册名。 | 字符串 | - |
| `tensor_4d.model_name` | 4D 压缩模型注册名。 | 字符串 | - |
| `tensor_4d.dataset_name` | 4D 数据集注册名。 | 字符串 | - |

### 1.5 下游任务验证重建质量

用途：比较原始数据和 AE 重建数据经过同一个 PDEBench forward/inverse operator 后的误差。

命令：

```bash
python scripts/evaluate_pdebench_downstream.py \
  --hdf5-path /data/PiERN/PDEbench/data/2d-ns/xxx.hdf5 \
  --sample-indices all \
  --compressor-checkpoint /data/wyx/runs/<ae_run>/checkpoints/best.pt \
  --batch-size 16 \
  --reconstructed-hdf5-output /data/wyx/reconstructed.hdf5 \
  --forward-operator-type pdebench-fno \
  --forward-checkpoint /data/PiERN/PDEbench/model/2d-ns/xxx_FNO.pt \
  --pdebench-root ./PDEBench_code/PDEBench-main \
  --num-channels 4 \
  --initial-step 10 \
  --t-train 21 \
  --modes 12 \
  --width 20 \
  --output /data/wyx/pdebench_downstream.json
```

命令行参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--hdf5-path` | PDEBench HDF5 文件。 | 路径，必填 | - |
| `--fields` | 参与评估的 HDF5 字段顺序。 | 逗号分隔字段、`null` | 不传：优先使用 checkpoint 中的训练字段顺序。 |
| `--sample-indices` | 要评估的样本索引。 | `all`、逗号分隔整数 | `all`：评估全部 sample。 |
| `--time-start` | 时间切片起点。 | 整数、`null` | `null`：从第一个时间步开始。 |
| `--time-stop` | 时间切片终点。 | 整数、`null` | `null`：直到最后。 |
| `--time-step` | 时间切片步长。 | 整数、`null` | `null`：步长为 1。 |
| `--spatial-stride` | 空间采样步长。 | 正整数 | `1`：不降采样。 |
| `--compressor-checkpoint` | AE checkpoint 路径。 | 路径、`null` | `null`：只评估 identity reconstruction。 |
| `--compressor-config` | AE config 路径。 | 路径、`null` | checkpoint 不含 config 时需要。 |
| `--batch-size` | AE 重建 batch 大小。 | 正整数 | - |
| `--device` | 运行设备。 | `auto`、`cpu`、`cuda`、`cuda:N` | `auto`：自动选择。 |
| `--forward-operator-type` | forward operator 类型。 | `none`、`callable`、`pdebench-fno`、`pdebench-unet` | `none`：不运行；`callable`：用户函数；`pdebench-fno/unet`：PDEBench 模型。 |
| `--forward-operator-spec` | callable forward 入口。 | `module.py:callable`、import path、`null` | 仅 `--forward-operator-type callable` 使用。 |
| `--forward-checkpoint` | PDEBench forward 模型 checkpoint。 | 路径、`null` | FNO/UNet operator 使用。 |
| `--inverse-operator-type` | inverse operator 类型。 | `none`、`callable` | 当前 inverse 只支持 `callable`。 |
| `--inverse-operator-spec` | callable inverse 入口。 | `module.py:callable`、import path、`null` | 仅 `--inverse-operator-type callable` 使用。 |
| `--pdebench-root` | PDEBench 仓库根目录。 | 路径 | - |
| `--num-channels` | PDEBench operator 输入通道数。 | 正整数、`null` | 通常等于字段数。 |
| `--initial-step` | PDEBench 初始时间步。 | 正整数 | - |
| `--t-train` | PDEBench 训练时间长度。 | 正整数、`null` | 需与 operator checkpoint 设置一致。 |
| `--modes` | FNO modes。 | 正整数 | 仅 FNO 使用。 |
| `--width` | FNO width。 | 正整数 | 仅 FNO 使用。 |
| `--init-features` | UNet 初始特征数。 | 正整数 | 仅 UNet 使用。 |
| `--output` | JSON 评估结果输出路径。 | 路径、`null` | `null`：使用默认输出目录。 |
| `--reconstructed-hdf5-output` | 重建 HDF5 输出路径。 | 路径、`null` | `null`：不写 HDF5。 |
| `--overwrite-reconstructed-hdf5` | 是否覆盖已有重建 HDF5。 | 开关 | 不加：已存在时报错；加上：允许覆盖。 |
| `--no-progress` | 是否关闭进度条。 | 开关 | 不加：显示进度；加上：关闭进度。 |

## 2. Tensor Editor

### 2.1 功能定位

Tensor Editor 是实验性功能。它读取输入 tensor、文本 prompt 和目标 tensor，在冻结 AE 的 latent map 上预测 `delta_z`，再通过 AE decoder 输出编辑后的 tensor。

它不属于当前 Adapter 主线。它主要用于验证“AE latent 空间是否能做文本条件编辑/修复”。

### 2.2 训练 Tensor Editor

检查配置：

```bash
python scripts/train_tensor_editor.py \
  --config configs/tensor_editor_2d.yaml \
  --dry-run
```

训练：

```bash
python scripts/train_tensor_editor.py \
  --config configs/tensor_editor_2d.yaml
```

临时覆盖路径和训练参数：

```bash
python scripts/train_tensor_editor.py \
  --config configs/tensor_editor_2d.yaml \
  --jsonl-path /data/wyx/tensor_editor/train.jsonl \
  --compressor-checkpoint /data/wyx/runs/<ae_run>/checkpoints/best.pt \
  --device cuda:0 \
  --output-root /data/wyx/tensor_editor_outputs \
  --epochs 20 \
  --batch-size 1
```

命令行参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--config` | Tensor editor YAML 配置路径。 | 路径，必填 | - |
| `--dry-run` | 只检查数据、模型和配置。 | 开关 | 不加：训练；加上：只检查并写 `setup_summary.json`。 |
| `--jsonl-path` | 覆盖编辑数据 JSONL 路径。 | 路径、`null` | 不传：使用 config。 |
| `--compressor-checkpoint` | 覆盖 AE checkpoint 路径。 | 路径、`null` | 不传：使用 config。 |
| `--compressor-config` | 覆盖 AE config 路径。 | 路径、`null` | checkpoint 无内嵌 config 时需要。 |
| `--device` | 覆盖运行设备。 | `auto`、`cpu`、`cuda`、`cuda:N` | - |
| `--output-root` | 覆盖输出根目录。 | 路径、`null` | 不传：使用 config。 |
| `--epochs` | 覆盖训练轮数。 | 正整数、`null` | 不传：使用 config。 |
| `--batch-size` | 覆盖 batch size。 | 正整数、`null` | 不传：使用 config。 |
| `--validation-ratio` | 覆盖验证集比例。 | 0 到 1、`null` | 不传：使用 config。 |

### 2.3 Tensor Editor 数据格式

JSONL 每行一个样本：

```json
{
  "id": "sample_000001",
  "prompt": "请将输入速度场中的局部噪声去除，并保持整体结构不变。",
  "tensor_path": "inputs/sample_000001.npy",
  "label_path": "targets/sample_000001.npy",
  "meta": {
    "type": "denoise"
  }
}
```

也可以直接内联小 tensor：

```json
{
  "prompt": "修复缺失区域。",
  "tensor": [[...]],
  "label": [[...]]
}
```

支持的 tensor 文件格式：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `tensor_path` / `label_path` | 输入和目标 tensor 文件路径。 | `.npy`、`.npz`、`.pt`、`.pth` | `.npy`：直接读取；`.npz`：读取第一个 array；`.pt/.pth`：读取 tensor 或 dict 中第一个 tensor。 |
| `tensor` / `label` | 输入和目标 tensor 内联数组。 | JSON array | 只建议小样本 debug 使用。 |
| `prompt` | 文本条件。 | 字符串 | 当前模型使用字符 hash + GRU 编码，不依赖 LLM tokenizer。 |
| `meta` | 样本元数据。 | JSON object、可缺省 | 可用于按任务类型统计验证指标。 |

### 2.4 Tensor Editor 配置文件

配置示例：`configs/tensor_editor_2d.yaml`。

#### `experiment`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `name` | 实验名称，用于 run 目录命名。 | 字符串 | - |
| `output_root` | 输出根目录。 | 路径 | - |
| `seed` | 随机种子。 | 整数 | - |
| `device` | 运行设备。 | `auto`、`cpu`、`cuda`、`cuda:N` | `auto`：自动选择。 |

#### `editor.data`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `jsonl_path` | 编辑数据 JSONL 路径。 | 路径 | - |
| `input_size` | 输入 tensor 尺寸。 | `[H,W]` | 必须与 AE checkpoint 输入尺寸一致。 |
| `channels` | 输入 tensor 通道数。 | 正整数 | 必须与 AE checkpoint 输入通道数一致。 |
| `validation_ratio` | 从 JSONL 中划分验证集的比例。 | 0 到 1 | - |
| `fix_prompt_mojibake` | 是否尝试修复中文乱码 prompt。 | `true`、`false` | `true`：尝试 GBK/CP936 等修复；`false`：原样使用。 |

#### `editor.data.loader`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `batch_size` | DataLoader batch 大小。 | 正整数 | - |
| `num_workers` | DataLoader worker 数。 | 非负整数 | `0`：主进程读取；大于 0：多进程读取。 |
| `pin_memory` | 是否启用 pinned memory。 | `true`、`false` | CUDA 训练通常设 `true`。 |
| `persistent_workers` | worker 是否常驻。 | `true`、`false` | 仅 `num_workers > 0` 时有效。 |

#### `editor.compressor`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `checkpoint_path` | 预训练 AE checkpoint 路径。 | 路径 | - |
| `config_path` | AE config 路径。 | 路径、`null` | checkpoint 无内嵌 config 时需要。 |
| `freeze` | 是否冻结 AE。 | `true`、`false` | `true`：只训练 editor；`false`：AE 也参与训练。 |

#### `editor.text`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `vocab_size` | 字符 hash 词表大小。 | 正整数，至少 8 | - |
| `embed_dim` | 字符 embedding 维度。 | 正整数 | - |
| `hidden_dim` | GRU hidden 维度。 | 正整数 | 双向 GRU 输出维度为 `2 * hidden_dim`。 |
| `max_length` | prompt 最大字符数。 | 正整数 | 超出部分截断。 |
| `dropout` | prompt encoder dropout。 | 0 到 1 | - |

#### `editor.model`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `name` | editor 模型注册名。 | `conditional_tensor_editor_2d` | 当前唯一实现。 |
| `latent_hidden_dim` | latent editor 隐层通道数。 | 正整数 | - |
| `num_res_blocks` | FiLM residual block 数。 | 非负整数 | - |
| `condition_dim` | prompt 条件向量维度。 | 正整数 | - |
| `activation` | 激活函数。 | `relu`、`gelu`、`silu` | - |
| `dropout` | latent editor dropout。 | 0 到 1 | - |
| `use_base_reconstruction` | 是否计算 AE base reconstruction。 | `true`、`false` | `true`：同时输出 AE 原始重建用于比较。 |
| `residual_latent` | 是否使用残差 latent 编辑。 | `true`、`false` | `true`：`z_edit = z + delta_z`；`false`：`z_edit = delta_z`。 |
| `latent_delta_scale` | latent delta 缩放系数。 | 数值 | - |
| `zero_init_delta` | 是否零初始化 delta 输出层。 | `true`、`false` | `true`：初始近似不编辑。 |
| `detach_latent_target` | target latent 是否停止梯度。 | `true`、`false` | 冻结 AE 时通常设 `true`。 |

#### `loss`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `name` | loss 注册名。 | `composite_reconstruction_loss` | - |
| `weights.mse` | 输出 tensor MSE 权重。 | 非负数 | - |
| `weights.l1` | 输出 tensor L1 权重。 | 非负数 | - |
| `weights.relative_l1` | 输出 tensor relative L1 权重。 | 非负数 | - |
| `weights.gradient` | 输出 tensor 空间梯度误差权重。 | 非负数 | - |
| `weights.latent_mse` | edited latent 与 target latent 的 MSE 权重。 | 非负数 | - |
| `eps` | 数值稳定项。 | 正数 | - |

#### `optimizer`、`scheduler`、`training`、`wandb`

这些 section 与压缩训练的同名 section 含义一致。Tensor Editor 额外使用：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `training.num_saved_val_examples` | 保存验证样例摘要数量。 | 非负整数 | `0`：不保存验证样例摘要。 |

输出文件：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `checkpoints/best.pt` | 验证 loss 最低的 editor checkpoint。 | 文件路径 | - |
| `checkpoints/last.pt` | 最后一轮 editor checkpoint。 | 文件路径 | - |
| `metrics_latest.json` | 训练和验证指标。 | JSON 文件 | - |
| `val_examples_latest.json` | 验证样例摘要。 | JSON 文件 | - |
| `setup_summary.json` | dry-run 检查摘要。 | JSON 文件 | 仅 `--dry-run` 生成。 |

## 3. Adapter

### 3.1 Adapter Pipeline 总览

当前路线：

```text
PDEBench HDF5
  -> build_tensor_readout_qa.py 生成 QA JSONL
  -> export_tensor_readout_latents.py 用 AE 导出 latent cache
  -> train_tensor_llm_adapter.py 冻结 LLM 训练 soft prompt adapter
```

推荐复制模板配置：

```bash
cp configs/tensor_llm_adapter_pipeline.yaml configs/tensor_llm_adapter_pipeline.yaml
```

### 3.2 准备模型与缓存目录

用途：读取 pipeline config，创建 asset/cache/output 目录，可选下载 HuggingFace 模型。

命令：

```bash
python scripts/prepare_tensor_llm_assets.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --create-dirs \
  --download-model
```

命令行参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--config` | Adapter pipeline YAML 配置路径。 | 路径 | - |
| `--create-dirs` | 是否创建配置中的目录。 | 开关 | 不加：只打印；加上：创建目录并写环境文件。 |
| `--download-model` | 是否下载 HF 模型。 | 开关 | 不加：不下载；加上：调用 `snapshot_download`。 |
| `--token` | HuggingFace token。 | 字符串、`null` | 私有或需授权模型使用；公开 Qwen 通常不需要。 |

### 3.3 Adapter Pipeline 配置文件

模板：`configs/tensor_llm_adapter_pipeline.yaml`。

#### `storage`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `candidate_roots` | 候选存储根目录。 | 路径列表 | 只用于 `prepare_tensor_llm_assets.py` 显示空间余量。 |
| `min_free_gb` | 最小期望空闲空间。 | 数值 | 低于该值时标记为 `LOW`，不阻止运行。 |
| `asset_root` | 资产根目录。 | 路径 | 可放 QA、latent、环境文件。 |
| `hf_home` | HuggingFace cache 根目录。 | 路径 | 用于 `HF_HOME` 和模型缓存。 |
| `output_root` | 输出根目录。 | 路径 | Adapter run 默认输出根目录。 |

#### `runtime`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `seed` | 默认随机种子。 | 整数 | - |
| `device` | 默认运行设备。 | `auto`、`cpu`、`cuda`、`cuda:N` | `auto`：自动选择。 |

#### `data`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `hdf5_path` | PDEBench HDF5 路径。 | 路径 | QA 生成和 latent 导出使用。 |
| `fields` | 使用的 HDF5 字段。 | 字符串列表 | 必须与 AE checkpoint 编码字段一致。 |
| `qa_dir` | QA JSONL 输出/读取目录。 | 路径 | 应包含 `train.jsonl`、`val.jsonl`、`test.jsonl`。 |
| `latent_dir` | latent cache 目录。 | 路径 | 应包含 `<state_ref>.pt`。 |

#### `compressor`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `config` | AE config 路径。 | 路径、`null` | checkpoint 不含 config 时需要。 |
| `checkpoint` | AE checkpoint 路径。 | 路径 | latent 导出必需。 |

#### `model`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `name_or_path` | HF 模型名或本地路径。 | repo id、路径 | 如 `Qwen/Qwen2.5-1.5B-Instruct`。 |
| `local_dir` | 本地模型目录。 | 路径、`null` | 非空时训练脚本优先使用该路径。 |
| `revision` | HF revision。 | branch、tag、commit | 默认常用 `main`。 |
| `trust_remote_code` | 是否执行远端模型仓库代码。 | `true`、`false` | 首轮建议 `false`，优先选 Transformers 内置模型。 |
| `torch_dtype` | LLM 权重 dtype。 | `auto`、`float32`、`float16`、`bfloat16` | A800 推荐 `bfloat16`。 |
| `allow_patterns` | 下载时允许的文件模式。 | glob 列表 | 限制只下载权重、配置、tokenizer 等文件。 |
| `ignore_patterns` | 下载时忽略的文件模式。 | glob 列表 | 排除不需要的大文件格式。 |

#### `qa_generation`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `sample_indices` | 参与生成 QA 的 sample 索引。 | `all`、列表、逗号字符串 | `all`：全部 sample。 |
| `time_indices` | 参与生成 QA 的 time 索引。 | `all`、列表、逗号字符串 | `all`：全部 time step。 |
| `max_states` | 最大 tensor state 数。 | 正整数、`null` | `null`：不限制。 |
| `train_ratio` | QA train split 比例。 | 0 到 1 | - |
| `val_ratio` | QA val split 比例。 | 0 到 1 | - |
| `test_ratio` | QA test split 比例。 | 0 到 1 | - |
| `spatial_stride` | 读取 HDF5 时的空间步长。 | 正整数 | `1`：不降采样。 |
| `num_bins` | quantile bin 数量。 | 大于等于 2 的整数 | 生成 `B00...` 标签。 |
| `quantile_samples_per_state` | 每个 state 用于估计 quantile 的采样点数。 | 正整数 | - |
| `patch_size` | patch 比较题的 patch 边长。 | 正整数 | - |
| `point_bin_per_state` | 每个 state 生成的点值 bin 题数。 | 非负整数 | `0`：关闭该任务。 |
| `point_compare_per_state` | 每个 state 生成的点比较题数。 | 非负整数 | `0`：关闭该任务。 |
| `patch_compare_per_state` | 每个 state 生成的 patch 比较题数。 | 非负整数 | `0`：关闭该任务。 |
| `max_quadrant_per_state` | 每个 state 生成的最大速度象限题数。 | `0`、`1` | 需要 `Vx,Vy`；单 `Vx` 应设 0。 |
| `global_stat_bin_per_state` | 每个 state 生成的速度统计 bin 题数。 | 0 到 3 | 需要 `Vx,Vy`；对应 mean/max/std speed。 |
| `compare_min_bin_distance` | 比较题要求的最小 quantile bin 距离。 | 非负整数 | 越大越排除近似平局。 |
| `compare_max_attempts` | 比较题重采样最大尝试次数。 | 正整数 | - |
| `include_oracle` | 是否保存 oracle 数值。 | `true`、`false` | debug 建议 `true`。 |

#### `latent_export`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `splits` | 要导出 latent 的 split。 | split 名列表 | 通常 `[train,val,test]`。 |
| `batch_size` | AE encode batch 大小。 | 正整数 | - |
| `device` | latent 导出设备。 | `auto`、`cpu`、`cuda`、`cuda:N` | `auto`：自动选择。 |
| `spatial_stride` | latent 导出读取 HDF5 的空间步长。 | 正整数 | 应与 QA 生成的数据语义一致。 |
| `storage_dtype` | latent 保存 dtype。 | `float32`、`float16`、`bfloat16` | `float16` 节省空间；`float32` 保留精度。 |
| `allow_field_mismatch` | 是否允许 QA 字段与 AE 字段不一致。 | `true`、`false` | 除非做消融，否则设 `false`。 |
| `overwrite` | 是否覆盖已有 latent 文件。 | `true`、`false` | `false`：跳过已有文件。 |

#### `adapter`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `soft_prompt_tokens` | 插入 LLM 的连续 prompt token 数。 | 正整数 | - |
| `adapter_dim` | adapter 内部维度。 | 正整数 | 必须能被 `adapter_heads` 整除。 |
| `adapter_layers` | cross-attention block 数。 | 正整数 | - |
| `adapter_heads` | adapter attention heads。 | 正整数 | - |
| `dropout` | adapter dropout。 | 0 到 1 | - |
| `latent_pos_encoding` | 是否给 latent token 加二维位置编码。 | `grid`、`none` | `grid`：根据 latent 的 H、W 坐标加入可学习投影；`none`：不加入位置。 |
| `question_conditioning` | 是否让文本问题条件化 adapter query token。 | `true`、`false` | `true`：同一个 tensor 面对不同问题会产生不同 soft prompt；`false`：同一个 tensor 的 soft prompt 与问题无关。 |
| `question_condition_gate_init` | 文本问题条件分支的初始门控强度。 | 浮点数 | `1.0`：默认开启；`0.0`：初始近似关闭，但训练中仍可学习。 |
| `structured_query_conditioning` | 旧版结构化 query 旁路。正式实验必须关闭，使 adapter 自己读取自然语言。 | `true`、`false` | `false`：推荐，使用自然语言 token；`true`：regex 解析坐标/任务，仅允许 sanity 调试。 |
| `local_question_input_mode` | local branch 的自然语言输入。 | `contextual_tokens`、`input_embeddings` | `contextual_tokens`：使用 frozen Qwen 浅层逐 token hidden state；`input_embeddings`：旧版静态 embedding 路径。 |
| `local_context_layer` | contextual local 使用的 Qwen hidden-state 层。 | 非负整数 | 当前 `6`：运行前六个 decoder block 后提前停止。 |
| `local_context_layers` | residual adapter 融合的 Qwen hidden-state 层。 | 非负整数列表 | 当前 `[2,6]`，一次前向截取浅层词法细节与中层上下文。 |
| `local_fusion_mode` | 问题 token 与 latent 的融合方式。 | `residual_spatial_transformer`、`residual_qformer`、`anchor_queries`、`text_latent_pool` | 当前使用第一项；其余值用于旧 checkpoint 兼容。 |
| `local_text_encoder_layers` | 旧 `input_embeddings` 路径的额外文本 Transformer 层数。 | 非负整数 | contextual 正式配置设为 `0`。 |
| `freeze_conditioned_backbone` | 是否冻结 residual branch 继承的 Stage-1 空间 backbone。 | `true`、`false` | 正式配置为 `true`，防止 conditioned clone 绕过问题分支学成第二个无条件 encoder。 |
| `local_text_gate_init` / `local_gate_init` | cross-attention 与 outer residual 的固定尺度。 | 浮点数 | 正式配置均为 `1.0`，避免两个小 gate 相乘压低问题梯度。 |
| `local_text_gate_trainable` / `local_residual_gate_trainable` | 是否训练上述 gate。 | `true`、`false` | 正式配置均关闭；问题增量由 cross-attention 权重学习。 |
| `zero_init_local_text_attention` | 是否把新增 cross-attention 的输出投影初始化为零。 | `true`、`false` | `true` 使 Stage 2 启动时逐元素复现 Stage 1，同时保留到输出投影的梯度。 |
| `soft_prompt_scale` | soft prompt 输出尺度限制。 | 非负数 | `0.05`：`tanh` 后限制每维约在 `[-0.05,0.05]`，使 soft prompt token 范数接近普通 token embedding；`0`：关闭尺度限制，保留线性输出。 |

当前正式 adapter 的信息流是单向的：local reader 接收完整自然语言题干、候选项及输出约束，并把末尾 `Answer:` 换成中性的 `Tensor evidence requested:` anchor；frozen Qwen 一次运行到第 6 层，同时截取第 2/6 层逐 token contextual states，detach 后进入 residual spatial adapter。每个固定空间位置先 cross-attend 完整问题 token，再做空间 self-attention。完整 QA prompt 也直接进入 frozen LLM。梯度不会更新 LLM；代码不会通过 regex 提前提取任务、坐标、区域或 mean/scale。

#### `llm_training`

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `run_name` | Adapter run 名称。 | 字符串 | - |
| `output_root` | Adapter 输出根目录。 | 路径 | - |
| `train_split` | 训练 split 名。 | 字符串 | 默认通常为 `train`。 |
| `val_split` | 验证 split 名。 | 字符串 | 默认通常为 `val`。 |
| `test_split` | 测试 split 名。 | 字符串 | 默认通常为 `test`。 |
| `max_train_records` | 最大训练记录数。 | 正整数、`null` | `null`：不限制。 |
| `max_val_records` | 最大验证记录数。 | 正整数、`null` | `null`：不限制。 |
| `max_test_records` | 最大测试记录数。 | 正整数、`null` | `null`：不限制。 |
| `prefer_record_latent_ref` | 是否优先读取 JSONL 内的 `latent_ref`。 | `true`、`false` | `false`：从 `latent_dir/state_ref.pt` 读取。 |
| `device` | Adapter 训练设备。 | `auto`、`cpu`、`cuda`、`cuda:N` | 可配合 `CUDA_VISIBLE_DEVICES`。 |
| `torch_dtype` | LLM 权重 dtype。 | `auto`、`float32`、`float16`、`bfloat16` | A800 推荐 `bfloat16`。 |
| `shuffle_seed` | `shuffled` baseline 的随机错配种子。 | 整数 | 固定后每次评估使用同一组随机错配 latent；错配时会排除相同 `state_ref`。 |
| `epochs` | 训练轮数。 | 正整数 | - |
| `batch_size` | 训练 batch 大小。 | 正整数 | - |
| `eval_batch_size` | 评估 record batch 大小。 | 正整数 | - |
| `eval_choice_batch_size` | 候选答案打分 batch 大小。 | 正整数 | - |
| `gradient_accumulation_steps` | 梯度累积步数。 | 正整数 | - |
| `lr` | adapter 学习率。 | 正数 | - |
| `weight_decay` | 权重衰减。 | 非负数 | - |
| `grad_clip_norm` | 梯度裁剪范数。 | 非负数 | `0`：不裁剪。 |
| `ce_loss_weight` | 普通答案 token-level CE loss 权重。 | 非负数 | 该项训练 LLM 在答案 token 位置生成真实答案；权重大时可能更多学习输出格式和标签先验。 |
| `choice_ce_loss_weight` | 候选项分类 CE loss 权重。 | 非负数 | `0`：关闭；大于 0 时把所有候选答案的 `-NLL` 当作分类 logits，直接优化“选中正确候选项”。 |
| `ranking_loss_weight` | ranking loss 权重。 | 非负数 | `0`：关闭；大于 0 时要求正确 latent 比错配 latent 更支持正确答案。 |
| `ranking_loss_margin` | ranking loss 的最小 restricted-choice CE 间隔。 | 非负数 | 希望 `ChoiceCE(negative)-ChoiceCE(correct)` 至少达到该值，不包含 EOS。 |
| `ranking_loss_negative` | ranking loss 使用的对照类型。 | `global_only`、`shuffled`、`random`、`no_latent`、`zero_latent` | 正式配置用 `global_only`，要求 question reader 优于同一 tensor 的冻结 Stage-1 前缀；其余保留兼容。 |
| `swapped_question_loss_weight` | 同 tensor 问题交换 grounding loss 权重。 | 非负数 | 大于 0 时要求自己的 conditioned prompt 比同 tensor/任务的另一问题 prompt 更支持正确答案。 |
| `swapped_question_loss_margin` | 问题交换目标的最小 restricted-choice CE 间隔。 | 非负数 | 当前 `0.1`。 |
| `swapped_question_max_records` | 每个 batch 参与交换评分的最大记录数。 | 正整数 | 当前 `8`，限制额外 LLM 前向的显存和时间。 |
| `swapped_question_require_different_answer` | 是否跳过答案标签相同的问题交换。 | `true`、`false` | 正式配置为 `true`，避免把可能等价的证据强制当负样本。 |
| `prompt_template` | Adapter 训练用文本 prompt 模板。 | `task_specific`、`generic` | `task_specific`：按 `task_type` 写明读数/比较/bin 规则；`generic`：旧版通用提示。 |
| `max_prompt_tokens` | 文本 prompt 最大 token 数。 | 正整数 | 超出会左截断。 |
| `max_target_tokens` | 答案最大 token 数。 | 正整数 | - |
| `append_eos` | target 后是否追加 EOS。 | `true`、`false` | - |
| `eval_baselines` | 评估 baseline 列表。 | `correct`、`global_only`、`local_only`、`no_latent`、`zero_latent`、`shuffled`、`random`、`shuffled_stats` | residual 模式下 `global_only` 是固定 stage-1 prompt，`local_only` 是 question-conditioned residual；其余测试无前缀、零/错配/随机 latent 和错配 mean/scale。 |
| `choice_score` | 候选答案 NLL 计分方式。 | `mean`、`sum` | `mean`：按 token 数平均；`sum`：总 NLL。 |
| `log_interval` | 训练日志间隔。 | 正整数 | - |

#### `wandb`

Adapter 训练脚本也支持 W&B。默认 `enabled: false`；服务器上建议先确认 `WANDB_API_KEY` 环境变量或在 offline 模式试跑。

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `enabled` | 是否启用 W&B。 | `true`、`false` | `false`：只写本地 JSON 指标。 |
| `api_key` | W&B API key。 | 字符串、`null` | `null`：从 `WANDB_API_KEY` 读取。 |
| `project` | W&B project 名称。 | 字符串 | - |
| `entity` | W&B entity/team。 | 字符串、`null` | `null`：使用账号默认 entity。 |
| `group` | W&B run group。 | 字符串、`null` | 用于把同一组实验聚合。 |
| `tags` | W&B tags。 | 字符串列表 | - |
| `mode` | W&B 运行模式。 | `online`、`offline`、`disabled` | `online`：实时上传；`offline`：本地缓存；`disabled`：禁用。 |
| `log_model` | 是否上传 adapter checkpoint artifact。 | `true`、`false` | 大文件或频繁试验建议 `false`。 |
| `detailed_metrics` | 是否递归展开全部字段/baseline/诊断指标到 W&B。 | `true`、`false` | 默认 `false`；完整数据始终保存在本地 JSON/PT。 |

### 3.4 生成 Tensor Readout QA

命令：

```bash
python scripts/build_tensor_readout_qa.py \
  --config configs/tensor_llm_adapter_pipeline.yaml
```

直接传参：

```bash
python scripts/build_tensor_readout_qa.py \
  --hdf5-path /data/PiERN/PDEbench/data/2d-ns/xxx.hdf5 \
  --output-dir /data/wyx/tensor_llm_assets/tensor_readout_qa_vx \
  --fields Vx \
  --sample-indices all \
  --time-indices all \
  --max-states 21000 \
  --num-bins 10 \
  --patch-size 32 \
  --latent-root /data/wyx/tensor_llm_assets/tensor_readout_latents_vx_2x
```

命令行参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--config` | Adapter pipeline 配置路径。 | 路径、`null` | 不传则全部依赖命令行参数和脚本默认值。 |
| `--hdf5-path` | PDEBench HDF5 文件路径。 | 路径 | - |
| `--output-dir` | QA 输出目录。 | 路径 | - |
| `--fields` | HDF5 字段列表。 | 逗号分隔字符串 | 如 `Vx` 或 `density,pressure,Vx,Vy`。 |
| `--sample-indices` | sample 索引。 | `all`、逗号分隔整数 | `all`：全部 sample。 |
| `--time-indices` | time 索引。 | `all`、逗号分隔整数 | `all`：全部 time step。 |
| `--max-states` | 最大 state 数。 | 正整数、`null` | `null`：不限制。 |
| `--seed` | 随机种子。 | 整数 | - |
| `--train-ratio` | train split 比例。 | 0 到 1 | - |
| `--val-ratio` | val split 比例。 | 0 到 1 | - |
| `--test-ratio` | test split 比例。 | 0 到 1 | - |
| `--spatial-stride` | 空间读取步长。 | 正整数 | `1`：不降采样。 |
| `--num-bins` | quantile bin 数量。 | 大于等于 2 的整数 | - |
| `--quantile-samples-per-state` | 每个 state 的 quantile 采样点数。 | 正整数 | - |
| `--patch-size` | patch 边长。 | 正整数 | - |
| `--point-bin-per-state` | 每个 state 的点值 bin 题数。 | 非负整数 | `0`：关闭该任务。 |
| `--point-compare-per-state` | 每个 state 的点比较题数。 | 非负整数 | `0`：关闭该任务。 |
| `--patch-compare-per-state` | 每个 state 的 patch 比较题数。 | 非负整数 | `0`：关闭该任务。 |
| `--max-quadrant-per-state` | 每个 state 的最大速度象限题数。 | `0`、`1` | 需要 `Vx,Vy`。 |
| `--global-stat-bin-per-state` | 每个 state 的速度统计题数。 | 0 到 3 | 需要 `Vx,Vy`。 |
| `--compare-min-bin-distance` | 比较题最小 quantile bin 距离。 | 非负整数 | - |
| `--compare-max-attempts` | 比较题重采样最大尝试次数。 | 正整数 | - |
| `--latent-root` | 写入 JSONL 的 latent 引用根目录。 | 路径、`null` | `null`：不写 `latent_ref`。 |
| `--include-oracle` / `--no-include-oracle` | 是否保存 oracle 数值。 | 布尔开关 | `--include-oracle`：保存；`--no-include-oracle`：不保存。 |

### 3.5 导出 Latent Cache

命令：

```bash
python scripts/export_tensor_readout_latents.py \
  --config configs/tensor_llm_adapter_pipeline.yaml
```

命令行参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--config` | Adapter pipeline 配置路径。 | 路径、`null` | - |
| `--qa-dir` | QA JSONL 目录。 | 路径 | - |
| `--splits` | 要扫描的 split。 | 逗号分隔字符串 | 例如 `train,val,test`。 |
| `--hdf5-path` | PDEBench HDF5 文件路径。 | 路径、`null` | `null`：尝试从 `qa_dir/metadata.json` 读取。 |
| `--compressor-checkpoint` | AE checkpoint 路径。 | 路径 | - |
| `--compressor-config` | AE config 路径。 | 路径、`null` | checkpoint 不含 config 时需要。 |
| `--fields` | AE 编码字段顺序。 | 逗号分隔字符串、`null` | `null`：从 checkpoint/config 读取。 |
| `--output-dir` | latent cache 输出目录。 | 路径 | - |
| `--batch-size` | AE encode batch 大小。 | 正整数 | - |
| `--device` | 运行设备。 | `auto`、`cpu`、`cuda`、`cuda:N` | - |
| `--spatial-stride` | 空间读取步长。 | 正整数 | `1`：不降采样。 |
| `--storage-dtype` | latent 保存 dtype。 | `float32`、`float16`、`bfloat16` | `float16`：省空间；`float32`：高精度。 |
| `--allow-field-mismatch` / `--no-allow-field-mismatch` | 是否允许 QA 字段和 AE 字段不一致。 | 布尔开关 | 默认不允许；除非做消融，否则不要允许。 |
| `--overwrite` / `--no-overwrite` | 是否覆盖已有 latent 文件。 | 布尔开关 | 默认不覆盖。 |

### 3.6 训练 Soft Prompt Adapter

命令：

```bash
source /data/wyx/tensor_llm_assets/env_tensor_llm.sh
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_llm_adapter.py \
  --config configs/tensor_llm_adapter_pipeline.yaml
```

命令行参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--config` | Adapter pipeline 配置路径。 | 路径、`null` | - |
| `--qa-dir` | QA JSONL 目录。 | 路径 | - |
| `--latent-dir` | latent cache 目录。 | 路径 | - |
| `--model-name-or-path` | LLM 模型名或路径。 | HF repo id、本地路径 | - |
| `--cache-dir` | HF cache 目录。 | 路径、`null` | - |
| `--hf-home` | HF_HOME 路径。 | 路径、`null` | - |
| `--output-root` | Adapter 输出根目录。 | 路径 | - |
| `--run-name` | run 名称。 | 字符串 | - |
| `--train-split` | 训练 split 名。 | 字符串 | - |
| `--val-split` | 验证 split 名。 | 字符串 | - |
| `--test-split` | 测试 split 名。 | 字符串 | - |
| `--max-train-records` | 最大训练记录数。 | 正整数、`null` | `null`：不限制。 |
| `--max-val-records` | 最大验证记录数。 | 正整数、`null` | `null`：不限制。 |
| `--max-test-records` | 最大测试记录数。 | 正整数、`null` | `null`：不限制。 |
| `--require-disjoint-splits` / `--no-require-disjoint-splits` | 是否强制 train/val/test 的 PDEBench sample 完全不交叉。 | 布尔开关 | 正式实验必须开启；overfit sanity 才关闭。 |
| `--require-untruncated-prompts` / `--no-require-untruncated-prompts` | 是否禁止自然语言 prompt 被静默截断。 | 布尔开关 | 正式实验建议开启；超长记录会在加载 LLM 权重前报错并写入 `prompt_audit.json`。 |
| `--prefer-record-latent-ref` / `--no-prefer-record-latent-ref` | 是否优先使用 JSONL 内的 `latent_ref`。 | 布尔开关 | 默认从 `latent_dir` 解析。 |
| `--device` | 训练设备。 | `auto`、`cpu`、`cuda`、`cuda:N` | - |
| `--torch-dtype` | LLM 权重 dtype。 | `auto`、`float32`、`float16`、`bfloat16` | A800 推荐 `bfloat16`。 |
| `--trust-remote-code` / `--no-trust-remote-code` | 是否执行 HF 仓库远端代码。 | 布尔开关 | Qwen2.5 通常不需要。 |
| `--seed` | 随机种子。 | 整数 | - |
| `--shuffle-seed` | `shuffled` baseline 的随机错配种子。 | 整数 | 固定后可复现实验；错配时排除相同 `state_ref`。 |
| `--epochs` | 训练轮数。 | 正整数 | - |
| `--batch-size` | 训练 batch 大小。 | 正整数 | - |
| `--eval-batch-size` | 评估 record batch 大小。 | 正整数 | - |
| `--eval-choice-batch-size` | 候选答案打分 batch 大小。 | 正整数 | - |
| `--gradient-accumulation-steps` | 梯度累积步数。 | 正整数 | - |
| `--lr` | adapter 学习率。 | 正数 | - |
| `--weight-decay` | 权重衰减。 | 非负数 | - |
| `--grad-clip-norm` | 梯度裁剪范数。 | 非负数 | `0`：不裁剪。 |
| `--ce-loss-weight` | 普通答案 token-level CE loss 权重。 | 非负数 | 例如 `0.1`；该项不应长期占主导，否则可能主要学习答案格式。 |
| `--choice-ce-loss-weight` | 候选项分类 CE loss 权重。 | 非负数 | 例如 `1.0`；该项更接近选择题正确率的可导 surrogate。 |
| `--ranking-loss-weight` | ranking loss 权重。 | 非负数 | `0`：关闭；默认从 config 读取。 |
| `--ranking-loss-margin` | ranking loss 的最小 restricted-choice CE 间隔。 | 非负数 | 要求正确 latent 的合法选项分类损失比负样本更低。 |
| `--ranking-loss-negative` | ranking loss 对照类型。 | `global_only`、`shuffled`、`random`、`no_latent`、`zero_latent` | 正式配置用不可训练的 `global_only`，避免随机 tensor 的同答案假负样本。 |
| `--swapped-question-loss-weight` | 同 tensor 问题交换 loss 权重。 | 非负数 | 需要 grouped sampling 和正的 choice CE 权重。 |
| `--swapped-question-loss-margin` | 问题交换最小 restricted-choice CE margin。 | 非负数 | 当前 `0.1`。 |
| `--swapped-question-max-records` | 每 batch 最多交换评分多少条记录。 | 正整数 | 当前 `8`。 |
| `--swapped-question-require-different-answer` / `--no-swapped-question-require-different-answer` | 是否只使用答案不同的问题交换。 | 布尔开关 | 正式训练开启。 |
| `--soft-prompt-tokens` | soft prompt token 数。 | 正整数 | - |
| `--adapter-dim` | adapter 内部维度。 | 正整数 | 必须能被 heads 整除。 |
| `--adapter-layers` | adapter 层数。 | 正整数 | - |
| `--adapter-heads` | adapter heads。 | 正整数 | - |
| `--dropout` | adapter dropout。 | 0 到 1 | - |
| `--latent-pos-encoding` | latent 位置编码方式。 | `grid`、`none` | `grid`：给二维 latent token 加坐标投影；`none`：不加位置。 |
| `--question-conditioning` / `--no-question-conditioning` | 是否用文本问题条件化 adapter query。 | 布尔开关 | 开启后同一 tensor 的 soft prompt 会随问题变化。 |
| `--question-condition-gate-init` | 文本问题条件分支的初始门控强度。 | 浮点数 | `1.0`：默认开启；`0.0`：初始近似关闭。 |
| `--structured-query-conditioning` / `--no-structured-query-conditioning` | 是否使用结构化 query 条件。 | 布尔开关 | 开启后从 query 字符串解析坐标和任务类型，不读取 oracle 数值。 |
| `--local-question-input-mode` | local branch 的问题表示来源。 | `contextual_tokens`、`input_embeddings` | 正式配置使用 frozen Qwen contextual states。 |
| `--local-context-layer` | contextual question 提前停止的 Qwen decoder 层数。 | 非负整数 | 当前配置为 `6`。 |
| `--local-context-layers` | residual 模式融合的 Qwen hidden-state 层。 | 逗号分隔非负整数 | 当前 `2,6`。 |
| `--local-fusion-mode` | local query 的文本/latent 融合路径。 | `residual_spatial_transformer`、`residual_qformer`、`anchor_queries`、`text_latent_pool` | 正式配置使用 `residual_spatial_transformer`；其余保留旧 checkpoint 兼容。 |
| `--local-text-encoder-layers` | 旧 input-embedding local branch 的额外文本 Transformer 层数。 | 非负整数 | contextual 正式配置为 `0`。 |
| `--freeze-conditioned-backbone` / `--no-freeze-conditioned-backbone` | 是否冻结 conditioned branch 继承的 Stage-1 backbone。 | 布尔开关 | 正式训练开启。 |
| `--local-text-gate-trainable` / `--no-local-text-gate-trainable` | 是否训练 text gate。 | 布尔开关 | 正式训练关闭并固定为 `1`。 |
| `--local-residual-gate-trainable` / `--no-local-residual-gate-trainable` | 是否训练 outer residual gate。 | 布尔开关 | 正式训练关闭并固定为 `1`。 |
| `--zero-init-local-text-attention` / `--no-zero-init-local-text-attention` | 是否零初始化新增 attention 输出。 | 布尔开关 | 正式训练开启。 |
| `--soft-prompt-scale` | soft prompt 输出尺度限制。 | 非负数 | `0.05`：推荐默认值；`0`：关闭限制。 |
| `--prompt-template` | 文本 prompt 模板。 | `task_specific`、`generic` | `task_specific`：按任务写规则；`generic`：旧版通用提示。 |
| `--max-prompt-tokens` | prompt 最大 token 数。 | 正整数 | 默认正式配置禁止截断；调试时关闭严格检查才会左截断。 |
| `--max-target-tokens` | target 最大 token 数。 | 正整数 | - |
| `--append-eos` / `--no-append-eos` | target 后是否追加 EOS。 | 布尔开关 | - |
| `--eval-baselines` | 评估 baseline 列表。 | 逗号分隔字符串 | 可包含 `correct,global_only,local_only,no_latent,zero_latent,shuffled,random,shuffled_stats`。 |
| `--final-eval-baselines` | 最终 best checkpoint 使用的完整 baseline 列表。 | 逗号分隔字符串 | 每轮通常只跑 `correct,shuffled`，最终再跑全部对照。 |
| `--choice-score` | 候选答案 NLL 计分方式。 | `mean`、`sum` | `mean`：按 token 平均；`sum`：累加。 |
| `--log-interval` | 训练日志间隔。 | 正整数 | - |
| `--console-progress` / `--no-console-progress` | 是否显示逐 batch 进度条。 | 布尔开关 | 默认关闭，控制台每 epoch 一行。 |
| `--save-step-metrics` / `--no-save-step-metrics` | 是否把每个 log interval 写入 `metrics_latest.json`。 | 布尔开关 | 默认关闭；step 曲线仍记录到 W&B。 |
| `--diagnostics-enabled` / `--no-diagnostics-enabled` | 是否在训练内运行固定小样本 hidden-state 诊断。 | 布尔开关 | 默认开启。 |
| `--diagnostics-every-epochs` | 诊断间隔 epoch 数。 | 非负整数 | `1`：每轮；`0`：关闭周期诊断。 |
| `--diagnostics-records-per-task` | 每种 task 固定诊断多少条。 | 正整数 | 默认 `1`，额外开销较小。 |
| `--diagnostics-layers` | 要保存的 LLM hidden-state 层。 | 逗号分隔整数 | `-1` 表示最后一层。 |
| `--wandb-enabled` / `--no-wandb-enabled` | 是否启用 W&B。 | 布尔开关 | - |
| `--wandb-api-key` | W&B API key。 | 字符串、`null` | 不建议写进命令历史；优先用环境变量。 |
| `--wandb-project` | W&B project 名称。 | 字符串 | - |
| `--wandb-entity` | W&B entity/team。 | 字符串、`null` | - |
| `--wandb-group` | W&B run group。 | 字符串、`null` | - |
| `--wandb-tags` | W&B tags。 | 逗号分隔字符串 | 例如 `adapter,tensor-llm,vx`。 |
| `--wandb-mode` | W&B 模式。 | `online`、`offline`、`disabled` | - |
| `--wandb-log-model` / `--no-wandb-log-model` | 是否上传 adapter checkpoint artifact。 | 布尔开关 | - |
| `--wandb-detailed-metrics` / `--no-wandb-detailed-metrics` | 是否把所有嵌套指标展开到 W&B。 | 布尔开关 | 默认关闭以保持面板可读。 |

训练目标默认包含普通 CE、候选项分类 CE 和 ranking 项：

```text
loss = ce_loss_weight * token_CE(answer + EOS | correct_latent)
     + choice_ce_loss_weight * ChoiceCE(correct_latent, own_question)
     + ranking_loss_weight * max(0, margin + ChoiceCE(correct_latent, own_question)
                                           - ChoiceCE(global_only, own_question))
     + swapped_question_loss_weight * max(0, margin + ChoiceCE(correct_latent, own_question)
                                                    - ChoiceCE(correct_latent, swapped_question))
```

`ChoiceCE` 只在该题合法的 A/B/C/D logits 内归一化，与正式 accuracy 使用同一评分空间；EOS 只留在低权重 token CE 中负责格式。`choice_01_loss = 1 - choice_accuracy` 是不可导日志指标，不参与反向传播。

### 3.7 Adapter 过拟合 Sanity Check

这个检查用于回答一个基础问题：当前 adapter+LLM 接口是否至少能在很小的数据集上学会依赖 tensor latent。它把同一个 split 的前 N 条记录同时作为 train/val/test，因此不是正式泛化评估。

命令：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_llm_adapter_overfit.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --records 2048 \
  --epochs 5 \
  --run-name tensor_llm_adapter_overfit_vx
```

需要透传主训练脚本参数时，直接追加即可：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_llm_adapter_overfit.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --records 2048 \
  --epochs 5 \
  --soft-prompt-tokens 256 \
  --adapter-layers 4 \
  --batch-size 16
```

命令行参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--config` | Adapter pipeline 配置路径。 | 路径 | 默认 `configs/tensor_llm_adapter_pipeline.yaml`。 |
| `--source-split` | 用作 train/val/test 的原始 split。 | split 名 | 默认 `train`。 |
| `--records` | 取前 N 条记录做过拟合检查。 | 正整数 | 建议先用 `1024` 或 `2048`。 |
| `--epochs` | 过拟合训练轮数。 | 正整数 | 建议 `5` 起步。 |
| `--run-name` | 输出 run 名称。 | 字符串 | 建议包含 `overfit`。 |
| `--eval-baselines` | baseline 列表。 | 逗号分隔字符串 | 默认 `correct,no_latent,zero_latent,shuffled`。 |
| `--dry-run` | 只打印实际调用命令，不执行。 | 开关 | 用于检查透传参数是否正确。 |
| `--diagnose` / `--no-diagnose` | 训练成功后是否自动运行 adapter 诊断脚本。 | 布尔开关 | 默认开启，使用 `adapter_best.pt`。 |
| `--diagnose-records` | 自动诊断输出的 record 数。 | 正整数 | 默认 64。 |
| `--diagnose-hidden-records` | 自动诊断中保存 hidden state 摘要的 record 数。 | 非负整数 | 默认 16；设 0 可跳过 hidden state。 |
| `--diagnose-split` | 自动诊断使用的 split。 | split 名、`null` | `null`：使用 `--source-split`。 |

结果解读：

| 现象 | 说明 | 下一步 |
|---|---|---|
| `correct` 明显高于 `zero_latent/shuffled` | adapter 至少能在小数据上利用正确 latent。 | 再讨论泛化、任务设计和数据规模。 |
| `correct` 仍接近 `zero_latent/shuffled` | 当前结构或训练目标仍没有迫使模型使用 latent。 | 新结构已包含 latent 2D 位置编码、query-conditioned adapter 和结构化 query 条件；若仍失败，优先考虑任务重构或开放 encoder。 |
| train loss 降但 `correct` 不涨 | 模型主要学到输出格式或标签先验。 | 不应只靠继续加 epoch 解决。 |

过拟合脚本默认会在训练成功后自动运行 `scripts/diagnose_tensor_llm_adapter.py`，输出到本次 run 目录，例如 `adapter_best_diagnostics_train.jsonl` 和对应 summary。若只想训练不诊断，传 `--no-diagnose`。

### 3.8 Direct Probe Latent 可读性检查

`scripts/train_tensor_direct_probe.py` 不加载 LLM，也不训练 soft prompt。它直接用 cached AE latent 和结构化 query 特征训练一个小分类器，用来判断 latent 本身是否包含完成 readout QA 的信息。

推荐先做 overfit 检查：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_direct_probe.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --overfit-records 2048 \
  --epochs 50 \
  --batch-size 64 \
  --run-name tensor_direct_probe_overfit_vx
```

这个命令会把 `train/val/test` 都设为同一个 `--source-split` 的前 2048 条，因此不是泛化评估。它的作用是定位问题：

| 现象 | 说明 | 下一步 |
|---|---|---|
| `correct` 能接近 100%，且明显高于 `zero_latent/shuffled` | AE latent 中有足够信息；失败主要在 adapter -> soft prompt -> LLM 接口。 | 给 adapter 加 local addressing 或重新设计 soft prompt 注入方式。 |
| `correct` 仍很低，且接近 `zero_latent/shuffled` | 当前 AE latent 或任务标签对 probe 也不可读。 | 检查 latent cache、AE 重建质量、bin 标签构造，或降低任务难度。 |
| `correct` 高但 `shuffled` 也高 | 模型可能利用了答案分布、query 先验或数据泄漏。 | 看分任务结果，必要时重采样更均衡的数据。 |

输出目录位于 `direct_probe.output_root` 或 `llm_training.output_root` 下，主要文件：

| 文件 | 说明 |
|---|---|
| `run_summary.json` | latent shape、记录数、probe 参数量。 |
| `metrics_latest.json` | 每个 epoch 的 train/val 指标。 |
| `test_metrics.json` | 使用 `probe_best.pt` 在 test split 上的最终指标。 |
| `probe_best.pt`、`probe_last.pt` | best/last checkpoint。 |

常用参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--config` | Adapter pipeline 配置路径。 | 路径 | 默认可用 `configs/tensor_llm_adapter_pipeline.yaml`。 |
| `--qa-dir` | QA JSONL 目录。 | 路径 | 不传则读 `data.qa_dir`。 |
| `--latent-dir` | latent cache 目录。 | 路径 | 不传则读 `data.latent_dir`。 |
| `--overfit-records` | 使用同一 split 前 N 条作为 train/val/test。 | 正整数、`null` | 用于 sanity check；会覆盖 split 和 max record 设置。 |
| `--source-split` | overfit 使用的来源 split。 | split 名 | 默认 `train`。 |
| `--train-split`、`--val-split`、`--test-split` | 正式训练/评估 split。 | split 名 | 不使用 `--overfit-records` 时生效。 |
| `--max-train-records`、`--max-val-records`、`--max-test-records` | 限制各 split 记录数。 | 正整数、`null` | `null`：使用完整 split。 |
| `--feature-mode` | probe 使用的 latent 特征。 | `global`、`local`、`local_global` | `global`：只用全局 mean/std；`local`：按 query 坐标采样局部 latent；`local_global`：二者都用。 |
| `--hidden-dim` | probe MLP 隐层维度。 | 正整数 | 默认 512。 |
| `--hidden-layers` | probe MLP 隐层数。 | 正整数 | 默认 2。 |
| `--dropout` | probe dropout。 | 0 到 1 | overfit 检查建议 0。 |
| `--eval-baselines` | 评估 latent baseline。 | `correct`、`zero_latent`、`shuffled`、`random` | `correct`：正确 latent；`zero_latent`：全 0 latent；`shuffled`：错配 state；`random`：随机噪声。 |

当前 direct probe 的 local 特征会根据 query 显式采样 latent grid：`point_bin` 采样 row/col，`point_compare` 采样 A/B 两点，`patch_compare` 对 A/B patch 位置做局部 pooling。它不读取 `oracle` 数值。

### 3.9 Tensor-as-Text Patch 对齐

`scripts/train_tensor_patch_text_alignment.py` 用于训练一个更直接的中间表示对齐任务。同一个 PDEBench patch 走两条路径：

```text
tensor path:
  patch -> value-preserving patch AE -> 16x16 latent grid -> spatial Transformer -> 256 soft prompt tokens
        -> frozen LLM -> middle-layer student hidden

text path:
  同一份 configured patch（当前为 per-patch z-score）序列化为文本
        -> frozen LLM -> middle-layer teacher hidden
```

当前正式路径使用 `alignment_text_layout: values_shared_suffix`。两条分支都先放置各自的 tensor 内容，再追加完全相同的设置相关 suffix；对齐位置是 suffix 最后一个 token 的 hidden state。当前候选 backbone 是 Qwen2.5-32B-Instruct；稀疏扫描中 Layer 2-40 的 target/control 约为 1，Layer 56 达到 1.81，因此正式配置使用 `teacher_layer: 56`，并建议长训练前用八个 point-value 模板在 48-64 层做一次局部确认。`hidden_states[0]` 只是上下文化前的输入 embedding，共享 readout token 在该位置看不到前面的 tensor，因此程序会拒绝 `teacher_layer <= 0`。

Transformer block 不会删除或重排序列位置。当前 256 个 soft embeddings 按 row-major 对应 `16x16` 网格；suffix 的最后位置是一基 `256 + suffix_token_count`。该位置在所有 LLM 层保持不变，变化的是 hidden vector。padding 只用于对齐 batch 长度，不会作为 readout。

在正式训练前可用只读扫描脚本比较 frozen teacher 的所有层。它沿用 config 中的数据切分、归一化、数值精度和 suffix，既不训练也不加载 AE、Q-Former 或 alignment projector：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/scan_tensor_teacher_layers.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --anchor-mode probe
```

EOS 对照只需把最后一项改成 `--anchor-mode eos`。当前 32B 配置先扫描 `4,8,12,16,24,32,40,48,56,64`，再围绕最佳候选做密集扫描；完整结果写入 `patch_alignment.output_root/teacher_layer_scan_*.json`。控制台的 `pair_cos` 越低表示跨样本角度坍缩越弱，`eff_rank` 越高表示变化占用的独立方向越多。probe 模式另外对被句子点名的数值施加扰动，并用数量和幅度相同、但位于 probe 支持域外的扰动作 control；`target/control > 1` 才说明该层对问题相关位置比无关位置更敏感。不能只凭最低 `pair_cos` 选层。

注意：空间适配器不直接预测某一层 hidden vector。它输出 native-width soft prompt embeddings，并把它们放到 frozen LLM 输入 embedding 前面；然后从同一个 frozen LLM 的 `teacher_layer` 取 student hidden state，与 text teacher hidden state 对齐。这样后续迁移到 soft prompt QA 时不会出现“训练时对齐中层、推理时却塞到输入层”的层级错配。旧 Q-Former 路径仍可由 `adapter_type: qformer` 复现。

Teacher branch 只包含配置后的 tensor 数值和形状分隔符：

```text
[[[...]; [...]; ...]]
<当前设置的共同 suffix>
```

当前 `patch_encoder.normalization.mode: zscore`、`scope: channel` 且 `teacher_text_source: normalized`。每个单字段 patch 使用自身 mean/std 标准化，AE 与文本 Teacher 接收同一份标准化 tensor；Teacher 文本按 `text_decimal_places: 3` 量化，因此来源相同但不是逐 bit 相同。若 normalization 或 clipping 改变了 AE 输入，程序会强制要求 `teacher_text_source: normalized`，防止两条分支静默读取不同数值。文本不含字段名、patch size、任务说明、`sample/time/top_left` 等信息。三层括号依次表示 channel、row、value 结构；当前 `field_sampling_mode: single` 下 channel 数为 1。

Student branch 不再使用旧的说明 prompt：

```text
[256 row-major spatial prefix embeddings]
<同一个共同 suffix>
```

`alignment_anchor_mode` 是单选实验设置，不会在一次训练中混合三档：

| 设置 | Teacher / Student 末尾 | 训练目标 |
|---|---|---|
| `eos` | tokenizer 的单个显式 EOS token | EOS 浅层 hidden InfoNCE |
| `representation` | `\nRepresentation:` | 冒号浅层 hidden InfoNCE |
| `probe` | batch 共享的自然句子 stem | shared-anchor hidden，经所选 feature transform 后做 InfoNCE |

正式配置选择 `alignment_anchor_mode: probe`。EOS 是序列边界/readout token，不归类为自然语言 probe。CLIP 的 EOT pooling 和 E5 的 EOS pooling 都伴随 text encoder 或 LLM 的对比式适配，不能据此假定冻结 Qwen 的 EOS hidden 天生适合作为 embedding；`eos` 和 `representation` 保留为可单独运行的基线，而不占用正式 probe 实验的训练 batch。

这一选择对应几类已验证但并不等价的做法：[CLIP](https://arxiv.org/abs/2103.00020) 读取 text encoder 最高层 EOT activation，并与 projection 一起端到端对比训练；[E5-Mistral](https://arxiv.org/abs/2401.00368) 使用最后 token/EOS pooling，但会对 LLM 做 embedding-oriented contrastive tuning；[BLIP-2](https://arxiv.org/abs/2301.12597) 第二阶段把 Q-Former 输出投到 LLM embedding 后作为前缀，并以语言建模行为训练，而不是假定 frozen LLM 的某个 EOS hidden 已经是对齐空间；[Prefix-Tuning](https://arxiv.org/abs/2101.00190) 同样通过完整生成似然约束连续前缀。由此，EOS 适合成为独立基线，短自然语言 probe 则更接近后续问答时的使用上下文。

`representation` 和 `probe` 都使用 `add_special_tokens=False` 单独编码，后面不追加 EOS；否则三种设置最终都会退化成“读取 EOS hidden”。probe 不在 prompt 中列出候选项，也不使用统一的 `Answer:` 标记。当前 32B 正式配置只启用 `point_value`：Stage 1 负责让 tensor 数值可被读取，差值、均值和其他组合运算留给冻结 LLM 的后续层。其余 probe family 的实现仍保留，但不进入本轮训练或 checkpoint 选择。坐标和措辞随 batch 改变；它们只规定读取 hidden state 的自然语言条件，不使用字段名、样本元数据或下游任务标签。

```text
The value at row 3, column 7 is
```

同一 batch 的所有样本、Teacher/Student 两侧以及全部 DDP rank 使用完全相同的 stem 和坐标，这样 InfoNCE 不能靠 prompt 身份识别正样本。`point_value` 有八种短模板，并使用 `is`、`equals`、`gives`、`contains` 四类自然 readout 结尾；程序读取结尾 token 在所选 transformer block 的 raw hidden，再经过固定 whitening 做对比，不计算 next-token logits，也不要求续写某个规范词。

代码强制执行以下 probe contract：

- stem 必须以换行开始、停在可自然接续数值的短 readout 词，且不得包含候选词、问号、`Answer:`、`Options:` 或 A/B 格式。
- 每个 family 的模板数由代码契约明确规定；当前正式 `point_value` family 为八个，训练均匀循环模板，坐标仍由 seed 随机生成。
- 双点操作的坐标必须互异；区域必须完整落在 patch 内；多通道时 stem 中的 channel 与 `probe_parameters` 一致。
- probe 的标量结果按 teacher 文本相同的小数位从实际可见数值计算，但绝不附加到 Teacher/Student 输入；它只用于识别不应作为负样本的等价结果和生成诊断。
- probe 不定义答案词表、类别索引、LM-head CE 或 Teacher logits KL；表示监督仍来自成对 hidden 的 InfoNCE。
- 同一 batch 和所有 DDP rank 使用完全相同的 stem/token IDs；数值正文不允许静默截断。

AE warmup 前会穷举全部已启用的 `1 family x 8 templates`。tokenization preflight 检查上述结构、八种模板是否保持不同 token 序列、所有 suffix 长度和正文截断。任一结构或 tokenization 契约失败都会在训练前终止。

`teacher_probe_preflight.json` 会记录每个 probe 的 frozen-teacher hidden 与数值目标几何相关性。正式配置使用 128 条 train record，并把八个措辞模板按 numeric family 聚合，以 family 中位数作为稳定诊断。`teacher_probe_warn_below_correlation` 只产生 warning，不阻止长训练；tokenization、正文截断和 probe contract 失败仍会硬终止。旧的 `teacher_probe_min_correlation` 硬门槛已删除，因为它把小样本的观测相关性错误地当成了 teacher 可训练性的必要条件。

Qwen 本身的数值矩阵能力使用独立脚本测试，不属于训练入口：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/test_qwen_numeric_matrix_tasks.py \
  --config configs/tensor_llm_adapter_pipeline.yaml
```

它分别向完整 Qwen 提供短合成矩阵和真实归一化 PDE patch，要求回答当前 probe family 的数值查找/运算，并将输入、生成文本、解析值和误差写入 `outputs/diagnostics/*_qwen_numeric_matrix_test.json`。该测试不使用 Q-Former，不向 prompt 泄露答案，也不参与 loss、样本筛选或 checkpoint 选择。训练脚本只保留 frozen teacher hidden 的 probe 预检。

正式目标不减 probe-only baseline。`alignment_transform.mode` 可选三档：

| mode | InfoNCE 输入 | 是否增加可训练参数 | 第二阶段是否使用 |
|---|---|---:|---:|
| `none` | raw hidden 直接 L2 normalize | 否 | 否 |
| `projection` | 两侧独立的 `LayerNorm + Linear` | 是 | 否 |
| `whitening` | teacher 训练集拟合的同一套 frozen PCA-whitening 变换 | 否 | 否 |

当前正式配置使用 `whitening`。程序先从 train split 收集 teacher hidden，估计均值 `mu` 和协方差，保留方差最大的 512 个 PCA 方向，再对这些方向做带 shrinkage 的 whitening，然后固定：

```text
Teacher = L2_normalize((teacher_hidden - mu) @ W)
Student = L2_normalize((student_hidden - mu) @ W)
loss = 0.25 * weighted_directional_InfoNCE(Student, Teacher; i2t=0.75, t2i=0.25)
     + 1.00 * ddp_global_centered_symmetric_InfoNCE(Student, Teacher)
     + 0.25 * ddp_global_centered_symmetric_InfoNCE(native_student_hidden, native_teacher_hidden)
     + 0.10 * transformed_and_native_branch_mean_alignment
```

两侧必须减同一个 teacher mean 并乘同一个矩阵；分别 whitening 会产生不同坐标系，代码不允许这种做法。PCA 特征值另受 `max_condition_number: 1000` 约束，避免放大低方差噪声。centered loss 使用所有 DDP rank 的同一个 128-record probe batch 均值，而不是每卡各减 32-record 均值。centered retrieval 只负责学习实例残差；额外的 transformed/native branch-mean loss 约束绝对分支位置，因此第二阶段和单样本推理不依赖 candidate-library centering。`projection` 档仍保留旧的一层独立 projection 以复现实验。

probe 模式的全局 retrieval 使用 `evaluation_probe_count` 个固定 sentence stem，每个 family 在整个验证 split 上单独编码和检索，再做平均；同一个候选库内绝不混入不同 probe。量化后标量结果相同的非配对候选会从 InfoNCE 分母排除，避免把相同语义硬当负样本；配对正样本不变。严格的一对一 i2t/t2i 仍在未屏蔽的完整候选集上报告，另增 `semantic_*` 指标，因此该修正不会把宽松检索冒充实例检索。EOS 和 representation 设置各自只运行自己的全局检索。数值正文和 suffix 分开 tokenization；正文超出 `max_text_tokens - suffix_tokens` 会直接失败。

旧版“说明 + 字段 + 数值 + anchor / soft embeddings + 说明 + anchor”仍可通过 `alignment_text_layout: legacy_prompt` 复现，届时 `text_prompt_template` 才生效。

默认 tensor path 不把 `16x16` patch resize 到 `512x512`。当前空间实验新建 value-preserving patch AE：

```text
16x16x1 patch -> 16x16x8 latent -> 256 row-major spatial tokens
```

latent 的第一个通道逐元素保留归一化输入，另外 7 个通道学习局部特征；`16*16*8` 与旧 `4*4*128` 的 latent 标量数相同，因此实验主要比较空间拓扑而不是容量。固定二维正弦编码标记 row/column，空间 self-attention 提供上下文，独立的局部残差路径避免上下文化抹掉对应位置的数值。

如果 `encoder_source: checkpoint`，则加载 `compressor_checkpoint`；这适合调用已经训练好的 patch AE，或者临时复用旧的 512x512 compressor。只有后一种情况才应设 `resize_patch_to_compressor_input: true`。

默认 patch size 是 `16x16`。`8x8` 信息量偏少，`32x32` 文本 token 开销明显增大；模型筛选显示 Qwen2.5-32B-Instruct 在当前 16x16 point-value 测试上达到 86% 容差正确率，因此作为本轮 teacher 候选。

当前默认 `split_mode: sample`，train/val/test 会使用互不重叠的 `sample_index`，每个 split 内再随机采样 time 和空间 patch。这比旧的 `random_record` 更严格，可以避免验证集来自训练集已见过的 simulation trajectory。`run_summary.json` 会记录每个 split 的 sample/time 数量、预览和 exact record overlap。

命令：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_patch_text_alignment.py \
  --config configs/tensor_llm_adapter_pipeline.yaml
```

多卡扩大对比学习负样本：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3,4,5 \
torchrun --standalone --nproc_per_node=4 \
  scripts/train_tensor_patch_text_alignment.py \
  --config configs/tensor_llm_adapter_pipeline.yaml
```

多卡模式会自动读取 `WORLD_SIZE/RANK/LOCAL_RANK`，`CUDA_VISIBLE_DEVICES` 中的物理 GPU 会在各进程内重新编号为 `cuda:0...`。`batch_size` 是**每卡 batch**，全局每步样本数和 InfoNCE 候选数均为 `batch_size * GPU数`；Layer 56 先用命令行覆盖 `--batch-size 1` 做 smoke，正式配置为每卡 4，并启用 frozen-backbone activation checkpointing。这里 Teacher 虽有约 1800 个文本 token，但无梯度；需要反向的 Student 只有 soft prefix 和短 suffix。smoke 显存充足后可测试每卡 8，优先扩大真实同-probe negatives。DDP 会在每卡复制一份截断后的 frozen teacher，不会分片 32B 权重。训练使用可微分 all-gather，远端样本作为负样本时的 candidate-side 梯度也会回传；参数梯度按 dtype/device 扁平分桶后同步，不再为每个参数单独发起 NCCL collective。验证/测试使用无 padding 的精确 rank 分片，每条样本只编码一次，再 gather 全局 retrieval 候选。`train_records` 是整个训练 split 的记录数，不会按 GPU 再复制。

`distributed_timeout_seconds` 默认 1800。epoch 和 checkpoint 边界会打印 `ddp_wait` / `ddp_synced` 阶段名；若某个 rank 失步，参数梯度和指标 key 的跨 rank schema 检查会尽量在真正的 collective 顺序分叉前给出明确错误。checkpoint 先写同目录临时文件再原子替换，避免中断留下半写文件。

小规模 smoke test：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_patch_text_alignment.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --train-records 128 \
  --val-records 32 \
  --test-records 32 \
  --epochs 1 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --patch-ae-pretrain-epochs 0 \
  --text-preflight-records 32 \
  --run-name tensor_patch_text_alignment_smoke
```

这个 smoke test 只检查完整数据/模型/反向传播/评估链路，不用于判断效果；它仍会运行当前 8 个 probe contract 和数值正文 tokenization preflight。当前正式配置已经复用已训练 AE 并将 warmup 设为 0，命令行保留 `--patch-ae-pretrain-epochs 0` 是为了让 smoke 的意图明确。

两卡分布式 smoke test：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,5 \
torchrun --standalone --nproc_per_node=2 \
  scripts/train_tensor_patch_text_alignment.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --train-records 128 \
  --val-records 32 \
  --test-records 32 \
  --epochs 1 \
  --batch-size 2 \
  --eval-batch-size 4 \
  --patch-ae-pretrain-epochs 0 \
  --text-preflight-records 32 \
  --run-name tensor_patch_text_alignment_ddp_smoke
```

与当前服务器布局一致的四卡 Layer-56 smoke（优先运行这一条）：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,4,5,6 \
torchrun --standalone --nproc_per_node=4 \
  scripts/train_tensor_patch_text_alignment.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --train-records 256 \
  --val-records 64 \
  --test-records 64 \
  --alignment-whitening-records 128 \
  --epochs 1 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --run-name tensor_patch_text_alignment_qwen25_32b_layer56_ddp_smoke
```

这条命令使用全局 batch 4，只检查 Layer 56 的显存、反向传播、扁平梯度同步、精确验证分片、checkpoint 和最终 test。成功时应出现 `ddp_synced stage=alignment_epoch_0001_checkpointed`、`ddp_synced stage=final_test_written`，并生成可读取的 `alignment_best.pt` 与 `test_metrics.json`。之后再按配置的每卡 batch 4 启动正式训练；若 OOM，依次尝试每卡 2、1，而不是降低训练记录数来掩盖显存问题。

主要 loss：

```text
loss = contrastive_loss_weight * weighted_directional_InfoNCE(tensor_embedding, text_teacher_embedding; i2t=0.75, t2i=0.25)
     + centered_contrastive_loss_weight * false_negative_masked_centered_symmetric_InfoNCE(...)
     + native_centered_contrastive_loss_weight * false_negative_masked_centered_symmetric_InfoNCE(native_hidden...)
     + mean_alignment_loss_weight * transformed_and_native_branch_mean_alignment(...)
     + reconstruction_loss_weight * MSE(patch_AE_reconstruction, normalized_patch)
```

第一阶段通过 `alignment_transform.mode` 在 `none`、`projection`、`whitening` 中三选一。当前为 `whitening`：只用 train split teacher hidden 拟合一次共享、冻结、截断 PCA-whitening；它没有优化器参数，也不进入第二阶段。`projection` 才会创建两侧 projection head。旧 `alignment_projection.enabled` 仅保留兼容读取，新配置不要与 mode 同时设置。probe target 只定义合法负样本，不进入模型。代码没有 baseline subtraction、答案分类、LM-head CE、Teacher logits KL、数值 decoder 或新增任务 head。

旧配置中的 `center_embeddings`、cosine loss 和 probe answer/KL 字段不会被静默忽略；脚本会在加载模型前列出残留字段并终止。`alignment_transform`、`alignment_projection` 的结构参数和 `centered_contrastive_loss_weight` 是当前受支持设置。

脚本同样会在长训练前拒绝无效边界：`epochs <= 0`、空 DataLoader、0 层 adapter、空间 token 数与网格不一致、单卡 batch 1 导致 InfoNCE 没有负样本、新建随机 AE 却被冻结，以及可训练 AE 却关闭 reconstruction loss。

`alignment_best.pt` 使用整个验证 split 的方向加权严格 CE：`contrastive_weight * (0.75 * global_i2t_CE + 0.25 * global_t2i_CE)`，再加 batch-level centered、native-centered 和 branch-mean 项；全局 candidate-library centered retrieval 和全局 branch mean 只作诊断，不参与 checkpoint 选择，因为单样本部署没有验证集均值。全局评估缺失时回退到对应 batch 指标。i2t 是实际 tensor→text 部署方向，t2i 保留为较小的 one-to-one/hubness 辅助约束，而不是被删除或与 i2t 等权。语义同值负样本屏蔽只影响训练 loss，不用于美化 checkpoint 选择。

`freeze_patch_ae_after_pretrain: false` 时，AE 在 reconstruction warmup 后继续随 alignment loss 更新；设为 `true` 时只训练 adapter。空间 adapter 的输出线性层直接生成与 Qwen input embedding 同维的连续 prefix embeddings。

新布局会在 AE warmup 前逐 anchor 检查 tokenization，并输出最坏情况下的 `content_token_max`、`suffix_tokens` 和 `content_truncated`。anchor 单独编码，因此不会被截断；默认 `fail_on_text_max_length_hit: true` 会在任意数值正文超出预算时直接报错。`fail_on_text_anchor_missing` 只用于 `legacy_prompt` 兼容路径。

输出文件：

| 文件 | 说明 |
|---|---|
| `run_summary.json` | 真实开始/结束时间与耗时，以及 patch 大小、字段、split plan、record overlap、encoder 来源、latent grid、LLM hidden size、teacher layer、adapter 参数量。 |
| `probe_contract.json` | 启动时穷举的已启用 stem、family/template、坐标和实际 token IDs；当前 point-value 配置为 8 个。 |
| `probe_tokenization_preflight.json` | 已启用 stem 的长度、token 序列和数值正文截断检查。 |
| `teacher_probe_preflight.json` | AE warmup 前对最多 128 条 train record 和全部已启用 family/template 做 frozen-teacher 只读诊断，按 family 聚合记录 hidden-目标相关性、最近邻目标误差与 target collision。 |
| `*_qwen_numeric_matrix_test.json` | 独立脚本产生的完整 frozen Qwen 数值 generation baseline，含逐例输入输出与分 source/family 指标。 |
| `alignment_whitening.json` | whitening 档的 train record 数、均值范数、协方差谱和正则后 condition number。其他两档不生成。 |
| `metrics_latest.json` | patch AE warmup、每轮 train/val loss、reconstruction loss、i2t/t2i retrieval accuracy。 |
| `test_metrics.json` | 使用 `alignment_best.pt` 的最终 test 指标。 |
| `patch_ae_pretrain_best.pt`、`patch_ae_pretrain_last.pt` | patch AE 验证最优和最后一轮 checkpoint；alignment 自动恢复 best。 |
| `alignment_best.pt`、`alignment_last.pt` | 对齐 adapter、所选 feature transform 状态和 compressor checkpoint；下游只加载 adapter/compressor，transform 仅用于第一阶段评估。 |

W&B 曲线：

| 曲线名 | 说明 |
|---|---|
| `patch_ae_pretrain_step/reconstruction_loss` | patch AE 预训练阶段的 step-level 平均重建误差。 |
| `patch_ae_pretrain_step/current_reconstruction_loss` | patch AE 当前 batch 重建误差。 |
| `patch_ae_pretrain/reconstruction_loss` | patch AE 每个预训练 epoch 的平均重建误差。 |
| `patch_ae_pretrain/val_reconstruction_loss` | patch AE 每个预训练 epoch 后的验证集重建误差，用于观察过拟合。 |
| `patch_ae_pretrain/relative_rmse_to_target_std` | patch AE 预训练 RMSE / patch 自身 std；小于 1 才说明优于只预测 patch 均值。 |
| `train/reconstruction_loss` | alignment 阶段训练集重建误差；`train_patch_ae: true` 时可观察 AE 是否继续变化。 |
| `val/reconstruction_loss` | alignment 阶段验证集重建误差。 |
| `train/reconstruction_relative_rmse_to_target_std`、`val/reconstruction_relative_rmse_to_target_std` | alignment 阶段重建相对误差诊断；用于判断裸 MSE 变大是量纲问题还是 AE 退化。 |
| `train/contrastive_loss`、`val/contrastive_loss` | tensor embedding 与 text teacher embedding 的对比学习 loss。 |
| `train/i2t_accuracy`、`val/i2t_accuracy` | batch 内 tensor-to-text retrieval accuracy。 |
| `train/t2i_accuracy`、`val/t2i_accuracy` | batch 内 text-to-tensor retrieval accuracy。 |
| `train/semantic_i2t_accuracy`、`val/semantic_t2i_accuracy` | top-1 候选的 probe 数值结果是否等价；只作语义诊断，不能替代严格 i2t/t2i。 |
| `train/semantic_collision_fraction` | 原候选中量化后结果相同、因而从 InfoNCE 负样本分母排除的比例。 |
| `train/teacher_probe_hidden_similarity_vs_negative_target_distance_pearson` | Teacher hidden 相似度与 probe 数值接近程度的相关性；越高越支持 frozen teacher 确实编码了所问数值。 |
| `train/alignment_soft_prompt_gradient_norm` | loss 对 soft prompts 的平均梯度范数；接近 0 表示所选 teacher layer 对前缀难以控制。 |
| `train/alignment_soft_prompt_active_token_fraction` | 梯度大于该样本最大 token 梯度 `1e-3` 的 token 比例；空间模式下用于发现只有少数位置收到更新。 |
| `train/alignment_soft_prompt_gradient_entropy` | token 梯度分布的归一化熵，范围约为 0 到 1；越接近 1 表示更新覆盖越均匀。 |
| `train/candidate_count` | 训练 InfoNCE 的候选数；单卡等于 batch size，多卡等于每卡 batch size 乘 GPU 数。 |
| `train/centered_i2t_accuracy`、`val/centered_i2t_accuracy` | 所选 transform 空间的 batch-centered retrieval；当前以配置中的 centered loss 权重参与主要残差目标，不作为单独 checkpoint 指标。 |
| `val/global_i2t_accuracy`、`val/global_t2i_accuracy` | 整个验证 split 内的未居中 retrieval accuracy，是主要全局指标。 |
| `val/global_centered_i2t_accuracy`、`val/global_centered_t2i_accuracy` | 整个验证 split 内的 centered retrieval，仅用于诊断 hidden 各向异性。 |
| `val/global_hidden_uncentered_i2t_accuracy`、`val/global_hidden_uncentered_t2i_accuracy` | 不经过任何 alignment transform 的 raw Qwen hidden 全局 retrieval；用于判断 transform 前后的差距。 |
| `train/teacher_anchor_missing_fraction`、`val/teacher_anchor_missing_fraction` | teacher text tokenization 后 anchor 缺失比例；默认应为 0，否则代码会报错。 |
| `train/teacher_duplicate_text_fraction`、`val/teacher_duplicate_text_fraction` | 当前 batch 中量化后 teacher tensor 文本的重复比例；非 0 表示 InfoNCE 可能把等价文本当作负样本。 |
| `train/teacher_max_length_hit_fraction`、`val/teacher_max_length_hit_fraction` | teacher text 达到 `max_text_tokens` 的比例；非 0 时应考虑增大上下文或减小 patch/text 精度。 |
| `train/student_soft_prompt_token_norm`、`val/student_soft_prompt_token_norm` | soft prompt token 的平均范数。 |
| `train/student_text_token_embedding_norm`、`val/student_text_token_embedding_norm` | 普通文本 token embedding 的平均范数，可和 soft prompt norm 对比。 |
| `train/alignment_student_embedding_pairwise_cosine`、`val/alignment_student_embedding_pairwise_cosine` | student embedding 的 batch 内非对角 cosine 均值，用于检查塌缩。 |
| `train/alignment_teacher_embedding_pairwise_cosine`、`val/alignment_teacher_embedding_pairwise_cosine` | teacher embedding 的 batch 内非对角 cosine 均值。 |

要启用 W&B：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_patch_text_alignment.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --wandb-enabled \
  --wandb-mode online
```

已训练 patch AE 可以按路径冻结复用。第一次训练后，优先用 `alignment_best.pt`；如果只想用 reconstruction warmup 后的 AE，则用 `patch_ae_pretrain_best.pt`：

```yaml
patch_alignment:
  encoder_source: checkpoint
  compressor_checkpoint: /data/wyx/tensor_llm_outputs/runs/CHANGE_ME/alignment_best.pt
  train_patch_ae: false
  patch_ae_pretrain_epochs: 0
  resize_patch_to_compressor_input: false
```

该 checkpoint 内会保存 `compressor_config` 和 `compressor_state_dict`，因此通常不需要再单独提供 `compressor_config`。如果复用旧的 `scripts/train_compressor.py` 产物，则仍然支持读取其中的 `model_state_dict` 和 `config`。

常用参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--patch-size` | 从 PDEBench 原始场中裁剪的方形 patch 边长。 | 正整数 | 默认 16；建议先在 16 和 32 中比较。 |
| `--fields` | 使用哪些 HDF5 字段。 | 逗号分隔字段 | 配合 `--field-sampling-mode single` 时，多个字段不会堆成通道，而是作为单通道 patch 的采样池。 |
| `--field-sampling-mode` | 多个 HDF5 字段如何进入 patch。 | `channels`、`single` | `channels`：旧行为，所有字段堆成 `[C,H,W]`；`single`：每条 record 随机选一个字段，patch 保持 `[1,H,W]`。 |
| `--train-records`、`--val-records`、`--test-records` | 随机采样 patch 数。 | 正整数 | 初始 smoke test 用小值，正式训练可增大。 |
| `--split-mode` | train/val/test 如何隔离采样轴。 | `sample`、`time`、`sample_time`、`random_record` | `sample`：默认，按 `sample_index` 隔离；`time`：按 `time_index` 隔离；`sample_time`：两者都隔离；`random_record`：旧随机记录方式，只适合 smoke test。 |
| `--split-train-ratio`、`--split-val-ratio`、`--split-test-ratio` | sample/time 轴隔离时的 split 比例。 | 正数 | 默认 `0.8/0.1/0.1`。 |
| `--unique-records` / `--no-unique-records` | 是否避免 split 内完全重复的 `(sample,time,row,col)`。 | 布尔开关 | 默认开启。 |
| `--ensure-disjoint-records` / `--no-ensure-disjoint-records` | 是否禁止 train/val/test 出现完全相同 record。 | 布尔开关 | 默认开启；触发说明 split 采样有泄漏。 |
| `--encoder-source` | encoder 从哪里来。 | `patch_ae_config`、`checkpoint` | `patch_ae_config`：按 YAML 构建 patch AE；`checkpoint`：加载 `compressor_checkpoint`。 |
| `--train-patch-ae` / `--no-train-patch-ae` | 是否更新 AE 参数。 | 布尔开关 | 新建 patch AE 建议 `true`；加载已训练 patch AE 做纯 adapter 对齐时可设 `false`。 |
| `--freeze-patch-ae-after-pretrain` / `--no-freeze-patch-ae-after-pretrain` | AE warmup 后 alignment 阶段是否冻结 AE。 | 布尔开关 | 当前为 `true`，warmup 后只训练空间 adapter。 |
| `--patch-ae-pretrain-epochs` | 对齐前 reconstruction-only warmup 轮数。 | 非负整数 | `0`：跳过；大于 0：先训练 patch AE 重建。 |
| `--compressor-checkpoint` | 已训练 encoder checkpoint。 | 路径、`null` | 仅 `encoder_source: checkpoint` 必需。 |
| `--resize-patch-to-compressor-input` / `--no-resize-patch-to-compressor-input` | 是否把 patch resize 到 encoder `input_size` 后再编码。 | 布尔开关 | patch AE 应设 `false`；旧 512x512 AE 才设 `true`。 |
| `--adapter-type` | tensor path 的对齐 adapter。 | `spatial_transformer`、`qformer`、`pooled_mlp` | 当前使用 `spatial_transformer`：每个 latent 位置产生一个同位置 token；其余值保留兼容。 |
| `--query-tokens` | soft token 数。 | 正整数 | spatial 模式必须严格等于 `latent_height * latent_width`；Q-Former 模式表示 learnable query 数。 |
| `--adapter-layers` | adapter block 数。 | 正整数 | 当前空间 adapter 为 2 层 self-attention。 |
| `--adapter-heads` | adapter attention heads。 | 正整数 | 必须整除 `adapter_dim`。 |
| `--soft-prompt-scale` | soft prompt 输出尺度限制。 | 非负数 | `0.05`：`tanh` 后限制每维约在 `[-0.05,0.05]`；`0`：关闭限制。 |
| `--reconstruction-loss-weight` | patch AE 重建 MSE 权重。 | 非负数 | 只在 `train_patch_ae: true` 时影响训练。 |
| `--teacher-text-source` | teacher branch 序列化哪一种 patch。 | `raw`、`normalized` | 当前为 `normalized`；只要 AE 启用 normalization/clipping，就必须使用该值。 |
| `--alignment-text-layout` | Teacher/Student 的文本布局。 | `values_shared_suffix`、`legacy_prompt` | 默认前者：两侧内容都在同一个短 suffix 前；后者复现旧的不对称说明 prompt。 |
| `--alignment-anchor-mode` | 本次训练使用的唯一对齐设置。 | `eos`、`representation`、`probe` | 正式配置为 `probe`；三档不会在一次训练中混合。 |
| `--representation-suffix` | `representation` 模式的极短文本后缀。 | 字符串 | 当前为换行后接 `Representation:`；末尾不追加 EOS。 |
| `--probe-families` | hidden readout probe 的通用连续数值语义。 | `point_value,point_difference,point_mean,region_mean,region_range` 的子集 | 每个 batch 生成共享 sentence stem/坐标；结果不进入模型，只排除等价 false negatives。 |
| `--probe-region-size` | 区域 probe 的正方形边长。 | 小于 patch size 的正整数 | 当前 4。 |
| `--evaluation-probe-count` | 验证和 global retrieval 使用的固定 probe 数。 | 正整数 | 当前 8，覆盖 point-value 的八种模板。 |
| `--max-shared-suffix-tokens` | 单个非 EOS suffix 的 token 数硬上限。 | 正整数 | 当前 32；实际 token 数由 preflight 记录，超限直接失败。 |
| `--fail-on-text-anchor-missing` / `--no-fail-on-text-anchor-missing` | tokenization 后 anchor 缺失时是否直接报错。 | 布尔开关 | 默认开启；关闭后只记录缺失比例。 |
| `--fail-on-text-max-length-hit` / `--no-fail-on-text-max-length-hit` | tokenized 文本打满 `max_text_tokens` 时是否直接报错。 | 布尔开关 | 默认开启；关闭后只记录 `*_max_length_hit_fraction`。 |
| `--global-retrieval-eval` / `--no-global-retrieval-eval` | eval 时是否额外计算整个 split 的 retrieval。 | 布尔开关 | 默认开启；比 batch 内 retrieval 更严格。 |
| `--global-retrieval-max-records` | 允许全局 retrieval 的最大 split 条数。 | 正整数 | 默认 8192；超过则跳过，避免相似度矩阵过大。 |
| `--global-retrieval-chunk-size` | 全局 retrieval 矩阵分块大小。 | 正整数 | 默认 1024；显存/内存紧张时调小。 |
| `--text-prompt-template` | 旧 Teacher prompt 模板。 | `compact`、`compact_with_metadata`、`plain` | 仅 `alignment_text_layout: legacy_prompt` 生效；新布局固定使用无字段的数值序列化。 |
| `--text-decimal-places` | 文本化 tensor 数值保留小数位。 | 非负整数 | 当前归一化 patch 使用 3。 |
| `--max-text-tokens` | LLM 文本路径最大 token 数。 | 正整数 | 当前配置为 3072；严格 preflight 会报告真实长度并拒绝截断。 |
| `--text-preflight-records` | AE warmup 前先检查多少条 teacher text 的 tokenization。 | 非负整数 | 默认 32；设 0 跳过预检查。 |
| `--teacher-layer` | 取 LLM 哪一层 hidden state。 | `1..num_hidden_layers` | 当前 32B 配置使用 56；0/负数和超过模型深度的索引会在 AE warmup 前终止。 |
| `--temperature` | InfoNCE 温度。 | 正数 | 默认 0.07。 |
| `--contrastive-loss-weight` | 未中心化 symmetric InfoNCE 权重。 | 非负数 | 当前 0.25，保留绝对空间约束但不让公共 probe 方向主导训练。 |
| `--contrastive-i2t-weight` / `--contrastive-t2i-weight` | 两个 retrieval 方向在每个 InfoNCE 中的相对权重，会自动归一化。 | 非负数，和为正 | 当前 i2t=0.75、t2i=0.25；i2t 对应最终 tensor→text 部署，t2i 保留为防 hubness 的辅助约束。 |
| `--projection-dim` | adapter 输出到 LLM 的 soft prompt 维度。 | `null` 或 LLM hidden size | 默认 `null` 自动匹配 LLM hidden size；这是输入桥接层，不是 post-hidden 对齐投影。 |
| `--alignment-transform-mode` | hidden readout 后的对比空间。 | `none`、`projection`、`whitening` | 当前 `whitening`；三档互斥。 |
| `--alignment-whitening-records` | 拟合固定 teacher whitening 的 train 记录数。 | `>=2` | 当前 2048；只在训练开始前读取一次。 |
| `--alignment-whitening-dim` | 保留的 teacher PCA 主方向数。 | `1..hidden_size` | 当前 512；丢弃不稳定低方差方向。 |
| `--alignment-whitening-shrinkage` | teacher covariance 向各向同性协方差收缩的比例。 | `[0,1]` | 当前 0.01，用于限制低方差方向的噪声放大。 |
| `--alignment-whitening-epsilon` | whitening 特征值相对下限。 | 正数 | 当前 `1e-5`。 |
| `--alignment-whitening-max-condition-number` | whitening 正则后协方差最大条件数。 | `>=1` | 当前 1000，限制低方差方向增益。 |
| `--centered-contrastive-loss-weight` | DDP 全局 batch centered InfoNCE 权重。 | `>=0` | 当前 1.0，作为主要实例残差目标。 |
| `--native-centered-contrastive-loss-weight` | 原生 LLM hidden 的 centered InfoNCE 权重。 | `>=0` | 当前 0.25；约束可迁移的 LLM 原生 hidden 空间。 |
| `--mean-alignment-loss-weight` | transformed/native 分支均值方向与范数匹配权重。 | `>=0` | 当前 0.1；避免推理时依赖中心化。 |
| `--alignment-patch-ae-lr-scale` | alignment 阶段 AE 相对 adapter 的学习率倍率。 | `(0,1]` | 当前 AE 在 warmup 后冻结，此值不生效。 |
| `--teacher-probe-warn-below-correlation` | probe family 中位相关性低于此值时打印 warning。 | `null` 或 `[-1,1]` | 当前 0.1；只告警，不阻断训练。 |
| `--teacher-probe-diagnostic-records` | 每个 probe 模板用于 frozen-teacher 语义诊断的 train record 数。 | `>=2` | 当前 128；八个措辞模板在 family 内聚合。 |
| `--alignment-projection-enabled` / `--no-alignment-projection-enabled` | 旧配置兼容开关。 | 布尔开关 | 新实验改用 `--alignment-transform-mode`；冲突设置会直接报错。 |
| `--alignment-projection-dim` | post-hidden 对齐空间宽度。 | 正整数 | 当前 512。 |
| `--alignment-projection-layers` | 每侧 projection head 层数。 | 正整数 | 仅 projection 档生效；当前 1，即 `LayerNorm + Linear`。 |
| `--alignment-projection-shared` / `--no-alignment-projection-shared` | student/teacher 是否共享 projection 参数。 | 布尔开关 | 仅 projection 档生效；默认 false。 |
| `--wandb-enabled` / `--no-wandb-enabled` | 是否启用 W&B。 | 布尔开关 | 默认读取 `wandb.enabled`。 |
| `--wandb-mode` | W&B 运行模式。 | `online`、`offline`、`disabled` | `online`：上传到云端；`offline`：本地缓存；`disabled`：禁用。 |
| `--wandb-log-model` / `--no-wandb-log-model` | 是否上传 patch AE/alignment checkpoint artifact。 | 布尔开关 | 默认读取 `wandb.log_model`。 |

`patch_alignment.patch_encoder` 是新建 patch AE 的模型配置，只有 `encoder_source: patch_ae_config` 时使用。当前 `field_sampling_mode: single` 下仍是单通道 patch AE，即使 `fields` 写了多个 HDF5 key：

```yaml
patch_encoder:
  model:
    input_size: [16, 16]
    channel_multipliers: []
    latent_dim: 8
    latent_grid: [16, 16]
    preserve_input_channels: true
```

空 `channel_multipliers` 表示不下采样。`preserve_input_channels` 只允许在 latent grid 与输入网格相同时启用，并把原输入通道原样放在 latent 的最前面。

这个脚本不是 QA 训练。它的目标是验证：tensor path 能否学到和“LLM 直接阅读文本形式 patch”一致的中间表示。若这里的 retrieval accuracy 训练不上去，说明 text teacher 表示或 tensor adapter 结构仍有问题；若训练有效，再把这个 adapter 初始化迁移到后续 readout QA。

### 3.10 16x16 Patch QA 迁移

第一阶段 patch alignment 完成后，用 `scripts/build_tensor_patch_qa.py` 从同一 PDEBench HDF5 裁剪单字段 `16x16` patch，同时生成 QA JSONL 和 `[8,16,16]` latent cache。脚本直接加载 `alignment_best.pt` 中的 patch AE 及其 normalization config，因此 latent 编码的是当前 per-patch z-score tensor，与第一阶段 encoder 输入一致。QA 仍从 raw patch 计算标准答案和可逆变换所需的 mean/scale。alignment projector/whitening 不进入下游；空间 adapter 的 256 个 soft tokens 保持 Qwen 原生 embedding 维度。

当前任务使用固定、清晰的自然语言问题：

| `task_type` | 目标 |
|---|---|
| `normalized_point_value` | 直接从 soft tokens 读取指定位置的 z-score，选择最接近的数值。 |
| `raw_point_value_with_stats` | 从 soft tokens 读取 z，再按题面 `x = mean + scale * z` 恢复 raw 点值。 |
| `point_compare` | 比较 patch 内两个 z-score；标准化保持 raw 值顺序。 |
| `region_mean_compare` | 比较两个局部区域的 z-score 均值；标准化保持 raw 均值顺序。 |
| `extreme_quadrant` | 定位 z-score 最大值或最小值所在象限；位置与 raw patch 相同。 |

先在配置中填写最新 checkpoint：

```yaml
patch_qa:
  alignment_checkpoint: /data/wyx/tensor_llm_outputs/runs/CHANGE_ME/alignment_best.pt
  # 默认拒绝使用未在第一阶段出现过的字段；只在明确的跨字段迁移实验中设为 true。
  allow_unseen_alignment_fields: false

adapter:
  architecture: residual_question_adapter
  init_checkpoint: /data/wyx/tensor_llm_outputs/runs/CHANGE_ME/alignment_best.pt
```

生成 QA 和 latent：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/build_tensor_patch_qa.py \
  --config configs/tensor_llm_adapter_pipeline.yaml
```

每条题目的坐标、区域和数值选项由 `seed + patch_id + variant` 独立确定。自然语言坐标统一为与第一阶段 probe 相同的 1-based `1..16`，内部 oracle/张量索引保持 0-based 并由 metadata 明确区分。数值选项生成后会确定性随机打乱显示顺序，不再固定为 `A < B < C < D`；正确 label 随数值一起移动。默认使用 16,384 个 train patches，并为同一个 train tensor 的每类操作生成三个独立自然语言问题；val 为两个，test 保持一个固定问题。这样每个 epoch 有 `16,384 x 5 x 3 = 245,760` 条训练记录。更换 Stage-1 checkpoint 后必须使用新的 `qa_dir`/`latent_dir`，或显式 `overwrite: true` 重新编码；生成器会拒绝静默复用来源 checkpoint 不同的 latent。

输出：

```text
patch_qa.qa_dir/train.jsonl
patch_qa.qa_dir/val.jsonl
patch_qa.qa_dir/test.jsonl
patch_qa.qa_dir/metadata.json
patch_qa.latent_dir/<patch_id>.pt
```

使用 alignment adapter 初始化下游 QA：

```bash
# 先做零成本语法检查，成功时不输出任何内容。
python -m py_compile \
  scripts/build_tensor_patch_qa.py \
  scripts/train_tensor_llm_adapter.py \
  scripts/diagnose_tensor_llm_adapter.py \
  scripts/inspect_tensor_llm_readout.py

# 先生成多 query QA；使用独立 spatial256 目录，不能复用旧 4x4 latent。
CUDA_VISIBLE_DEVICES=1 python scripts/build_tensor_patch_qa.py \
  --config configs/tensor_llm_adapter_pipeline.yaml

# 正式四卡训练。每个进程各持有一份 frozen Qwen，只同步 adapter 梯度。
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3,4,5 \
torchrun --standalone --nproc_per_node=4 scripts/train_tensor_llm_adapter.py \
  --config configs/tensor_llm_adapter_pipeline.yaml
```

`llm_training.batch_size` 是每张卡的 batch。当前 `batch_size: 3`、
`gradient_accumulation_steps: 1` 在四卡下得到 effective batch 12。分布式 sampler 保留完整的
same-tensor/same-task 问题组；验证和测试精确分片且不补齐重复记录。checkpoint、诊断、终端输出和
W&B 只由 rank 0 写入。单卡回退时使用原来的 Python 命令，并加
`--gradient-accumulation-steps 5` 恢复 effective batch 15。

Qwen2.5-32B 的参数虽然冻结，答案损失仍要穿过全部 decoder layers 回传到 tensor prefix。Stage 2
因此默认启用 non-reentrant `llm_gradient_checkpointing`。启动日志会在模型加载前报告可见卡及空闲
显存，低于总容量 95% 时标记
`warning=visible_gpu_not_empty`。训练数据的 shuffled-negative 索引按 sample bucket 线性构造，latent
使用有界 CPU LRU cache，避免大数据集启动时的近二次扫描和同一 `.pt` 文件反复读取。

当前正式任务的 choices（A/B/C/D 或 A/B）都是单 token。`llm_training.choice_scoring_mode: auto`
会在正确答案 teacher-forcing 的同一次 Qwen forward 中，从 `Answer:` 位置的 restricted-label logits
计算 choice CE；普通 answer CE 仍计算正确 label 和 EOS，因此没有移除格式监督。ranking 与
swapped-question 也使用同一 restricted-choice CE，不把 EOS 或完整词表概率混入 grounding margin。
负向序列会合并并复用同一 batch 的第 2/6 层问题上下文；超过
`train_grounding_batch_size: 8` 时只对该合并 batch 分块，所有 margin 项仍会参与同一次 loss。
若未来使用多 token choices，代码会自动回退到原始 sequence-likelihood scorer；也可显式设置
`choice_scoring_mode: sequence` 做严格旧实现复现。`eval_batch_size: 8` 只影响无梯度评估，不改变指标。
`batch_size: 8` 在当前完整三问题分组下会被 sampler
装入两个完整 group，实际 batch 为 6；`summary.json` 的 `grouped_batch_size_epoch_zero` 会记录真实
范围。主 batch 从 3 提至 8 会让每 epoch 的优化器更新数减半，因此它是训练日程变更，不作为默认
速度优化；应先用短 smoke test 比较稳定的 `samples/s` 和梯度，再决定是否调整正式训练日程。

#### Stage 2 空闲 GPU sweep 调度

`scripts/run_tensor_llm_adapter_sweep.py` 扫描 `nvidia-smi` 报告的全部 GPU。每当一张卡的
空闲显存达到配置阈值，就在该卡启动下一个编号实验；调度器内部排除已经分配给其他 sweep 的卡。
默认一张 A800 对应一个独立实验，并用梯度累积 5 保持 effective batch 15。这样比把一个实验拆到
多卡更适合并行比较参数。

默认清单位于 `configs/tensor_llm_adapter_stage2_sweep.yaml`，只做单因素修改：

| 编号 | 相对 S001 的修改 | 目的 |
|---|---|---|
| `S001` | 无 | 当前配置基线。 |
| `S002` | `lr: 5e-5` | 更保守地训练自然语言读取器。 |
| `S003` | `swapped_question_loss_weight: 0.2` | 加强自然语言问题 grounding。 |

在 `tmux` 中启动：

```bash
python scripts/run_tensor_llm_adapter_sweep.py \
  --config configs/tensor_llm_adapter_stage2_sweep.yaml
```

交互终端只有一条动态等待状态，并在开始和结束时输出编号：

```text
waiting elapsed=02:13:45 pending=4 running=1
S001 START time=... gpu=6 pid=...
S001 END time=... duration=... exit=0 run_dir=/data/.../timestamp_..._S001
```

每次调度会创建独立 session 目录。`sweep_state.json` 记录所有编号的 PID、GPU、开始/结束时间、
退出码和实际训练目录；`events.jsonl` 记录事件；`logs/S001.log` 等文件保存各训练进程的完整输出。
重定向调度器输出时，waiting 默认每五分钟写一行，避免产生数千行日志。`Ctrl+C` 会终止本 session
启动的所有训练进程并将其标记为 interrupted。

显存扫描不是系统级原子预留；其他用户仍可能在检查和模型加载之间抢占同一张卡。正式共享集群应优先
使用 Slurm。`gpus_per_run` 可改为多卡，但此时必须同时重新计算
`gradient_accumulation_steps`，确保不同编号的 effective batch 可比。

长时间训练前先检查启动摘要包含配置中的 loss weights 和 `checkpoint_load=stage1_frozen_backbone_question_residual`。这表示固定 reference 与冻结的 conditioned backbone 都由第一阶段空间 adapter 初始化，而不是复用旧 downstream checkpoint。

`adapter.architecture: alignment_adapter` 会按 checkpoint 的 adapter 类型、网格、层数和 hidden size 重建第一阶段结构，并以 `strict=True` 加载 `adapter_state_dict`。旧 `alignment_qformer` 名称仍用于旧 checkpoint。

局部读取增强模式不重跑第一阶段，也不使用 AE decoder。`residual_question_adapter` 从 `alignment_best.pt` 严格载入 256-token 空间 adapter，并复制为两份：reference branch 永久冻结；conditioned branch 的逐位置 latent projection、固定二维编码、空间 blocks、局部残差和输出映射同样冻结。每个空间 block 前新增 trainable text cross-attention，使每个位置 token 根据完整自然语言问题选择证据；不读取 task id、正则表达式坐标、oracle 或答案。新增 attention 的输出投影为零初始化，因此 conditioned branch 在第一个优化器更新前逐元素复现 Stage 1。

问题 token 由 frozen Qwen 同一次 early-exit 前向提取第 2 层和第 6 层完整序列。第 2 层偏重数字、坐标和词法细节，第 6 层提供上下文语义；两层经过各自 `LayerNorm + Linear` 后以 learned softmax 权重融合。最终输出保持第一阶段相同的 256 个位置：

```text
global = frozen_stage1_spatial_adapter(latent)
conditioned = frozen_stage1_backbone_with_trainable_text_attention(latent, full_question_tokens)
soft_prompt = global + (conditioned - global)
```

```yaml
adapter:
  architecture: residual_question_adapter
  init_checkpoint: /data/wyx/tensor_llm_outputs/runs/CHANGE_ME/alignment_best.pt
  local_soft_prompt_tokens: 256
  local_adapter_layers: 2
  local_question_input_mode: contextual_tokens
  local_context_layer: 6
  local_context_layers: [2, 6]
  local_fusion_mode: residual_spatial_transformer
  local_text_encoder_layers: 0
  structured_query_conditioning: false
  local_text_gate_init: 1.0
  local_text_gate_trainable: false
  local_gate_init: 1.0
  local_residual_gate_trainable: false
  zero_init_local_text_attention: true
  freeze_conditioned_backbone: true
  freeze_global_adapter: true
  global_unfreeze_epoch: 0
  global_lr: 1.0e-5
  global_prompt_dropout: 0.0

llm_training:
  epochs: 2
  batch_size: 3
  lr_scheduler: cosine
  warmup_ratio: 0.03
  group_questions_by_state: true
  questions_per_state_group: 3
  weight_decay: 1.0e-4
  ranking_loss_weight: 0.1
  ranking_loss_negative: global_only
  swapped_question_loss_weight: 0.1
  swapped_question_loss_margin: 0.1
  swapped_question_require_different_answer: true
  swapped_question_max_records: 8
  checkpoint_metric: macro_latent_gain
```

最终前缀为 `[256 residual-conditioned spatial tokens][完整QA prompt]`，不会额外拼接第二组 256 token。同一问题的所有候选答案共享一次 soft prompt 计算。训练 batch 按 `state_ref + task_type + field` 组织同一 tensor 的不同问题；swapped-question loss 互换 conditioned prompt，约束自然语言条件敏感性。QA 文件中的 `oracle` 在 dataset 载入时会被删除。

每道题的主 prompt 会按该题实际 choices 明确写出输出契约，例如 `Required output: exactly one of A, B, C, D`，并要求不输出解释、标点或其他文字。训练和正式准确率仍使用每个合法标签的候选 NLL，避免自由生成的格式噪声改变优化目标；自由生成只在内置诊断中运行，用于区分“语义上选对但格式不合规”和“答案本身错误”。

当前正式实验不复用旧 downstream hybrid checkpoint，新架构会主动拒绝这样的路径。启动后可在 `run_summary.json` 核对 `question_input_mode=contextual_tokens`、`local_context_layers=[2,6]`、`local_fusion_mode=residual_spatial_transformer` 和 `stage1_frozen_backbone_question_residual` load report。程序还会对一条真实 latent 做训练前恒等检查；`stage1_identity_error` 应接近 `0`，超出数值容差会直接停止，不会进入长训练。

`freeze_global_adapter: true` 表示 reference global branch 始终冻结；`freeze_conditioned_backbone: true` 也禁止其副本形成无条件捷径。local 参数组只包含问题层融合、文本投影和 cross-attention。两个固定 gate 均为 1，问题路径不会被双重缩小；`global_prompt_dropout` 关闭，问题敏感性由自然语言 cross-attention 与 swapped-question loss 约束。

`residual_question_adapter` 必须提供第一阶段 checkpoint，不能传 `--adapter-init-checkpoint none`。

训练器默认先在 512 条验证记录上运行训练前评估，再开始微调。每个 epoch 只评估 `correct` 和 `shuffled`；最终 best checkpoint 再完整评估 `zero_latent`、`no_latent` 和 `shuffled_stats`。patch QA 的 shuffled baseline 优先随机换成同字段、同任务、但不同 `sample_index` 的 patch，避免把同一 PDE 轨迹的相邻时间步误当成强负样本；只有 sanity 数据没有第二个 sample 时才退回不同 state。`shuffled_stats` 会同时替换自然语言和记录元数据中的 mean/scale。

启动前会生成 `data_audit.json`，检查 train/val/test 的 PDEBench sample 是否交叉、task/field/答案标签覆盖是否一致、QA id 是否重复、latent 文件是否存在、答案是否属于 choices、数值选项显示后是否重复。`prompt_audit.json` 记录每个 split/task 的 token 长度；正式配置禁止自然语言指令或 query 被静默截断。正式运行默认 `require_disjoint_splits: true`，并拒绝开启 `structured_query_conditioning`；overfit sanity wrapper 会显式标记为 `sanity_only`。

控制台默认每个 epoch 只输出一行。W&B 的 step 级日志只保留 loss、choice accuracy、ranking/swap loss 与 LR；epoch 级保留 overall、各任务 accuracy 和主要 latent gain。完整字段、task-field、全部 baseline 和 hidden-state 明细继续写入本地 `metrics_latest.json`、`test_metrics.json` 与 `diagnostics/`。需要恢复旧的全量 W&B 展开时设置 `wandb.detailed_metrics: true`。真实开始时间、结束时间、时区、耗时和 `completed/failed/interrupted` 状态写入 `run_timing.json`，并同步到 `run_summary.json`。

### 3.11 Adapter 诊断输出

正式训练已经内置轻量诊断，不需要另跑实验。训练前和每个 epoch 后默认从验证集为每种 task 固定取 4 条记录，对比正确 latent、同字段错配 latent，以及同一 tensor 上的另一个同任务问题，并保存：

- `diagnostics/epoch_XXXX_summary.json`：主 prompt、local 完整条件 prompt、候选 NLL 预测、自由生成原文、解析标签、严格格式是否合法、正确答案 margin、soft prompt 差异，以及指定 LLM 层的 hidden-state cosine/relative L2；
- `diagnostics/epoch_XXXX_states.pt`：latent、mask、global prompt、conditioned residual、combined prompt、文本/latent 投影、query self-attention/cross-attention 权重和各指定 LLM 层 hidden state 的原始张量快照。

默认层为 `[0,2,8,14,-1]`。诊断同时记录跨任务 `question_sensitivity`、按任务拆分的 correct/shuffled accuracy 与 answer margin、same-tensor/same-task residual 敏感度与 swapped margin、residual/global RMS、固定 residual/text gate、conditioned-backbone 与 question-reader 的 trainable parameter 数、layer 2/6 learned fusion 权重、256 个空间输出的平均非对角余弦相似度，以及各 block 对问题 token 和 `16x16` latent cell 的 top attention。它还把 conditioned prompt 换成另一个问题的结果重新评分；只有正确问题比 swapped question 获得更高答案 margin，才说明问题差异与任务输出相关。最终测试额外报告 `local_only`（仅 residual）和 `global_only`（固定 stage-1 prompt）。摘要 JSON 会先于较大的 states PT 写入；诊断与 checkpoint 都先写临时文件再原子替换正式文件名，避免下载到仍在写入的 0 字节或半截文件。

W&B 和 `metrics_latest.json` 还会记录 `train_local_grad_norm`、`train_global_grad_norm`、`train_total_grad_norm` 与 local gate。若 hidden state 已有差异但 local gradient 长期接近 0，应先排查门控、mask 或 loss 路径，而不是继续增加 epoch。

`scripts/diagnose_tensor_llm_adapter.py` 仍保留为手动深度扫描工具，用于检查更多记录。它现在可以从 checkpoint 重建 legacy、alignment Q-Former 和 hybrid adapter。

命令：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/diagnose_tensor_llm_adapter.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --checkpoint /data/wyx/tensor_llm_outputs/runs/CHANGE_ME/adapter_best.pt \
  --split train \
  --records 64 \
  --output /data/wyx/tensor_llm_outputs/runs/CHANGE_ME/diagnostics_train.jsonl
```

常用参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--config` | Adapter pipeline 配置路径。 | 路径 | 默认 `configs/tensor_llm_adapter_pipeline.yaml`。 |
| `--checkpoint` | adapter checkpoint 路径。 | `adapter_best.pt`、`adapter_last.pt` | 必填。 |
| `--split` | 诊断的 QA split。 | split 名 | 默认 `train`。 |
| `--records` | 诊断记录数。 | 正整数 | 建议先用 32 或 64。 |
| `--start-index` | 从 split 的第几个 record 开始。 | 非负整数 | 用于换一批样本看。 |
| `--output` | JSONL 输出路径。 | 路径、`null` | `null`：写到 checkpoint 同目录。 |
| `--hidden-layers` | 要汇总的 LLM hidden state 层号。 | 逗号分隔整数 | 例如 `0,-1` 表示 embedding 后和最后一层。 |
| `--max-choice-records` | 保存 hidden state 摘要的最大记录数。 | 非负整数 | hidden state 计算较慢，可小于 `--records`。 |

每行 JSONL 主要包含：

| 字段 | 作用 |
|---|---|
| `record`、`shuffled_record` | 检查正确样本和随机错配样本的 `state_ref/sample_index/time_index/query/answer/oracle`。 |
| `latent` | 正确 latent 与 shuffled latent 的统计、L2 距离、cosine similarity。 |
| `soft_prompt` | 正确 latent、shuffled latent、零 soft prompt 的统计与差异。 |
| `nll` | 每个候选答案在 correct/shuffled/no_latent/zero_latent 下的 NLL 和正确答案 margin。 |
| `hidden_states` | 指定 LLM 层的全序列、soft token 区、text token 区 hidden state 摘要。 |

结果解读：

| 现象 | 说明 |
|---|---|
| shuffled 的 `same_sample_as_shuffled` 大量为 `true` 且 `delta_time_to_shuffled` 很小 | 负样本仍可能太接近。 |
| `latent.correct_vs_shuffled` 差异大，但 `soft_prompt.correct_vs_shuffled` 差异小 | adapter 没有把 latent 差异传到 soft prompt。 |
| soft prompt 差异大，但 NLL margin 接近 0 | LLM 没有有效使用 soft prompt。 |
| `answer_margin_shuffled_minus_correct` 多数为正 | 正确 latent 正在降低正确答案 NLL，是有效信号。 |

### 3.12 配置模型对话 Smoke Test

`tests/chat_with_config_model.py` 会读取 `configs/tensor_llm_adapter_pipeline.yaml` 中的 `model.local_dir` 或 `model.name_or_path`，并使用同一份配置里的 `storage.hf_home`、`model.torch_dtype`、`model.trust_remote_code` 加载模型。这个脚本是手动检查下载模型是否能正常加载和生成文本，不属于自动单元测试。

单轮对话：

```bash
CUDA_VISIBLE_DEVICES=1 python tests/chat_with_config_model.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --prompt "用一句话解释什么是 soft prompt。"
```

交互式对话：

```bash
CUDA_VISIBLE_DEVICES=1 python tests/chat_with_config_model.py \
  --config configs/tensor_llm_adapter_pipeline.yaml
```

常用参数：

| 参数 | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `--config` | Adapter pipeline 配置路径。 | 路径 | 默认 `configs/tensor_llm_adapter_pipeline.yaml`。 |
| `--prompt` | 单轮用户输入。 | 字符串、`null` | 不传则进入交互式模式。 |
| `--system` | system prompt。 | 字符串 | 默认 `You are a helpful assistant.` |
| `--device` | 推理设备。 | `auto`、`cpu`、`cuda`、`cuda:N` | 不传则读 `runtime.device`。 |
| `--max-new-tokens` | 最大生成 token 数。 | 正整数 | 默认 128。 |
| `--temperature` | 采样温度。 | 正数 | 越低越确定。 |
| `--top-p` | nucleus sampling 阈值。 | 0 到 1 | - |
| `--do-sample` / `--no-do-sample` | 是否采样生成。 | 布尔开关 | `--no-do-sample`：贪心/确定性生成。 |

### 3.13 模型选择建议

| 阶段 | 模型 | 说明 |
|---|---|---|
| 快速 debug | `Qwen/Qwen2.5-0.5B-Instruct` | 速度快，能力较弱。 |
| 推荐 pilot | `Qwen/Qwen2.5-1.5B-Instruct` | 中英能力、速度、显存占用较平衡。 |
| 正式结果 | `Qwen/Qwen2.5-7B-Instruct` | A800 80GB 可承受，结果更有说服力。 |

当前 QA 是英文 DSL，第一阶段不强依赖中文能力；后续如果要中文提问，优先选 Qwen 系列。

### 3.14 评估逻辑

训练脚本默认做 choice likelihood 评估，而不是只看 loss。

| baseline | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `correct` | 使用当前样本的正确 tensor latent。 | baseline 名 | 目标能力。 |
| `no_latent` | 使用全 0 soft prompt。 | baseline 名 | 检查 LLM 是否只靠文本先验答题。 |
| `zero_latent` | latent 置零，但仍经过 adapter 和 query 条件。 | baseline 名 | 新结构下更干净地检查“没有 tensor 内容、只有 query 条件”能答到什么程度。 |
| `shuffled` | 使用其他 state 的 latent。 | baseline 名 | 检查 latent 是否绑定当前 tensor。 |
| `random` | 使用随机 latent。 | baseline 名 | 可选消融。 |

只有当 `correct` 明显优于 `zero_latent/shuffled` 时，才说明 adapter 可能学到了读取 tensor latent 的能力。正式配置关闭 `structured_query_conditioning`；该旧开关只保留给显式标记为 sanity-only 的兼容调试。

### 3.15 LLM Readout Inspection

这个脚本用于回答一个更具体的问题：在答案位置，冻结 LLM 到底更偏向哪些候选项。它不会解释 LLM 的内部语义，只输出可观测的候选答案 NLL、归一化概率、rank，以及 correct latent 相比 `zero_latent/shuffled/no_latent` 对正确答案概率的改变。

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/inspect_tensor_llm_readout.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --checkpoint /data/wyx/tensor_llm_outputs/runs/<RUN>/adapter_best.pt \
  --split train \
  --records 64
```

输出在 checkpoint 同目录下：

```text
adapter_best_readout_inspection_train.jsonl
adapter_best_readout_inspection_train.summary.json
```

重点看这些字段：

| 字段 | 说明 |
|---|---|
| `modes.correct.choices` | 正确 latent 下每个候选答案的 NLL、概率、rank。 |
| `modes.zero_latent.choices` | 无 tensor 内容但保留 adapter/query 条件时的候选分布。 |
| `modes.shuffled.choices` | 错配 tensor state 时的候选分布。 |
| `deltas_from_correct.*.answer_prob_correct_minus_mode` | 正确 latent 是否提高了正确答案概率。 |
| `deltas_from_correct.*.answer_nll_mode_minus_correct` | 正确 latent 是否降低了正确答案 NLL。 |
| `soft_prompt.correct_vs_*` | 不同 baseline 的 soft prompt 距离和 cosine。 |
