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
| `structured_query_conditioning` | 是否把 query 中的结构化坐标显式编码进 adapter query。 | `true`、`false` | `true`：解析 row/col、A/B 点位、patch 范围、任务类型和选项数；不使用 oracle 数值答案。 |
| `soft_prompt_scale` | soft prompt 输出尺度限制。 | 非负数 | `0.05`：`tanh` 后限制每维约在 `[-0.05,0.05]`，使 soft prompt token 范数接近普通 token embedding；`0`：关闭尺度限制，保留线性输出。 |

当前 adapter 的信息流是单向的：文本 prompt 的冻结 embedding 和结构化 query 特征只用于生成 query 条件，latent token 仍然是 cross-attention 的唯一 key/value 来源，并且文本 embedding 在进入条件分支前会 detach。因此文本可以告诉 adapter “应该读哪里/读什么”，但训练梯度不会更新 LLM，也不会把 tensor latent 和文本 embedding 混成同一个可写空间。

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
| `ranking_loss_margin` | ranking loss 的最小 NLL 间隔。 | 非负数 | 希望 `NLL(shuffled)-NLL(correct)` 至少达到该值。 |
| `ranking_loss_negative` | ranking loss 使用的负样本类型。 | `shuffled`、`random`、`no_latent`、`zero_latent` | `shuffled`：随机错配 latent；`random`：随机噪声；`no_latent`：零 soft prompt；`zero_latent`：latent 置零但仍经过 adapter 和 query 条件。 |
| `prompt_template` | Adapter 训练用文本 prompt 模板。 | `task_specific`、`generic` | `task_specific`：按 `task_type` 写明读数/比较/bin 规则；`generic`：旧版通用提示。 |
| `max_prompt_tokens` | 文本 prompt 最大 token 数。 | 正整数 | 超出会左截断。 |
| `max_target_tokens` | 答案最大 token 数。 | 正整数 | - |
| `append_eos` | target 后是否追加 EOS。 | `true`、`false` | - |
| `eval_baselines` | 评估 baseline 列表。 | `correct`、`no_latent`、`zero_latent`、`shuffled`、`random` | `correct`：正确 latent；`no_latent`：零 soft prompt；`zero_latent`：latent 置零但保留 adapter 和 query 条件；`shuffled`：固定 seed 的全局随机错配 latent；`random`：随机噪声 latent。 |
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
| `--ranking-loss-margin` | ranking loss 的最小 NLL 间隔。 | 非负数 | 要求正确 latent 的正确答案 NLL 比负样本更低。 |
| `--ranking-loss-negative` | ranking loss 负样本类型。 | `shuffled`、`random`、`no_latent`、`zero_latent` | `shuffled`：错配 tensor state；`zero_latent`：更严格地检查是否超过 query 条件先验。 |
| `--soft-prompt-tokens` | soft prompt token 数。 | 正整数 | - |
| `--adapter-dim` | adapter 内部维度。 | 正整数 | 必须能被 heads 整除。 |
| `--adapter-layers` | adapter 层数。 | 正整数 | - |
| `--adapter-heads` | adapter heads。 | 正整数 | - |
| `--dropout` | adapter dropout。 | 0 到 1 | - |
| `--latent-pos-encoding` | latent 位置编码方式。 | `grid`、`none` | `grid`：给二维 latent token 加坐标投影；`none`：不加位置。 |
| `--question-conditioning` / `--no-question-conditioning` | 是否用文本问题条件化 adapter query。 | 布尔开关 | 开启后同一 tensor 的 soft prompt 会随问题变化。 |
| `--question-condition-gate-init` | 文本问题条件分支的初始门控强度。 | 浮点数 | `1.0`：默认开启；`0.0`：初始近似关闭。 |
| `--structured-query-conditioning` / `--no-structured-query-conditioning` | 是否使用结构化 query 条件。 | 布尔开关 | 开启后从 query 字符串解析坐标和任务类型，不读取 oracle 数值。 |
| `--soft-prompt-scale` | soft prompt 输出尺度限制。 | 非负数 | `0.05`：推荐默认值；`0`：关闭限制。 |
| `--prompt-template` | 文本 prompt 模板。 | `task_specific`、`generic` | `task_specific`：按任务写规则；`generic`：旧版通用提示。 |
| `--max-prompt-tokens` | prompt 最大 token 数。 | 正整数 | 超出会左截断。 |
| `--max-target-tokens` | target 最大 token 数。 | 正整数 | - |
| `--append-eos` / `--no-append-eos` | target 后是否追加 EOS。 | 布尔开关 | - |
| `--eval-baselines` | 评估 baseline 列表。 | 逗号分隔字符串 | 可包含 `correct,no_latent,zero_latent,shuffled,random`；`zero_latent` 是新结构下的重要对照。 |
| `--choice-score` | 候选答案 NLL 计分方式。 | `mean`、`sum` | `mean`：按 token 平均；`sum`：累加。 |
| `--log-interval` | 训练日志间隔。 | 正整数 | - |
| `--wandb-enabled` / `--no-wandb-enabled` | 是否启用 W&B。 | 布尔开关 | - |
| `--wandb-api-key` | W&B API key。 | 字符串、`null` | 不建议写进命令历史；优先用环境变量。 |
| `--wandb-project` | W&B project 名称。 | 字符串 | - |
| `--wandb-entity` | W&B entity/team。 | 字符串、`null` | - |
| `--wandb-group` | W&B run group。 | 字符串、`null` | - |
| `--wandb-tags` | W&B tags。 | 逗号分隔字符串 | 例如 `adapter,tensor-llm,vx`。 |
| `--wandb-mode` | W&B 模式。 | `online`、`offline`、`disabled` | - |
| `--wandb-log-model` / `--no-wandb-log-model` | 是否上传 adapter checkpoint artifact。 | 布尔开关 | - |

训练目标默认包含普通 CE、候选项分类 CE 和 ranking 项：

```text
loss = ce_loss_weight * token_CE(answer | correct_latent)
     + choice_ce_loss_weight * CE(softmax(-NLL(candidate_i | correct_latent)), correct_candidate)
     + ranking_loss_weight * max(0, margin + NLL(answer | correct_latent) - NLL(answer | negative_latent))
```

`choice_01_loss = 1 - choice_accuracy` 会被记录到日志中，但它是硬 argmax 后的不可导指标，不参与反向传播。`choice_ce_loss` 是它的可导替代项，更接近最终 choice accuracy；ranking 项用于惩罚“错配 latent 也同样支持正确答案”的情况。做消融时可以设置 `ranking_loss_weight: 0` 或命令行传 `--ranking-loss-weight 0`。

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
  patch -> patch AE encoder -> latent tokens -> Q-Former adapter -> soft prompt tokens
        -> frozen LLM -> middle-layer student hidden state

text path:
  同一份 normalized/resized patch 序列化为文本 -> frozen LLM -> middle-layer teacher hidden state
```

当前默认对齐位置是：**冻结 LLM 中层、最后一个非 padding token 的 hidden state**。teacher branch 的文本和 student branch 的短 anchor prompt 都以 `Representation:` 结尾；tokenizer 会把同一个 batch 中较短文本补 padding，最后一个非 padding token 就是每条真实输入文本的最后一个 token。Qwen2.5-1.5B 有 28 个 decoder layers，当前配置使用 `teacher_layer: 8`，避免直接对齐最后层的 next-token 决策状态。这个值不是理论常数，后续可以系统比较 `4/8/14/20/-1`。

注意：Q-Former 不再直接预测某一层 hidden vector。它输出 soft prompt tokens，并把这些 tokens 放到 frozen LLM 输入 embedding 前面；然后从同一个 frozen LLM 的 `teacher_layer` 取 student hidden state，与 text teacher hidden state 对齐。这样后续迁移到 soft prompt QA 时不会出现“训练时对齐中层、推理时却塞到输入层”的层级错配。

Prompt 设置：

Teacher branch 的默认 `compact` prompt 包含任务说明、字段名、patch 尺寸、完整数值矩阵和 anchor。默认 `teacher_text_source: normalized`，因此这里的数值矩阵来自 **AE 实际输入的 normalized/resized patch**，不是 HDF5 原始值；这样 teacher branch 和 tensor path 才表示同一个对象。

```text
Represent this PDE tensor patch for numeric reasoning.
fields=Vx patch_size=16
Vx=[[...]; [...]; ...]
Representation:
```

它不再包含 `sample_index`、`time_index`、`top_left`，因为 tensor path 只能看到 patch 数值，看不到这些采样元数据。旧格式可通过 `text_prompt_template: compact_with_metadata` 复现，但只建议做消融。若设置 `teacher_text_source: raw`，teacher branch 会读取原始 HDF5 数值；这会和默认 zscore AE 输入产生信息不一致，只建议作为 ablation。

Student branch 的短 anchor prompt 不包含数值矩阵：

```text
Represent this PDE tensor patch for numeric reasoning.
fields=Vx patch_size=16
Representation:
```

实际输入 LLM 的形式是：

```text
[soft prompt tokens from tensor] + student anchor prompt
```

因此 student hidden 必须从 soft prompt 中获得数值信息，而不能从文本里偷看数值。

默认 tensor path 不再把 `16x16` patch resize 到 `512x512`。脚本会按 `patch_alignment.patch_encoder` 构建一个 patch-sized AE：

```text
16x16 patch -> patch AE -> 4x4 latent tokens
```

如果 `encoder_source: checkpoint`，则加载 `compressor_checkpoint`；这适合调用已经训练好的 patch AE，或者临时复用旧的 512x512 compressor。只有后一种情况才应设 `resize_patch_to_compressor_input: true`。

默认 patch size 是 `16x16`。`8x8` 信息量偏少，`32x32` 文本 token 开销明显增大；`16x16` 单字段在 Qwen2.5-1.5B 的上下文内比较适合做第一轮实验。

命令：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_patch_text_alignment.py \
  --config configs/tensor_llm_adapter_pipeline.yaml
```

小规模 smoke test：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_patch_text_alignment.py \
  --config configs/tensor_llm_adapter_pipeline.yaml \
  --train-records 128 \
  --val-records 32 \
  --test-records 32 \
  --epochs 1 \
  --batch-size 4 \
  --run-name tensor_patch_text_alignment_smoke
```

主要 loss：

```text
loss = contrastive_loss_weight * symmetric_InfoNCE(tensor_embedding, text_teacher_embedding)
     + cosine_loss_weight * (1 - cosine_similarity)
     + reconstruction_loss_weight * MSE(patch_AE_reconstruction, normalized_patch)
```

这里的 `tensor_embedding` 实际是 student branch 经过 frozen LLM 后取出的 anchor hidden state。`text_teacher_embedding` 是 teacher branch 读完整数值矩阵文本后的同层 anchor hidden state。默认会先对两个 batch hidden 分别减去 batch mean，再 L2 normalize 后进入 InfoNCE，以减弱 LLM hidden space 的公共方向。当前默认 `cosine_loss_weight: 0.0`，因为 raw cosine 可能只鼓励公共方向接近，却不提高 batch retrieval。`freeze_patch_ae_after_pretrain: true` 时，reconstruction 项只在 AE warmup 阶段训练 encoder；alignment 阶段默认冻结 patch AE，只训练 Q-Former/soft prompt bridge。

脚本会检查 teacher/student tokenization 后的最后 token 附近是否仍包含 `Representation:`。默认 `fail_on_text_anchor_missing: true`，一旦文本过长导致 anchor 被截断，会直接报错，而不是继续训练一个语义位置已经错位的目标。默认 `fail_on_text_max_length_hit: true`，只要序列打满 `max_text_tokens` 也会报错；这比静默截断更严格，若触发应优先增大 `max_text_tokens`、减小 `patch_size` 或降低 `text_decimal_places`。

输出文件：

| 文件 | 说明 |
|---|---|
| `run_summary.json` | patch 大小、字段、encoder 来源、latent grid、LLM hidden size、teacher layer、adapter 参数量。 |
| `metrics_latest.json` | patch AE warmup、每轮 train/val loss、reconstruction loss、i2t/t2i retrieval accuracy。 |
| `test_metrics.json` | 使用 `alignment_best.pt` 的最终 test 指标。 |
| `patch_ae_pretrain_last.pt` | 可选 patch AE reconstruction warmup 后的 checkpoint。 |
| `alignment_best.pt`、`alignment_last.pt` | 对齐 adapter checkpoint；若开放 AE 训练，也会保存 compressor state。 |

W&B 曲线：

| 曲线名 | 说明 |
|---|---|
| `patch_ae_pretrain_step/reconstruction_loss` | patch AE 预训练阶段的 step-level 平均重建误差。 |
| `patch_ae_pretrain_step/current_reconstruction_loss` | patch AE 当前 batch 重建误差。 |
| `patch_ae_pretrain/reconstruction_loss` | patch AE 每个预训练 epoch 的平均重建误差。 |
| `patch_ae_pretrain/val_reconstruction_loss` | patch AE 每个预训练 epoch 后的验证集重建误差，用于观察过拟合。 |
| `train/reconstruction_loss` | alignment 阶段训练集重建误差；`train_patch_ae: true` 时可观察 AE 是否继续变化。 |
| `val/reconstruction_loss` | alignment 阶段验证集重建误差。 |
| `train/contrastive_loss`、`val/contrastive_loss` | tensor embedding 与 text teacher embedding 的对比学习 loss。 |
| `train/i2t_accuracy`、`val/i2t_accuracy` | batch 内 tensor-to-text retrieval accuracy。 |
| `train/t2i_accuracy`、`val/t2i_accuracy` | batch 内 text-to-tensor retrieval accuracy。 |
| `train/teacher_anchor_missing_fraction`、`val/teacher_anchor_missing_fraction` | teacher text tokenization 后 anchor 缺失比例；默认应为 0，否则代码会报错。 |
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

已训练 patch AE 可以按路径冻结复用。第一次训练后，优先用 `alignment_best.pt`；如果只想用 reconstruction warmup 后的 AE，则用 `patch_ae_pretrain_last.pt`：

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
| `--fields` | 使用哪些 HDF5 字段。 | 逗号分隔字段 | 当前建议先用单字段 `Vx`，多字段会显著增加文本长度。 |
| `--train-records`、`--val-records`、`--test-records` | 随机采样 patch 数。 | 正整数 | 初始 smoke test 用小值，正式训练可增大。 |
| `--encoder-source` | encoder 从哪里来。 | `patch_ae_config`、`checkpoint` | `patch_ae_config`：按 YAML 构建 patch AE；`checkpoint`：加载 `compressor_checkpoint`。 |
| `--train-patch-ae` / `--no-train-patch-ae` | 是否更新 AE 参数。 | 布尔开关 | 新建 patch AE 建议 `true`；加载已训练 patch AE 做纯 adapter 对齐时可设 `false`。 |
| `--freeze-patch-ae-after-pretrain` / `--no-freeze-patch-ae-after-pretrain` | AE warmup 后 alignment 阶段是否冻结 AE。 | 布尔开关 | 默认 `true`：减少 encoder 记忆训练集 teacher hidden 的风险。 |
| `--patch-ae-pretrain-epochs` | 对齐前 reconstruction-only warmup 轮数。 | 非负整数 | `0`：跳过；大于 0：先训练 patch AE 重建。 |
| `--compressor-checkpoint` | 已训练 encoder checkpoint。 | 路径、`null` | 仅 `encoder_source: checkpoint` 必需。 |
| `--resize-patch-to-compressor-input` / `--no-resize-patch-to-compressor-input` | 是否把 patch resize 到 encoder `input_size` 后再编码。 | 布尔开关 | patch AE 应设 `false`；旧 512x512 AE 才设 `true`。 |
| `--adapter-type` | tensor path 的对齐 adapter。 | `qformer`、`pooled_mlp` | `qformer`：learnable queries cross-attend latent tokens 并输出 soft prompt tokens；`pooled_mlp`：mean/std pooling 旧实现，仅作 ablation。 |
| `--query-tokens` | Q-Former learnable query 数。 | 正整数 | query 越多，tensor path 容量越大，显存也更高。 |
| `--adapter-layers` | Q-Former cross-attention block 数。 | 正整数 | 默认 2。 |
| `--adapter-heads` | Q-Former attention heads。 | 正整数 | 必须整除 `adapter_dim`。 |
| `--soft-prompt-scale` | soft prompt 输出尺度限制。 | 非负数 | `0.05`：`tanh` 后限制每维约在 `[-0.05,0.05]`；`0`：关闭限制。 |
| `--reconstruction-loss-weight` | patch AE 重建 MSE 权重。 | 非负数 | 只在 `train_patch_ae: true` 时影响训练。 |
| `--teacher-text-source` | teacher branch 序列化哪一种 patch。 | `normalized`、`raw` | `normalized`：使用 AE 实际输入，默认；`raw`：使用 HDF5 原始值，只建议消融。 |
| `--center-embeddings` / `--no-center-embeddings` | InfoNCE 前是否分别对 student/teacher batch hidden 减均值。 | 布尔开关 | 默认开启，用于减弱 LLM hidden 公共方向。 |
| `--fail-on-text-anchor-missing` / `--no-fail-on-text-anchor-missing` | tokenization 后 anchor 缺失时是否直接报错。 | 布尔开关 | 默认开启；关闭后只记录缺失比例。 |
| `--fail-on-text-max-length-hit` / `--no-fail-on-text-max-length-hit` | tokenized 文本打满 `max_text_tokens` 时是否直接报错。 | 布尔开关 | 默认开启；关闭后只记录 `*_max_length_hit_fraction`。 |
| `--text-prompt-template` | teacher branch 的数值矩阵 prompt 模板。 | `compact`、`compact_with_metadata`、`plain` | `compact`：不含不可见元数据；`compact_with_metadata`：旧格式，含 sample/time/top_left；`plain`：仅矩阵文本。 |
| `--text-decimal-places` | 文本化 tensor 数值保留小数位。 | 非负整数 | 默认 3；位数越多 token 越多。 |
| `--max-text-tokens` | LLM 文本路径最大 token 数。 | 正整数 | 默认 1024。 |
| `--text-preflight-records` | AE warmup 前先检查多少条 teacher text 的 tokenization。 | 非负整数 | 默认 32；设 0 跳过预检查。 |
| `--teacher-layer` | 取 LLM 哪一层 hidden state。 | 整数 | 当前配置为 8；`-1` 表示最后一层。 |
| `--temperature` | InfoNCE 温度。 | 正数 | 默认 0.07。 |
| `--contrastive-loss-weight` | symmetric InfoNCE 权重。 | 非负数 | 当前主要优化项，默认 1.0。 |
| `--cosine-loss-weight` | 正样本 cosine loss 权重。 | 非负数 | 当前默认 0.0；非 0 时注意它可能提高公共方向相似度但不提高 retrieval。 |
| `--projection-dim` | tensor embedding 维度。 | `null` 或 LLM hidden size | 当前 text teacher 不训练 projection，因此必须等于 LLM hidden size；默认 `null` 自动匹配。 |
| `--wandb-enabled` / `--no-wandb-enabled` | 是否启用 W&B。 | 布尔开关 | 默认读取 `wandb.enabled`。 |
| `--wandb-mode` | W&B 运行模式。 | `online`、`offline`、`disabled` | `online`：上传到云端；`offline`：本地缓存；`disabled`：禁用。 |
| `--wandb-log-model` / `--no-wandb-log-model` | 是否上传 patch AE/alignment checkpoint artifact。 | 布尔开关 | 默认读取 `wandb.log_model`。 |

`patch_alignment.patch_encoder` 是新建 patch AE 的模型配置，只有 `encoder_source: patch_ae_config` 时使用。默认单字段 `Vx` 的结构是：

```yaml
patch_encoder:
  model:
    input_size: [16, 16]
    channel_multipliers: [1, 2]
    latent_dim: 128
    latent_grid: [4, 4]
```

`channel_multipliers` 的长度决定下采样次数。默认长度为 2，因此下采样因子是 `2^2=4`，`16x16` 输入对应 `4x4` latent tokens。

这个脚本不是 QA 训练。它的目标是验证：tensor path 能否学到和“LLM 直接阅读文本形式 patch”一致的中间表示。若这里的 retrieval accuracy 训练不上去，说明 text teacher 表示或 tensor adapter 结构仍有问题；若训练有效，再把这个 adapter 初始化迁移到后续 readout QA。

### 3.10 Adapter 诊断输出

`scripts/diagnose_tensor_llm_adapter.py` 用于检查三个问题：QA 记录和 latent 是否对齐、不同 state 的 latent/soft prompt 是否真的不同、正确 latent 是否系统性降低正确答案的 NLL。输出是 JSONL，不会改训练结果。

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

### 3.11 配置模型对话 Smoke Test

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

### 3.12 模型选择建议

| 阶段 | 模型 | 说明 |
|---|---|---|
| 快速 debug | `Qwen/Qwen2.5-0.5B-Instruct` | 速度快，能力较弱。 |
| 推荐 pilot | `Qwen/Qwen2.5-1.5B-Instruct` | 中英能力、速度、显存占用较平衡。 |
| 正式结果 | `Qwen/Qwen2.5-7B-Instruct` | A800 80GB 可承受，结果更有说服力。 |

当前 QA 是英文 DSL，第一阶段不强依赖中文能力；后续如果要中文提问，优先选 Qwen 系列。

### 3.13 评估逻辑

训练脚本默认做 choice likelihood 评估，而不是只看 loss。

| baseline | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `correct` | 使用当前样本的正确 tensor latent。 | baseline 名 | 目标能力。 |
| `no_latent` | 使用全 0 soft prompt。 | baseline 名 | 检查 LLM 是否只靠文本先验答题。 |
| `zero_latent` | latent 置零，但仍经过 adapter 和 query 条件。 | baseline 名 | 新结构下更干净地检查“没有 tensor 内容、只有 query 条件”能答到什么程度。 |
| `shuffled` | 使用其他 state 的 latent。 | baseline 名 | 检查 latent 是否绑定当前 tensor。 |
| `random` | 使用随机 latent。 | baseline 名 | 可选消融。 |

只有当 `correct` 明显优于 `zero_latent/shuffled` 时，才说明 adapter 可能学到了读取 tensor latent 的能力。开启 `structured_query_conditioning` 后，`no_latent` 会同时移除 soft prompt 和 query-conditioned adapter 输出，因此不再是唯一关键对照。

### 3.14 LLM Readout Inspection

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
