# Tensor Compression 2.0

本仓库当前按三个功能块组织：

1. **压缩**：训练 tensor autoencoder，检查 HDF5 数据结构，并用 PDEBench 下游算子验证重建质量。
2. **Tensor Editor**：基于冻结 AE，在 latent 空间训练一个文本条件编辑器。这是实验性功能。
3. **Adapter**：导出 AE latent cache，冻结 LLM，训练 soft prompt adapter，让 LLM 回答 tensor readout QA。

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
cp configs/tensor_llm_adapter_pipeline.yaml configs/local_tensor_llm_adapter_pipeline.yaml
```

`configs/local_*.yaml` 已被 `.gitignore` 忽略，适合写服务器真实路径和 checkpoint。

### 3.2 准备模型与缓存目录

用途：读取 pipeline config，创建 asset/cache/output 目录，可选下载 HuggingFace 模型。

命令：

```bash
python scripts/prepare_tensor_llm_assets.py \
  --config configs/local_tensor_llm_adapter_pipeline.yaml \
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
| `epochs` | 训练轮数。 | 正整数 | - |
| `batch_size` | 训练 batch 大小。 | 正整数 | - |
| `eval_batch_size` | 评估 record batch 大小。 | 正整数 | - |
| `eval_choice_batch_size` | 候选答案打分 batch 大小。 | 正整数 | - |
| `gradient_accumulation_steps` | 梯度累积步数。 | 正整数 | - |
| `lr` | adapter 学习率。 | 正数 | - |
| `weight_decay` | 权重衰减。 | 非负数 | - |
| `grad_clip_norm` | 梯度裁剪范数。 | 非负数 | `0`：不裁剪。 |
| `max_prompt_tokens` | 文本 prompt 最大 token 数。 | 正整数 | 超出会左截断。 |
| `max_target_tokens` | 答案最大 token 数。 | 正整数 | - |
| `append_eos` | target 后是否追加 EOS。 | `true`、`false` | - |
| `eval_baselines` | 评估 baseline 列表。 | `correct`、`no_latent`、`shuffled`、`random` | `correct`：正确 latent；`no_latent`：零 soft prompt；`shuffled`：错配 latent；`random`：随机 latent。 |
| `choice_score` | 候选答案 NLL 计分方式。 | `mean`、`sum` | `mean`：按 token 数平均；`sum`：总 NLL。 |
| `log_interval` | 训练日志间隔。 | 正整数 | - |

### 3.4 生成 Tensor Readout QA

命令：

```bash
python scripts/build_tensor_readout_qa.py \
  --config configs/local_tensor_llm_adapter_pipeline.yaml
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
  --config configs/local_tensor_llm_adapter_pipeline.yaml
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
  --config configs/local_tensor_llm_adapter_pipeline.yaml
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
| `--epochs` | 训练轮数。 | 正整数 | - |
| `--batch-size` | 训练 batch 大小。 | 正整数 | - |
| `--eval-batch-size` | 评估 record batch 大小。 | 正整数 | - |
| `--eval-choice-batch-size` | 候选答案打分 batch 大小。 | 正整数 | - |
| `--gradient-accumulation-steps` | 梯度累积步数。 | 正整数 | - |
| `--lr` | adapter 学习率。 | 正数 | - |
| `--weight-decay` | 权重衰减。 | 非负数 | - |
| `--grad-clip-norm` | 梯度裁剪范数。 | 非负数 | `0`：不裁剪。 |
| `--soft-prompt-tokens` | soft prompt token 数。 | 正整数 | - |
| `--adapter-dim` | adapter 内部维度。 | 正整数 | 必须能被 heads 整除。 |
| `--adapter-layers` | adapter 层数。 | 正整数 | - |
| `--adapter-heads` | adapter heads。 | 正整数 | - |
| `--dropout` | adapter dropout。 | 0 到 1 | - |
| `--max-prompt-tokens` | prompt 最大 token 数。 | 正整数 | 超出会左截断。 |
| `--max-target-tokens` | target 最大 token 数。 | 正整数 | - |
| `--append-eos` / `--no-append-eos` | target 后是否追加 EOS。 | 布尔开关 | - |
| `--eval-baselines` | 评估 baseline 列表。 | 逗号分隔字符串 | 可包含 `correct,no_latent,shuffled,random`。 |
| `--choice-score` | 候选答案 NLL 计分方式。 | `mean`、`sum` | `mean`：按 token 平均；`sum`：累加。 |
| `--log-interval` | 训练日志间隔。 | 正整数 | - |

### 3.7 模型选择建议

| 阶段 | 模型 | 说明 |
|---|---|---|
| 快速 debug | `Qwen/Qwen2.5-0.5B-Instruct` | 速度快，能力较弱。 |
| 推荐 pilot | `Qwen/Qwen2.5-1.5B-Instruct` | 中英能力、速度、显存占用较平衡。 |
| 正式结果 | `Qwen/Qwen2.5-7B-Instruct` | A800 80GB 可承受，结果更有说服力。 |

当前 QA 是英文 DSL，第一阶段不强依赖中文能力；后续如果要中文提问，优先选 Qwen 系列。

### 3.8 评估逻辑

训练脚本默认做 choice likelihood 评估，而不是只看 loss。

| baseline | 说明 | 可选值 | 可选值说明 |
|---|---|---|---|
| `correct` | 使用当前样本的正确 tensor latent。 | baseline 名 | 目标能力。 |
| `no_latent` | 使用全 0 soft prompt。 | baseline 名 | 检查 LLM 是否只靠文本先验答题。 |
| `shuffled` | 使用其他 state 的 latent。 | baseline 名 | 检查 latent 是否绑定当前 tensor。 |
| `random` | 使用随机 latent。 | baseline 名 | 可选消融。 |

只有当 `correct` 明显优于 `no_latent/shuffled` 时，才说明 adapter 可能学到了读取 tensor latent 的能力。
