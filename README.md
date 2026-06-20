# Tensor Compression 2.0

本仓库当前包含两条主线：

- **压缩**：训练数值张量 autoencoder，评估重建质量，并可用 PDEBench 下游算子验证重建后的物理任务误差。
- **Adapter**：把训练好的 AE latent cache 接入冻结 LLM，训练 soft prompt adapter，让 LLM 回答基于张量状态的 readout QA。

安装依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

建议服务器上把 HDF5、模型权重、latent cache、运行输出放在仓库外部，例如 `/data/<user>/...`。仓库只保存代码、模板配置和轻量文档。

## 1. 压缩

### 1.1 探明 HDF5 文件 Key

用途：确认 PDEBench HDF5 里有哪些 dataset、shape 和 dtype，决定 `hdf5_dataset_key` 或 `hdf5_dataset_keys` 应该怎么写。

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

若没有设置 `PDEBENCH_HDF5_PATH`，测试会回退读取 `configs/compressor_2d.yaml` 中的 `data.source_roots.all_primary`。

常见输出：

```text
density: shape=(N, T, H, W), dtype=float32
pressure: shape=(N, T, H, W), dtype=float32
Vx: shape=(N, T, H, W), dtype=float32
Vy: shape=(N, T, H, W), dtype=float32
```

### 1.2 下载或定位 PDEBench 数据

用途：从 PDEBench 官方 CSV 中列出可下载文件，或直接下载到指定目录。

命令：

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

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `--pdebench-root` | PDEBench 仓库根目录 | 路径 | 默认 `./PDEBench_code/PDEBench-main`；脚本读取其中的 URL CSV。 |
| `--pde-name` | 按 PDE 类型筛选 | 字符串，可重复 | 例如 `2d_cfd`；可多次传入。 |
| `--filename-contains` | 按文件名子串筛选 | 字符串或不传 | 例如 `2D_CFD_Turb_M0.1`。 |
| `--root-folder` | 下载目标根目录 | 路径 | 建议放仓库外部，如 `/data/...`。 |
| `--download` | 是否实际下载 | 开关 | 不加时只打印匹配文件和命令。 |
| `--skip-existing` | 已存在文件是否跳过 | 开关 | 只在 `--download` 时生效。 |

### 1.3 训练压缩模型

用途：训练 2D/3D/4D tensor autoencoder。当前主要实验是 PDEBench 2D HDF5。

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

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `--config` | 压缩模型 YAML 配置 | 路径，必填 | 例如 `configs/compressor_2d.yaml` 或 `configs/compressor_2d_vx_2x.yaml`。 |
| `--dry-run` | 只构建对象并检查配置 | 开关 | 不启动训练；适合先检查数据路径、模型尺寸、loss 配置。 |

### 1.4 压缩配置文件

压缩训练配置示例：`configs/compressor_2d.yaml`。

#### `experiment`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `name` | 实验名 | 字符串 | 会进入 run 目录名。 |
| `output_root` | 输出根目录 | 路径 | 保存 checkpoint、metrics、可视化结果。 |
| `seed` | 随机种子 | 整数 | 控制 split、初始化等随机行为。 |
| `device` | 训练设备 | `auto`、`cpu`、`cuda`、`cuda:N` | `auto` 优先使用 CUDA。 |
| `save_top_k` | 保留最佳 checkpoint 数 | 整数 | 当前训练器主要保存 `best.pt` 和 `last.pt`。 |

#### `data.source_roots`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `all_primary` | 总数据入口 | 文件或目录路径 | HDF5 单文件时直接指向 `.hdf5`。 |
| `all_extra` | 总数据额外来源 | 路径列表 | 可为空。 |
| `train_primary` | 预定义训练目录 | 路径 | `split.mode: predefined` 时使用。 |
| `train_extra` | 训练额外来源 | 路径列表 | 可为空。 |
| `val_primary` | 预定义验证目录 | 路径 | `split.mode: predefined` 时使用。 |
| `val_extra` | 验证额外来源 | 路径列表 | 可为空。 |
| `test_primary` | 预定义测试目录 | 路径 | `split.mode: predefined` 时使用。 |
| `test_extra` | 测试额外来源 | 路径列表 | 可为空。 |
| `external_reference_roots` | 外部参考数据 | 路径列表 | 预留给外部参考数据。 |

#### `data.split`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `mode` | 数据划分模式 | `auto`、`predefined` | `auto` 从 `all_primary` 自动切分；`predefined` 使用 train/val/test 目录。 |
| `seed` | split 随机种子 | 整数 | `shuffle: true` 时生效。 |
| `shuffle` | 切分前是否打乱 | `true`、`false` | `auto` 模式常用 `true`。 |
| `train_ratio` | 训练比例 | 0 到 1 | 三个比例总和应为 1。 |
| `val_ratio` | 验证比例 | 0 到 1 | 同上。 |
| `test_ratio` | 测试比例 | 0 到 1 | 同上。 |

#### `data.dataset`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `recursive` | 是否递归扫描目录 | `true`、`false` | 目录输入时生效。 |
| `allow_empty` | 是否允许空数据集 | `true`、`false` | 训练时建议 `false`。 |
| `extensions` | 允许文件扩展名 | 列表 | 如 `.npy`、`.npz`、`.h5`、`.hdf5`、图片格式。 |
| `npz_key` | NPZ 数组 key | 字符串或 `null` | 读取 `.npz` 时使用。 |
| `hdf5_dataset_key` | 单字段 HDF5 key | 字符串或 `null` | 如 `Vx`；通道数自动设为 1。 |
| `hdf5_dataset_keys` | 多字段 HDF5 key | 字符串列表 | 如 `[density, pressure, Vx, Vy]`；按顺序堆成通道。 |
| `hdf5_key_candidates` | HDF5 key 候选 | 列表 | 未显式指定 key 时尝试匹配。 |
| `detect_hdf5_by_signature` | 是否按文件签名识别 HDF5 | `true`、`false` | 扩展名不标准时有用。 |
| `hdf5_index_mode` | HDF5 内部索引方式 | `auto`、`sample` | PDEBench `[sample,time,H,W]` 建议 `sample`。 |
| `hdf5_sample_axes` | 作为样本展开的轴 | 列表或 `null` | PDEBench 常用 `[0,1]`，即 sample 和 time 都展开。 |
| `hdf5_sample_axis` | 单样本轴 | 整数或 `null` | 向后兼容单轴配置。 |
| `allow_images` | 是否允许图片输入 | `true`、`false` | 科学张量实验建议 `false`。 |
| `input_size` | 模型输入空间尺寸 | `[H,W]` | 2D 模型如 `[512,512]`。 |
| `strict_size` | 是否强制输入尺寸一致 | `true`、`false` | `false` 时可 resize。 |
| `resize_mode` | resize 插值方式 | `bilinear` 等 | 2D 连续场常用 `bilinear`。 |

#### `data.dataset.normalization`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `mode` | 归一化方式 | `none`、`zscore`、`minmax` | PDEBench 当前常用 `zscore`。 |
| `scope` | 统计范围 | `global`、`channel` | `channel` 按通道分别统计。 |
| `stats_path` | 外部统计文件 | 路径或 `null` | 当前实现主要使用样本内统计。 |
| `clip_min` | 最小裁剪值 | 数值或 `null` | 不裁剪时为 `null`。 |
| `clip_max` | 最大裁剪值 | 数值或 `null` | 不裁剪时为 `null`。 |

#### `data.loader`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `batch_size` | batch 大小 | 正整数 | 512x512 多通道显存大，必要时设 1 或 2。 |
| `num_workers` | DataLoader worker 数 | 非负整数 | Linux 服务器可用 4 或更多。 |
| `shuffle_train` | 训练集是否打乱 | `true`、`false` | 常用 `true`。 |
| `pin_memory` | 是否 pin memory | `true`、`false` | CUDA 训练常用 `true`。 |
| `drop_last` | 是否丢弃最后小 batch | `true`、`false` | 小数据可设 `false`。 |
| `persistent_workers` | worker 是否常驻 | `true`、`false` | `num_workers > 0` 时可考虑。 |

#### `model`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `name` | 模型名 | `conv_token_autoencoder_2d`、`conv_token_autoencoder_3d`、`factorized_autoencoder_4d` | 从 registry 构建模型。 |
| `input_size` | 输入尺寸 | `[H,W]` | 必须和 encoder 下采样、`latent_grid` 对齐。 |
| `base_channels` | 基础通道数 | 正整数 | 越大容量越强，显存越高。 |
| `channel_multipliers` | 各层通道倍率 | 整数列表 | 长度决定下采样次数；2D 每层下采样 2 倍。 |
| `num_res_blocks` | 每层残差块数 | 非负整数 | 增加表达能力和计算量。 |
| `latent_dim` | latent 通道数 | 正整数 | 影响压缩率和 adapter 输入通道。 |
| `latent_dim_base` | latent 基准维度 | 正整数 | `latent_dim_scale_with_channels: true` 时使用。 |
| `latent_dim_scale_with_channels` | 是否按输入通道缩放 latent_dim | `true`、`false` | 多通道 AE 可设 `true`；固定压缩率可设 `false`。 |
| `latent_dim_reference_channels` | 缩放参考通道数 | 正整数 | 通常为 1。 |
| `latent_dim_round_to` | latent_dim 对齐粒度 | 正整数 | 如 32。 |
| `latent_grid` | latent 空间网格 | `[H_lat,W_lat]` | token 数为 `H_lat * W_lat`。 |
| `dropout` | dropout 概率 | 0 到 1 | AE baseline 常设 0。 |
| `norm` | 归一化层 | `group`、`batch`、`identity` | 小 batch 推荐 `group`。 |
| `activation` | 激活函数 | `relu`、`gelu`、`silu` | 默认常用 `gelu`。 |
| `output_activation` | 输出激活 | `identity`、`sigmoid`、`tanh` | 标准化连续场常用 `identity`。 |

压缩率估算：

```text
input scalars = C_in * H * W
latent scalars = latent_dim * H_lat * W_lat
float compression ratio = input scalars / latent scalars
```

例如单字段 `Vx`，输入 `[1,512,512]`，latent `[512,16,16]`，则约为 `262144 / 131072 = 2x`。

#### `loss`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `name` | loss 名 | `composite_reconstruction_loss` | 当前重建组合 loss。 |
| `weights.mse` | MSE 权重 | 非负数 | 主重建项。 |
| `weights.l1` | L1 权重 | 非负数 | 增强绝对误差约束。 |
| `weights.relative_l1` | 相对 L1 权重 | 非负数 | 关注相对误差。 |
| `weights.gradient` | 梯度误差权重 | 非负数 | 约束空间变化。 |
| `eps` | 数值稳定项 | 正数 | 防止除零。 |

#### `optimizer`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `name` | 优化器 | `adamw`、`adam` | 默认推荐 `adamw`。 |
| `lr` | 学习率 | 正数 | 如 `3.0e-4`。 |
| `weight_decay` | 权重衰减 | 非负数 | 如 `1.0e-2`。 |

#### `scheduler`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `name` | 调度器 | `cosine`、`none` | `cosine` 使用余弦退火。 |
| `t_max` | cosine 周期 | 正整数 | 通常等于 epoch 数。 |
| `min_lr` | 最小学习率 | 非负数 | 如 `1.0e-6`。 |

#### `training`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `epochs` | 训练轮数 | 正整数 | 如 50。 |
| `mixed_precision` | 混合精度 | `true`、`false` | CUDA 上建议 `true`。 |
| `grad_clip_norm` | 梯度裁剪 | 数值或 0 | 0 或空表示不裁剪。 |
| `log_interval` | step 日志间隔 | 正整数 | 控制训练日志频率。 |
| `val_interval` | 验证间隔 | 正整数 | 当前训练器每 epoch 验证。 |
| `checkpoint_interval` | checkpoint 间隔 | 正整数 | 当前训练器保存 best/last。 |

#### `visualization`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `enabled` | 是否可视化 | `true`、`false` | 保存原图/重建/误差图。 |
| `num_samples` | 可视化样本数 | 正整数 | 从验证集取样。 |
| `every_n_epochs` | 可视化间隔 | 正整数 | 如每 1 个 epoch。 |
| `field_cmap` | 场图 colormap | Matplotlib cmap | 如 `turbo`。 |
| `error_cmap` | 误差图 colormap | Matplotlib cmap | 如 `inferno`。 |
| `robust_percentile` | 鲁棒显示百分位 | 0 到 50 | 降低极值影响。 |
| `display_channel` | 显示通道 | 整数 | 多通道时选择一个通道。 |
| `add_colorbar` | 是否加色条 | `true`、`false` | 可视化用。 |
| `save_dirname` | 保存目录名 | 字符串 | 位于 run 目录下。 |

#### `wandb`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `enabled` | 是否启用 W&B | `true`、`false` | 无账号或离线时设 `false`。 |
| `api_key` | W&B key | 字符串或 `null` | 不建议写入文件，优先环境变量。 |
| `project` | W&B project | 字符串 | 如 `tensor-compression`。 |
| `entity` | W&B entity | 字符串或 `null` | 个人或团队名。 |
| `group` | W&B group | 字符串 | 实验分组。 |
| `tags` | 标签 | 字符串列表 | 便于筛选。 |
| `mode` | W&B 模式 | `online`、`offline`、`disabled` | 服务器无网可用 `offline`。 |
| `log_model` | 是否上传模型 | `true`、`false` | 大 checkpoint 建议 `false`。 |

#### `future`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `adapters.enabled` | adapter 预留开关 | `true`、`false` | 压缩训练链路暂不使用。 |
| `adapters.module_name` | adapter 模块名 | 字符串或 `null` | 预留。 |
| `llm.enabled` | LLM 预留开关 | `true`、`false` | 压缩训练链路暂不使用。 |
| `llm.model_name` | LLM 名称 | 字符串或 `null` | 预留。 |
| `llm.prompt_token_count` | prompt token 数 | 正整数 | 预留。 |
| `tensor_3d.model_name` | 3D 模型名 | 字符串 | 默认 `conv_token_autoencoder_3d`。 |
| `tensor_3d.dataset_name` | 3D 数据集名 | 字符串 | 默认 `tensor_folder_3d`。 |
| `tensor_4d.model_name` | 4D 模型名 | 字符串 | 默认 `factorized_autoencoder_4d`。 |
| `tensor_4d.dataset_name` | 4D 数据集名 | 字符串 | 默认 `tensor_folder_4d`。 |

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

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `--hdf5-path` | PDEBench HDF5 文件 | 路径，必填 | 原始数据文件。 |
| `--fields` | 参与评估字段 | 逗号分隔或不传 | 有 checkpoint 时默认使用 checkpoint 字段顺序；显式传入必须完全匹配。 |
| `--sample-indices` | 样本索引 | `all` 或逗号分隔整数 | 默认 `0`。 |
| `--time-start` | 时间切片起点 | 整数或不传 | Python slice start。 |
| `--time-stop` | 时间切片终点 | 整数或不传 | Python slice stop。 |
| `--time-step` | 时间切片步长 | 整数或不传 | Python slice step。 |
| `--spatial-stride` | 空间降采样步长 | 正整数 | 默认 1。 |
| `--compressor-checkpoint` | AE checkpoint | 路径或不传 | 不传时评估 identity reconstruction。 |
| `--compressor-config` | AE config | 路径或不传 | checkpoint 无内嵌 config 时使用。 |
| `--batch-size` | 重建 batch 大小 | 正整数 | 显存不足时调小。 |
| `--device` | 设备 | `auto`、`cpu`、`cuda`、`cuda:N` | 默认 `auto`。 |
| `--forward-operator-type` | forward operator 类型 | `none`、`callable`、`pdebench-fno`、`pdebench-unet` | `none` 表示不跑 forward operator。 |
| `--forward-operator-spec` | callable forward 入口 | `module.py:callable` 或 import path | 仅 `forward-operator-type=callable` 使用。 |
| `--forward-checkpoint` | PDEBench forward checkpoint | 路径或不传 | FNO/UNet operator 使用。 |
| `--inverse-operator-type` | inverse operator 类型 | `none`、`callable` | 当前 inverse 只支持 callable。 |
| `--inverse-operator-spec` | callable inverse 入口 | `module.py:callable` 或 import path | 仅 inverse callable 使用。 |
| `--pdebench-root` | PDEBench 仓库根目录 | 路径 | FNO/UNet operator 需要。 |
| `--num-channels` | operator 输入通道数 | 正整数或不传 | 通常等于字段数。 |
| `--initial-step` | PDEBench 初始时间步 | 正整数 | 默认 10。 |
| `--t-train` | PDEBench 训练时间长度 | 正整数或不传 | 与对应 checkpoint 设置一致。 |
| `--modes` | FNO modes | 正整数 | 默认 12。 |
| `--width` | FNO width | 正整数 | 默认 20。 |
| `--init-features` | UNet 初始特征数 | 正整数 | 默认 32。 |
| `--output` | JSON 输出路径 | 路径或不传 | 不传时保存到默认输出目录。 |
| `--reconstructed-hdf5-output` | 重建 HDF5 输出 | 路径或不传 | 会复制源文件并替换选定字段。 |
| `--overwrite-reconstructed-hdf5` | 是否覆盖重建 HDF5 | 开关 | 输出文件已存在时需要。 |
| `--no-progress` | 关闭进度条 | 开关 | 后台日志较干净。 |

## 2. Adapter

### 2.1 Adapter Pipeline 总览

当前路线：

```text
PDEBench HDF5
  -> build_tensor_readout_qa.py 生成 QA JSONL
  -> export_tensor_readout_latents.py 用 AE 导出 latent cache
  -> train_tensor_llm_adapter.py 冻结 LLM 训练 soft prompt adapter
```

推荐使用同一个配置：

```bash
cp configs/tensor_llm_adapter_pipeline.yaml configs/local_tensor_llm_adapter_pipeline.yaml
```

`configs/local_*.yaml` 已被 `.gitignore` 忽略，适合写服务器真实路径和 checkpoint。

### 2.2 准备模型与缓存目录

用途：读取 pipeline config，创建 asset/cache/output 目录，可选下载 HuggingFace 模型。

命令：

```bash
python scripts/prepare_tensor_llm_assets.py \
  --config configs/local_tensor_llm_adapter_pipeline.yaml \
  --create-dirs \
  --download-model
```

命令行参数：

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `--config` | pipeline YAML 配置 | 路径 | 默认 `configs/tensor_llm_adapter_pipeline.yaml`。 |
| `--create-dirs` | 创建配置中的目录 | 开关 | 创建 `asset_root`、`hf_home`、`qa_dir`、`latent_dir`、`output_root` 等。 |
| `--download-model` | 下载 HF 模型 | 开关 | 使用 `huggingface_hub.snapshot_download`。 |
| `--token` | HF token | 字符串或不传 | 私有模型或需授权模型使用；公开 Qwen 通常不需要。 |

### 2.3 Adapter Pipeline 配置文件

模板：`configs/tensor_llm_adapter_pipeline.yaml`。

#### `storage`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `candidate_roots` | 候选存储根目录 | 路径列表 | `prepare_tensor_llm_assets.py` 用于显示空间余量；不参与训练。 |
| `min_free_gb` | 最小期望空闲空间 | 数值 | 低于该值会标记为 `LOW`；不阻止运行。 |
| `asset_root` | 资产根目录 | 路径 | 可放 QA、latent、环境文件。 |
| `hf_home` | HuggingFace cache 根目录 | 路径 | 会用于 `HF_HOME`、模型缓存。 |
| `output_root` | 输出根目录 | 路径 | adapter run 输出的默认根目录。 |

#### `runtime`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `seed` | 全局随机种子 | 整数 | QA 生成和 adapter 训练可复用。 |
| `device` | 默认设备 | `auto`、`cpu`、`cuda`、`cuda:N` | 被 latent 导出和 adapter 训练读取。 |

#### `data`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `hdf5_path` | PDEBench HDF5 路径 | 路径 | QA 生成和 latent 导出使用。 |
| `fields` | 使用字段 | 字符串列表 | 必须和 AE checkpoint 编码字段一致，如 `[Vx]`。 |
| `qa_dir` | QA JSONL 输出/读取目录 | 路径 | 包含 `train.jsonl`、`val.jsonl`、`test.jsonl`。 |
| `latent_dir` | latent cache 目录 | 路径 | 包含 `<state_ref>.pt`。 |

#### `compressor`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `config` | AE 配置路径 | 路径或 `null` | checkpoint 不含 config 时需要。 |
| `checkpoint` | AE checkpoint 路径 | 路径 | latent 导出必需。 |

#### `model`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `name_or_path` | HF 模型名或路径 | repo id 或路径 | 如 `Qwen/Qwen2.5-1.5B-Instruct`。 |
| `local_dir` | 本地模型目录 | 路径或 `null` | 非空时训练脚本优先使用本地目录。 |
| `revision` | HF revision | 分支、tag、commit | 默认 `main`。 |
| `trust_remote_code` | 是否执行远端模型代码 | `true`、`false` | 首轮建议 `false`，选 Transformers 内置模型。 |
| `torch_dtype` | LLM 权重 dtype | `auto`、`float32`、`float16`、`bfloat16` | A800 推荐 `bfloat16`。 |
| `allow_patterns` | 下载允许文件模式 | glob 列表 | 限制下载文件，如 `*.safetensors`、`*.json`。 |
| `ignore_patterns` | 下载忽略文件模式 | glob 列表 | 排除不需要的大文件格式。 |

#### `qa_generation`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `sample_indices` | 样本索引 | `all` 或列表/逗号字符串 | 选择 HDF5 的 sample 维。 |
| `time_indices` | 时间索引 | `all` 或列表/逗号字符串 | 选择 HDF5 的 time 维。 |
| `max_states` | 最大 state 数 | 正整数或 `null` | 控制数据规模。 |
| `train_ratio` | train split 比例 | 0 到 1 | 与 val/test 总和为 1。 |
| `val_ratio` | val split 比例 | 0 到 1 | 同上。 |
| `test_ratio` | test split 比例 | 0 到 1 | 同上。 |
| `spatial_stride` | 空间采样步长 | 正整数 | 1 表示不降采样。 |
| `num_bins` | quantile bin 数 | 大于等于 2 | 生成 `B00...` 标签。 |
| `quantile_samples_per_state` | 每个 state 采样点数 | 正整数 | 用于估计 quantile 边界。 |
| `patch_size` | patch 边长 | 正整数 | `patch_compare` 使用。 |
| `point_bin_per_state` | 每个 state 的点值 bin 问题数 | 非负整数 | 可设 0 关闭。 |
| `point_compare_per_state` | 每个 state 的点比较问题数 | 非负整数 | 可设 0 关闭。 |
| `patch_compare_per_state` | 每个 state 的 patch 比较问题数 | 非负整数 | 可设 0 关闭。 |
| `max_quadrant_per_state` | 最大速度象限问题数 | 0 或 1 | 需要 `Vx,Vy`；单 `Vx` 应设 0。 |
| `global_stat_bin_per_state` | 速度统计 bin 问题数 | 0 到 3 | 需要 `Vx,Vy`；统计量为 mean/max/std speed。 |
| `compare_min_bin_distance` | 比较题最小 bin 间隔 | 非负整数 | 越大越避免近似平局。 |
| `compare_max_attempts` | 比较题重采样次数 | 正整数 | 找不到充分分离样本时停止尝试。 |
| `include_oracle` | 是否保存 oracle 数值 | `true`、`false` | debug 建议 `true`。 |

#### `latent_export`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `splits` | 导出哪些 split | 列表 | 通常 `[train,val,test]`。 |
| `batch_size` | AE encode batch | 正整数 | 显存不足时调小。 |
| `device` | 导出设备 | `auto`、`cpu`、`cuda`、`cuda:N` | 默认可继承 `runtime.device`。 |
| `spatial_stride` | 空间采样步长 | 正整数 | 必须与 QA 生成所用数据语义一致。 |
| `storage_dtype` | latent 保存 dtype | `float32`、`float16`、`bfloat16` | 默认 `float16` 节省空间。 |
| `allow_field_mismatch` | 允许 QA 字段和 AE 字段不一致 | `true`、`false` | 除非做消融，否则必须 `false`。 |
| `overwrite` | 是否覆盖已有 latent | `true`、`false` | 默认跳过已有文件。 |

#### `adapter`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `soft_prompt_tokens` | 插入 LLM 的连续 prompt token 数 | 正整数 | 如 32；越大容量越强，计算越多。 |
| `adapter_dim` | adapter 内部维度 | 正整数 | 需能被 `adapter_heads` 整除。 |
| `adapter_layers` | cross-attention 层数 | 正整数 | 首轮 2 层即可。 |
| `adapter_heads` | attention heads | 正整数 | 与 `adapter_dim` 匹配。 |
| `dropout` | dropout 概率 | 0 到 1 | 小数据可适当使用。 |

#### `llm_training`

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `run_name` | adapter run 名 | 字符串 | 进入输出目录名。 |
| `output_root` | adapter 输出根目录 | 路径 | 保存 `adapter_best.pt`、metrics。 |
| `train_split` | 训练 split 名 | 字符串 | 默认 `train`。 |
| `val_split` | 验证 split 名 | 字符串 | 默认 `val`。 |
| `test_split` | 测试 split 名 | 字符串 | 默认 `test`。 |
| `max_train_records` | 最大训练记录数 | 正整数或 `null` | 小规模 debug 可设置。 |
| `max_val_records` | 最大验证记录数 | 正整数或 `null` | 同上。 |
| `max_test_records` | 最大测试记录数 | 正整数或 `null` | 同上。 |
| `prefer_record_latent_ref` | 优先使用 JSONL 中的 latent_ref | `true`、`false` | 默认从 `latent_dir/state_ref.pt` 读取。 |
| `device` | 训练设备 | `auto`、`cpu`、`cuda`、`cuda:N` | 搭配 `CUDA_VISIBLE_DEVICES` 使用。 |
| `torch_dtype` | LLM dtype | `auto`、`float32`、`float16`、`bfloat16` | A800 推荐 `bfloat16`。 |
| `epochs` | 训练轮数 | 正整数 | 首轮 3。 |
| `batch_size` | 训练 batch | 正整数 | 显存足够可增大。 |
| `eval_batch_size` | 评估 batch | 正整数 | choice scoring 时的 record batch。 |
| `eval_choice_batch_size` | 候选答案打分 batch | 正整数 | 越大越快但占显存。 |
| `gradient_accumulation_steps` | 梯度累积步数 | 正整数 | 等效扩大 batch。 |
| `lr` | adapter 学习率 | 正数 | 默认 `1.0e-4`。 |
| `weight_decay` | 权重衰减 | 非负数 | 默认 `1.0e-2`。 |
| `grad_clip_norm` | 梯度裁剪 | 非负数 | 0 表示不裁剪。 |
| `max_prompt_tokens` | 文本 prompt 最大 token 数 | 正整数 | 过长会左截断。 |
| `max_target_tokens` | 答案最大 token 数 | 正整数 | 短标签通常 8 足够。 |
| `append_eos` | 答案后是否追加 EOS | `true`、`false` | 默认 `true`。 |
| `eval_baselines` | 评估 baseline | `correct`、`no_latent`、`shuffled`、`random` 的列表 | 必须至少看 `correct` 与 `shuffled/no_latent` 差距。 |
| `choice_score` | 候选答案 NLL 归一化 | `mean`、`sum` | 短答案推荐 `mean`。 |
| `log_interval` | 训练日志间隔 | 正整数 | 每多少 step 写一次 metrics。 |

### 2.4 生成 Tensor Readout QA

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

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `--config` | pipeline 配置 | 路径或不传 | 命令行显式参数优先。 |
| `--hdf5-path` | PDEBench HDF5 | 路径 | 必填，除非 config 提供。 |
| `--output-dir` | QA 输出目录 | 路径 | 写入 `train/val/test.jsonl` 和 `metadata.json`。 |
| `--fields` | 字段 | 逗号分隔字符串 | 如 `Vx` 或 `density,pressure,Vx,Vy`。 |
| `--sample-indices` | 样本索引 | `all` 或逗号分隔整数 | 控制 sample 维。 |
| `--time-indices` | 时间索引 | `all` 或逗号分隔整数 | 控制 time 维。 |
| `--max-states` | 最大 state 数 | 正整数或不传 | 限制总状态数。 |
| `--seed` | 随机种子 | 整数 | 控制 split 和采样。 |
| `--train-ratio` | train 比例 | 0 到 1 | split 比例。 |
| `--val-ratio` | val 比例 | 0 到 1 | split 比例。 |
| `--test-ratio` | test 比例 | 0 到 1 | split 比例。 |
| `--spatial-stride` | 空间步长 | 正整数 | 降采样读取。 |
| `--num-bins` | quantile bin 数 | 大于等于 2 | 生成 bin 标签。 |
| `--quantile-samples-per-state` | quantile 采样点数 | 正整数 | 估计 bin 边界。 |
| `--patch-size` | patch 大小 | 正整数 | patch 任务使用。 |
| `--point-bin-per-state` | 点值 bin 数 | 非负整数 | 每 state 生成多少题。 |
| `--point-compare-per-state` | 点比较题数 | 非负整数 | 每 state 生成多少题。 |
| `--patch-compare-per-state` | patch 比较题数 | 非负整数 | 每 state 生成多少题。 |
| `--max-quadrant-per-state` | 最大速度象限题数 | 0 或 1 | 需要 `Vx,Vy`。 |
| `--global-stat-bin-per-state` | 速度统计题数 | 0 到 3 | 需要 `Vx,Vy`。 |
| `--compare-min-bin-distance` | 比较题最小 bin 距离 | 非负整数 | 避免近似平局。 |
| `--compare-max-attempts` | 比较题采样尝试数 | 正整数 | 控制生成耗时。 |
| `--latent-root` | latent 引用根目录 | 路径或不传 | 写入 JSONL 的 `latent_ref`，不创建文件。 |
| `--include-oracle` / `--no-include-oracle` | 是否保存 oracle | 布尔开关 | debug 推荐保留。 |

### 2.5 导出 Latent Cache

命令：

```bash
python scripts/export_tensor_readout_latents.py \
  --config configs/local_tensor_llm_adapter_pipeline.yaml
```

命令行参数：

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `--config` | pipeline 配置 | 路径或不传 | 命令行显式参数优先。 |
| `--qa-dir` | QA 目录 | 路径 | 包含 split JSONL。 |
| `--splits` | 读取 split | 逗号分隔字符串 | 默认 `train,val,test`。 |
| `--hdf5-path` | PDEBench HDF5 | 路径或不传 | 不传时尝试读 `qa_dir/metadata.json`。 |
| `--compressor-checkpoint` | AE checkpoint | 路径 | 必填，除非 config 提供。 |
| `--compressor-config` | AE config | 路径或不传 | checkpoint 无 config 时需要。 |
| `--fields` | 字段顺序 | 逗号分隔或不传 | 默认从 checkpoint/config 读取；显式传入必须匹配。 |
| `--output-dir` | latent 输出目录 | 路径 | 写 `<state_ref>.pt`。 |
| `--batch-size` | AE encode batch | 正整数 | 显存不足时调小。 |
| `--device` | 设备 | `auto`、`cpu`、`cuda`、`cuda:N` | 默认 `auto`。 |
| `--spatial-stride` | 空间步长 | 正整数 | 与 QA 数据语义保持一致。 |
| `--storage-dtype` | 保存 dtype | `float32`、`float16`、`bfloat16` | 默认 `float16`。 |
| `--allow-field-mismatch` / `--no-allow-field-mismatch` | 是否允许字段不一致 | 布尔开关 | 默认不允许。 |
| `--overwrite` / `--no-overwrite` | 是否覆盖已有 latent | 布尔开关 | 默认不覆盖。 |

### 2.6 训练 Soft Prompt Adapter

命令：

```bash
source /data/wyx/tensor_llm_assets/env_tensor_llm.sh
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_llm_adapter.py \
  --config configs/local_tensor_llm_adapter_pipeline.yaml
```

命令行参数：

| 参数 | 作用 | 可选值 | 说明 |
|---|---|---|---|
| `--config` | pipeline 配置 | 路径或不传 | 命令行显式参数优先。 |
| `--qa-dir` | QA 目录 | 路径 | 读取 split JSONL。 |
| `--latent-dir` | latent cache 目录 | 路径 | 读取 `<state_ref>.pt`。 |
| `--model-name-or-path` | LLM 模型 | HF repo id 或本地路径 | 如 `Qwen/Qwen2.5-1.5B-Instruct`。 |
| `--cache-dir` | HF cache 目录 | 路径或不传 | 传给 tokenizer/model `from_pretrained`。 |
| `--hf-home` | HF_HOME | 路径或不传 | 脚本内设置环境变量默认值。 |
| `--output-root` | 输出根目录 | 路径 | 保存 adapter run。 |
| `--run-name` | run 名 | 字符串 | 输出目录名一部分。 |
| `--train-split` | 训练 split | 字符串 | 默认 `train`。 |
| `--val-split` | 验证 split | 字符串 | 默认 `val`。 |
| `--test-split` | 测试 split | 字符串 | 默认 `test`。 |
| `--max-train-records` | 最大训练记录数 | 正整数或不传 | debug 用。 |
| `--max-val-records` | 最大验证记录数 | 正整数或不传 | debug 用。 |
| `--max-test-records` | 最大测试记录数 | 正整数或不传 | debug 用。 |
| `--prefer-record-latent-ref` / `--no-prefer-record-latent-ref` | 是否优先 JSONL latent_ref | 布尔开关 | 默认从 `latent_dir` 解析。 |
| `--device` | 设备 | `auto`、`cpu`、`cuda`、`cuda:N` | 搭配 `CUDA_VISIBLE_DEVICES`。 |
| `--torch-dtype` | LLM dtype | `auto`、`float32`、`float16`、`bfloat16` | A800 推荐 `bfloat16`。 |
| `--trust-remote-code` / `--no-trust-remote-code` | 是否信任远端代码 | 布尔开关 | Qwen2.5 通常不需要。 |
| `--seed` | 随机种子 | 整数 | 控制 adapter 初始化等。 |
| `--epochs` | 训练轮数 | 正整数 | 首轮可设 3。 |
| `--batch-size` | 训练 batch | 正整数 | 显存足够可增大。 |
| `--eval-batch-size` | 评估 record batch | 正整数 | 影响评估显存。 |
| `--eval-choice-batch-size` | choice scoring batch | 正整数 | 候选答案打分批量。 |
| `--gradient-accumulation-steps` | 梯度累积 | 正整数 | 等效扩大 batch。 |
| `--lr` | 学习率 | 正数 | 默认 `1.0e-4`。 |
| `--weight-decay` | 权重衰减 | 非负数 | 默认 `1.0e-2`。 |
| `--grad-clip-norm` | 梯度裁剪 | 非负数 | 0 表示不裁剪。 |
| `--soft-prompt-tokens` | soft prompt token 数 | 正整数 | 如 32。 |
| `--adapter-dim` | adapter 维度 | 正整数 | 需被 heads 整除。 |
| `--adapter-layers` | adapter 层数 | 正整数 | cross-attention block 数。 |
| `--adapter-heads` | adapter heads | 正整数 | attention heads。 |
| `--dropout` | dropout | 0 到 1 | adapter dropout。 |
| `--max-prompt-tokens` | prompt 最大 token | 正整数 | 过长左截断。 |
| `--max-target-tokens` | target 最大 token | 正整数 | 短标签 8 足够。 |
| `--append-eos` / `--no-append-eos` | target 后是否加 EOS | 布尔开关 | 默认加。 |
| `--eval-baselines` | baseline 列表 | 逗号分隔 | `correct,no_latent,shuffled,random` 中选择。 |
| `--choice-score` | NLL 计分方式 | `mean`、`sum` | 推荐 `mean`。 |
| `--log-interval` | 日志间隔 | 正整数 | 每多少 step 写 metrics。 |

### 2.7 模型选择建议

| 阶段 | 模型 | 说明 |
|---|---|---|
| 快速 debug | `Qwen/Qwen2.5-0.5B-Instruct` | 速度快，能力较弱。 |
| 推荐 pilot | `Qwen/Qwen2.5-1.5B-Instruct` | 中英能力、速度、显存占用较平衡。 |
| 正式结果 | `Qwen/Qwen2.5-7B-Instruct` | A800 80GB 可承受，结果更有说服力。 |

当前 QA 是英文 DSL，所以第一阶段不强依赖中文能力；但后续如果要中文提问，优先选 Qwen 系列。

### 2.8 评估逻辑

训练脚本默认不只看 loss，还会做 choice likelihood 评估：

| baseline | 含义 | 用途 |
|---|---|---|
| `correct` | 使用正确 tensor latent | 目标能力。 |
| `no_latent` | soft prompt 全 0 | 检查 LLM 是否只靠文本先验答题。 |
| `shuffled` | 换成其他 state 的 latent | 检查 latent 是否真的绑定当前 tensor。 |
| `random` | 随机 latent | 可选消融。 |

只有当 `correct` 明显优于 `no_latent/shuffled` 时，才能说明 adapter 可能学到了读取 tensor latent 的能力。
