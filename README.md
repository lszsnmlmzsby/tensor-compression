# Tensor Compression 2.0

该仓库用于训练面向 **2D / 3D / 4D 数值张量** 的压缩-重建模型，并为后续接入 **LLM soft prompt / adapter / downstream tasks** 预留清晰的扩展接口。

当前版本已经完成：

- 仓库基础结构与模块划分
- 2D 压缩-重建训练链路
- config 驱动的数据、模型、训练控制
- W&B 记录入口
- 重建结果可视化
- 3D / 4D 数据与模型注册入口预留
- 后续 adapter / downstream / LLM 命名空间预留

当前版本尚未完成：

- 实际数据集内容
- 3D / 4D 具体数据处理与模型实现
- latent 对齐 LLM 的 adapter
- 下游任务训练

## 1. 快速开始

### 1.1 依赖安装

当前仓库已经按 **CUDA 12.4** 配置了 PyTorch 安装源。

原因：

- 你的服务器 `nvidia-smi` 显示 **CUDA Version: 12.4**
- 当前 `requirements.txt` 已将 PyTorch 的索引切换到官方 `cu124` wheel 源

也就是说，执行：

```bash
python -m pip install -r requirements.txt
```

会优先安装 **CUDA 12.4 版本的 `torch==2.5.1`**，而不是 CPU 版。

建议在 Linux 服务器上按下面步骤执行：

1. 创建并激活虚拟环境（推荐 Python 3.10）

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

2. 升级 `pip`

```bash
python -m pip install --upgrade pip
```

3. 安装依赖

```bash
python -m pip install -r requirements.txt
```

4. 验证 PyTorch 是否正确识别 CUDA

```bash
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("compiled cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

预期结果：

- `torch.version.cuda` 显示 `12.4`
- `torch.cuda.is_available()` 显示 `True`

如果这两项不满足，说明当前安装到的不是可用的 CUDA 版 PyTorch，或服务器驱动 / 环境仍有问题。

### 1.2 Linux 路径与命令说明

当前仓库默认按 Linux 优先写法给出命令示例：

- 相对路径使用 `./...`
- 目录分隔符使用 `/`
- 虚拟环境激活使用 `source .venv/bin/activate`

配置文件中的路径目前使用的是相对路径，例如：

```yaml
./data/raw/train
./outputs/runs
./configs/compressor_2d.yaml
```

这些写法在 Linux 下可以直接使用，不需要改成别的形式。

另外，代码现在会把这些相对路径统一解释为 **相对于项目根目录**，而不是相对于 `configs/` 目录。
因此像下面这样的配置：

```yaml
output_root: ./outputs/runs
```

会被解析到：

```bash
<project_root>/outputs/runs
```

而不会错误地落到：

```bash
<project_root>/configs/outputs/runs
```

只有在你手动改成绝对路径时，才需要写成 Linux 风格，例如：

```bash
/home/yourname/tensor-compression/data/raw/train
```

仅检查配置和对象构建，不启动训练：

```bash
python ./scripts/train_compressor.py --config ./configs/compressor_2d.yaml --dry-run
```

正式训练：

```bash
python ./scripts/train_compressor.py --config ./configs/compressor_2d.yaml
```

说明：

- 当前仓库不会主动安装任何依赖，也不会执行任何污染环境的操作。
- 当前 `requirements.txt` 默认面向 **CUDA 12.4** 环境。
- 当前配置默认允许数据目录为空，以便先完成工程搭建；真正开始训练时，如果训练集或验证集为空，程序会明确报错并停止。

## 2. 仓库结构

```text
tensor compression2.0/
├─ configs/
│  └─ compressor_2d.yaml
├─ data/
│  ├─ external/
│  ├─ interim/
│  ├─ processed/
│  └─ raw/
│     ├─ train/
│     ├─ val/
│     └─ test/
├─ outputs/
├─ scripts/
│  └─ train_compressor.py
├─ src/
│  └─ tensor_compression/
│     ├─ adapters/
│     ├─ data/
│     │  ├─ builders.py
│     │  └─ datasets/
│     ├─ downstream/
│     ├─ engine/
│     │  └─ trainer.py
│     ├─ integrations/
│     │  └─ wandb_logger.py
│     ├─ losses/
│     ├─ metrics/
│     ├─ models/
│     │  └─ compressors/
│     ├─ utils/
│     ├─ config.py
│     └─ registry.py
├─ README.md
├─ requirements.txt
└─ training_workflow.md
```

### 结构说明

- `configs/`：训练配置文件。
- `data/`：数据目录占位。当前只建结构，不放实际数据。
- `scripts/train_compressor.py`：压缩-重建训练入口脚本。
- `src/tensor_compression/data/`：数据集和 DataLoader 组装逻辑。当前已实现 2D 文件夹数据集，3D/4D 入口已预留。
- `src/tensor_compression/models/compressors/`：压缩模型定义。当前实现 2D 卷积式 token autoencoder。
- `src/tensor_compression/losses/`：重建损失定义。
- `src/tensor_compression/metrics/`：重建评估指标。
- `src/tensor_compression/engine/trainer.py`：训练主循环，包括进度条、日志记录、checkpoint、验证与可视化保存。
- `src/tensor_compression/integrations/wandb_logger.py`：W&B 初始化、登录和日志封装。
- `src/tensor_compression/utils/visualization.py`：重建可视化模块。
- `src/tensor_compression/adapters/`：预留给 latent -> soft prompt / embedding adapter。
- `src/tensor_compression/downstream/`：预留给分类、检索、问答、摘要等下游任务。

## 3. 当前已实现的训练能力

### 3.1 数据输入

当前支持 2D 数据：

- `.npy`
- `.npz`
- `.h5`
- `.hdf5`
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`

当前默认行为：

- 自动扫描 `train / val / test` 目录
- 支持从单一数据池自动切分 `train / val / test`
- 支持多来源路径
- 支持递归扫描
- 支持 HDF5 文件读取
- 支持通过文件头自动识别 HDF5，即使后缀不完全标准也会尽量识别
- 支持通道数对齐
- 支持自动 resize 到固定输入尺寸
- 支持简单归一化

### 3.2 模型

当前模型为：

- `conv_token_autoencoder_2d`

特点：

- 编码器：卷积下采样 + 残差块
- latent：连续 latent map 与 latent tokens
- 解码器：反卷积上采样 + 残差块
- 适合先完成 2D 数值张量压缩-重建基线

### 3.3 训练

当前训练链路支持：

- `tqdm` 进度条
- 混合精度开关
- 梯度裁剪
- train/val 指标记录
- checkpoint 保存
- 重建图保存
- W&B 日志记录

## 4. 可视化设计

当前可视化专门针对“数值张量场的人工观察”做了适配，而不是把它当普通图片处理。

每个样本输出三张图：

1. **原张量场**
2. **重建张量场**
3. **差值绝对值**

可视化策略：

- 对原张量场和重建张量场使用相同的颜色范围
- 原场和重建场使用独立于误差图的色图与色条
- 误差图显示 `|reconstruction - input|`
- 支持稳健分位数裁剪，避免极端值让整张图“糊成一片”
- 默认选择适合数值场观察的热图方式，而不是普通图片显示方式

## 5. 配置文件说明

主配置文件为 `configs/compressor_2d.yaml`。

### 5.1 `experiment`

- `name`：实验名称，用于输出目录和 W&B run 名称。
- `output_root`：训练输出根目录。
- `seed`：随机种子。
- `device`：训练设备，可设为 `auto`、`cpu`、`cuda`。
- `save_top_k`：预留参数，后续可扩展为保留多个最优 checkpoint。

### 5.2 `data`

- `dimensions`：张量维度标记。当前 `2` 表示 2D 数据。
- `dataset_name`：数据集注册名。当前使用 `tensor_folder_2d`。

说明：

- `tensor_folder_2d` 不是 PyTorch 内置数据集名字。
- 它是本项目里单独定义并注册的一种通用 2D 数据读取逻辑。
- 当前对应实现文件是 `src/tensor_compression/data/datasets/tensor_folder_2d.py`。
- 这个读取器负责统一处理 `.npy / .npz / .h5 / .hdf5 / 图片` 等 2D 数据源，并完成样本索引、HDF5 dataset 选择、通道整理、resize、归一化等步骤。

#### `data.source_roots`

- `all_primary`：当 `data.split.mode: auto` 时，作为未切分数据池的主目录。
- `all_extra`：当 `data.split.mode: auto` 时，作为未切分数据池的额外目录列表。
- `train_primary`：主训练目录。
- `train_extra`：额外训练目录列表，可放其他来源数据。
- `val_primary`：主验证目录。
- `val_extra`：额外验证目录列表。
- `test_primary`：主测试目录。
- `test_extra`：额外测试目录列表。
- `external_reference_roots`：预留给外部参考数据源。

说明：

- `all_primary / train_primary / val_primary / test_primary` 既可以指向目录，也可以直接指向单个文件。
- 当它们直接指向单个 `.h5/.hdf5/.npy/.npz/图片` 文件时，数据集会只读取这个文件，而不会继续扫描同目录下其他文件。

#### `data.split`

- `mode`：数据切分模式。
  - `predefined`：从 `train / val / test` 目录直接读取。
  - `auto`：从 `all_primary / all_extra` 统一扫描后，按比例自动切分。

- `seed`：自动切分随机种子。

- `shuffle`：自动切分前是否先打乱文件顺序。

- `train_ratio`：自动切分时训练集比例。

- `val_ratio`：自动切分时验证集比例。

- `test_ratio`：自动切分时测试集比例。

说明：

- 当前 `train_ratio + val_ratio + test_ratio` 必须等于 `1.0`。
- 自动切分是按“最终样本列表”做的。
- 对普通 `.npy / .npz / 图片` 文件来说，最终样本通常就是文件本身。
- 对 HDF5 来说，如果 `hdf5_index_mode: file`，则按文件内选中的整个 dataset 作为一个样本切分。
- 对 HDF5 来说，如果 `hdf5_index_mode: sample`，则会先按 `hdf5_sample_axis` 展开文件内多个样本，再按展开后的样本切分。
- 自动切分结果是确定性的：同一组文件、同一个 `seed` 会得到同样的切分。

#### `data.dataset`

- `recursive`：是否递归扫描子目录。
- `allow_empty`：是否允许目录为空。默认 `true`，方便先搭工程。
- `extensions`：支持读取的文件后缀。
- `npz_key`：`.npz` 文件中要读取的 key；为 `null` 时读取第一个数组。
- `hdf5_dataset_key`：指定 `.h5/.hdf5` 文件中要读取的 dataset 路径。
- `field_key`：`hdf5_dataset_key` 的兼容别名，方便兼容旧配置写法。
- `hdf5_key_candidates`：当 `hdf5_dataset_key` 为空时，按顺序尝试的候选 dataset 路径列表。
- `detect_hdf5_by_signature`：是否通过文件头自动识别 HDF5。开启后，即使后缀写得不标准，也会尽量识别。
- `hdf5_index_mode`：HDF5 索引模式。
  - `file`：整个 HDF5 dataset 当成一个样本。
  - `sample`：按 `hdf5_sample_axis` 将一个 HDF5 dataset 展开成多个样本。
  - `auto`：自动判断是否应展开为多个样本。
- `hdf5_sample_axis`：当 `hdf5_index_mode: sample` 时，指定样本维。
- `allow_images`：是否允许图片类文件作为 2D 输入读取。
- `channels`：输入通道数。
- `input_size`：模型输入尺寸。
- `strict_size`：尺寸不匹配时是否直接报错。
- `resize_mode`：resize 插值方法。

##### `data.dataset.normalization`

- `mode`：归一化方式。当前支持 `none`、`minmax`、`zscore`。
- `stats_path`：预留给离线统计文件路径。
- `clip_min`：最小裁剪值。
- `clip_max`：最大裁剪值。

#### `data.loader`

- `batch_size`：batch 大小。
- `num_workers`：DataLoader worker 数。
- `shuffle_train`：训练集是否打乱。
- `pin_memory`：是否启用 pinned memory。
- `drop_last`：训练集是否丢弃最后一个不足 batch 的样本。
- `persistent_workers`：是否保留 worker 进程。

### 5.3 `model`

- `name`：模型注册名。当前 2D 使用 `conv_token_autoencoder_2d`。
- `in_channels`：输入通道数。
- `out_channels`：输出通道数。
- `input_size`：输入分辨率，需要与 `latent_grid` 和总下采样因子匹配。
- `base_channels`：主干起始通道数。
- `channel_multipliers`：各层通道倍率，长度也决定下采样层数。
- `num_res_blocks`：每个尺度上的残差块个数。
- `latent_dim`：latent map 的通道维度。
- `latent_grid`：latent map 的空间尺寸。
- `dropout`：残差块内 dropout。
- `norm`：归一化类型，当前支持 `batch`、`group`、`identity`。
- `activation`：激活函数，当前支持 `relu`、`gelu`、`silu`。
- `output_activation`：输出层激活，当前支持 `identity`、`sigmoid`、`tanh`。

### 5.4 `loss`

- `name`：当前损失函数名，固定为组合重建损失。
- `weights.mse`：MSE 权重。
- `weights.l1`：L1 权重。
- `weights.relative_l1`：相对误差权重。
- `weights.gradient`：梯度差异损失权重。
- `eps`：相对误差计算中的数值稳定项。

### 5.5 `optimizer`

- `name`：当前支持 `adamw`、`adam`。
- `lr`：学习率。
- `weight_decay`：权重衰减。

### 5.6 `scheduler`

- `name`：当前支持 `cosine`、`none`。
- `t_max`：cosine 调度周期。
- `min_lr`：最小学习率。

### 5.7 `training`

- `epochs`：总训练轮数。
- `mixed_precision`：是否启用混合精度，仅在 GPU 上生效。
- `grad_clip_norm`：梯度裁剪阈值。
- `log_interval`：训练阶段 step 级日志间隔。
- `val_interval`：预留参数，后续可扩展为控制验证频率。
- `checkpoint_interval`：预留参数，后续可扩展为控制 checkpoint 保存频率。

### 5.8 `visualization`

- `enabled`：是否保存重建可视化。
- `num_samples`：每次验证可视化的样本数。
- `every_n_epochs`：每隔多少个 epoch 保存一次图像。
- `field_cmap`：原场和重建场使用的色图。
- `error_cmap`：误差图使用的色图。
- `robust_percentile`：用于稳健裁剪显示范围的百分位数。
- `display_channel`：多通道输入时，选择哪个通道进行可视化。
- `add_colorbar`：是否为每个子图添加色条。
- `save_dirname`：可视化图像输出目录名。

### 5.9 `wandb`

- `enabled`：是否启用 W&B。
- `api_key`：W&B 登录 API Key。若填写，代码会在初始化前调用 `wandb.login(key=...)`。
- 若 `api_key` 为空，代码会自动尝试读取环境变量 `WANDB_API_KEY`。
- `project`：W&B project 名称。
- `entity`：W&B entity。
- `group`：W&B 分组名。
- `tags`：W&B 标签。
- `mode`：W&B 运行模式，例如 `offline`、`online`、`disabled`。
- `log_model`：预留参数，当前代码未启用 model artifact 上传。

说明：

- 当前支持把 `api_key` 写在 config 中。
- 当前也支持通过环境变量 `WANDB_API_KEY` 提供密钥。
- 为避免泄露，训练过程中落盘的 `config_resolved.yaml` 和 checkpoint 会自动将该字段打码。
- 上传 GitHub 前更推荐保持 `api_key: null`，并在服务器上通过环境变量管理密钥。

### 5.10 `future`

这是后续扩展保留的命名空间：

- `future.adapters`：后续 latent -> prompt adapter 的配置入口。
- `future.llm`：后续接入 LLM 的配置入口。
- `future.tensor_3d`：3D 数据与模型注册名预留。
- `future.tensor_4d`：4D 数据与模型注册名预留。

## 6. 当前默认数据放置方式

将训练、验证、测试数据分别放到：

- `data/raw/train/`
- `data/raw/val/`
- `data/raw/test/`

如果有其他来源目录，可以直接在配置中填写：

- `data.source_roots.train_extra`
- `data.source_roots.val_extra`
- `data.source_roots.test_extra`

如果你希望程序自动切分数据，而不是手动准备 `train/val/test` 三个目录，则把所有原始文件放到：

- `data/raw/all/`

并在配置中打开：

```yaml
data:
  split:
    mode: auto
```

## 7. HDF5 数据集配置示例

下面给一个以 `.h5/.hdf5` 数据集为例的完整配置思路。

### 7.1 场景 A：已经有 train / val / test 三个目录

假设目录结构如下：

```text
data/raw/train/
  sample_0001.h5
  sample_0002.h5
data/raw/val/
  sample_0101.h5
data/raw/test/
  sample_0201.h5
```

推荐配置：

```yaml
data:
  dimensions: 2
  dataset_name: tensor_folder_2d
  source_roots:
    all_primary: ./data/raw/all
    all_extra: []
    train_primary: ./data/raw/train
    train_extra: []
    val_primary: ./data/raw/val
    val_extra: []
    test_primary: ./data/raw/test
    test_extra: []
    external_reference_roots: []
  split:
    mode: predefined
    seed: 42
    shuffle: true
    train_ratio: 0.8
    val_ratio: 0.1
    test_ratio: 0.1
  dataset:
    recursive: true
    allow_empty: false
    extensions:
      - .h5
      - .hdf5
    npz_key: null
    hdf5_dataset_key: /fields/pressure
    hdf5_key_candidates:
      - /fields/pressure
      - pressure
      - data
    detect_hdf5_by_signature: true
    allow_images: false
    channels: 1
    input_size: [128, 128]
    strict_size: false
    resize_mode: bilinear
    normalization:
      mode: zscore
      stats_path: null
      clip_min: null
      clip_max: null
```

说明：

- `extensions` 里只保留 HDF5 后缀，避免混入图片或其他格式。
- `hdf5_dataset_key` 用于显式指定你要读取的 dataset 路径。
- 如果你不确定 dataset 路径，可以先把 `hdf5_dataset_key` 设为 `null`，并把常见候选路径写到 `hdf5_key_candidates`。
- `detect_hdf5_by_signature: true` 时，即使后缀写得不完全标准，只要文件本体是 HDF5，也会尽量识别。

### 7.2 场景 B：只有一个总目录，希望自动切分

假设所有 HDF5 文件都先放在一个目录里：

```text
data/raw/all/
  sample_0001.h5
  sample_0002.h5
  sample_0003.hdf5
  sample_0004.data
```

其中 `sample_0004.data` 虽然后缀不标准，但如果文件头是真正的 HDF5，代码也会尝试识别。

推荐配置：

```yaml
data:
  dimensions: 2
  dataset_name: tensor_folder_2d
  source_roots:
    all_primary: ./data/raw/all
    all_extra: []
    train_primary: ./data/raw/train
    train_extra: []
    val_primary: ./data/raw/val
    val_extra: []
    test_primary: ./data/raw/test
    test_extra: []
    external_reference_roots: []
  split:
    mode: auto
    seed: 42
    shuffle: true
    train_ratio: 0.8
    val_ratio: 0.1
    test_ratio: 0.1
  dataset:
    recursive: true
    allow_empty: false
    extensions:
      - .h5
      - .hdf5
    npz_key: null
    hdf5_dataset_key: null
    hdf5_key_candidates:
      - /fields/pressure
      - /data
      - pressure
      - data
    detect_hdf5_by_signature: true
    allow_images: false
    channels: 1
    input_size: [128, 128]
    strict_size: false
    resize_mode: bilinear
    normalization:
      mode: none
      stats_path: null
      clip_min: null
      clip_max: null
```

说明：

- `mode: auto` 后，程序会扫描 `all_primary` 和 `all_extra` 中的文件，再自动分成 `train / val / test`。
- 这个切分是文件级切分，所以适合“一个文件对应一个样本”的场景。
- 如果一个 HDF5 文件里本身包含很多子样本，当前代码不会自动在文件内部继续切分；那种情况建议后续单独扩展 dataset 类。

### 7.3 HDF5 dataset 选择规则

当前读取顺序是：

1. 如果配置了 `hdf5_dataset_key`，优先读取它。
2. 否则按 `hdf5_key_candidates` 的顺序尝试。
3. 如果还没找到，就自动遍历文件，选择第一个“数值型且维度为 2D / 3D / 4D”的 dataset。

对 2D 数据加载器来说：

- 2D dataset 会被视为单通道张量场。
- 3D dataset 会被尝试解释为带通道的 2D 张量。
- 4D dataset 在合适配置下可以被解释为“单文件多样本 2D 数据”。

如果你的 HDF5 结构比较复杂，最稳妥的方式仍然是直接指定 `hdf5_dataset_key`。

### 7.4 单个 HDF5 文件内部有多个样本时

现在已经支持“一个 `.h5/.hdf5` 文件里保存很多 2D 样本”的情况。

常见形状包括：

- `[N, H, W]`
- `[N, C, H, W]`
- `[N, H, W, C]`

其中：

- `N` 是样本数
- `H, W` 是空间尺寸
- `C` 是通道数

推荐配置示例：

```yaml
data:
  split:
    mode: auto
    seed: 42
    shuffle: true
    train_ratio: 0.8
    val_ratio: 0.1
    test_ratio: 0.1
  dataset:
    recursive: true
    allow_empty: false
    extensions:
      - .h5
      - .hdf5
    hdf5_dataset_key: /data
    hdf5_key_candidates:
      - /data
      - data
    detect_hdf5_by_signature: true
    hdf5_index_mode: sample
    hdf5_sample_axis: 0
    allow_images: false
    channels: 1
    input_size: [128, 128]
    strict_size: false
    resize_mode: bilinear
    normalization:
      mode: none
      stats_path: null
      clip_min: null
      clip_max: null
```

这段配置的含义是：

- 从指定目录扫描 HDF5 文件。
- 读取 `/data` 这个 dataset。
- 把第 `0` 维视为样本维。
- 将单个 HDF5 文件内部的多个样本展开为独立样本。
- 在 `data.split.mode: auto` 下，按展开后的样本再划分 `train / val / test`。

如果你的 HDF5 整体就是一个样本，可以改成：

```yaml
hdf5_index_mode: file
```

如果暂时不确定，就先用：

```yaml
hdf5_index_mode: auto
```

## 8. GitHub 上传与下载

当前仓库已经补好了适合上传 GitHub 的基础文件：

- `.gitignore`
- `.gitattributes`
- `.editorconfig`

这些文件的作用：

- `.gitignore`：避免把数据、训练输出、虚拟环境、checkpoint、W&B 本地缓存等无关内容推上 GitHub。
- `.gitattributes`：统一行尾，并把二进制数据格式标记为 binary。
- `.editorconfig`：统一基础文本格式习惯。

### 8.1 上传到 GitHub

假设你已经在 GitHub 网站上创建了一个空仓库，下面是在 Linux 服务器或本地 Linux 环境中的常见流程。

1. 进入项目目录

```bash
cd /path/to/tensor-compression2.0
```

2. 初始化 Git 仓库

```bash
git init
```

3. 查看当前状态

```bash
git status
```

4. 添加文件

```bash
git add .
```

5. 提交

```bash
git commit -m "Initial commit"
```

6. 将默认分支设为 `main`

```bash
git branch -M main
```

7. 绑定远程仓库

HTTPS 方式：

```bash
git remote add origin https://github.com/<your-name>/<repo-name>.git
```

SSH 方式：

```bash
git remote add origin git@github.com:<your-name>/<repo-name>.git
```

8. 推送到 GitHub

```bash
git push -u origin main
```

说明：

- 如果你使用 HTTPS，GitHub 现在通常要求使用 **Personal Access Token**，而不是账户密码。
- 上传前请确认 `configs/compressor_2d.yaml` 中的 `wandb.api_key` 仍然是 `null`，不要把真实密钥提交到 GitHub。

### 8.2 从 GitHub 下载

使用 Git 克隆：

```bash
git clone https://github.com/<your-name>/<repo-name>.git
cd <repo-name>
```

或使用 SSH：

```bash
git clone git@github.com:<your-name>/<repo-name>.git
cd <repo-name>
```

### 8.3 拉取更新

```bash
git pull
```

### 8.4 查看远程仓库地址

```bash
git remote -v
```

### 8.5 修改远程仓库地址

```bash
git remote set-url origin https://github.com/<your-name>/<repo-name>.git
```

### 8.6 不使用 Git，直接下载 ZIP

如果只是下载仓库代码，不想使用 Git：

1. 打开 GitHub 仓库网页。
2. 点击 `Code`。
3. 选择 `Download ZIP`。

这种方式适合快速拿代码，但不方便后续同步更新。

## 9. 后续扩展建议

下一步建议优先做以下几项：

1. 增加数据统计与 manifest 生成脚本。
2. 固化归一化统计并接入 `stats_path`。
3. 完成 3D / 4D 数据读取器与模型。
4. 增加 latent 导出和缓存逻辑。
5. 接入 adapter 与 LLM。
6. 增加下游任务训练脚本。

## 10. 相关文档

- `training_workflow.md`：完整训练流程文档。
- `./configs/compressor_2d.yaml`：默认 2D 配置。
- `./scripts/train_compressor.py`：训练入口。

## 11. PyTorch 安装说明

当前仓库使用的是：

- `torch==2.5.1`
- 官方 CUDA 12.4 wheel 源：`https://download.pytorch.org/whl/cu124`

这和 PyTorch 官方 `2.5.1` 的安装说明一致。对于 Linux / Windows 的 pip 安装，官方给出的 CUDA 12.4 命令是：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

本仓库没有用到 `torchvision` 和 `torchaudio`，所以 `requirements.txt` 中只保留了 `torch` 本体以及项目所需的其他依赖。
