# Tensor Compression 2.0

该仓库用于训练面向 **2D / 3D / 4D 数值张量** 的压缩-重建模型，并为后续接入 **LLM soft prompt / adapter / downstream tasks** 预留清晰的扩展接口。

当前实验配置已经切到 **PDEBench** 数据集，当前 2D 示例默认按 PDEBench 的 HDF5 文件组织方式来配置与说明。

当前版本已经完成：

- 仓库基础结构与模块划分
- 2D 压缩-重建训练链路
- 3D 压缩-重建 baseline 训练链路
- config 驱动的数据、模型、训练控制
- W&B 记录入口
- 重建结果可视化
- PDEBench HDF5 多字段训练与下游算子评估入口
- 4D 数据与模型注册入口预留
- 后续 adapter / LLM 命名空间预留

当前版本尚未完成：

- PDEBench 之外的数据整理、标准化统计与 manifest 生成
- 4D 具体数据处理与模型实现
- latent 对齐 LLM 的 adapter
- PDEBench 官方 forward / inverse 算子的训练本身

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

### 1.2 快速开始

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

### 1.3 检查 PDEBench HDF5 的 key 和 shape

仓库新增了测试文件 `tests/test_inspect_pdebench_hdf5.py`，用于在不启动训练的情况下检查 `.h5/.hdf5` 文件中的：

- 顶层 `key`
- 所有 dataset 路径
- 每个 dataset 的 `shape`
- 每个 dataset 的 `dtype`

这个测试会优先读取环境变量 `PDEBENCH_HDF5_PATH` 指向的文件；如果没有设置，则回退到 `configs/compressor_2d.yaml` 中的 `data.source_roots.all_primary`。

Linux / macOS 示例：

```bash
export PDEBENCH_HDF5_PATH=/path/to/your_pdebench_file.hdf5
python -m unittest discover -s tests -p "test_inspect_pdebench_hdf5.py" -v
```

PowerShell 示例：

```powershell
$env:PDEBENCH_HDF5_PATH="E:\path\to\your_pdebench_file.hdf5"
python -m unittest discover -s tests -p "test_inspect_pdebench_hdf5.py" -v
```

如果输出里出现类似：

```text
- density: shape=(N, T, H, W), dtype=float32
- pressure: shape=(N, T, H, W), dtype=float32
- Vx: shape=(N, T, H, W), dtype=float32
- Vy: shape=(N, T, H, W), dtype=float32
```

那么当前 2D 配置可以把这些物理字段作为一个 4 通道样本一起训练：

```yaml
data:
  dataset:
    hdf5_dataset_key: null
    hdf5_dataset_keys: [density, pressure, Vx, Vy]
    hdf5_index_mode: sample
    hdf5_sample_axes: [0, 1]
```

也就是把 PDEBench 常见的 `[sample, time, height, width]` 展开成 `sample * time` 个 2D 样本，并把 `density/pressure/Vx/Vy` 堆叠为通道维。当前代码会根据 `hdf5_dataset_keys` 的长度自动同步 `data.dataset.channels`、`model.in_channels`、`model.out_channels`，所以从 4 通道扩到 `n` 通道时，通常只需要改字段列表本身。若显存不够，可以退回单字段训练：设置 `hdf5_dataset_key: Vx`、删除或置空 `hdf5_dataset_keys`，代码会自动把通道数回退到 `1`。

### 1.4 PDEBench 下载辅助

本仓库不会自动下载 PDEBench 大文件，但可以基于 `PDEBench_code/PDEBench-main/pdebench/data_download/pdebench_data_urls.csv` 列出匹配条目和官方下载命令：

```bash
python ./scripts/pdebench_download_helper.py \
  --pdebench-root ./PDEBench_code/PDEBench-main \
  --pde-name 2d_cfd \
  --filename-contains 2D_CFD_Turb_M0.1 \
  --root-folder ./data/external/pdebench
```

如果已经手动下载了单个 HDF5 文件，也可以直接把 `configs/compressor_2d.yaml` 里的 `data.source_roots.all_primary` 指向该文件，不必把文件复制到仓库里。

如果你确认要直接下载匹配文件，可以显式加上 `--download`，脚本会用标准库下载并在 CSV 里有 MD5 时校验完整性：

```bash
python ./scripts/pdebench_download_helper.py \
  --pdebench-root ./PDEBench_code/PDEBench-main \
  --pde-name 2d_cfd \
  --filename-contains 2D_CFD_Turb_M0.1 \
  --root-folder ./data/external/pdebench \
  --download \
  --skip-existing
```

### 1.5 PDEBench 下游算子评估

训练好 AE 后，可以比较“原始数据经过 PDEBench 算子”和“AE 重建数据经过同一算子”的输出误差：

```bash
python ./scripts/evaluate_pdebench_downstream.py \
  --hdf5-path /path/to/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 \
  --sample-indices 0,1 \
  --compressor-checkpoint ./outputs/runs/<run>/checkpoints/best.pt \
  --batch-size 1 \
  --forward-operator-type callable \
  --forward-operator-spec ./path/to/your_forward_wrapper.py:forward_operator \
  --inverse-operator-type callable \
  --inverse-operator-spec ./path/to/your_inverse_wrapper.py:inverse_operator
```

当提供 `--compressor-checkpoint` 时，脚本会优先使用 checkpoint 中保存的训练字段顺序，例如训练时如果是 `hdf5_dataset_keys: [density, pressure, Vx, Vy]`，评估和导出也会自动沿用这个顺序。这样可以避免通道语义错位。此时如果你仍然显式传入 `--fields`，它必须与训练顺序完全一致，否则脚本会直接报错，而不是带着错误通道顺序继续运行。

如果还希望生成一个新的 HDF5 文件，里面只把 `--fields` 指定的数据集替换成 AE 重建结果，其他数据集和文件元信息保持原样，可以额外加：

```bash
python ./scripts/evaluate_pdebench_downstream.py \
  --hdf5-path /path/to/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 \
  --sample-indices all \
  --compressor-checkpoint ./outputs/runs/<run>/checkpoints/best.pt \
  --batch-size 1 \
  --reconstructed-hdf5-output ./outputs/pdebench_downstream/reconstructed_all.hdf5
```

对类似 `density/pressure/Vx/Vy: [sample, time, height, width]`、`t-coordinate: [22]`、`x-coordinate/y-coordinate: [512]` 的 PDEBench 2D CFD 文件，这个导出会复制原始 HDF5，再只覆盖 `density/pressure/Vx/Vy` 中 `--sample-indices` 和可选 `--time-start/--time-stop/--time-step` 对应的切片。`--sample-indices all` 表示覆盖所有 sample；也可以写 `0,1,2` 只导出部分样本。`t-coordinate`、`x-coordinate`、`y-coordinate` 以及其他未指定数据集会保留为原文件内容，便于后续直接把原始文件和重建文件分别送入同一个科学计算算子做对比。若输出路径已存在，需要显式加 `--overwrite-reconstructed-hdf5`。

`forward_operator` 和 `inverse_operator` 都接收一个 `payload: dict`，其中最重要的字段是：

- `payload["data"]`：形状为 `[height, width, time, channel]` 的 `torch.Tensor`
- `payload["grid"]`：形状为 `[height, width, 2]` 的坐标网格，如果 HDF5 中有 `x-coordinate/y-coordinate`
- `payload["t_coordinates"]`：时间坐标，PDEBench CFD 文件中可能比实际数据时间步多一个边界点
- `payload["field_names"]`：例如 `("density", "pressure", "Vx", "Vy")`

建议：如果你是从训练好的 checkpoint 出发做 PDEBench 评估或导出，优先不要手写 `--fields`，直接让脚本从 checkpoint 继承训练时的字段顺序。这是当前最不容易出错的用法。

脚本会分别对 `original` 和 `reconstructed` 调用同一个算子，并输出重建本身的 `mse/mae/relative_l1/psnr`，以及 `forward/...`、`inverse/...` 前缀下的算子输出误差。这里的 downstream `reconstruction/*` 是在反归一化后的原始物理量空间上计算的。完整 JSON 默认写入 `outputs/pdebench_downstream/`。

默认情况下，脚本还会在终端显示一个 `tqdm` 进度条，并持续更新当前 sample、AE 重建帧批次、算子阶段以及 PDEBench rollout 步数；如果你在日志重定向场景下不想看到这些进度输出，可以额外传 `--no-progress`。

注意：训练阶段的 `tensor_folder_2d` 会把 `[N, T, H, W]` 展开为单帧 `[C, H, W]` 样本；评估阶段的 `evaluate_pdebench_downstream.py` 会重新读取同一个 HDF5 文件，把某个样本的完整时间序列整理为 `[H, W, T, C]`，逐帧通过 AE 重建后再送入下游算子。

如果要直接使用 PDEBench 官方 FNO / UNet 类，可以使用内建 wrapper，但需要提供官方 checkpoint 和架构参数：

```bash
python ./scripts/evaluate_pdebench_downstream.py \
  --hdf5-path /path/to/file.hdf5 \
  --sample-indices 0 \
  --compressor-checkpoint ./outputs/runs/<run>/checkpoints/best.pt \
  --forward-operator-type pdebench-fno \
  --forward-checkpoint /path/to/2D_CFD_..._FNO.pt \
  --pdebench-root ./PDEBench_code/PDEBench-main \
  --num-channels 4 \
  --initial-step 10 \
  --t-train 21 \
  --modes 12 \
  --width 20
```

### 1.6 命令行参数说明

本节覆盖 README 中出现的主要命令行参数。`<path>`、`<int>`、`<float>`、`<string>` 表示需要替换成你自己的路径、整数、浮点数或字符串。

#### 通用安装、测试命令

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `python -m <module>` | 让 Python 以模块方式启动工具。 | `pip`、`unittest`、`venv` 等 Python 模块名 | `pip` 表示启动包管理器；`unittest` 表示启动 Python 标准测试框架；`venv` 表示创建虚拟环境。 |
| `pip install -r <file>` | 按依赖清单安装包。 | `<file>` | 指向 requirements 文件；本项目示例为 `requirements.txt`。 |
| `pip install --upgrade <package>` | 升级指定包。 | `<package>` | 示例里的 `pip` 表示升级 pip 自身。 |
| `python3.10 -m venv <dir>` | 用 Python 3.10 创建虚拟环境。 | `<dir>` | 虚拟环境目录；示例 `.venv` 表示在项目根目录创建 `.venv`。 |
| `python -` | 从标准输入读取并执行 Python 代码。 | `-` | `-` 是固定写法，表示代码来自后续输入而不是 `.py` 文件。 |
| `python -m unittest discover -s <dir>` | 指定 unittest 自动发现测试的起始目录。 | `<dir>` | 示例 `tests` 表示从 `tests/` 目录开始搜索测试。 |
| `python -m unittest discover -p <pattern>` | 指定测试文件名匹配模式。 | `<glob>` | 示例 `test_inspect_pdebench_hdf5.py` 表示只运行这个测试文件。 |
| `python -m unittest discover -v` | 打印更详细的测试输出。 | 无值开关 | 写上 `-v` 表示 verbose；不写则使用默认简略输出。 |
| `pip install --index-url <url>` | 指定 pip 主包索引。 | `<url>` | README 示例里的 `https://download.pytorch.org/whl/cu124` 表示优先从 PyTorch CUDA 12.4 wheel 源安装。 |

#### `scripts/train_compressor.py`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `--config` | 指定训练 YAML 配置文件。 | `<path>` | 指向任意 YAML 配置；示例 `./configs/compressor_2d.yaml` 表示使用默认 2D 配置。 |
| `--dry-run` | 只构建模型、损失、数据集并校验配置，不启动训练。 | 无值开关 | 写上 `--dry-run` 表示检查配置；不写则正式训练。 |

#### `scripts/pdebench_download_helper.py`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `--pdebench-root` | PDEBench 官方仓库根目录。 | `<path>` | 指向包含 `pdebench/data_download/pdebench_data_urls.csv` 的目录；默认是 `./PDEBench_code/PDEBench-main`。 |
| `--pde-name` | 按 PDE 类型过滤下载条目。 | `<string>`，可重复传入 | 示例 `2d_cfd` 表示只匹配 CSV 中 PDE 列为 `2d_cfd` 的文件；重复传入表示允许多个 PDE 类型。 |
| `--filename-contains` | 按文件名子串过滤。 | `null` 或 `<string>` | 不传表示不过滤文件名；传入 `2D_CFD_Turb_M0.1` 表示只保留文件名包含该子串的条目。 |
| `--root-folder` | 下载目标根目录，也用于打印官方下载命令。 | `<path>` | 示例 `./data/external/pdebench` 表示把 PDEBench 数据放到该目录下。 |
| `--download` | 是否由本脚本直接下载匹配文件。 | 无值开关 | 写上表示实际下载并在 CSV 有 MD5 时校验；不写只列出匹配条目和下载命令。 |
| `--skip-existing` | 下载时是否跳过已存在文件。 | 无值开关 | 写上表示目标文件已存在时跳过；不写则重新下载并覆盖写入目标文件。 |

#### `scripts/evaluate_pdebench_downstream.py`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `--hdf5-path` | 待评估的 PDEBench HDF5 文件。 | `<path>` | 必填；指向原始 `.h5/.hdf5` 文件。 |
| `--fields` | 指定参与评估和导出的 HDF5 字段顺序。 | `null` 或逗号分隔字段名 | 不传时优先使用 checkpoint 中的训练字段顺序；没有 checkpoint 顺序时自动发现字段；传入如 `density,pressure,Vx,Vy` 时必须与 checkpoint 顺序一致。 |
| `--sample-indices` | 指定评估哪些 sample。 | `all` 或逗号分隔整数 | `all` 表示所有 sample；`0,1,2` 表示只评估第 0、1、2 个 sample；默认 `0`。 |
| `--time-start` | 时间切片起点。 | `null` 或 `<int>` | 不传表示从第一个时间步开始；传整数表示 Python slice 的 start。 |
| `--time-stop` | 时间切片终点。 | `null` 或 `<int>` | 不传表示直到最后；传整数表示 Python slice 的 stop，左闭右开。 |
| `--time-step` | 时间切片步长。 | `null` 或 `<int>` | 不传表示步长为 1；传 `2` 表示每隔一个时间步取一次。 |
| `--spatial-stride` | 空间下采样步长。 | 正整数 | `1` 表示不下采样；`2` 表示空间维每隔一个点取一次。 |
| `--compressor-checkpoint` | 训练好的 AE checkpoint。 | `null` 或 `<path>` | 不传时把重建视为恒等复制，只评估原始流程；传入 `best.pt/last.pt` 时会先用 AE 重建再评估。 |
| `--compressor-config` | checkpoint 内没有 config 时额外提供训练配置。 | `null` 或 `<path>` | 不传表示从 checkpoint 的 `config` 字段读取；传入 YAML 时用该配置构建 AE。 |
| `--batch-size` | AE 逐帧重建时的 batch 大小。 | 正整数 | 数值越大通常越快但占用显存越多；默认 `1`。 |
| `--device` | 评估设备。 | `auto`、`cpu`、`cuda`、其他 PyTorch 设备字符串 | `auto` 表示有 CUDA 就用 GPU，否则用 CPU；`cpu` 强制 CPU；`cuda` 使用默认 CUDA 设备；`cuda:1` 等字符串可指定具体 GPU。 |
| `--forward-operator-type` | 正问题算子类型。 | `none`、`callable`、`pdebench-fno`、`pdebench-unet` | `none` 表示不评估 forward；`callable` 表示加载你提供的 Python callable；`pdebench-fno` 表示加载 PDEBench 官方 FNO；`pdebench-unet` 表示加载 PDEBench 官方 UNet。 |
| `--forward-operator-spec` | callable forward 算子位置。 | `null`、`module.py:function`、`import.path:function`、`<torch-load-path>` | `module.py:function` 从文件加载函数或类；`import.path:function` 从 Python import 路径加载；无冒号时用 `torch.load`，对象必须可调用。 |
| `--forward-checkpoint` | PDEBench forward 算子的 checkpoint。 | `null` 或 `<path>` | 使用 `pdebench-fno` 或 `pdebench-unet` 时必填；`none/callable` 时不用。 |
| `--inverse-operator-type` | 反问题算子类型。 | `none`、`callable` | `none` 表示不评估 inverse；`callable` 表示加载你提供的 callable wrapper。 |
| `--inverse-operator-spec` | callable inverse 算子位置。 | `null`、`module.py:function`、`import.path:function`、`<torch-load-path>` | 含义同 `--forward-operator-spec`，但用于 inverse 算子。 |
| `--pdebench-root` | PDEBench 官方仓库根目录。 | `<path>` | 使用 `pdebench-fno/pdebench-unet` 时用于 import 官方模型；默认 `./PDEBench_code/PDEBench-main`。 |
| `--num-channels` | PDEBench 官方模型输入/输出物理字段数。 | `null` 或正整数 | 不传时使用当前字段数；传 `4` 表示 `density/pressure/Vx/Vy` 四通道。 |
| `--initial-step` | PDEBench autoregressive 模型使用的历史时间步数。 | 正整数 | 示例 `10` 表示用前 10 个时间步预测后续时间步。 |
| `--t-train` | PDEBench 算子 rollout 的时间长度上限。 | `null` 或正整数 | 不传表示使用数据里可用时间长度；传 `21` 表示最多 rollout 到第 21 个时间步。 |
| `--modes` | FNO 频域 mode 数。 | 正整数 | 仅 `pdebench-fno` 使用；数值越大频域容量越高，需与官方 checkpoint 架构一致。 |
| `--width` | FNO 网络宽度。 | 正整数 | 仅 `pdebench-fno` 使用；需与官方 checkpoint 架构一致。 |
| `--init-features` | UNet 初始特征通道数。 | 正整数 | 仅 `pdebench-unet` 使用；需与官方 checkpoint 架构一致。 |
| `--output` | 评估结果 JSON 输出路径。 | `null` 或 `<path>` | 不传时写入 `outputs/pdebench_downstream/<timestamp>_pdebench_downstream.json`；传入路径则写到指定文件。 |
| `--reconstructed-hdf5-output` | 是否额外导出替换了重建字段的新 HDF5。 | `null` 或 `<path>` | 不传表示不导出 HDF5；传入路径表示复制原文件并覆盖指定字段切片。 |
| `--overwrite-reconstructed-hdf5` | 允许覆盖已存在的重建 HDF5 输出。 | 无值开关 | 写上表示目标存在时可覆盖；不写且目标存在会报错。 |
| `--no-progress` | 关闭评估阶段的进度条和阶段状态输出。 | 无值开关 | 默认会显示 sample 级 `tqdm` 和阶段状态；写上后改为静默执行，仅保留最终结果输出。 |

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
│  ├─ evaluate_pdebench_downstream.py
│  ├─ pdebench_download_helper.py
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
- `src/tensor_compression/data/`：数据集和 DataLoader 组装逻辑。当前已实现 2D / 3D 文件夹数据集，4D 入口已预留。
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

当前也支持 3D 数据 baseline：

- `.npy`
- `.npz`
- `.h5`
- `.hdf5`

3D baseline 当前能力：

- 支持规则体数据读取
- 支持 HDF5 dataset 选择
- 支持单文件多样本展开
- 支持通道数对齐
- 支持 3D resize 到固定输入尺寸
- 支持简单归一化

### 3.2 模型

当前模型为：

- `conv_token_autoencoder_2d`
- `conv_token_autoencoder_3d`

特点：

- 编码器：卷积下采样 + 残差块
- latent：连续 latent map 与 latent tokens
- 解码器：反卷积上采样 + 残差块
- 适合先完成 2D / 3D 数值张量压缩-重建基线

### 3.3 训练

当前训练链路支持：

- `tqdm` 进度条
- 混合精度开关
- 梯度裁剪
- train/val 指标记录
- checkpoint 保存
- 重建图保存
- W&B 日志记录
- 验证阶段重建图直接上传到 W&B

当前 W&B 记录策略如下：

- 保留一组精简后的 `train_step/*` 标量，便于直接观察训练趋势。
- 额外上传一个 `reconstruction_panel` 图片面板，用于查看 `reconstructions/` 目录中的三联图。

当前默认保留的 `train_step/*` 指标含义如下：

- `loss_total`：总目标函数值，等于各损失项按权重加权后的结果，是训练时真正反向传播的目标。
- `psnr` / `mse` / `mae` / `relative_l1` / `max_abs_error`：训练阶段默认指标，都是在归一化后的张量空间上计算的，适合观察优化是否稳定。
- `physical_psnr` / `physical_mse` / `physical_mae` / `physical_relative_l1` / `physical_max_abs_error`：把每个样本先反归一化回原始物理量空间后再计算，更适合与 PDEBench downstream 评估结果对齐。
- `loss_gradient`：梯度差损失，越低越好，用于约束局部变化和边缘结构。

当前已去掉的重复项如下：

- `loss_mse` 与 `mse` 数值相同。
- `loss_l1` 与 `mae` 数值相同。
- `loss_relative_l1` 与 `relative_l1` 数值相同。

如果你问“哪个更合理”，答案通常是两个都合理，但用途不同：

- 看训练是否收敛、不同实验是否更好优化：优先看归一化空间指标，也就是默认的 `mse/mae/relative_l1/psnr`。
- 看科学量纲下到底重建得怎样、以及和 downstream 误差是否一致：优先看 `physical_*` 指标。

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

本节按“意义 / 可选值 / 每个可选值的意义”说明配置项。没有固定枚举的参数会写成类型或范围，例如 `<path>`、`<int>`、`<float>`、`true/false`、`null`。

### 5.1 `experiment`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `name` | 实验名称，用于输出目录名和 W&B run 名称。 | `<string>` | 任意非空字符串；例如 `compressor_2d_baseline` 会生成类似 `<timestamp>_compressor_2d_baseline` 的 run 目录。 |
| `output_root` | 训练输出根目录。 | `<path>` | 可写相对路径或绝对路径；以 `./` 开头的相对路径会按项目根目录解析。 |
| `seed` | 随机种子，用于 Python、NumPy、PyTorch 和自动切分。 | `<int>` | 相同数据和相同 seed 下，自动切分与大部分初始化更可复现。 |
| `device` | 训练设备。 | `auto`、`cpu`、`cuda`、其他 PyTorch 设备字符串 | `auto` 表示 CUDA 可用时用 GPU，否则用 CPU；`cpu` 强制 CPU；`cuda` 使用默认 CUDA 设备；`cuda:0`、`cuda:1` 等可指定 GPU。 |
| `save_top_k` | 预留参数，当前训练代码未实际使用。 | `<int>` | 当前无运行效果；默认 `1` 表示未来可扩展为保留 1 个最优 checkpoint。 |

### 5.2 `data`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `dimensions` | 张量空间维度标记，也决定可视化器类型。 | `2`、`3`、其他整数 | `2` 表示 2D 张量场，使用 2D 可视化；`3` 表示 3D 体数据，使用三视图切片可视化；其他整数当前不会报错但只会得到未实现可视化器。 |
| `dataset_name` | 数据集注册名，用于从项目 registry 构建数据集。 | `tensor_folder_2d`、`tensor_folder_3d`、`tensor_folder_4d` | `tensor_folder_2d` 已实现 2D 文件夹/HDF5/图片读取；`tensor_folder_3d` 已实现 3D `.npy/.npz/.h5/.hdf5` baseline；`tensor_folder_4d` 只是预留入口，当前会抛 `NotImplementedError`。 |

说明：

- `tensor_folder_2d` 不是 PyTorch 内置数据集名字。
- 它是本项目里单独定义并注册的一种通用 2D 数据读取逻辑。
- 当前对应实现文件是 `src/tensor_compression/data/datasets/tensor_folder_2d.py`。
- 这个读取器负责统一处理 `.npy / .npz / .h5 / .hdf5 / 图片` 等 2D 数据源，并完成样本索引、HDF5 dataset 选择、通道整理、resize、归一化等步骤。
- `dataset_name` 和 `dimensions` 应保持一致：2D 推荐 `dimensions: 2` + `tensor_folder_2d`；3D 推荐 `dimensions: 3` + `tensor_folder_3d`。

#### `data.source_roots`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `all_primary` | `data.split.mode: auto` 时的主数据池。 | `null` 或 `<path>` | `null` 表示不使用主数据池；路径可指目录或单个文件。 |
| `all_extra` | `data.split.mode: auto` 时的额外数据池列表。 | `[]` 或路径列表 | `[]` 表示没有额外数据源；列表中的每个路径都会和 `all_primary` 合并后再自动切分。 |
| `train_primary` | `data.split.mode: predefined` 时的主训练数据源。 | `null` 或 `<path>` | `null` 表示无主训练源；路径可指训练目录或单个训练文件。 |
| `train_extra` | `predefined` 模式下的额外训练数据源列表。 | `[]` 或路径列表 | `[]` 表示没有额外训练源；路径列表会追加到训练集扫描范围。 |
| `val_primary` | `predefined` 模式下的主验证数据源。 | `null` 或 `<path>` | 含义同 `train_primary`，但用于验证集。 |
| `val_extra` | `predefined` 模式下的额外验证数据源列表。 | `[]` 或路径列表 | 含义同 `train_extra`，但用于验证集。 |
| `test_primary` | `predefined` 模式下的主测试数据源。 | `null` 或 `<path>` | 含义同 `train_primary`，但用于测试集。 |
| `test_extra` | `predefined` 模式下的额外测试数据源列表。 | `[]` 或路径列表 | 含义同 `train_extra`，但用于测试集。 |
| `external_reference_roots` | 外部参考数据源预留字段。 | `[]` 或路径列表 | 当前代码不会读取该字段；保留给后续评估、对齐或外部参考数据。 |

说明：

- `all_primary / train_primary / val_primary / test_primary` 既可以指向目录，也可以直接指向单个文件。
- 当它们直接指向单个 `.h5/.hdf5/.npy/.npz/图片` 文件时，数据集会只读取这个文件，而不会继续扫描同目录下其他文件。

#### `data.split`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `mode` | 数据切分模式。 | `predefined`、`auto` | `predefined` 表示分别读取 `train/val/test` 数据源；`auto` 表示从 `all_primary/all_extra` 扫描出全部样本后按比例切分。 |
| `seed` | 自动切分随机种子。 | `<int>` | 仅 `mode: auto` 且 `shuffle: true` 时影响样本顺序。 |
| `shuffle` | 自动切分前是否打乱样本列表。 | `true`、`false` | `true` 表示用 `seed` 打乱后再切分；`false` 表示按扫描排序结果直接切分。 |
| `train_ratio` | 自动切分训练集比例。 | `0.0` 到 `1.0` 的 `<float>` | 例如 `0.8` 表示 80% 样本进入训练集。 |
| `val_ratio` | 自动切分验证集比例。 | `0.0` 到 `1.0` 的 `<float>` | 例如 `0.1` 表示 10% 样本进入验证集。 |
| `test_ratio` | 自动切分测试集比例。 | `0.0` 到 `1.0` 的 `<float>` | 例如 `0.1` 表示剩余 10% 样本进入测试集。 |

说明：

- 当前 `train_ratio + val_ratio + test_ratio` 必须等于 `1.0`。
- 自动切分是按“最终样本列表”做的。
- 对普通 `.npy / .npz / 图片` 文件来说，最终样本通常就是文件本身。
- 对 HDF5 来说，如果 `hdf5_index_mode: file`，则按文件内选中的整个 dataset 作为一个样本切分。
- 对 HDF5 来说，如果 `hdf5_index_mode: sample`，则会先按 `hdf5_sample_axes` 或 `hdf5_sample_axis` 展开文件内多个样本，再按展开后的样本切分。
- 自动切分结果是确定性的：同一组文件、同一个 `seed` 会得到同样的切分。

#### `data.dataset`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `recursive` | 扫描目录时是否递归进入子目录。 | `true`、`false` | `true` 表示扫描所有层级子目录；`false` 表示只扫描当前目录第一层文件。 |
| `allow_empty` | 是否允许当前 split 没有样本。 | `true`、`false` | `true` 允许先搭工程但训练时 train/val 为空仍会停止；`false` 表示数据集构建阶段就报错。 |
| `extensions` | 按文件后缀筛选可读取文件。 | 后缀字符串列表 | 2D 已支持 `.npy/.npz/.h5/.hdf5/.png/.jpg/.jpeg/.bmp/.tif/.tiff`；3D 已支持 `.npy/.npz/.h5/.hdf5`。 |
| `npz_key` | `.npz` 文件中读取哪个数组。 | `null` 或 `<string>` | `null` 表示读取 `.npz` 中第一个数组；字符串表示读取同名 key。 |
| `hdf5_dataset_key` | 单字段 HDF5 dataset 路径。 | `null` 或 `<string>` | `null` 表示不显式指定单字段；字符串如 `Vx` 或 `/fields/pressure` 表示读取该 dataset，并按单通道处理。 |
| `field_key` | `hdf5_dataset_key` 的兼容别名。 | `null` 或 `<string>` | 旧配置可继续用该字段；当 `hdf5_dataset_key` 为空时生效。 |
| `hdf5_dataset_keys` | 多字段 HDF5 dataset 路径列表。 | `null`、`[]`、字符串或字符串列表 | `null/[]` 表示不启用多字段；字符串会被当成单元素列表；列表如 `[density, pressure, Vx, Vy]` 会按顺序读取并堆叠成通道维。 |
| `hdf5_key_candidates` | 自动选择 HDF5 dataset 时的候选路径。 | `[]` 或字符串列表 | `[]` 表示不设置候选；列表会按顺序尝试，找到第一个存在且数值型的 dataset。 |
| `detect_hdf5_by_signature` | 是否通过文件头识别 HDF5。 | `true`、`false` | `true` 表示即使后缀不标准也尝试识别 HDF5；`false` 表示主要按后缀和 HDF5-like 后缀判断。 |
| `hdf5_index_mode` | HDF5 样本索引模式。 | `auto`、`file`、`sample` | `auto` 根据 dataset 维度和通道数推断；`file` 把整个 dataset 当一个样本；`sample` 按样本轴展开为多个样本。 |
| `hdf5_sample_axes` | 多个样本维配置。 | `null`、单个整数、整数列表 | `null` 表示使用 `hdf5_sample_axis` 并自动补足；`0` 表示第 0 维是样本维；`[0, 1]` 表示第 0 和第 1 维都展开，如 `[N,T,H,W]` 展开为 `N*T` 个样本。 |
| `hdf5_sample_axis` | 单个样本维配置。 | 整数，支持负索引 | 仅在 `hdf5_sample_axes: null` 时作为初始样本轴；`0` 表示第 0 维；`-1` 表示最后一维，但必须保证剩余维度能组成合法样本。 |
| `allow_images` | 2D 数据集是否允许读取图片。 | `true`、`false` | `true` 表示允许 `.png/.jpg/.jpeg/.bmp/.tif/.tiff`；`false` 表示遇到图片会报错。3D 数据集不读取图片。 |
| `channels` | 数据输入通道数。 | 正整数，或由 HDF5 字段自动推断 | 手写时表示最终输入通道数；配置 `hdf5_dataset_keys` 时会自动等于字段个数；配置单个 `hdf5_dataset_key/field_key` 时必须为 `1`。 |
| `input_size` | 数据加载后送入模型前的空间尺寸。 | 2D `[H, W]`，3D `[D, H, W]` | 样本尺寸不一致且 `strict_size: false` 时会 resize 到该尺寸；应与 `model.input_size` 保持一致。 |
| `strict_size` | 尺寸不匹配时是否直接报错。 | `true`、`false` | `true` 表示样本尺寸必须已经等于 `input_size`；`false` 表示自动 resize。 |
| `resize_mode` | PyTorch `F.interpolate` 插值模式。 | 2D 常用 `nearest`、`bilinear`、`bicubic`、`area`；3D 常用 `nearest`、`trilinear`、`area` | `nearest` 最近邻；`bilinear` 2D 双线性；`bicubic` 2D 双三次；`trilinear` 3D 三线性；`area` 面积插值。 |

##### `data.dataset.normalization`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `mode` | 归一化方式。 | `none`、`minmax`、`zscore` | `none` 不归一化；`minmax` 按最小值/最大值缩放到约 `[0,1]`；`zscore` 按均值和标准差标准化。 |
| `scope` | 统计归一化参数的范围。 | `global`、`channel` | `global` 对整个样本一起统计；`channel` 对每个通道分别统计，更适合不同量纲的多物理量。 |
| `stats_path` | 离线统计文件路径预留字段。 | `null` 或 `<path>` | 当前代码未读取该字段；`null` 表示不用离线统计，路径值留给后续扩展。 |
| `clip_min` | 归一化前的最小裁剪值。 | `null` 或 `<float>` | `null` 表示不设下界；浮点数表示低于该值的输入会被裁到该值。 |
| `clip_max` | 归一化前的最大裁剪值。 | `null` 或 `<float>` | `null` 表示不设上界；浮点数表示高于该值的输入会被裁到该值。 |

#### `data.loader`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `batch_size` | DataLoader 每个 batch 的样本数。 | 正整数 | 越大吞吐可能越高但显存占用越高；显存不足时先降到 `1` 或 `2`。 |
| `num_workers` | DataLoader 子进程数量。 | 非负整数 | `0` 表示主进程读取数据；大于 `0` 表示用多个 worker 并行加载。 |
| `shuffle_train` | 训练集 DataLoader 是否打乱样本。 | `true`、`false` | `true` 表示每个 epoch 打乱训练样本；`false` 表示按数据集顺序读取。 |
| `pin_memory` | 是否启用 pinned memory。 | `true`、`false` | `true` 通常有利于 CPU 到 GPU 拷贝；CPU 训练或内存紧张时可设 `false`。 |
| `drop_last` | 训练集是否丢弃最后一个不足 batch 的批次。 | `true`、`false` | `true` 保持训练 batch 尺寸一致；`false` 保留所有样本。验证和测试当前总是不丢弃。 |
| `persistent_workers` | 多 worker 时是否在 epoch 间保留 worker 进程。 | `true`、`false` | `true` 且 `num_workers > 0` 时减少重复启动开销；`false` 每轮按默认方式管理 worker。 |

### 5.3 `model`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `name` | 模型注册名。 | `conv_token_autoencoder_2d`、`conv_token_autoencoder_3d`、`factorized_autoencoder_4d` | `conv_token_autoencoder_2d` 已实现 2D 卷积 token autoencoder；`conv_token_autoencoder_3d` 已实现 3D baseline；`factorized_autoencoder_4d` 只是预留入口，当前会抛 `NotImplementedError`。 |
| `in_channels` | 模型输入通道数。 | 正整数，或由配置加载自动同步 | 多字段 HDF5 会自动设为 `hdf5_dataset_keys` 个数；单字段 HDF5 必须为 `1`。 |
| `out_channels` | 模型输出通道数。 | 正整数，或由配置加载自动同步 | 压缩-重建任务通常应等于 `in_channels`；不一致会导致目标通道不匹配。 |
| `input_size` | 模型输入空间尺寸。 | 2D `[H, W]`，3D `[D, H, W]` | 必须与 `data.dataset.input_size` 对齐，并满足 `input_size = latent_grid * 2^len(channel_multipliers)`。 |
| `base_channels` | 主干网络起始通道数。 | 正整数 | 数值越大模型容量越强、显存和计算开销越高。 |
| `channel_multipliers` | 每个下采样层的通道倍率列表。 | 正整数列表 | 列表长度决定下采样层数；每个元素 `m` 表示该层输出通道为 `base_channels * m`。 |
| `num_res_blocks` | 每个尺度上的残差块数量。 | 非负整数 | `0` 表示每个尺度不加残差块；更大值增加容量和计算量。 |
| `latent_dim` | latent map 的通道维度，也是每个 latent token 的特征维。 | 正整数 | 越小压缩率越高但表达能力越弱；启用自动扩张时会被重写为缩放后的值。 |
| `latent_dim_base` | 自动扩张 latent 时使用的基准 latent 维度。 | `null` 或正整数 | `null` 时代码会以 `latent_dim` 作为基准；正整数表示参考通道数下的 latent 维度。 |
| `latent_dim_scale_with_channels` | 是否按输入通道数自动扩张 `latent_dim`。 | `true`、`false` | `true` 表示按 `latent_dim_base * in_channels / latent_dim_reference_channels` 放大；`false` 表示完全使用手写 `latent_dim`。 |
| `latent_dim_reference_channels` | `latent_dim_base` 对应的参考通道数。 | 正整数 | 通常为 `1`，表示单通道时的 latent 预算。 |
| `latent_dim_round_to` | 自动扩张后向上取整的倍数。 | 正整数 | `1` 表示不额外取整；`32` 表示向上取到 32 的倍数。 |
| `latent_grid` | latent map 的空间尺寸。 | 2D `[H_lat, W_lat]`，3D `[D_lat, H_lat, W_lat]` | 与 `input_size` 和下采样层数严格绑定；越小 token 数越少、压缩越强。 |
| `dropout` | 残差块内部 dropout 概率。 | `0.0` 到 `1.0` 的 `<float>` | `0.0` 表示关闭 dropout；更大值增强正则但可能降低重建细节。 |
| `norm` | 网络归一化层类型。 | `batch`、`group`、`identity` | `batch` 使用 BatchNorm2d/3d；`group` 使用最多 8 组的 GroupNorm，小 batch 更稳；`identity` 不使用归一化。 |
| `activation` | 隐藏层激活函数。 | `relu`、`gelu`、`silu` | `relu` 经典 ReLU；`gelu` 平滑激活，当前默认；`silu` 也称 Swish。 |
| `output_activation` | 输出层激活函数。 | `identity`、`sigmoid`、`tanh` | `identity` 不限制输出范围，适合标准化后的数值场；`sigmoid` 把输出限制到 `[0,1]`；`tanh` 把输出限制到 `[-1,1]`。 |

3D baseline 使用同一套参数语义，只是：

- `input_size` 变成 `[D, H, W]`
- `latent_grid` 变成 `[D_lat, H_lat, W_lat]`
- `name` 可切换为 `conv_token_autoencoder_3d`

补充说明：

- 当配置了 `data.dataset.hdf5_dataset_keys` 时，代码会自动把 `data.dataset.channels`、`model.in_channels`、`model.out_channels` 同步为字段个数，并在配置不一致时直接报错。
- 当配置了单个 `hdf5_dataset_key` 或 `field_key` 时，代码会自动按单通道处理，并校验 `channels == in_channels == out_channels == 1`。
- 如果启用了 `latent_dim_scale_with_channels`，代码会基于 `latent_dim_base * in_channels / latent_dim_reference_channels` 自动放大 `latent_dim`，并按 `latent_dim_round_to` 向上取整。
- 如果你更在意固定压缩率，也可以关闭 `latent_dim_scale_with_channels`，继续手动控制 `latent_dim`。

#### 5.3.1 `conv_token_autoencoder_2d` 结构直观说明

当前 2D 模型可以粗略理解为：

1. 一个输入投影层
2. 多层卷积下采样
3. 每个尺度上若干残差块
4. 一个 `1x1` 卷积把特征投影到 latent
5. 对称的反卷积解码器把 latent 还原回原尺寸

其中：

- `stem conv`：指编码器开头那一层卷积，也就是“输入投影层”。
- 它当前对应代码里的：

```python
nn.Conv2d(self.in_channels, base_channels, kernel_size=3, padding=1)
```

- 它的作用是把原始输入从 `in_channels` 映射到模型主干使用的 `base_channels`。

例如当前常见配置是：

```yaml
in_channels: 1
base_channels: 32
```

那么 `stem conv` 会把：

```text
[B, 1, H, W] -> [B, 32, H, W]
```

这里的 `channel` 可以理解为“每个像素位置上的特征维度”。

对 2D 张量来说：

- `1` 个 channel：常见于单标量场，如压力场、速度某一分量等。
- `3` 个 channel：常见于 RGB 图像。
- 更大的 channel 数：表示网络内部学习到的多组特征，而不是人类直接可见的颜色通道。

例如：

```text
[B, 1, 512, 512]
```

表示一个 batch 中，每个样本是单通道、分辨率 `512x512` 的张量场。

```text
[B, 32, 512, 512]
```

表示同样的空间位置上，现在每个位置不再只有 1 个数，而是有 32 个特征值。


#### 5.3.2 当前配置下从样本到 latent 的形状变化

以当前常见配置为例：

```yaml
in_channels: 1
input_size: [512, 512]
base_channels: 32
channel_multipliers: [1, 2, 4, 8]
latent_dim: 128
latent_grid: [32, 32]
```

单个样本进入模型时的形状是：

```text
[1, 512, 512]
```

如果 batch size 为 `B`，则实际输入形状是：

```text
[B, 1, 512, 512]
```

编码过程形状变化如下：

```text
输入                         [B, 1,   512, 512]
stem conv                    [B, 32,  512, 512]

downsample 1                 [B, 32,  256, 256]
res blocks                   [B, 32,  256, 256]

downsample 2                 [B, 64,  128, 128]
res blocks                   [B, 64,  128, 128]

downsample 3                 [B, 128, 64,  64]
res blocks                   [B, 128, 64,  64]

downsample 4                 [B, 256, 32,  32]
res blocks                   [B, 256, 32,  32]

to_latent (1x1 conv)         [B, 128, 32,  32]   <- latent_map
flatten + transpose          [B, 1024, 128]      <- latent_tokens
```

这里：

- `latent_map` 是更接近卷积网络内部表示的 latent。
- `latent_tokens` 是把空间维展平成 token 后的表示。
- 因为 `32 * 32 = 1024`，所以 token 数是 `1024`。

#### 5.3.3 压缩率如何计算

这套模型里常见有两种压缩率口径：

1. 按 token 数计算的压缩率
2. 按内存占用计算的压缩率

##### 按 token 数计算

对输入大小为 `[H, W]` 的 2D 样本：

- 原始空间位置数：`H * W`
- latent token 数：`latent_grid[0] * latent_grid[1]`

所以：

```text
token 压缩率 = (H * W) / (latent_grid[0] * latent_grid[1])
```

当前 `512x512 -> 32x32` 时：

```text
token 压缩率 = (512 * 512) / (32 * 32) = 256:1
```

##### 按内存占用计算

若输入和 latent 使用相同精度存储，例如都按 `float32` 保存，则：

- 输入标量数：`in_channels * H * W`
- latent 标量数：`latent_dim * latent_grid[0] * latent_grid[1]`

所以：

```text
内存压缩率 = (in_channels * H * W) / (latent_dim * latent_grid[0] * latent_grid[1])
```

当前配置下：

```text
in_channels = 1
H = W = 512
latent_dim = 128
latent_grid = [32, 32]
```

因此：

```text
内存压缩率 = (1 * 512 * 512) / (128 * 32 * 32) = 2:1
```

注意：

- token 压缩率高，不代表真实内存压缩率也同样高。
- 当前 latent 仍然是连续浮点张量，没有做量化或熵编码，所以按内存算的压缩率会更保守。

#### 5.3.4 哪些 config 会影响压缩率

最关键的是两个：

- `latent_dim`
- `channel_multipliers`

其中：

- `latent_dim` 决定每个 latent token 有多少特征维。
- `channel_multipliers` 的长度决定下采样层数，也决定 `latent_grid` 的大小。

src\tensor_compression\models\compressors\conv_token_autoencoder_2d.py代码中
```python
for mult in multipliers:
    next_channels = base_channels * mult
```
即假设
```
multipliers = [1, 2, 4]
```
那么channels会变成
```
base → base → 2base → 4base
```
其中base指base_channel，默认值为32，即在进入卷积层之前，每个空间位置的量会由特征维度为1的数字先转换成特征维度为32的向量。（channel即通道，例如RBG图像为3通道）

如果把 `channel_multipliers` 的长度记为 `L`，则总下采样因子为：

```text
down_factor = 2^L
```

因此必须满足：

```text
input_size = latent_grid * down_factor
```

更展开地写，就是：

```text
latent_grid[0] = input_size[0] / 2^L
latent_grid[1] = input_size[1] / 2^L
```

其中 `L = len(channel_multipliers)`。

也就是说：

- `input_size[0]` 和 `input_size[1]` 必须都能被 `2^L` 整除。
- `latent_grid` 必须与 `input_size` 和下采样层数严格匹配。
- `latent_dim` 本身不会影响这个几何关系，但会影响最终内存压缩率。

例如当前 `input_size: [512, 512]` 时：

- 若 `channel_multipliers` 长度为 `4`，则 `down_factor = 16`，必须有 `latent_grid = [32, 32]`
- 若 `channel_multipliers` 长度为 `5`，则 `down_factor = 32`，必须有 `latent_grid = [16, 16]`
- 若 `channel_multipliers` 长度为 `6`，则 `down_factor = 64`，必须有 `latent_grid = [8, 8]`

例如下面这组配置：

```yaml
input_size: [512, 512]
channel_multipliers: [1, 2, 4, 8, 8, 8]
latent_grid: [8, 8]
latent_dim: 256
```

是满足几何关系的，因为：

```text
L = 6
down_factor = 2^6 = 64
512 / 64 = 8
```

因此：

```text
latent_grid = [8, 8]
```

正好与 `input_size` 匹配。

但这组配置虽然“能建模成功”，并不代表一定“训练最优”。

它对应的压缩率是：

- token 压缩率：`(512*512) / (8*8) = 4096:1`
- 内存压缩率：`(1*512*512) / (256*8*8) = 16:1`

可见：

- token 压缩率非常高
- 内存压缩率也明显高于当前基线
- 但 latent 更小、压缩更强，重建任务会更难，训练也更容易掉细节

所以判断一组配置时，通常需要同时检查两件事：

1. 数学上是否满足：

```text
input_size = latent_grid * 2^len(channel_multipliers)
```

2. 压缩是否过强，是否可能明显损伤重建质量

##### `latent_dim` 的作用

在不改变 `latent_grid` 的情况下：

- `latent_dim` 越小，内存压缩率越高。
- `latent_dim` 越大，latent 表达能力越强，但压缩率越低。

例如在 `512 -> 32x32` 不变时：

- `latent_dim: 128` -> 内存压缩率 `2:1`
- `latent_dim: 64` -> 内存压缩率 `4:1`
- `latent_dim: 32` -> 内存压缩率 `8:1`

##### `channel_multipliers` 长度的作用

如果把：

```yaml
channel_multipliers: [1, 2, 4, 8]
```

改成：

```yaml
channel_multipliers: [1, 2, 4, 8, 8]
```

那么下采样层数从 `4` 变成 `5`，总下采样因子从：

```text
16 -> 32
```

这时在 `input_size: [512, 512]` 下，`latent_grid` 需要从：

```text
[32, 32] -> [16, 16]
```

此时：

- token 压缩率从 `256:1` 提高到 `1024:1`
- 若 `latent_dim` 仍为 `128`，则内存压缩率从 `2:1` 提高到 `8:1`

#### 5.3.5 推荐压缩率分档

下面给出几组常见配置档位，便于按压缩率做实验。

##### 档位 A：当前基线

```yaml
input_size: [512, 512]
channel_multipliers: [1, 2, 4, 8]
latent_grid: [32, 32]
latent_dim: 128
```

- token 压缩率：`256:1`
- 内存压缩率：`2:1`

##### 档位 B：中等压缩

```yaml
input_size: [512, 512]
channel_multipliers: [1, 2, 4, 8]
latent_grid: [32, 32]
latent_dim: 64
```

- token 压缩率：`256:1`
- 内存压缩率：`4:1`

##### 档位 C：较强压缩

```yaml
input_size: [512, 512]
channel_multipliers: [1, 2, 4, 8]
latent_grid: [32, 32]
latent_dim: 32
```

- token 压缩率：`256:1`
- 内存压缩率：`8:1`

##### 档位 D：减少 token 数

```yaml
input_size: [512, 512]
channel_multipliers: [1, 2, 4, 8, 8]
latent_grid: [16, 16]
latent_dim: 128
```

- token 压缩率：`1024:1`
- 内存压缩率：`8:1`

##### 档位 E：高压缩

```yaml
input_size: [512, 512]
channel_multipliers: [1, 2, 4, 8, 8]
latent_grid: [16, 16]
latent_dim: 64
```

- token 压缩率：`1024:1`
- 内存压缩率：`16:1`

实验建议：

- 如果你优先关心“真实内存压缩率太低”，先减小 `latent_dim`。
- 如果你优先关心“token 数太多，不利于后续接 Transformer / LLM”，先增加下采样层数，减小 `latent_grid`。
- 如果两者都关心，就同时减小 `latent_dim` 与 `latent_grid`。

### 5.4 `loss`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `name` | 损失函数注册名。 | `composite_reconstruction_loss` | 当前唯一支持值，表示组合重建损失。 |
| `weights.mse` | MSE 损失权重。 | 非负浮点数 | `0` 表示关闭该项；大于 `0` 表示加入总 loss。 |
| `weights.l1` | L1 损失权重。 | 非负浮点数 | `0` 表示关闭该项；大于 `0` 表示约束平均绝对误差。 |
| `weights.relative_l1` | 相对 L1 损失权重。 | 非负浮点数 | `0` 表示关闭该项；大于 `0` 表示按目标幅值归一化误差。 |
| `weights.gradient` | 梯度差异损失权重。 | 非负浮点数 | `0` 表示关闭该项；大于 `0` 表示约束局部变化、边缘和场梯度。 |
| `eps` | 相对误差计算中的数值稳定项。 | 正浮点数 | 防止目标值接近 0 时除零；常用 `1.0e-6`。 |

### 5.5 `optimizer`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `name` | 优化器类型。 | `adamw`、`adam` | `adamw` 使用解耦权重衰减，当前默认；`adam` 使用经典 Adam。 |
| `lr` | 学习率。 | 正浮点数 | 控制参数更新步长；过大可能发散，过小训练慢。 |
| `weight_decay` | 权重衰减系数。 | 非负浮点数 | `0` 表示不做权重衰减；大于 `0` 表示加入 L2/解耦衰减正则。 |

### 5.6 `scheduler`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `name` | 学习率调度器类型。 | `cosine`、`none` | `cosine` 使用 `CosineAnnealingLR`；`none` 表示不使用调度器。 |
| `t_max` | cosine 调度周期。 | 正整数 | 仅 `name: cosine` 生效；通常可设为总 epoch 数。 |
| `min_lr` | cosine 调度最低学习率。 | 非负浮点数 | 仅 `name: cosine` 生效；表示退火末端的最小学习率。 |

### 5.7 `training`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `epochs` | 总训练轮数。 | 正整数 | 训练循环会从第 1 个 epoch 运行到该值。 |
| `mixed_precision` | 是否启用自动混合精度。 | `true`、`false` | `true` 且设备为 CUDA 时启用 AMP；CPU 上即使为 `true` 也不会启用 CUDA AMP；`false` 全精度训练。 |
| `grad_clip_norm` | 梯度范数裁剪阈值。 | `null`、`0`、正浮点数 | `null/0` 表示不裁剪；正数表示把梯度范数裁到该上限。 |
| `log_interval` | 训练 step 级日志间隔。 | 正整数 | 每隔多少个 training step 向 W&B 记录一次精简指标。 |
| `val_interval` | 验证频率预留字段。 | 正整数 | 当前代码每个 epoch 都验证，该字段暂未实际使用。 |
| `checkpoint_interval` | checkpoint 保存频率预留字段。 | 正整数 | 当前代码每个 epoch 都保存 `last.pt`，并在验证更优时保存 `best.pt`；该字段暂未实际使用。 |

### 5.8 `visualization`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `enabled` | 是否保存重建可视化。 | `true`、`false` | `true` 表示按周期保存图像；`false` 表示不保存。 |
| `num_samples` | 每次验证可视化的样本数。 | 正整数 | 实际数量为 `min(num_samples, 当前验证 batch 大小)`。 |
| `every_n_epochs` | 可视化保存间隔。 | 正整数 | `1` 表示每个 epoch 保存；`10` 表示每 10 个 epoch 保存一次。 |
| `field_cmap` | 原场和重建场的 Matplotlib colormap。 | 任意 Matplotlib colormap 名称 | 示例 `turbo` 适合数值场；也可用 `viridis`、`plasma`、`gray` 等。 |
| `error_cmap` | 误差图的 Matplotlib colormap。 | 任意 Matplotlib colormap 名称 | 示例 `inferno` 用于强调误差强度；也可用其他 Matplotlib 色图。 |
| `robust_percentile` | 稳健显示范围裁剪百分位。 | `0.0` 到小于 `50.0` 的浮点数 | `0.0` 接近使用全范围；`1.0` 表示忽略两端约 1% 极端值，避免色条被异常值支配。 |
| `display_channel` | 多通道输入时可视化哪些通道。 | `all` 或非负整数 | `all` 表示所有通道都输出；`0` 表示第 0 通道；整数超过通道数时会被截到最后一个通道。 |
| `add_colorbar` | 是否为每个子图添加色条。 | `true`、`false` | `true` 更便于读数但图更占空间；`false` 更紧凑。 |
| `save_dirname` | 可视化图像输出目录名。 | `<string>` 或相对目录名 | 会在每个 run 目录下创建该子目录；默认 `reconstructions`。 |

说明：

- 2D 输入会导出“原场 / 重建场 / 绝对误差”的标准三联图。
- 3D 输入现在支持三视图切片可视化：
  - 第 1 行：`axial` 中心切片的原场 / 重建 / 误差
  - 第 2 行：`coronal` 中心切片的原场 / 重建 / 误差
  - 第 3 行：`sagittal` 中心切片的原场 / 重建 / 误差
- 因而单个 3D 样本会导出一个 `3 x 3` 面板，便于直接观察三维体数据在三个正交方向上的重建质量。

### 5.9 `wandb`

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `enabled` | 是否启用 W&B 日志。 | `true`、`false` | `true` 会初始化 W&B 并记录指标/图片；`false` 不调用 W&B。 |
| `api_key` | W&B 登录 API Key。 | `null` 或 `<string>` | `null` 表示尝试读取环境变量 `WANDB_API_KEY`；字符串表示显式用该 key 登录。 |
| `project` | W&B project 名称。 | `<string>` | 指定 run 归属的项目名，例如 `tensor-compression`。 |
| `entity` | W&B entity/team/user。 | `null` 或 `<string>` | `null` 使用 W&B 默认 entity；字符串表示指定团队或用户名。 |
| `group` | W&B run 分组名。 | `null` 或 `<string>` | `null` 表示不分组；字符串表示同组实验，例如 `compression`。 |
| `tags` | W&B 标签列表。 | `[]` 或字符串列表 | `[]` 表示无标签；列表如 `[compression, reconstruction, 2d]` 便于筛选。 |
| `mode` | W&B 运行模式。 | `online`、`offline`、`disabled` 等 wandb 支持值 | `online` 实时上传；`offline` 本地记录后续可同步；`disabled` 禁用 W&B 运行。 |
| `log_model` | 模型 artifact 上传预留字段。 | `true`、`false` | 当前代码未使用；`false` 表示不上传模型 artifact，`true` 当前也不会产生额外效果。 |

说明：

- 当前支持把 `api_key` 写在 config 中。
- 当前也支持通过环境变量 `WANDB_API_KEY` 提供密钥。
- 为避免泄露，训练过程中落盘的 `config_resolved.yaml` 和 checkpoint 会自动将该字段打码。
- 上传 GitHub 前更推荐保持 `api_key: null`，并在服务器上通过环境变量管理密钥。

### 5.10 `future`

这是后续扩展保留的命名空间：

| 参数 | 意义 | 可选值 | 每个可选值的意义 |
|---|---|---|---|
| `future.adapters.enabled` | 是否启用 latent -> prompt adapter 的预留开关。 | `true`、`false` | 当前代码未读取；`false` 表示不启用，`true` 仅作为未来配置意图。 |
| `future.adapters.module_name` | adapter 模块名预留字段。 | `null` 或 `<string>` | 当前代码未读取；未来可用于指定 adapter 实现。 |
| `future.llm.enabled` | 是否启用 LLM 接入的预留开关。 | `true`、`false` | 当前代码未读取；`false` 表示不接入，`true` 仅作为未来配置意图。 |
| `future.llm.model_name` | LLM 模型名预留字段。 | `null` 或 `<string>` | 当前代码未读取；未来可用于指定 LLM。 |
| `future.llm.prompt_token_count` | prompt token 数预留字段。 | 正整数 | 当前代码未读取；未来可表示 latent 转换出的 soft prompt token 数。 |
| `future.tensor_3d.model_name` | 3D 模型注册名预留/提示字段。 | `conv_token_autoencoder_3d` 或其他未来注册模型名 | 当前只是记录推荐 3D 模型名；不会自动切换主 `model.name`。 |
| `future.tensor_3d.dataset_name` | 3D 数据集注册名预留/提示字段。 | `tensor_folder_3d` 或其他未来注册数据集名 | 当前只是记录推荐 3D 数据集名；不会自动切换主 `data.dataset_name`。 |
| `future.tensor_4d.model_name` | 4D 模型注册名预留/提示字段。 | `factorized_autoencoder_4d` 或其他未来注册模型名 | 当前 `factorized_autoencoder_4d` 已注册但未实现。 |
| `future.tensor_4d.dataset_name` | 4D 数据集注册名预留/提示字段。 | `tensor_folder_4d` 或其他未来注册数据集名 | 当前 `tensor_folder_4d` 已注册但未实现。 |

## 6. 当前数据来源与放置方式

### 6.1 当前 PDEBench 示例

当前实验默认使用 **PDEBench** 的 HDF5 文件作为 2D 数据源。

`configs/compressor_2d.yaml` 里的当前示例配置是：

- `data.source_roots.all_primary`：直接指向一个 PDEBench `.hdf5` 文件
- `data.dataset.hdf5_dataset_key`：当前设为 `null`
- `data.dataset.hdf5_dataset_keys`：当前设为 `density / pressure / Vx / Vy`
- `data.dataset.hdf5_index_mode`：当前设为 `sample`
- `data.dataset.hdf5_sample_axes`：当前设为 `[0, 1]`
- `data.dataset.normalization`：当前默认设为 `mode: zscore` 且 `scope: channel`
- `model.norm`：当前默认设为 `group`
- `model.latent_dim_scale_with_channels`：当前默认设为 `true`

这表示当前默认把 PDEBench 2D CFD 中需要压缩重建的 `density / pressure / Vx / Vy` 四个字段作为一个 4 通道样本训练，并将常见的 `[sample, time, height, width]` 结构展开成多个 2D 时刻样本。

现在切换通道数时，最稳妥的做法是优先改 `hdf5_dataset_keys` 或 `hdf5_dataset_key`，让代码自动同步通道数。比如如果你只想训练单个物理量，例如只压缩 `Vx`，可以先运行 `tests/test_inspect_pdebench_hdf5.py` 看实际 key 和 shape，再把 `hdf5_dataset_key` 改成 `Vx`、删除或置空 `hdf5_dataset_keys`，其余通道配置会自动回到单通道语义。

对于多物理量输入，当前默认推荐：

- 使用 `normalization.mode: zscore` 且 `normalization.scope: channel`，避免不同量纲的通道互相干扰。
- 在小 batch 训练下优先使用 `model.norm: group`，比 `batch` 更稳。
- 对 2D 默认配置，`latent_dim` 会按通道数自动扩张；例如单通道基准为 `128` 时，4 通道会自动扩成 `512`。
- `latent_grid` 仍然建议按你的压缩目标手动设定；如果你更想固定真实压缩率，也可以关闭自动扩张并手动设置 `latent_dim`。

### 6.2 通用目录放置方式

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
      scope: global
      stats_path: null
      clip_min: null
      clip_max: null
```

说明：

- `extensions` 里只保留 HDF5 后缀，避免混入图片或其他格式。
- `hdf5_dataset_key` 用于显式指定你要读取的 dataset 路径。
- 如果你不知道 key 是什么，建议先运行 `python -m unittest discover -s tests -p "test_inspect_pdebench_hdf5.py" -v`。
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
      scope: global
      stats_path: null
      clip_min: null
      clip_max: null
```

说明：

- `mode: auto` 后，程序会扫描 `all_primary` 和 `all_extra` 中的文件，再自动分成 `train / val / test`。
- 这个切分是文件级切分，所以适合“一个文件对应一个样本”的场景。
- 如果一个 HDF5 文件里本身包含很多子样本，可以使用 `hdf5_index_mode: sample` 配合 `hdf5_sample_axis` 或 `hdf5_sample_axes` 在文件内部展开样本。

### 7.3 HDF5 dataset 选择规则

当前读取顺序是：

1. 如果配置了 `hdf5_dataset_keys`，会把这些 dataset 按配置顺序读取，并堆叠成通道维。
2. 否则如果配置了 `hdf5_dataset_key`，优先读取它。
3. 否则按 `hdf5_key_candidates` 的顺序尝试。
4. 如果还没找到，就自动遍历文件，选择第一个“数值型且维度为 2D / 3D / 4D / 5D”的 dataset。

对 2D 数据加载器来说：

- 2D dataset 会被视为单通道张量场。
- 3D dataset 会被尝试解释为带通道的 2D 张量。
- 4D dataset 在合适配置下可以被解释为“单文件多样本 2D 数据”。
- 多个 HDF5 dataset 通过 `hdf5_dataset_keys` 堆叠时，所有字段 shape 必须一致，例如 PDEBench 2D CFD 的 `density / pressure / Vx / Vy` 均为 `[N, T, H, W]`。

如果你的 HDF5 结构比较复杂，最稳妥的方式仍然是直接指定 `hdf5_dataset_key`。

### 7.4 单个 HDF5 文件内部有多个样本时

现在已经支持“一个 `.h5/.hdf5` 文件里保存很多 2D 样本”的情况。

常见形状包括：

- `[N, H, W]`
- `[N, T, H, W]`
- `[N, C, H, W]`
- `[N, H, W, C]`

对当前项目正在使用的 PDEBench 2D CFD 数据来说，`density / pressure / Vx / Vy` 这些字段通常都接近 `[N, T, H, W]`，因此当前配置才会使用：

```yaml
hdf5_dataset_key: null
hdf5_dataset_keys: [density, pressure, Vx, Vy]
hdf5_index_mode: sample
hdf5_sample_axes: [0, 1]
channels: 4
```

其中：

- `N` 是样本数
- `T` 可以是时间步或序列长度
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
    hdf5_sample_axes: [0, 1]
    hdf5_sample_axis: 0
    allow_images: false
    channels: 1
    input_size: [128, 128]
    strict_size: false
    resize_mode: bilinear
    normalization:
      mode: none
      scope: global
      stats_path: null
      clip_min: null
      clip_max: null
```

这段配置的含义是：

- 从指定目录扫描 HDF5 文件。
- 读取 `/data` 这个 dataset。
- 优先按 `hdf5_sample_axes: [0, 1]` 把前两维视为样本维。
- 因而像 `[N, T, H, W]` 这样的数据会被展开成 `N*T` 个独立 2D 样本。
- 在 `data.split.mode: auto` 下，按展开后的样本再划分 `train / val / test`。

如果你的 HDF5 整体就是一个样本，可以改成：

```yaml
hdf5_index_mode: file
```

如果暂时不确定，就先用：

```yaml
hdf5_index_mode: auto
```

### 7.5 适配 `n` 通道输入的建议

当前代码已经针对 `n` 通道输入做了两类自动适配：

- 如果使用 `hdf5_dataset_keys`，通道数会自动从字段列表长度推断，不需要再手动同步 `channels / in_channels / out_channels`。
- 如果使用单个 `hdf5_dataset_key`，代码会自动按单通道处理，并校验配置一致性。
- 如果启用了 `latent_dim_scale_with_channels`，`latent_dim` 会按通道数自动扩张，减少“多通道共享同一瓶颈导致明显变糊”的问题。

推荐做法：

```yaml
data:
  dataset:
    hdf5_dataset_keys: [field_0, field_1, field_2, field_3, field_4]
    normalization:
      mode: zscore
      scope: channel
model:
  norm: group
  latent_dim: 128
  latent_dim_scale_with_channels: true
  latent_dim_reference_channels: 1
  latent_dim_round_to: 32
```

这样做的原因是：

- 多个物理量往往量纲不同，按通道归一化更稳。
- 小 batch 下 `GroupNorm` 通常比 `BatchNorm` 更适合多通道张量重建。
- 保持单通道基准 latent 不变的同时，按通道数自动扩张 latent，比每次手动重算更不容易漏改。

当前仍然建议人工确定的部分主要是 `latent_grid`，以及是否启用 `latent_dim_scale_with_channels`。如果你更想维持与单通道近似的每通道表示预算，可以打开自动扩张；如果你更看重固定压缩率，可以关闭它并手动设定 `latent_dim`。

## 8. 后续扩展建议

下一步建议优先做以下几项：

1. 增加数据统计与 manifest 生成脚本。
2. 固化归一化统计并接入 `stats_path`。
3. 完成 3D / 4D 数据读取器与模型。
4. 增加 latent 导出和缓存逻辑。
5. 接入 adapter 与 LLM。
6. 增加下游任务训练脚本。

## 9. 相关文档

- `training_workflow.md`：完整训练流程文档。
- `./configs/compressor_2d.yaml`：默认 2D 配置。
- `./scripts/train_compressor.py`：训练入口。

## 10. PyTorch 安装说明

当前仓库使用的是：

- `torch==2.5.1`
- 官方 CUDA 12.4 wheel 源：`https://download.pytorch.org/whl/cu124`

这和 PyTorch 官方 `2.5.1` 的安装说明一致。对于 Linux / Windows 的 pip 安装，官方给出的 CUDA 12.4 命令是：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

本仓库没有用到 `torchvision` 和 `torchaudio`，所以 `requirements.txt` 中只保留了 `torch` 本体以及项目所需的其他依赖。
