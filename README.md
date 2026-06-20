# Tensor Compression 2.0

本仓库当前围绕两条链路展开：

1. **Tensor autoencoder**：训练 2D / 3D / 4D 数值张量压缩与重建模型。
2. **Tensor latent -> LLM soft prompt**：把 AE latent cache 接入冻结大语言模型，训练一个小 adapter，让 LLM 完成基于张量状态的 readout QA。

当前 LLM 方向已经实现：

- 从 PDEBench HDF5 生成自监督 readout QA 数据。
- 从训练好的 AE checkpoint 导出每个 tensor state 的 latent cache。
- 冻结 HuggingFace causal LM，训练 soft prompt adapter。
- 用 `correct / no_latent / shuffled` baseline 做 choice likelihood 评估。
- 用配置文件统一管理模型、AE、数据、缓存和输出路径。

## 1. 环境安装

服务器 CUDA 为 12.4 时，直接安装：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

验证 CUDA：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("devices:", torch.cuda.device_count())
PY
```

`requirements.txt` 包含 PyTorch、HDF5、W&B、Transformers、HuggingFace Hub 等依赖。

## 2. 服务器存储与模型下载

大模型、PDEBench、latent cache 都不建议放在仓库目录。推荐在服务器上选择一个空间充足的盘，例如 `/data/<user>/...`。

先看 GPU：

```bash
nvidia-smi
```

再看磁盘：

```bash
df -h /data /scratch /mnt /home 2>/dev/null
du -h --max-depth=1 /data 2>/dev/null | sort -h | tail
```

复制模板配置：

```bash
cp configs/tensor_llm_adapter_pipeline.yaml configs/local_tensor_llm_adapter_pipeline.yaml
```

编辑 `configs/local_tensor_llm_adapter_pipeline.yaml`，重点改：

```yaml
storage:
  asset_root: /data/wyx/tensor_llm_assets
  hf_home: /data/wyx/hf_cache
  output_root: /data/wyx/tensor_llm_outputs

data:
  hdf5_path: /data/PiERN/PDEbench/data/2d-ns/xxx.hdf5
  fields: [Vx]
  qa_dir: /data/wyx/tensor_llm_assets/tensor_readout_qa_vx
  latent_dir: /data/wyx/tensor_llm_assets/tensor_readout_latents_vx_2x

compressor:
  config: ./configs/compressor_2d_vx_2x.yaml
  checkpoint: /data/wyx/tensor_llm_outputs/runs/<ae_run>/checkpoints/best.pt

model:
  name_or_path: Qwen/Qwen2.5-1.5B-Instruct
  local_dir: null
```

查看候选存储、创建目录、写入 HF 环境文件：

```bash
python scripts/prepare_tensor_llm_assets.py \
  --config configs/local_tensor_llm_adapter_pipeline.yaml \
  --create-dirs
```

下载模型：

```bash
python scripts/prepare_tensor_llm_assets.py \
  --config configs/local_tensor_llm_adapter_pipeline.yaml \
  --create-dirs \
  --download-model
```

如果配置了 `storage.asset_root` 和 `storage.hf_home`，脚本会写：

```bash
source /data/wyx/tensor_llm_assets/env_tensor_llm.sh
```

也可以手动下载：

```bash
export HF_HOME=/data/wyx/hf_cache
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir /data/wyx/models/Qwen2.5-1.5B-Instruct
```

如果使用本地模型目录，把配置改成：

```yaml
model:
  name_or_path: Qwen/Qwen2.5-1.5B-Instruct
  local_dir: /data/wyx/models/Qwen2.5-1.5B-Instruct
```

## 3. 功能与脚本

| 功能 | 脚本 | 主要输出 |
|---|---|---|
| 训练 AE | `scripts/train_compressor.py` | AE checkpoint |
| 生成 readout QA | `scripts/build_tensor_readout_qa.py` | `train/val/test.jsonl` |
| 导出 latent cache | `scripts/export_tensor_readout_latents.py` | `<state_ref>.pt` |
| 训练 LLM adapter | `scripts/train_tensor_llm_adapter.py` | `adapter_best.pt` |
| 准备存储和 HF 模型 | `scripts/prepare_tensor_llm_assets.py` | 目录、模型、环境文件 |
| PDEBench 下游评估 | `scripts/evaluate_pdebench_downstream.py` | JSON/HDF5 评估结果 |

所有 LLM pipeline 脚本都支持：

```bash
--config configs/local_tensor_llm_adapter_pipeline.yaml
```

命令行参数会覆盖配置文件中的默认值。

## 4. AE 训练

检查配置：

```bash
python scripts/train_compressor.py \
  --config configs/compressor_2d.yaml \
  --dry-run
```

开始训练：

```bash
python scripts/train_compressor.py \
  --config configs/compressor_2d.yaml
```

如果只训练 `Vx` 且希望得到约 2x float latent compression，可使用类似：

```yaml
data:
  dataset:
    hdf5_dataset_key: Vx
    hdf5_dataset_keys: []
    input_size: [512, 512]

model:
  channel_multipliers: [1, 2, 4, 8, 8]
  latent_grid: [16, 16]
  latent_dim: 512
  latent_dim_scale_with_channels: false
```

此时输入标量数为 `1 * 512 * 512`，latent 标量数为 `512 * 16 * 16`，约为 2:1。注意：如果 AE 只编码 `Vx`，后续 QA 也必须只生成 `Vx` 问题。

关键配置：

| section | 作用 |
|---|---|
| `data.source_roots.all_primary` | 单个 HDF5 文件路径 |
| `data.dataset.hdf5_dataset_key` | 单字段训练，如 `Vx` |
| `data.dataset.hdf5_dataset_keys` | 多字段训练，如 `[density, pressure, Vx, Vy]` |
| `data.dataset.hdf5_sample_axes` | PDEBench `[sample, time, H, W]` 展开轴 |
| `model.latent_grid` | latent 空间 token 网格 |
| `model.latent_dim` | 每个 latent token 的通道维度 |
| `training.epochs` | 训练轮数 |
| `wandb.enabled` | 是否启用 W&B |

## 5. 生成 Tensor Readout QA

用配置运行：

```bash
python scripts/build_tensor_readout_qa.py \
  --config configs/local_tensor_llm_adapter_pipeline.yaml
```

或直接传参：

```bash
python scripts/build_tensor_readout_qa.py \
  --hdf5-path /data/PiERN/PDEbench/data/2d-ns/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 \
  --output-dir /data/wyx/tensor_llm_assets/tensor_readout_qa_vx \
  --fields Vx \
  --sample-indices all \
  --time-indices all \
  --max-states 21000 \
  --num-bins 10 \
  --patch-size 32 \
  --latent-root /data/wyx/tensor_llm_assets/tensor_readout_latents_vx_2x
```

输出格式为 JSONL，每行一个 QA：

```json
{
  "qa_id": "sample000001_t0003_point_bin_0000",
  "sample_index": 1,
  "time_index": 3,
  "state_ref": "sample000001_t0003",
  "task_type": "point_bin",
  "query": "VALUE_BIN field=Vx time=3 row=12 col=34 choices=B00,...,B09",
  "choices": ["B00", "B01"],
  "answer": "B03",
  "latent_ref": "/data/wyx/tensor_llm_assets/tensor_readout_latents_vx_2x/sample000001_t0003.pt"
}
```

已实现任务：

| task | 含义 | 适用字段 |
|---|---|---|
| `point_bin` | 某点数值落在哪个 quantile bin | 单字段/多字段 |
| `point_compare` | 两个点哪个值更大 | 单字段/多字段 |
| `patch_compare` | 两个 patch 的均值哪个更大 | 单字段/多字段 |
| `max_speed_quadrant` | 最大速度所在象限 | 需要 `Vx,Vy` |
| `global_stat_bin` | 速度统计量所在 bin | 需要 `Vx,Vy` |

如果 `fields: [Vx]`，速度相关任务会自动关闭。

## 6. 导出 AE Latent Cache

用配置运行：

```bash
python scripts/export_tensor_readout_latents.py \
  --config configs/local_tensor_llm_adapter_pipeline.yaml
```

或直接传参：

```bash
python scripts/export_tensor_readout_latents.py \
  --qa-dir /data/wyx/tensor_llm_assets/tensor_readout_qa_vx \
  --hdf5-path /data/PiERN/PDEbench/data/2d-ns/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 \
  --compressor-checkpoint /data/wyx/tensor_llm_outputs/runs/<ae_run>/checkpoints/best.pt \
  --compressor-config configs/compressor_2d_vx_2x.yaml \
  --output-dir /data/wyx/tensor_llm_assets/tensor_readout_latents_vx_2x \
  --batch-size 4
```

输出为每个 state 一个 `.pt`：

```python
{
    "latent_map": Tensor[C, H_lat, W_lat],
    "sample_index": int,
    "time_index": int,
    "state_ref": str,
    "field_keys": list[str],
}
```

脚本会检查 QA 记录中的字段是否与 AE checkpoint 字段一致。除非做专门消融，否则不要使用 `--allow-field-mismatch`。

关键参数：

| 参数 | 作用 |
|---|---|
| `--qa-dir` | QA JSONL 所在目录 |
| `--compressor-checkpoint` | AE checkpoint |
| `--compressor-config` | checkpoint 没有 config 时使用 |
| `--fields` | 手动指定字段顺序，必须与 AE 训练一致 |
| `--storage-dtype` | latent 保存 dtype，默认 `float16` |
| `--overwrite` | 覆盖已有 latent |

## 7. 训练 LLM Soft Prompt Adapter

推荐先用一张空闲 A800：

```bash
source /data/wyx/tensor_llm_assets/env_tensor_llm.sh
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_llm_adapter.py \
  --config configs/local_tensor_llm_adapter_pipeline.yaml
```

直接传参版本：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_llm_adapter.py \
  --qa-dir /data/wyx/tensor_llm_assets/tensor_readout_qa_vx \
  --latent-dir /data/wyx/tensor_llm_assets/tensor_readout_latents_vx_2x \
  --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --cache-dir /data/wyx/hf_cache \
  --output-root /data/wyx/tensor_llm_outputs/runs \
  --torch-dtype bfloat16 \
  --batch-size 2 \
  --epochs 3
```

训练脚本内置 `TensorSoftPromptAdapter`：

```text
AE latent map -> flatten latent tokens -> cross-attention adapter -> K soft prompt embeddings
soft prompt embeddings + text token embeddings -> frozen causal LM
```

LLM 参数全部冻结，只训练 adapter。

关键参数：

| 参数 | 作用 |
|---|---|
| `--model-name-or-path` | HF model id 或本地模型目录 |
| `--cache-dir` | HF 下载/cache 目录 |
| `--soft-prompt-tokens` | 插入 LLM 前的连续 prompt token 数 |
| `--adapter-dim` | adapter 内部维度 |
| `--adapter-layers` | cross-attention 层数 |
| `--adapter-heads` | attention heads |
| `--eval-baselines` | 默认 `correct,no_latent,shuffled` |
| `--choice-score` | 候选答案 NLL 使用 `mean` 或 `sum` |

模型建议：

| 阶段 | 模型 |
|---|---|
| 最快 debug | `Qwen/Qwen2.5-0.5B-Instruct` |
| 推荐 pilot | `Qwen/Qwen2.5-1.5B-Instruct` |
| 更强结果 | `Qwen/Qwen2.5-7B-Instruct` |

当前 QA 是英文 DSL，第一阶段不强依赖中文能力。但为了后续中文提问和论文展示，建议优先使用 Qwen 系列。

## 8. Pipeline 配置说明

模板文件：

```text
configs/tensor_llm_adapter_pipeline.yaml
```

本地私有配置：

```text
configs/local_tensor_llm_adapter_pipeline.yaml
```

主要 section：

| section | 内容 |
|---|---|
| `storage` | 服务器存储候选、HF cache、输出目录 |
| `runtime` | seed、device |
| `data` | HDF5、字段、QA 目录、latent 目录 |
| `compressor` | AE config 和 checkpoint |
| `model` | HF 模型名、本地目录、dtype、下载模式 |
| `qa_generation` | QA 生成任务数量和 split |
| `latent_export` | latent 导出 batch、dtype、覆盖策略 |
| `adapter` | soft prompt adapter 结构 |
| `llm_training` | 训练轮数、batch、学习率、评估 baseline |

命令行参数优先级高于配置。例如临时换模型：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_llm_adapter.py \
  --config configs/local_tensor_llm_adapter_pipeline.yaml \
  --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct
```

## 9. PDEBench 下游评估

训练好 AE 后，可以比较原始数据和 AE 重建数据经过同一 PDEBench forward/inverse operator 后的误差：

```bash
python scripts/evaluate_pdebench_downstream.py \
  --hdf5-path /data/PiERN/PDEbench/data/2d-ns/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 \
  --sample-indices all \
  --compressor-checkpoint /data/wyx/tensor_llm_outputs/runs/<ae_run>/checkpoints/best.pt \
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

当 checkpoint 中保存了训练字段顺序时，评估脚本会优先沿用 checkpoint 字段，避免通道语义错位。

## 10. 常见注意事项

- **字段必须一致**：`Vx` AE 只能回答 `Vx` 相关 QA；四字段 QA 需要四通道 AE。
- **不要只看 loss**：至少比较 `correct`、`no_latent`、`shuffled`，确认模型真的使用了 tensor latent。
- **不要把大文件放仓库**：模型、HDF5、latent cache、checkpoint 都放 `/data/...`，仓库只保存代码和模板配置。
- **API 模型不适合当前训练**：soft prompt 需要 `inputs_embeds` 和反向传播，优先使用本地 HuggingFace causal LM。
- **`trust_remote_code` 默认关闭**：首轮建议选 Transformers 已支持的模型，如 Qwen2.5。

## 11. Git 忽略策略

已忽略：

```text
outputs/
wandb/
*.pt
*.ckpt
data/raw/
data/processed/
data/external/
configs/local_*.yaml
```

因此服务器真实路径、模型缓存和私有 checkpoint 建议只写在 `configs/local_*.yaml` 中。
