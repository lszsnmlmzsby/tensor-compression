# 可变形状张量实验

同一模型接收不同的二维单通道数值场 `[1,H,W]`。每个格点保留一个 memory token，
编码器输出 `[8,H,W]`，空间位置编码按实际行、列生成。训练从原始 HDF5 开始，
不需要旧 Stage 1、Direct-QA 或 scratch 实验的 checkpoint。
本轮保持现有五类 QA 和训练损失，默认只运行 `correct` 输入，不运行消融。

## 实验配置

| 项目 | `--profile pilot` | `--profile full` |
|---|---:|---:|
| 训练场状态 | 1,344 | 15,969 |
| 训练 QA | 12,096 | 143,721 |
| 更新步数（三卡、每卡 3 条、累积 1） | 1,500 | 31,938 |
| epoch 上限 | 3 | 2 |
| 每种形状的验证状态 / QA | 32 / 160 | 500 / 2,500 |
| 完整验证 QA | 2,560 | 40,000 |
| 用于选择 checkpoint 的验证 QA | 350 | 3,500 |
| 总时间上限 / 末尾评估预留 | 6 小时 / 45 分钟 | 24 小时 / 120 分钟 |

时间上限是可调整的保护参数，不是运行时间预测。到达时间上限但没有完成步数时，
`run_summary.json` 会标记 `training_incomplete`。小实验用于确认整条路径和初步学习趋势；
它的训练数据量、重复次数和余弦学习率日程不同，不能当作全量收敛结果。

| 划分 | 形状 H×W |
|---|---|
| 参与训练 | 8×8、8×16、16×8、16×16、16×32、32×16、32×32 |
| 未见形状 | 12×20、20×12、16×24、24×16、24×24、8×32、32×8 |
| 更大网格 | 40×40、48×48 |

原始 HDF5 的 sample / trajectory 按固定种子划分为 80% / 10% / 10%，不同时间、物理量、
裁剪尺寸仍然服从同一个 sample 划分。训练形状均衡抽样；验证和测试每种形状的状态数相等。
每个验证状态包含五类题目。区域固定为 4×4，点比较最小间隔 0.5 z，区域均值比较最小间隔 0.2 z。
训练保留原有每状态 9 条、每组三条的 matched QA 组织。

最佳 checkpoint 只按**训练中出现过的形状**的验证子集选择。完整验证分别报告
`seen`、`heldout`、`extrapolation` 和“形状 × 任务”；后两类不进入模型选择。
测试集默认不评估，最终配置确定后再打开。此处变化的是同一源网格上的裁剪大小和长宽比，
不等同于不同物理分辨率或三维、四维张量实验。

同一个 batch 内 H、W 完全一致，不在编码前补零，不做 resize、插值或池化。
三卡同一步使用同一种形状，保留完整 matched group。默认配置每个 epoch 的分桶补齐数为零；
修改 batch size / 卡数后如果需要重复完整组，会写入 `run_contract.json` 的
`parameters.shape_padding_records_per_epoch`。验证按实际样本划分，不补齐、不重复计数。

## 1. 本机上传 GitHub（PowerShell）

下面创建独立实验分支。第一次执行 `git switch -c`；分支已经存在时用
`git switch experiment/variable-shape`。提交内容包含代码、配置、测试和这份说明。

```powershell
Set-Location 'E:\Projects\中石油国重项目\tensor compression2.0'
git status --short
git switch -c experiment/variable-shape
git add .gitignore README.md VARIABLE_SHAPE_RUNBOOK.md configs/field_to_llm_variable_shape.yaml scripts/build_variable_shape_qa.py scripts/build_tensor_patch_matched_qa.py scripts/train_tensor_qwen_cross_attention.py scripts/summarize_variable_shape_run.py src/tensor_compression/downstream/variable_shape.py src/tensor_compression/downstream/patch_qa_prompt.py tests/test_variable_shape_fields.py
git diff --cached --stat
git diff --cached --check
git commit -m "Add variable-shape field QA pilot and full training"
git push -u origin experiment/variable-shape
git rev-parse HEAD
```

远端仍是当前项目的 `git@github.com:lszsnmlmzsby/tensor-compression.git`。
记录最后输出的提交 SHA，服务器应使用同一个 SHA。

## 2. 服务器获取代码并检查环境（Bash）

通过你已有的 SSH 连接方式登录 `lvhong`。建议在 tmux 中运行，避免终端断开影响任务：

```bash
tmux new -s variable_shape
```

然后执行：

```bash
set -euo pipefail
cd /home/lpr/wyx/tensor-compression
git status --short
git fetch origin
git switch --track origin/experiment/variable-shape
git pull --ff-only
git rev-parse HEAD

conda activate tcenv
export FIELD_TO_LLM_ROOT=/data/wyx
export FIELD_TO_LLM_HF_HOME=/data/wyx/hf_cache
export FIELD_TO_LLM_RUNS_DIR=/data/wyx/tensor_llm_outputs/runs
export FIELD_TO_LLM_MODEL_DIR=/data/wyx/tensor_llm_assets/models/Qwen2.5-14B-Instruct
export PDEBENCH_HDF5=/data/PiERN/PDEbench/data/2d-ns/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
unset FIELD_TO_LLM_VARIABLE_QA_DIR
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$FIELD_TO_LLM_RUNS_DIR/launch_logs"

test -f "$PDEBENCH_HDF5"
test -f "$FIELD_TO_LLM_MODEL_DIR/config.json"
nvidia-smi
python -c 'import torch, transformers, h5py, yaml; print("torch", torch.__version__, "transformers", transformers.__version__, "cuda", torch.cuda.is_available())'
python -m pytest tests/test_variable_shape_fields.py tests/test_tensor_qwen_cross_attention.py tests/test_tensor_patch_matched_qa.py -q
```

服务器分支已存在时，把 `git switch --track ...` 替换为 `git switch experiment/variable-shape`。
Git 如果报告本地修改冲突，先保留这些修改再切换；这里不执行清理或强制覆盖。
沿用成功完成上一轮训练的 `tcenv`，无需重新安装 CUDA / 模型。如果仅缺 pytest：
`python -m pip install 'pytest>=8'`。此扩展没有新增训练依赖。

## 3. 试探性小实验

构建一次 pilot 数据集：

```bash
python scripts/build_variable_shape_qa.py \
  --config configs/field_to_llm_variable_shape.yaml \
  --profile pilot \
  2>&1 | tee "$FIELD_TO_LLM_RUNS_DIR/launch_logs/build_variable_pilot.log"
```

成功时末尾应出现 `completed dataset=... train_records=12096`。
构建器直接读取原始数据，保存 JSONL / metadata，不生成 latent 文件。
逐条 QA 会按实际 FP16 z 值重算标签；每个状态的数值 hash 还会在训练启动时重新核对。

开始训练（默认沿用 GPU 4、5、6）：

```bash
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 \
  scripts/train_tensor_qwen_cross_attention.py \
  --config configs/field_to_llm_variable_shape.yaml \
  --profile pilot \
  2>&1 | tee "$FIELD_TO_LLM_RUNS_DIR/launch_logs/train_variable_pilot_$(date +%Y%m%d_%H%M%S).log"
```

启动日志应显示 `train/val/test/screen=12096/2560/2560/350`、
`padding_records_per_epoch=0` 和 `planned_updates=1500`。
训练目录会自动生成时间戳，例如 `.../runs/<timestamp>_variable_shape_pilot`。
按 `Ctrl+B`，再按 `D` 脱离 tmux；用 `tmux attach -t variable_shape` 回来。

训练结束后，选择最新的 pilot 目录并查看结果：

```bash
export RUN_DIR="$(python -c 'import os; from pathlib import Path; root=Path(os.environ["FIELD_TO_LLM_RUNS_DIR"]); runs=[p for p in root.glob("*_variable_shape_pilot") if p.is_dir()]; print(max(runs, key=lambda p:p.name))')"
printf '%s\n' "$RUN_DIR"
python scripts/summarize_variable_shape_run.py --run-dir "$RUN_DIR" --csv "$RUN_DIR/val_by_shape.csv"
```

检查 `status=complete`、`updates=1500/1500`，确认所有 16 种形状均有输出。
重点查看：矩形方向互换是否有明显差异、32×32 的点读数是否落后、40×40 / 48×48
是否能完成推理。pilot 精度偏低可以作为后续诊断线索，不能直接判定全量模型无法学习。

## 4. 全量训练

pilot 的流程检查通过后，生成独立 full 数据集并启动全量训练：

```bash
python scripts/build_variable_shape_qa.py \
  --config configs/field_to_llm_variable_shape.yaml \
  --profile full \
  2>&1 | tee "$FIELD_TO_LLM_RUNS_DIR/launch_logs/build_variable_full.log"

CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 \
  scripts/train_tensor_qwen_cross_attention.py \
  --config configs/field_to_llm_variable_shape.yaml \
  --profile full \
  2>&1 | tee "$FIELD_TO_LLM_RUNS_DIR/launch_logs/train_variable_full_$(date +%Y%m%d_%H%M%S).log"
```

启动日志应显示 `143721/40000/40000/3500` 和 `planned_updates=31938`。
不要加 pilot 的 `--resume`：全量训练使用独立初始化和完整训练日程。
已有同名 QA 目录时直接复用，跳过构建步骤；构建器会拒绝覆盖非空目录。
失败构建留下 `.build_in_progress.json`，训练会拒绝读取；修复原因后用一个新目录重建：

```bash
python scripts/build_variable_shape_qa.py --profile full --output-dir /data/wyx/data/variable_shape_qa/full_v2
# 对应训练命令额外加：--qa-dir /data/wyx/data/variable_shape_qa/full_v2
```

汇总全量验证结果：

```bash
export RUN_DIR="$(python -c 'import os; from pathlib import Path; root=Path(os.environ["FIELD_TO_LLM_RUNS_DIR"]); runs=[p for p in root.glob("*_variable_shape_full") if p.is_dir()]; print(max(runs, key=lambda p:p.name))')"
printf '%s\n' "$RUN_DIR"
python scripts/summarize_variable_shape_run.py --run-dir "$RUN_DIR" --csv "$RUN_DIR/val_by_shape.csv"
```

`final_val_metrics.json` 中的 `modes.correct.by_shape_partition` 提供三类形状汇总；
`by_shape` 和 `by_shape_task` 提供逐形状、逐任务结果。CSV 中准确率取值为 0～1，终端为百分数。
新数据集的抽样和题目不同，结果不应直接当作对旧 16×16 实验的严格配对提升。

## 5. 断点恢复、仅评估和最终测试

保持 `RUN_DIR` 指向需要恢复的**具体目录**；使用前打印检查。下面以 full 为例，
pilot 对应改为 `--profile pilot`。每 250 次更新保存 `cross_attention_last.pt`，
最佳模型单独保存为 `cross_attention_best.pt`。恢复应使用 last。

```bash
printf '%s\n' "$RUN_DIR"
test -f "$RUN_DIR/cross_attention_last.pt"
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 \
  scripts/train_tensor_qwen_cross_attention.py \
  --config configs/field_to_llm_variable_shape.yaml \
  --profile full \
  --resume "$RUN_DIR/cross_attention_last.pt" \
  --max-wall-clock-hours 48 \
  2>&1 | tee "$RUN_DIR/resume_$(date +%Y%m%d_%H%M%S).log"
```

时间上限包含 checkpoint 已记录的累计耗时，因此原先因为 24 小时上限停止时，应提高到例如 48。
恢复会核对数据 hash、形状集合、profile、卡数、batch、优化器和学习率日程；
不能通过改 profile、训练步数或卡数把旧运行冒充成另一个实验。
如果只需降低最终评估的显存，可加 `--eval-batch-size 1`，不改变训练日程。

只重跑最佳模型的验证，不增加训练更新：

```bash
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 \
  scripts/train_tensor_qwen_cross_attention.py \
  --config configs/field_to_llm_variable_shape.yaml \
  --profile full \
  --resume "$RUN_DIR/cross_attention_last.pt" \
  --evaluate-only --eval-batch-size 1
```

最终实验配置确定后，运行最佳模型的验证和测试：

```bash
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 \
  scripts/train_tensor_qwen_cross_attention.py \
  --config configs/field_to_llm_variable_shape.yaml \
  --profile full \
  --resume "$RUN_DIR/cross_attention_last.pt" \
  --evaluate-only --evaluate-test
python scripts/summarize_variable_shape_run.py --run-dir "$RUN_DIR" --split test --csv "$RUN_DIR/test_by_shape.csv"
```

请保留或发回 `run_summary.json`、`final_val_metrics.json`、`val_by_shape.csv`、
`run_contract.json` 和 `train_metrics.jsonl`；测试运行后增加 `final_test_metrics.json`。
它们足够判断学习是否完成、问题集中在哪些形状或任务，不需要传输 Qwen 权重。
