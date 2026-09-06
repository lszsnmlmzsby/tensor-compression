# 可变形状混合场实验 v2

本轮扩展真实场和合成场的覆盖，提高独立训练状态数和更新预算。继续使用冻结的 Qwen2.5-14B、完整网格 memory 和原有损失权重，默认只评估正常输入。旧配置 `configs/field_to_llm_variable_shape.yaml`、旧 QA 和旧 checkpoint 保留为 v1 基线。本轮从头训练，使用 `configs/field_to_llm_variable_shape_mixed.yaml`。

## 数据、形状与训练预算

每种形状按状态数精确分配：50% PDEBench，25% 空间相关合成场，25% IID 合成场。真实部分四个物理量各占总状态数的 12.5%；合成部分标记为 `scalar`，不冒充物理量。相关场混合波、局部峰、坡度和弱噪声；IID 混合高斯和均匀分布。合成均值均匀采样于 [-10,10]，尺度在 [0.01,100] 对数均匀采样。这是本轮预先选定的起始比例，不代表最优比例。

这里的“随机”是固定种子、预先生成的随机样本。实际 float32 合成输入写入 QA 目录下的 `synthetic_fields.hdf5`，附文件 hash；训练时读取这些输入。真实和合成输入都重新计算逐场 mean/std，然后按 FP16 舍入并逐题验证答案，跨机器不依赖重新生成随机数组。`raw` 题的目标仍是题面 mean/scale 对保存的 z 值进行反标准化，遵循既有 QA 数值约定。

真实轨迹仍用 v1 的 `split_seed=20260905` 按 80/10/10 划分，同一轨迹的时间、物理量和裁剪共用划分。合成状态使用独立的 split 命名空间和种子。两轮训练重复同一套训练状态，不在验证时临时生成数据。

| 项目 | pilot | full |
|---|---:|---:|
| 训练形状 | 32 | 32 |
| 每形状训练状态 | 64 | 2,048 |
| 独立训练状态 | 2,048 | 65,536 |
| 训练 QA，每状态 9 条 | 18,432 | 589,824 |
| 三卡、每卡 batch=3、累积=1 的更新数 | 3,000 | 131,072 |
| epoch 上限 | 2 | 2 |
| 每形状验证/测试状态 | 16 | 256 |
| 完整验证 QA；测试集同规模 | 3,520 | 56,320 |
| 固定筛选验证 QA | 1,280 | 5,120 |
| 最终评估每卡 batch | 1 | 1 |
| 累计时间上限 / 末尾评估预留 | 6 小时 / 60 分钟 | 48 小时 / 180 分钟 |

full 更新数是上轮 31,938 的约 4.1 倍。实际耗时还受较大网格影响，不能只按更新数比例预测；48 小时是停止上限，不是预计耗时。三卡默认配置不需要重复补齐训练组，`shape_padding_records_per_epoch=0`。改变卡数或 batch 会改变更新预算，实验中途保持不变。

训练形状的单轴范围为 8～96，面积不超过 2048，包含正方形、长方形、转置、奇数尺寸、细长网格，例如 `9×17`、`15×31`、`15×88`、`19×80`、`16×96`、`21×95` 及转置。完整列表写在配置中。

| 评估分组 | 形状 | 解释 |
|---|---|---|
| seen | 32 种训练形状 | 使用独立轨迹/状态 |
| heldout | 17×90、90×17、13×47、47×13、29×43、43×29、40×40、33×55 | 尺寸组合未见，单轴与面积仍在训练范围内 |
| extrapolation / length | 48×48、64×64 | 坐标在 96 以内，但面积超过训练上限 |
| extrapolation / coordinate | 16×128、128×16 | 面积仍为 2048，但某一轴超出训练坐标范围 |

行、列的绝对正弦二维位置编码保持原实现，每格一个 token；按行展开仅用于排列 token。编码前不补零，不 resize，不池化。奇数尺寸的象限规则直接写进题面：上半部到 `floor(H/2)` 行，下半部从下一行开始，列同理。例如 17×90 的上半部是 1～8 行，下半部是 9～17 行。

每个训练状态仍有 3 条 norm、3 条 raw、2 条空间角色交换题和 1 条极值题。每种来源、每种形状内，一半状态保留原数值匹配组，另一半使用与验证相同的均匀坐标和等间距选项；后一半只作为原子三题 batch，不应用要求“共享选项且答案不同”的坐标 margin。空间角色交换 margin 保留。点比较改为均匀候选位置、差值至少 0.1 z，区域比较保持 4×4、差值至少 0.2 z。记录实际差值以便分难度分析。归一化和反归一化题使用统一题面表达。

最佳 checkpoint 仅由 seen 形状、固定 50/25/25 来源比例、五类任务均衡的验证子集选择。未见形状与两种外推不参与选择。测试集默认不评估。

## 1. 本机上传代码：PowerShell

当前工作分支为 `experiment/variable-shape`；以下在同一分支追加本轮改动。这里仅给出上传命令，代码修改本身不会自动提交或推送。

```powershell
Set-Location 'E:\Projects\中石油国重项目\tensor compression2.0'
git status --short
git switch experiment/variable-shape
git add .gitignore README.md VARIABLE_SHAPE_MIXED_RUNBOOK.md configs/field_to_llm_variable_shape_mixed.yaml scripts/build_variable_shape_qa.py scripts/mixed_shape_qa.py scripts/train_tensor_qwen_cross_attention.py scripts/summarize_variable_shape_run.py scripts/analyze_field_predictions.py scripts/evaluate_variable_shape_checkpoint.py src/tensor_compression/downstream/variable_shape.py src/tensor_compression/downstream/field_diagnostics.py tests/test_mixed_shape_fields.py
git diff --cached --check
git diff --cached --stat
git commit -m "Expand variable-shape training with mixed fields and prediction diagnostics"
git push -u origin experiment/variable-shape
git rev-parse HEAD
```

记下最后的提交 SHA，服务器更新后应相同。

## 2. 服务器准备：Bash

沿用现有 SSH 连接方式登录 `lvhong`。长实验可先进入 `tmux new -s mixed_shape_v2`。已有该会话时使用 `tmux attach -t mixed_shape_v2`。

```bash
set -euo pipefail
cd /home/lpr/wyx/tensor-compression
git status --short
git fetch origin
git switch experiment/variable-shape
git pull --ff-only origin experiment/variable-shape
git rev-parse HEAD

conda activate tcenv
export FIELD_TO_LLM_ROOT=/data/wyx
export FIELD_TO_LLM_HF_HOME=/data/wyx/hf_cache
export FIELD_TO_LLM_RUNS_DIR=/data/wyx/tensor_llm_outputs/runs
export FIELD_TO_LLM_MODEL_DIR=/data/wyx/tensor_llm_assets/models/Qwen2.5-14B-Instruct
export PDEBENCH_HDF5=/data/PiERN/PDEbench/data/2d-ns/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
unset FIELD_TO_LLM_VARIABLE_QA_DIR
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
mkdir -p "$FIELD_TO_LLM_RUNS_DIR/launch_logs"
test -f "$PDEBENCH_HDF5"
test -d "$FIELD_TO_LLM_MODEL_DIR"
nvidia-smi
```

GPU 编号沿用 4、5、6。使用现有已成功训练的 tcenv，无新增运行依赖。

## 3. pilot：覆盖全部形状的试跑

```bash
python scripts/build_variable_shape_qa.py \
  --config configs/field_to_llm_variable_shape_mixed.yaml --profile pilot \
  2>&1 | tee "$FIELD_TO_LLM_RUNS_DIR/launch_logs/build_mixed_pilot.log"

CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 \
  scripts/train_tensor_qwen_cross_attention.py \
  --config configs/field_to_llm_variable_shape_mixed.yaml --profile pilot \
  2>&1 | tee "$FIELD_TO_LLM_RUNS_DIR/launch_logs/train_mixed_pilot_$(date +%Y%m%d_%H%M%S).log"

export PILOT_RUN="$(python -c 'import os; from pathlib import Path; print(max(Path(os.environ["FIELD_TO_LLM_RUNS_DIR"]).glob("*_variable_shape_mixed_v2_pilot"), key=lambda p:p.name))')"
printf '%s\n' "$PILOT_RUN"
python scripts/summarize_variable_shape_run.py --run-dir "$PILOT_RUN" --csv "$PILOT_RUN/val_by_shape.csv"
python scripts/analyze_field_predictions.py "$PILOT_RUN/predictions/final_val.correct.manifest.json" --output "$PILOT_RUN/val_diagnostics.json"
```

启动应显示 `train/val/test/screen=18432/3520/3520/1280`、`planned_updates=3000`。完成应为 `status=complete`、`3000/3000`，44 种形状都应有结果。确认训练最大 2048 格与评估最大 4096 格不发生显存问题、数值损失有限，seen 的 norm/raw 相对初始化有学习趋势。pilot 评估样本少，不用单个形状的几个百分点波动决定方向。

## 4. full：扩大数据与完整训练

```bash
python scripts/build_variable_shape_qa.py \
  --config configs/field_to_llm_variable_shape_mixed.yaml --profile full \
  2>&1 | tee "$FIELD_TO_LLM_RUNS_DIR/launch_logs/build_mixed_full.log"

CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 \
  scripts/train_tensor_qwen_cross_attention.py \
  --config configs/field_to_llm_variable_shape_mixed.yaml --profile full \
  2>&1 | tee "$FIELD_TO_LLM_RUNS_DIR/launch_logs/train_mixed_full_$(date +%Y%m%d_%H%M%S).log"

export FULL_RUN="$(python -c 'import os; from pathlib import Path; print(max(Path(os.environ["FIELD_TO_LLM_RUNS_DIR"]).glob("*_variable_shape_mixed_v2_full"), key=lambda p:p.name))')"
printf '%s\n' "$FULL_RUN"
python scripts/summarize_variable_shape_run.py --run-dir "$FULL_RUN" --csv "$FULL_RUN/val_by_shape.csv"
python scripts/analyze_field_predictions.py "$FULL_RUN/predictions/final_val.correct.manifest.json" --output "$FULL_RUN/val_diagnostics.json"
```

启动应显示 `589824/56320/56320/5120`、`planned_updates=131072`。full 使用独立初始化和新日程。不要附加旧 v1 或 pilot 的 `--resume`。数据生成器拒绝覆盖非空目录；同配置已成功生成时跳过构建命令即可。若构建中断，使用新目录，例如构建加 `--output-dir /data/wyx/data/variable_shape_mixed_v2_qa/full_retry`，训练及续训对应加 `--qa-dir` 指向该目录；不要使用带 `.build_in_progress.json` 的未完成数据。

## 5. 中断后继续同一次 full

将 `FULL_RUN` 设成启动日志中的具体路径。每 1000 次更新保存 last，正常收到中断信号也会保存已提交更新。续训使用相同三卡、配置、QA 和训练预算；可以提高累计时间上限。

```bash
# 若是新终端，先执行第 2 节环境设置，再设置实际目录：
# export FULL_RUN=/data/wyx/tensor_llm_outputs/runs/实际时间戳_variable_shape_mixed_v2_full
printf '%s\n' "$FULL_RUN"
test -f "$FULL_RUN/cross_attention_last.pt"
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 \
  scripts/train_tensor_qwen_cross_attention.py \
  --config configs/field_to_llm_variable_shape_mixed.yaml --profile full \
  --resume "$FULL_RUN/cross_attention_last.pt" --max-wall-clock-hours 72 \
  2>&1 | tee "$FULL_RUN/resume_$(date +%Y%m%d_%H%M%S).log"
```

如果只需重跑本次实验的最终验证，可以在上面命令增加 `--evaluate-only`。它会评估本次运行的 best，不增加训练更新。不要通过修改 `max_updates` 或数据目录延长已绑定日程的实验。

## 6. 同一套题上的旧模型与新模型比较

独立评估入口读取明确指定的 checkpoint，验证架构与参数，再校验独立评估数据。它不恢复优化器、不重新选择 best、不修改原训练结果。还会读取 checkpoint 绑定的原训练 JSONL，检查新评估题与训练轨迹/合成种子不重叠；如果原 QA 迁移了位置，用 `--training-qa-dir` 指定。

下面用 v2 full 验证集同时评估已完成的 v1 和本轮 full，得到可直接比较的逐题结果：

```bash
export OLD_RUN=/data/wyx/tensor_llm_outputs/runs/20260906_012032_variable_shape_full
export COMMON_QA=/data/wyx/data/variable_shape_mixed_v2_qa/full
export COMPARISON_ROOT="$FIELD_TO_LLM_RUNS_DIR/mixed_comparison_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 \
  scripts/evaluate_variable_shape_checkpoint.py \
  --checkpoint "$OLD_RUN/cross_attention_best.pt" \
  --config configs/field_to_llm_variable_shape_mixed.yaml --profile full \
  --qa-dir "$COMMON_QA" --split val --output-dir "$COMPARISON_ROOT/v1"

CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 \
  scripts/evaluate_variable_shape_checkpoint.py \
  --checkpoint "$FULL_RUN/cross_attention_best.pt" \
  --config configs/field_to_llm_variable_shape_mixed.yaml --profile full \
  --qa-dir "$COMMON_QA" --split val --output-dir "$COMPARISON_ROOT/v2"

python scripts/analyze_field_predictions.py "$COMPARISON_ROOT/v1/predictions/val.correct.manifest.json"
python scripts/analyze_field_predictions.py "$COMPARISON_ROOT/v2/predictions/val.correct.manifest.json"
```

也可以将新模型评估在旧验证集上：上面的 `--config` 改成旧 v1 配置、`--qa-dir` 改成 `/data/wyx/data/variable_shape_qa/full`，输出指定一个新目录。旧模型与新模型的数据、训练预算同时发生了变化，这个比较衡量整套扩展的效果，不能单独归因于合成场或训练量。

## 7. 结果判读和最终测试

`final_val_metrics.json` 提供形状、来源、来源×任务、外推类型×任务汇总。`predictions/*.manifest.json` 列出本次完成评估的所有 rank 分片、条数及 hash；分析器只读取清单中的文件，并拒绝重复、缺失或篡改结果。

`val_diagnostics.json` 重点查看：

- `source_task`：真实场的 norm/raw 是否改善，合成两类是否都学会读取，避免只看混合总分。
- `source_shape_task`：同来源、同形状、同任务的表现，特别是 17×90 与其转置。
- `coordinate32_task` 和 `coordinate96_task`：实际被查询的位置/区域是否超过旧 32、新 96 的范围；极值题归为 global。
- `partition_task`：组合泛化、token 数外推、坐标范围外推分别分析。
- `numeric_pairs`：norm 对但 raw 错、两者都错、两者是否选中同一个数值顺序位置。norm/raw 的字母选项重新排列，不能直接比较预测字母。
- `mean_absolute_z_error` 与 `gap_task`：误读幅度和比较难度。选项舍入会使正确题也有很小的数值误差。

最终方案确定后再评估 test。直接用独立入口即可，不需要重新跑训练或验证：

```bash
export TEST_OUTPUT="$FULL_RUN/test_evaluation_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 \
  scripts/evaluate_variable_shape_checkpoint.py \
  --checkpoint "$FULL_RUN/cross_attention_best.pt" \
  --config configs/field_to_llm_variable_shape_mixed.yaml --profile full \
  --qa-dir /data/wyx/data/variable_shape_mixed_v2_qa/full --split test \
  --output-dir "$TEST_OUTPUT"
python scripts/analyze_field_predictions.py "$TEST_OUTPUT/predictions/test.correct.manifest.json"
```

下一次分析请保留 `run_summary.json`、`run_contract.json`、`train_metrics.jsonl`、`final_val_metrics.json`、`val_diagnostics.json` 和逐题预测清单/分片。仅检查总体结果时，前五个文件即可。
