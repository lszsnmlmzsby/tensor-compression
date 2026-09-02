# Field-to-LLM

Official implementation of the paper's interface for querying two-dimensional
scientific fields with a frozen large language model (LLM).

The released pipeline maps each `16 x 16` field patch to 256 grid-aligned field
tokens, preserves the normalized scalar channel up to FP16 storage precision,
and exposes the complete field to selected frozen-LLM blocks through
cross-attention. The LLM receives only the natural-language QA prompt; parsed
coordinates, structured task IDs, oracle values, and task-specific readout
heads are not model inputs.

## Method

The paper pipeline has two training stages:

1. **Field-text alignment.** A trainable value-preserving 2D encoder and spatial
   adapter produce 256 grid-aligned tokens. Their shallow frozen-LLM hidden
   states are aligned with hidden states from a textual serialization of the
   same normalized field. The encoder remains trainable during alignment.
2. **Field QA reasoning.** Stage 2 begins with a one-epoch Direct-QA warm start,
   where the aligned spatial adapter acts as a 256-token prefix. The final phase
   freezes that spatial backbone and exposes the full 256-cell memory at LLM
   blocks 8, 20, and 32 while training the cross-attention bridges and continuous
   scalar-value branch.

The five QA tasks are:

- `normalized_point_value`: choose the standardized value at one grid cell.
- `raw_point_value_with_stats`: recover a raw value using the supplied patch
  mean and scale.
- `point_compare`: compare values at two cells.
- `region_mean_compare`: compare the means of two `4 x 4` regions.
- `extreme_quadrant`: locate the quadrant containing a requested extreme.

## Paper Results

The test set contains 5,035 questions from 1,007 sample-disjoint field states.
All results below use the same frozen Qwen2.5-14B-Instruct backbone.

| Interface | Macro | Extreme quadrant | Normalized point | Point compare | Raw point | Region compare |
|---|---:|---:|---:|---:|---:|---:|
| Serialized field | 53.11 | 47.17 | 70.21 | 63.46 | 27.71 | 57.00 |
| Cross-attention field interface | **90.39** | **96.13** | **89.37** | **97.42** | **71.10** | **97.91** |

The paired three-A800 inference benchmark reports:

| Metric | Serialized field | Cross-attention interface |
|---|---:|---:|
| Mean discrete prompt tokens | 2,856.97 | 225.44 |
| End-to-end latency per question | 164.45 ms | 14.19 ms |
| Throughput | 6.08 questions/s | 70.49 questions/s |
| Incremental inference memory | 1.408 GiB | 0.128 GiB |
| Total host-to-device payload (5,035 questions) | 222.10 MiB | 59.06 MiB |

This corresponds to `+37.28` macro-accuracy points and `11.59x` faster
end-to-end inference. The 225.44-token value counts only discrete textual prompt
tokens: the 256 continuous field-memory positions enter through cross-attention
and are intentionally not counted as tokenizer tokens. In the separate
representation-length audit, 256 field tokens replace 1,538 serialized numeric
tokens on average at two decimal places, an 83.36% reduction.

## Repository Layout

```text
configs/
  field_to_llm_stage1.yaml          # Stage 1 and QA-data construction
  field_to_llm_direct_qa.yaml       # Direct-QA memory initialization
  field_to_llm_cross_attention.yaml # Final paper model
  field_to_llm_cross_attention_scratch.yaml # Direct-Cross scratch ablation
  field_to_llm_benchmark.yaml       # Paired accuracy and cost benchmark
scripts/
  train_tensor_patch_text_alignment.py
  build_tensor_patch_qa.py
  build_tensor_patch_matched_qa.py
  train_tensor_llm_adapter.py
  train_tensor_qwen_cross_attention.py
  evaluate_frozen_qwen_patch_qa.py
  benchmark_tensor_qwen_inference.py
src/tensor_compression/              # Shared field, model, prompt, and contract code
tests/                               # Tests for the released paper pipeline
```

Some internal filenames, prompt/metadata identifiers, and checkpoint contract
strings retain the historical words `tensor` and `dense`. They are kept only
for compatibility with the paper checkpoints. In the method and public
interface, these concepts are called **field** and **cross-attention**,
respectively. The Direct-QA trainer also retains shared runtime and legacy
checkpoint-loading paths imported by the final trainer; the release
configurations never select the superseded model variants.

## Installation

Python 3.10 or newer and CUDA-capable PyTorch are recommended. The paper runs
used PyTorch 2.5.1 with CUDA 12.4, Transformers 5.12.1, and NVIDIA A800 80GB
GPUs.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Weights can be downloaded by Transformers from
`Qwen/Qwen2.5-14B-Instruct`. For an offline model copy, set `model.local_dir` in
the release configurations to its absolute path. The Direct-QA and
cross-attention entry points also accept `FIELD_TO_LLM_MODEL_DIR`, which takes
precedence without requiring a tracked configuration edit.

## Data And Paths

The experiments use the PDEBench 2D CFD training HDF5 file
`2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5`. It must contain
the datasets `density`, `pressure`, `Vx`, and `Vy`. The official PDEBench
manifest lists the download URL
`https://darus.uni-stuttgart.de/api/access/datafile/164685` and MD5
`844555000d342d2947162c6cf46798e7` for this file.

Set a writable experiment root and the source HDF5 path:

```bash
export FIELD_TO_LLM_ROOT=/absolute/path/to/field_to_llm_assets
export PDEBENCH_HDF5=/absolute/path/to/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
mkdir -p "$FIELD_TO_LLM_ROOT"/{data,hf_cache,runs}
```

For machines whose existing assets do not share that directory layout, the
Direct-QA and cross-attention entry points accept the following optional path
overrides while retaining `FIELD_TO_LLM_ROOT` as the fallback:

```bash
export FIELD_TO_LLM_HF_HOME=/absolute/path/to/hf_cache
export FIELD_TO_LLM_RUNS_DIR=/absolute/path/to/runs
export FIELD_TO_LLM_MODEL_DIR=/absolute/path/to/Qwen2.5-14B-Instruct
export FIELD_TO_LLM_DIRECT_QA_DIR=/absolute/path/to/direct_qa
export FIELD_TO_LLM_MATCHED_QA_DIR=/absolute/path/to/matched_qa
export FIELD_TO_LLM_LATENT_DIR=/absolute/path/to/patch_latents
```

Verify the source data before training:

```bash
echo "844555000d342d2947162c6cf46798e7  $PDEBENCH_HDF5" | md5sum -c -
```

Generated data, model weights, checkpoints, and run outputs are deliberately
excluded from Git.

## Run The Paper Pipeline

The commands below implement the paper training recipe. Per-rank batch sizes
remain configurable through the YAML files and corresponding CLI flags.

### 1. Train field-text alignment

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_tensor_patch_text_alignment.py \
  --config configs/field_to_llm_stage1.yaml
```

Point the following steps to the selected Stage 1 checkpoint:

```bash
export FIELD_TO_LLM_ALIGNMENT_CHECKPOINT="$FIELD_TO_LLM_ROOT/runs/<stage1-run>/alignment_best.pt"
```

### 2. Build QA and field latents

```bash
python scripts/build_tensor_patch_qa.py \
  --config configs/field_to_llm_stage1.yaml

python scripts/build_tensor_patch_matched_qa.py \
  --config configs/field_to_llm_stage1.yaml
```

The first command writes the value-preserving `[8, 16, 16]` latent cache and
base QA. The second creates matched questions while replay-validating every
answer against the stored FP16 scalar channel.

### 3. Train the Direct-QA warm start

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_tensor_llm_adapter.py \
  --config configs/field_to_llm_direct_qa.yaml
```

```bash
export FIELD_TO_LLM_DIRECT_CHECKPOINT="$FIELD_TO_LLM_ROOT/runs/<direct-qa-run>/adapter_best.pt"
```

### 4. Train the final cross-attention model

The paper's final run used three A800 80GB GPUs and an eight-hour wall-clock
budget:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  scripts/train_tensor_qwen_cross_attention.py \
  --config configs/field_to_llm_cross_attention.yaml
```

```bash
export FIELD_TO_LLM_CROSS_CHECKPOINT="$FIELD_TO_LLM_ROOT/runs/<cross-attention-run>/cross_attention_best.pt"
```

### 5. Run the paired paper benchmark

```bash
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  scripts/benchmark_tensor_qwen_inference.py \
  --config configs/field_to_llm_benchmark.yaml
```

This evaluates serialized-field and cross-attention inference on identical
records and order, then records accuracy, paired bootstrap uncertainty, prompt
length, wall time, throughput, GPU memory, GPU-hours, and transfer volume in
`benchmark_results.json`.

For a standalone frozen-LLM serialized baseline:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  scripts/evaluate_frozen_qwen_patch_qa.py \
  --config configs/field_to_llm_stage1.yaml \
  --splits test
```

## Direct-Cross Scratch Ablation

The primary Stage 1 / QA-warm-start ablation trains the final full-grid field
memory path directly, using
`configs/field_to_llm_cross_attention_scratch.yaml`. Its defining settings are
`data.input_source=raw_hdf5` and `memory.init_mode=scratch`.

This run reuses the existing matched-QA JSONL questions, answers, and
sample-disjoint splits so that its supervision is directly comparable with the
paper run. For every QA state, however, the model input is reread from the
PDEBench HDF5 file and normalized at runtime. The trainer does not open a Stage
1 checkpoint, a Direct-QA checkpoint, or a learned latent-cache file. The field
encoder, full-grid spatial backbone, pointwise scalar-value encoder, and
cross-attention bridges are initialized from scratch and trained together;
Qwen remains frozen. The scratch spatial backbone contains no prefix/soft-prompt
output head. Runtime values are rounded to the same FP16 value space used when
the matched-QA targets were built, and every selected normalized patch is
content-hashed so resuming cannot silently switch HDF5 contents.

The released scratch configuration is a 3,000-update validation screen and
keeps test evaluation disabled. On physical GPUs 4, 5, and 6, run:

```bash
CUDA_VISIBLE_DEVICES=4,5,6 \
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  scripts/train_tensor_qwen_cross_attention.py \
  --config configs/field_to_llm_cross_attention_scratch.yaml
```

Only `FIELD_TO_LLM_ROOT` and `PDEBENCH_HDF5` are required by this configuration
(plus an optional machine-local model path such as `FIELD_TO_LLM_MODEL_DIR`).
The existing matched-QA directory is still required under
`$FIELD_TO_LLM_ROOT/data/matched_patch_qa`. Consequently, this experiment
removes Stage 1 and Direct-QA from the model-training path while holding the QA
dataset fixed; it does not claim that the current matched-QA data-generation
utilities are themselves Stage-1-free.

## Earlier Stage 1 Initialization Diagnostics

The following three-condition experiment predates the Direct-Cross scratch
design. It remains useful for diagnosing which learned Stage 1 components
survive initialization changes, but it still runs a Direct-QA prefix warm start
and therefore is not the primary test of whether both upstream phases can be
removed.

One matched reference and two isolated ablations reuse the release Direct-QA
recipe and a common 3,000-update cross-attention screening budget without
changing the cached files:

- `full_stage1_reference` preserves the learned Stage-1 spatial adapter and all
  latent channels under the same current commit, seed, and downstream budgets.
- `adapter_only` replaces the learned Stage-1 spatial adapter with
  a deterministic random adapter of identical architecture while retaining all
  latent channels.
- `no_learned_stage1` applies the same adapter reset and exposes only latent
  channel 0, the explicitly preserved standardized scalar. Learned channels
  1 through N are zeroed at load time.

Run the Direct and dense phases with the same condition config. Start with the
matched reference shown here, then repeat with each ablation config:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_tensor_qwen_stage1_ablation.py direct \
  --config configs/field_to_llm_stage1_reference.yaml

python -m torch.distributed.run --standalone --nproc_per_node=3 \
  scripts/train_tensor_qwen_stage1_ablation.py dense \
  --config configs/field_to_llm_stage1_reference.yaml \
  --spatial-init-checkpoint "$FIELD_TO_LLM_ROOT/runs/<reference-direct-run>/adapter_best.pt"
```

The two ablation configs are
`configs/field_to_llm_stage1_adapter_ablation.yaml` and
`configs/field_to_llm_no_learned_stage1_ablation.yaml`. Test evaluation is
forbidden in the Direct phase and disabled by default in the dense phase; pass
`--resume <same-run>/cross_attention_last.pt --evaluate-test --protocol-lock
<filled-lock.json>` to the same completed dense run only after the validation
comparison is complete and the protocol is locked. The launcher verifies the
locked validation run summary, data audit, best/last dense checkpoints, Direct
checkpoint, and Direct companion audits before evaluating test metrics; a lock
cannot authorize fresh training. Both trainers still construct all declared
split records and perform integrity audits at startup, so “sealed” here means
that test predictions and metrics are not computed before the lock, not that
test JSONL bytes are never read.
Use `configs/stage1_test_protocol_lock.example.json` as the lock template and
copy the filled lock outside the Git worktree (for example under
`$FIELD_TO_LLM_ROOT/runs/`), recording the validation comparison file's
SHA-256. An interrupted validation phase can be continued with
`--resume <run>/cross_attention_last.pt` while retaining the same lineage and
world/effective-batch contract. Elapsed wall-clock time is cumulative across
resume, so resume recovers an interruption but does not extend the configured
wall-clock budget.

The wrapper records the source checkpoint, source and effective initial adapter
state digests, condition, config hashes, source commit, tracked-diff hash, and
Direct checkpoint SHA. It refuses formal execution from a dirty tracked tree
or while any experiment script/config is absent from the current commit.
The adapter-only condition does **not** establish that the whole Stage 1 is
unnecessary because its learned encoder channels remain active. A negative
result from either screening condition also requires a clean shared-initializer
fork to separate representation benefit from extra pretraining compute.

After all three dense validation runs complete, audit invariants and apply the
pre-registered non-inferiority guards:

```bash
python scripts/compare_tensor_qwen_stage1_ablation.py \
  --reference "$FIELD_TO_LLM_ROOT/runs/<reference-dense-run>" \
  --adapter-only "$FIELD_TO_LLM_ROOT/runs/<adapter-only-dense-run>" \
  --no-learned-stage1 "$FIELD_TO_LLM_ROOT/runs/<no-learned-dense-run>" \
  --split validation \
  --output "$FIELD_TO_LLM_ROOT/runs/stage1_validation_comparison.json"
```

The comparison is rejected before scoring if the source Stage-1 SHA, tracked
commit, QA-record fingerprints, model contract, trainable parameter count,
record counts, or optimizer-update budget differ across conditions.
Passing this screen is a reason to run a longer fixed-update confirmation, not
by itself a final claim that every possible Stage 1 is unnecessary.

The released Stage-1 artifact binds its embedded encoder state by SHA-256, but
its older upstream compressor checkpoint path has no recorded checkpoint SHA
(`source_encoder_lineage_complete=false`). This limits complete historical
reconstruction of that ancestor; it does not weaken the three-condition
comparison because every condition consumes the same byte-identical embedded
encoder state.

## Paper Checkpoint Provenance

The paper artifacts are identified by SHA-256 rather than machine-specific
paths:

| Artifact | SHA-256 |
|---|---|
| Stage 1 `alignment_best.pt` | `cadad31b15bf448d2c37a2d35d47bd3deabc9304d8e9388658815ced42202bb3` |
| Direct-QA `adapter_best.pt` | `154fca725763dcb308a37c2f3320bb0a5b8f3fb9c1f03f5b554735e7e6bef989` |
| Final `cross_attention_best.pt` | `7893fa55938798b9476c75dff17b2f2a49432bce4430ae4271646b856f4fa60a` |

The builders and trainers also record and verify these identities in QA
metadata and checkpoint contracts, so relocating an artifact does not change
its provenance.

## Tests

```bash
python -m pytest -q
```

The release tests cover normalization and configuration, Stage 1 alignment,
QA/latent contracts, matched-QA replay, Direct-QA configuration, the frozen
serialized baseline, cross-attention, distributed evaluation, and the paired
inference benchmark.

## Scope

The released experiments cover single-variable `16 x 16` patches sampled from
four PDEBench fields and restricted-choice QA. Extending the interface to larger
grids, multivariate or temporal fields, and open-ended scientific reasoning is
future work.
