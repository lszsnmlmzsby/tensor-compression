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
the four release configurations to its absolute path.

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
