# ReGraph-VLM

Repetition-aware graph-vision-language learning for natural-image fMRI.

This project studies repeated natural-image responses in the Natural Scenes Dataset (NSD). Each trial-level fMRI response is represented as a brain graph:

```text
G_{s,i,r} = (V, A, X_{s,i,r})
```

where `s` is subject, `i` is image, `r` is ordinal presentation index, `V` contains 180 HCP-MMP ROI nodes, `X` contains scalar ROI beta-response features, and `A` is a training-only ROI relation matrix.

The main task is same-image cross-repetition brain graph matching: given two trial-level ROI graphs from matched repeat types, predict whether they were elicited by the same natural image.

## Project Goal

The final goal is `ReGraph-VLM`: an anatomy-preserving brain graph encoder aligned with a frozen image/VLM encoder such as CLIP. The model should learn brain graph embeddings that are:

- stable across repeated presentations of the same image,
- less sensitive to subject/session noise,
- aligned with image-level semantic representations,
- interpretable through ROI and edge importance.

The current repository contains the data-processing, neuroscience-analysis, baseline, and BNT-adaptation code used to reach the current proposal state.

## Core Scientific Question

Repeated image viewing changes neural responses, but same-image information should remain partly stable. This project asks:

> Can a graph model learn repeat-stable and eventually cross-subject brain graph representations of natural images while preserving anatomical ROI identity?

## Current Data Formulation

The active dataset line is strict `T=3` repeated-trial NSD ROI graphs.

- Dataset: Natural Scenes Dataset.
- Subjects: 8 NSD subjects.
- Nodes: 180 HCP-MMP ROIs.
- Node features: scalar4 ROI summaries.
- Features per ROI:
  - `mean_beta`
  - `std_beta`
  - `q90_beta`
  - `positive_fraction`
- Smoke folds:
  - `fold_01`
  - `fold_04`
- Pair construction:
  - positives: same subject, same image, different repeats.
  - negatives: same subject, different image, matched repeat-pair type.

Large NSD data, beta volumes, `.pt` datasets, checkpoints, and array artifacts are not stored in this GitHub repository.

## Main Tasks

### 1. Same-Image Cross-Repetition Matching

This is the primary modeling task.

Input:

```text
(G_{s,i,r_a}, G_{s,j,r_b})
```

Target:

```text
y = 1 if i == j else 0
```

Metrics:

- AUROC
- AUPRC
- Recall@1
- Recall@5
- MRR

### 2. First-vs-Repeated Classification

This is an auxiliary diagnostic, not the main task. It is graph-level single-trial classification:

```text
repeat1 vs repeat2/repeat3
```

This task is useful for checking repetition/familiarity signal, but it is session/order-confounded in the current experiments.

### 3. Future Cross-Subject Matching

The planned extension uses same-image positives across different subjects:

```text
G_{s,i,r} <-> G_{s',i,r'},  s != s'
```

This tests whether the model can learn subject-invariant brain graph responses to the same natural image.

### 4. Future Brain-Image Alignment

The final ReGraph-VLM model will align brain graph embeddings with frozen image embeddings:

```text
z_graph = f_graph(G)
z_image = f_CLIP(I)
```

This adds visual-semantic grounding to repeat-stable brain graph learning.

## Baseline Hierarchy

The proposal uses the following baseline hierarchy:

1. Raw ROI similarity.
2. ROI MLP pair encoder.
3. GCN/GAT graph encoders.
4. BrainGNN as a classical interpretable fMRI GNN baseline.
5. BNT-style fixed-order ROI transformer baseline.
6. ReGraph-VLM as the planned final model.

## Current Results Snapshot

The current result summary is in:

- `reports/repetition_familiarity_results_snapshot_2026-05-02.md`
- `reports/repetition_familiarity_results_snapshot_2026-05-02.json`

### Neuroscience Findings

Strict `T=3` repeated-response data is feasible and clean.

Key observations:

- 41 ROIs show FDR-significant repeat2-repeat1 effects.
- 56 ROIs show FDR-significant repeat3-repeat1 effects.
- Same-image repeated graphs are more similar than different-image controls.
- Global ROI correlation structure is stable across repeats.
- Local repetition-sensitive edge changes exist.

Same-image representational stability:

| Repeat pair | Same-image similarity | Different-image similarity | Gap |
|---|---:|---:|---:|
| 1-2 | 0.8916 | 0.8305 | 0.0612 |
| 1-3 | 0.8851 | 0.8296 | 0.0555 |
| 2-3 | 0.8871 | 0.8292 | 0.0579 |

### Modeling Findings

The current strongest baseline is ROI MLP with BCE+InfoNCE:

| Model | AUROC | AUPRC | R@5 | MRR |
|---|---:|---:|---:|---:|
| Raw Pearson flat | 0.6182 | 0.6332 | 0.0522 | 0.0433 |
| ROI MLP + BCE | 0.7672 | 0.7336 | 0.0543 | 0.0442 |
| ROI MLP + BCE+InfoNCE | 0.7778 | 0.7540 | 0.0758 | 0.0592 |
| BrainGNN pair encoder | 0.5000 | 0.5013 | 0.0065 | 0.0104 |
| BNT-token + flatten + BCE+InfoNCE | 0.7693 | 0.7512 | 0.0718 | 0.0584 |

Interpretation:

- Raw ROI similarity already contains same-image repeat signal.
- Contrastive learning improves same-image repeat matching.
- Naive GCN/GAT and BrainGNN collapse in the repeat-matching setting.
- BNT-token with flatten readout is the strongest graph baseline so far and nearly matches ROI MLP.
- Transformer interaction appears useful relative to no-transformer token MLP controls.
- CLS/native BNT readouts collapse, suggesting the task requires preserving fixed ROI-token layout.

## Planned ReGraph-VLM Model

The planned model follows the proposal:

```text
ROI scalar features
  -> ROI feature encoder
  -> ROI identity embedding
  -> adjacency-biased ROI transformer / graph attention
  -> anatomy-preserving readout
  -> brain graph embedding
```

Then the brain embedding is trained with:

- within-subject repeat contrast,
- cross-subject same-image contrast,
- brain-image CLIP alignment.

The intended final objective is:

```text
L = L_BCE + lambda_1 L_repeat + lambda_2 L_cross_subject + lambda_3 L_brain_image
```

## Repository Layout

```text
scripts/
  inspect_repetition_inventory.py
  export_trial_level_hcp_roi_scalar4.py
  build_repetition_t3_dataset.py
  analyze_roi_repetition_suppression.py
  analyze_repeat_representational_stability.py
  analyze_repeat_graph_reconfiguration.py
  run_repeat_pair_similarity_baseline.py
  run_repeat_pair_encoder_fold.py
  run_repeat_pair_braingnn_fold.py
  run_repeat_state_baseline_fold.py
  summarize_repeat_pair_encoder_results.py
  summarize_repeat_state_baselines.py

models/
  bnt_encoder.py

external/
  BrainGNN_Pytorch/

reports/
  repetition_familiarity_results_snapshot_2026-05-02.md
  repetition_familiarity_results_snapshot_2026-05-02.json

results/
  hpc_repetition_familiarity/
  version0_shared_unit/
```

## Reproducibility Notes

The code assumes NSD-derived artifacts exist on HPC under a project root similar to:

```text
/scratch/xh2906/final_project_nsd/v0_shared_unit/
```

Example repeat-pair encoder command:

```bash
python scripts/run_repeat_pair_encoder_fold.py \
  --root /scratch/xh2906/final_project_nsd/v0_shared_unit \
  --fold fold_01 \
  --model bnt_token \
  --readout flat \
  --roi-id-mode normal \
  --loss-mode bce_infonce \
  --seed 11
```

Summarize repeat-pair encoder results:

```bash
python scripts/summarize_repeat_pair_encoder_results.py \
  --root /scratch/xh2906/final_project_nsd/v0_shared_unit
```

## What Is Not Included

This GitHub repository intentionally excludes:

- NSD raw data,
- beta volumes,
- generated `.pt` datasets,
- checkpoints,
- large `.npy/.npz` arrays,
- local virtual environments,
- HPC scratch-only intermediate artifacts.

Only source code, project documents, and lightweight result summaries are tracked.

## Proposal Alignment

This repository corresponds to the proposal:

```text
ReGraph-VLM: Repetition-Aware Graph-Vision-Language Learning for Natural-Image fMRI
```

The current stage has established:

1. repeated natural-image responses show measurable ROI and graph-level structure,
2. same-image repeat matching is a clean and learnable graph representation task,
3. existing GCN/GAT/BrainGNN baselines are not sufficient,
4. BNT-style fixed-order ROI transformers are the strongest graph baseline so far,
5. the next method step is ReGraph-VLM with anatomy-preserving graph encoding and CLIP/VLM alignment.
