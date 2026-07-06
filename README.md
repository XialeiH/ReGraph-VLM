# ReGraph-VLM

Minimal code release for the ReGraph-VLM paper.

This repository intentionally contains only source code needed for the paper:

- `models/`: fixed-order ROI-token encoders and ReGraph-VLM modules.
- `scripts/run_regraph_vlm_fold.py`: main training/evaluation entry point.
- `scripts/build_*` and `scripts/export_*`: dataset construction utilities for NSD-style ROI features and LAION-fMRI validation.
- `scripts/analyze_*`, `scripts/run_*`, and `scripts/make_publication_stat_tests.py`: mechanism checks used in the manuscript, including ROI-order/gate controls, adjacency controls, gate analysis, deletion tests, leakage QC, and summary statistics.

No raw fMRI data, derived datasets, checkpoints, generated result tables, figures, reports, or manuscript documents are stored in this repository. Those artifacts should be generated or stored outside Git according to the original dataset access rules.

## Install

```bash
python -m pip install -e .
```

Optional CLIP/image and neuroimaging utilities are declared as extras:

```bash
python -m pip install -e ".[clip,neuro]"
```

## Basic Checks

```bash
make syntax
make parameter-counts
```

`make syntax` compiles all Python files. `make parameter-counts` instantiates the main model variants and prints trainable parameter counts.

## Main Training Entry Point

```bash
python scripts/run_regraph_vlm_fold.py \
  --root /path/to/workdir \
  --fold fold_01 \
  --dataset-root /path/to/scalar4_T3_clip \
  --output-root /path/to/results \
  --graph-encoder roi_transformer_noadj \
  --readout gated_flat \
  --lambda-clip 2.0
```

Expected dataset files are not bundled. The run script expects fold directories containing serialized train/validation/test pair files, CLIP embeddings, and optional adjacency files produced by the dataset-building scripts.

## Scope

The supported paper claim is fixed anatomical ROI-token modeling with learned interactions, gated ROI-preserving readout, and CLIP alignment. The code includes adjacency and edge-bias controls because the manuscript reports that explicit fixed adjacency is not the main source of the gain.
