# Bridge-Sensitive Synthetic Generator

This repository implements a bridge-sensitive synthetic graph suite built from scratch for testing whether explicit perception-then-reasoning outperforms direct graph reasoning on mesoscopic and global structure.

## What is implemented

- A three-layer graph generator: fine primitives -> mesoscopic cells -> coarse skeleton.
- A frozen master-corpus format with graph labels and ground-truth mesoscopic annotations.
- Preliminary validation probes for structural validity, annotation consistency, local-shortcut leakage, split sanity, and size control.
- Training and evaluation entry points for direct GNNs, generic pooling baselines, and a perception-then-reasoning prototype.

## Quick start

```bash
python3 -m pip install -e .[dev]
python3 -m bridgegen.generate_corpus --config configs/bridge_v1.yaml
python3 -m bridgegen.run_preliminary --config configs/bridge_v1.yaml
python3 -m bridgegen.run_pilot --config configs/pilot.yaml
```

## HPC pipeline

On Torch HPC, use the project-local bootstrap and Slurm scripts from a fresh remote project directory:

```bash
bash scripts/hpc_bootstrap.sh
bash scripts/hpc_submit_pipeline.sh
```

The pipeline submits:

- `hpc_generate_prelim.sbatch`
- `hpc_pilot.sbatch`
- `hpc_pilot_gate.sbatch`
- `hpc_full_matrix_array.sbatch` only if the pilot gate passes

## Local smoke path

For fast local validation, use the smoke configs instead of the full corpus:

```bash
python3 -m pytest tests/test_generator.py tests/test_models.py
python3 -m bridgegen.generate_corpus --config configs/smoke_bridge.yaml
python3 -m bridgegen.run_pilot --config configs/smoke_pilot.yaml
```

If your macOS Python blocks compiled `scikit-learn` extensions, run the preliminary probe from a local virtualenv instead:

```bash
.venv/bin/python -m bridgegen.run_preliminary --config configs/smoke_bridge.yaml
```

## Dataset contract

Each `torch_geometric.data.Data` sample contains:

- `edge_index`
- `x`
- `num_nodes`
- `y_topology`
- `y_bridge_count`
- `y_attack_disconnect`
- `y_redundant`
- `cell_id`
- `cell_role`
- `coarse_edge_index`
- `critical_cell_mask`
- `critical_edge_mask`
- `meta`

## Academic anchors

- DiffPool: [Ying et al., 2018](https://arxiv.org/abs/1806.08804)
- MinCutPool: [Bianchi et al., 2020](https://proceedings.mlr.press/v119/bianchi20a.html)
- GraphGPS: [Rampasek et al., 2022](https://arxiv.org/abs/2205.12454)
- Size generalization: [Yehudai et al., 2021](https://proceedings.mlr.press/v139/yehudai21a.html)
- Oversquashing / bottlenecks: [Alon and Yahav, 2020](https://arxiv.org/abs/2006.05205)
- Long-range evaluation context: [Dwivedi et al., 2022](https://arxiv.org/abs/2206.08164)
