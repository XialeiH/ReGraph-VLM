# Repetition/Familiarity Results Snapshot

Date: 2026-05-02

This snapshot summarizes the current NSD HCP-MMP ROI graph line. Heavy data artifacts, checkpoints, beta volumes, and `.pt` datasets are intentionally excluded from Git. Lightweight CSV/JSON/MD result artifacts are included under `results/hpc_repetition_familiarity/`.

## Dataset

- Node set: 180 HCP-MMP ROI parcels.
- Trial feature set: scalar4 ROI summaries: `mean_beta`, `std_beta`, `q90_beta`, `positive_fraction`.
- Repetition dataset: strict T=3 repeated presentations.
- Smoke folds: `fold_01` and `fold_04`.
- Pair task: same-image repeat matching with same-subject, repeat-pair-matched negatives.

## Neuroscience Analyses

### ROI Repetition Effects

- FDR-significant ROIs for repeat2-repeat1: 41.
- FDR-significant ROIs for repeat3-repeat1: 56.
- Mean absolute real delta21: 10.90.
- Mean absolute shuffled delta21: 2.56.
- Mean absolute real delta31: 14.33.
- Mean absolute shuffled delta31: 2.48.

Interpretation: trial-level ROI responses change systematically from first to repeated viewing, with stronger effects by the third presentation.

### Same-Image Representational Stability

Flattened `[180, 4]` ROI graph features show consistent same-image repeat structure.

| Repeat pair | Same-image similarity | Different-image similarity | Gap |
|---|---:|---:|---:|
| 1-2 | 0.8916 | 0.8305 | 0.0612 |
| 1-3 | 0.8851 | 0.8296 | 0.0555 |
| 2-3 | 0.8871 | 0.8292 | 0.0579 |

Interpretation: same-image repeats preserve image-specific ROI graph structure beyond different-image controls.

### Graph Reconfiguration

- Global repeat-specific adjacency matrices are stable.
- Local repeat-related edge changes exist.
- Fold-consistent train-only changed edges: 9 total.
- Fold-consistent delta21 edges: 3.
- Fold-consistent delta31 edges: 6.

Interpretation: repeated viewing does not globally rewire the ROI graph, but local edge-level changes are measurable.

## Pair-Matching Baselines

### Raw Similarity

Best raw baseline: `pearson_flat`.

| Fold | AUROC | AUPRC | R@1 | R@5 | MRR |
|---|---:|---:|---:|---:|---:|
| fold_01 | 0.6410 | 0.6555 | 0.0196 | 0.0592 | 0.0465 |
| fold_04 | 0.5953 | 0.6110 | 0.0110 | 0.0453 | 0.0401 |
| mean | 0.6182 | 0.6333 | 0.0153 | 0.0522 | 0.0433 |

### Learned Pair Encoders

| Model | Mean AUROC | Mean AUPRC | Mean R@1 | Mean R@5 | Mean MRR | Current status |
|---|---:|---:|---:|---:|---:|---|
| ROI MLP + BCE | 0.7672 | 0.7336 | 0.0114 | 0.0543 | 0.0442 | Strong binary baseline |
| ROI MLP + BCE+InfoNCE | 0.7778 | 0.7540 | 0.0194 | 0.0758 | 0.0592 | Current strongest baseline |
| Naive GCN | ~0.50 | ~0.50 | low | low | low | Failed |
| BrainGNN Siamese pair encoder | ~0.50 | ~0.50 | low | low | low | Failed |
| GCN/GAT + ROI identity | 0.71-0.72 | 0.68-0.69 | low | 0.038-0.044 | 0.035-0.038 | Better than naive GCN, below ROI MLP |

Interpretation: contrastive ROI MLP is the strongest current repeat-matching baseline. Naive GCN and BrainGNN pooling are poorly matched to aligned ROI-pattern retrieval.

## Repeat-State Classification

Task: first-vs-repeated single-trial ROI graph classification.

| Model/control | Mean AUROC | Mean AUPRC | Notes |
|---|---:|---:|---|
| Session-only MLP | 0.7394 | 0.8551 | Strong session confound |
| ROI MLP | 0.6355 | 0.7638 | Above shuffle but weaker than session-only |
| GCN | 0.5897 | not summarized here | Weak |
| GAT | 0.5798 | not summarized here | Weak |
| BrainGNN pool-ratio variants | 0.50 | ~class prior | Collapsed |

Interpretation: first-vs-repeated is biologically meaningful but session-confounded. It is not the cleanest main modeling target.

## BNT Baseline

Implemented BNT adaptations:

- `bnt_token`: scalar4 ROI tokens with transformer over fixed-order ROIs.
- `bnt_native_*`: sample-specific ROI connection-profile adaptation.
- `token_mlp`: no-transformer capacity control.

Current completed BNT results:

| Model | Readout | ROI ID | Loss | Mean AUROC | Mean AUPRC | Mean R@5 | Mean MRR |
|---|---|---|---|---:|---:|---:|---:|
| BNT-token | flat | normal | BCE+InfoNCE | 0.7693 | 0.7512 | 0.0718 | 0.0584 |
| BNT-token | flat | normal | BCE only | 0.7550 | 0.7281 | 0.0563 | 0.0475 |
| BNT-token | flat | shuffled | BCE+InfoNCE | 0.7712 | 0.7454 | 0.0727 | 0.0589 |
| BNT-token | cls | normal | BCE+InfoNCE | 0.5057 | 0.5043 | 0.0082 | 0.0122 |
| BNT-native hybrid | cls | normal | BCE+InfoNCE | 0.5087 | 0.5101 | 0.0112 | 0.0135 |
| Token MLP | flat | normal | BCE+InfoNCE | 0.7064 | 0.6827 | 0.0393 | 0.0356 |
| Token MLP | flat | none | BCE+InfoNCE | 0.7163 | 0.6910 | 0.0440 | 0.0377 |
| Token MLP | flat | shuffled | BCE+InfoNCE | 0.7228 | 0.6979 | 0.0387 | 0.0367 |

Pending at snapshot time:

- `bnt_token + ROI ID normal + flat + BCE+InfoNCE`, `fold_04`, `seed=33`.
- `bnt_token + no ROI ID + flat + BCE+InfoNCE`, `fold_01`, `seed=11`.

Current BNT interpretation:

- BNT-token with flatten readout is competitive with ROI MLP but does not yet beat it.
- Transformer interaction helps versus the no-transformer token MLP control.
- BCE+InfoNCE improves retrieval relative to BCE-only.
- CLS readout and BNT-native connection-profile adaptations collapse in this scalar4 trial-beta setting.
- ROI-ID necessity is not resolved yet because no-ID and shuffled-ID controls are competitive so far.

## Current Scientific Narrative

Repeated natural-image viewing produces measurable ROI activity changes and preserves image-specific ROI graph structure across repetitions. Raw ROI similarity already captures this effect, and contrastive learning improves repeat matching. Standard graph pooling methods such as BrainGNN fail in the Siamese retrieval setting, while fixed-order ROI transformers are more appropriate but still need targeted ablations. The next model direction should preserve anatomical ROI identity, avoid destructive pooling, and use repeat-aware contrastive objectives.
