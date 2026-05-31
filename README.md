# ReGraph-VLM

Fixed-order ROI-token graph/vision alignment for repeated natural-image fMRI.

This repository contains the code and lightweight publication artifacts for the ReGraph-VLM project. The active manuscript is:

```text
reports/neurips_report/may30.tex
```

The project studies repeated natural-image fMRI responses from the Natural Scenes Dataset (NSD). Each trial-level response is represented as a fixed-order HCP-MMP ROI graph:

```text
G_{s,i,r} = (V, A, X_{s,i,r})
```

where `s` is subject, `i` is image, `r` is repetition index, `V` contains 180 HCP-MMP cortical ROIs, `X` contains four scalar ROI beta-summary features, and `A` is a train-only ROI relation matrix used in adjacency-aware variants.

## Current Scientific Framing

The central result is not that explicit fixed adjacency explains the gain. The strongest supported claim is:

```text
Fixed anatomical ROI-token modeling, gated ROI-preserving readout, and image alignment improve cross-subject brain-image retrieval.
```

The no-adjacency gated ROI Transformer is statistically tied with the adjacency-based BNT/ReGraph variant, while both outperform ROI-MLP+CLIP. Therefore, the manuscript frames ReGraph-VLM as a fixed-order anatomical ROI-token graph/vision model rather than an adjacency-driven GNN.

## Main Task

The main task is cross-subject same-image brain graph matching and retrieval. A positive pair contains brain responses from different subjects viewing the same image; a negative pair contains different images. The model is evaluated by:

- AUROC
- AUPRC
- Recall@5
- MRR
- image retrieval Recall@5
- brain retrieval Recall@5

## Data Representation

The active NSD representation uses:

- 8 NSD subjects
- 180 HCP-MMP ROI tokens
- strict T=3 repeated-image sequences
- four ROI features per trial: `mean_beta`, `std_beta`, `q90_beta`, `positive_fraction`
- train-only adjacency construction for adjacency-aware controls

Large NSD data, beta volumes, `.pt` datasets, checkpoints, and Slurm outputs are not stored in this GitHub repository.

## Main Findings

### Neuroscience checks

Repeated natural-image responses show measurable ROI-level and graph-level structure:

- 41 ROIs show FDR-significant repeat2-repeat1 effects.
- 56 ROIs show FDR-significant repeat3-repeat1 effects.
- Same-image repeated graphs are consistently more similar than different-image controls.
- Repeat-specific ROI correlation graphs are globally stable, with local repeat-sensitive edge changes.

Same-image representational stability:

| Repeat pair | Same-image similarity | Different-image similarity | Gap |
|---|---:|---:|---:|
| 1-2 | 0.8916 | 0.8305 | 0.0612 |
| 1-3 | 0.8851 | 0.8296 | 0.0555 |
| 2-3 | 0.8871 | 0.8292 | 0.0579 |

### Main all-fold cross-subject results

Values are mean +/- std over 8 folds x 3 seeds.

| Model | AUROC | AUPRC | R@5 | MRR | Img R@5 | Brain R@5 |
|---|---:|---:|---:|---:|---:|---:|
| ROI-MLP+CLIP | 0.8164 +/- 0.0518 | 0.7896 +/- 0.0532 | 0.0782 +/- 0.0273 | 0.0636 +/- 0.0190 | 0.0729 +/- 0.0280 | 0.0837 +/- 0.0291 |
| Flat ReGraph+CLIP | 0.8210 +/- 0.0481 | 0.8022 +/- 0.0460 | 0.0865 +/- 0.0318 | 0.0677 +/- 0.0213 | 0.0812 +/- 0.0305 | 0.0966 +/- 0.0299 |
| Gated ReGraph+CLIP | 0.8259 +/- 0.0523 | 0.8065 +/- 0.0528 | 0.0899 +/- 0.0357 | 0.0695 +/- 0.0240 | 0.0847 +/- 0.0318 | 0.0996 +/- 0.0310 |

### Controlled adjacency conclusion

The clean adjacency ablation shows that explicit fixed adjacency is not the source of the gain:

| Model | AUROC | AUPRC | R@5 | MRR | Brain R@5 |
|---|---:|---:|---:|---:|---:|
| ROI-MLP+CLIP | 0.8164 | 0.7896 | 0.0782 | 0.0636 | 0.0837 |
| No-adj gated ROI Transformer+CLIP | 0.8258 | 0.8061 | 0.0884 | 0.0692 | 0.0965 |
| Gated ReGraph/BNT+CLIP | 0.8259 | 0.8065 | 0.0899 | 0.0695 | 0.0996 |

The recommended paper framing is:

```text
Gated ReGraph-VLM is a fixed-order anatomical ROI-token Transformer-VLM with gated ROI-preserving readout and CLIP alignment.
```

## External Validation Status

External smoke validations were run on public visual-ROI summaries or derivatives from:

- CNeuroMod-THINGS
- BOLD5000
- THINGS-fMRI
- LAION-fMRI

These are useful feasibility checks but are not full HCP-MMP 180-ROI external replications. LAION-fMRI uses trial-wise public beta maps summarized with public visual ROI masks and padded to the 180-token interface.

LAION-fMRI visual-ROI validation:

| Model | n | AUROC | AUPRC | R@5 | MRR |
|---|---:|---:|---:|---:|---:|
| ROI-MLP | 30 | 0.5315 +/- 0.0213 | 0.0284 +/- 0.0022 | 0.1481 +/- 0.0185 | 0.1215 +/- 0.0103 |
| Gated ROI Transformer | 30 | 0.5296 +/- 0.0221 | 0.0284 +/- 0.0023 | 0.1574 +/- 0.0283 | 0.1252 +/- 0.0126 |

Paired LAION tests show no significant AUROC/AUPRC difference and only a trend toward higher R@5 for the gated ROI Transformer. The external validation conclusion is intentionally conservative: the public compact visual-ROI settings show above-chance cross-subject signal outside NSD but do not reproduce a consistent gated-transformer advantage.

## Report Preflight

Run the full preflight from the project root before compiling or submitting:

```bash
python3 scripts/run_publication_preflight.py
```

This regenerates lightweight result artifacts, runs the AAAI artifact audit, verifies publication artifact provenance, runs the full manuscript/result audit, checks README/BUILD consistency, verifies key manuscript table values and statistical claims against committed CSV artifacts, runs the manuscript-only audit, and reports whether a local TeX compiler is available.
GitHub Actions installs a TeX distribution with recommended/extra LaTeX packages and runs the compile-required preflight on pushes to `main` and pull requests. The workflow also requires a clean Git working tree after preflight, so both tracked-file changes and newly generated untracked artifacts fail CI.

Run the manuscript-only audit directly when you only need a fast TeX-facing check:

```bash
python3 scripts/audit_manuscript_publication_claims.py \
  --tex reports/neurips_report/may30.tex \
  --manuscript-only \
  --output-dir /tmp/regraph_report_preflight
```

This checks:

- anonymity strings
- fixed-adjacency overclaim phrases
- duplicate labels and unresolved refs
- required labels
- citation coverage in `references.bib`
- figure file availability
- LaTeX table/figure/equation balance

To compile on a machine with a TeX distribution:

```bash
python3 scripts/run_publication_preflight.py --compile
```

This local machine may not have a TeX compiler installed. In that case, use the preflight audit plus a TeX-enabled machine for final PDF compilation.

## Repository Layout

```text
models/
  bnt_encoder.py

scripts/
  run_regraph_vlm_fold.py
  audit_manuscript_publication_claims.py
  audit_aaai_publication_readiness.py
  materialize_publication_readiness_artifacts.py
  run_publication_preflight.py
  summarize_laion_fmri_external_results.py
  export_laion_fmri_visual_roi_scalar4.py
  shanghai_*.sbatch

reports/neurips_report/
  may30.tex
  references.bib
  neurips_2025.sty
  BUILD.md
  figures/

external_validation/summary/
  laion_fmri_visual_roi_summary.csv
  laion_fmri_visual_roi_pairwise_tests.csv
  laion_fmri_visual_roi_latex.txt
```

## What Is Not Included

This GitHub repository intentionally excludes:

- NSD raw data
- large beta volumes
- generated `.pt` datasets
- checkpoints
- large `.npy/.npz` arrays
- local virtual environments
- HPC scratch-only intermediate artifacts

Only source code, lightweight summaries, manuscript files, and publication-facing helper artifacts are tracked.
