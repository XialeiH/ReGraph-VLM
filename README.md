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
For environment details, dependency tiers, and the large-data policy, see
`REPRODUCIBILITY.md`.
For fold-level split counts, session/order QC, external-validation scope, and
large-data handling, see `DATASET_CARD.md`.
For model scope, intended use, supported claims, non-claims, and limitations,
see `MODEL_CARD.md`.
For a concise map from likely reviewer concerns to committed evidence, see
`REVIEWER_RESPONSE.md`.

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
# or
make preflight
```

This regenerates lightweight result artifacts, runs the AAAI artifact audit, verifies publication artifact provenance for manuscript-backed result tables and generated audit CSVs, runs the full manuscript/result audit, checks README/BUILD consistency, verifies the external data policy audit, verifies package metadata, verifies reviewer-response readiness, verifies key manuscript table values and statistical claims against committed CSV artifacts, runs the manuscript-only audit, and reports whether a local TeX compiler is available. The manuscript audit also enforces framing guardrails for adjacency, task-matched component baselines, external smoke validation, fold_07 robustness, and implementation details.
GitHub Actions installs a TeX distribution with recommended/extra LaTeX packages and runs the compile-required preflight on pushes to `main` and pull requests. The workflow uses `--require-clean`, so both tracked-file changes and newly generated untracked artifacts fail CI.

When PyTorch is installed, the preflight also verifies `model_parameter_counts.csv`
against instantiated `ReGraphVLM` modules. Environments without PyTorch skip only
that optional code-level check; the formula-generated parameter-count artifact is
still audited against the manuscript table.

The reviewer-response readiness audit maps likely reviewer concerns to concrete manuscript/result evidence: dataset accounting, session/order controls, adjacency limitations, ROI-token/gate mechanism controls, implementation detail, paired statistics, component-baseline framing, semantic-alignment controls, external-validation caveats, and fold_07 robustness.

The external data policy audit checks that protected external fMRI download and
probe scripts use the shared HPC scratch path guard, that large data remains out
of Git, and that the anonymous bundle includes only the policy/audit code plus
lightweight summary artifacts, not external download scripts.

The Makefile target audit checks that reviewer-facing commands stay wired to the
canonical publication scripts for `may30.tex`. It is implemented in
`scripts/audit_makefile_targets.py` and writes
`preproc_v0/repetition_familiarity/results/final_tables/makefile_targets_audit.csv`.
The CI workflow audit checks that the GitHub Actions publication workflow still
uses the compile-required, clean-worktree preflight path. It is implemented in
`scripts/audit_ci_workflow.py` and writes
`preproc_v0/repetition_familiarity/results/final_tables/ci_workflow_audit.csv`.
The result artifact schema audit checks that committed reviewer-facing CSVs keep
the required columns, numeric fields, minimum row counts, and nonempty source
metadata. It is implemented in `scripts/audit_result_artifact_schemas.py` and
writes
`preproc_v0/repetition_familiarity/results/final_tables/result_artifact_schema_audit.csv`.
The result value-range audit checks that reported metrics, p-values,
correlations, standard deviations, counts, and split/QC invariants stay in
valid ranges. It is implemented in `scripts/audit_result_value_ranges.py` and
writes
`preproc_v0/repetition_familiarity/results/final_tables/result_value_range_audit.csv`.
The bundle allowlist audit checks that anonymous-bundle source paths, figures,
publication artifacts, generated audits, and reviewer-facing scripts are present
and tracked or staged before packaging. It is implemented in
`scripts/audit_bundle_allowlist.py` and writes
`preproc_v0/repetition_familiarity/results/final_tables/bundle_allowlist_audit.csv`.
The citation integrity audit checks that manuscript citation keys are present,
that every cited key is defined in `references.bib`, and that bibliography keys
are not duplicated. It is implemented in `scripts/audit_citation_integrity.py`
and writes
`preproc_v0/repetition_familiarity/results/final_tables/citation_integrity_audit.csv`.

The preflight also writes a compact Publication Evidence Manifest:

```text
preproc_v0/repetition_familiarity/results/final_tables/publication_evidence_manifest.md
```

Use it as the reviewer-facing index from each major claim or caveat to the committed manuscript table and result artifact that supports it.
For dataset accounting and external-validation scope, also see `DATASET_CARD.md`.
For model scope and non-claims, also see `MODEL_CARD.md`.
For a prose reviewer-response checklist, also see `REVIEWER_RESPONSE.md`.

For double-blind review, do not submit the public GitHub URL or a Git clone.
Use the Anonymous Submission Bundle workflow instead; see `ANONYMIZATION.md`.
The non-mutating bundle check is included in the publication preflight:

```bash
python3 scripts/make_anonymous_submission_bundle.py --dry-run
# or
make bundle-check
```

To build the archive for submission:

```bash
python3 scripts/make_anonymous_submission_bundle.py
# or
make bundle
```

The bundle writer normalizes archive metadata and reports a SHA-256 checksum,
so repeated builds from the same committed inputs should be byte-stable.
It also scans every included file at the byte level for deanonymizing strings,
including PDFs and image metadata.
The publication preflight also regenerates a per-file source manifest for the
anonymous bundle:

```text
preproc_v0/repetition_familiarity/results/final_tables/anonymous_bundle_manifest.csv
```

The manifest lists each included source path, source byte count, and SHA-256
checksum. It intentionally excludes its own row to avoid self-referential
checksum drift.
`--manifest-output` also supports external sidecar paths outside the repository;
the publication preflight tests both the in-bundle manifest path and a temporary
sidecar manifest path.
To verify the manifest against either the committed source tree or an extracted
anonymous bundle, run:

```bash
python3 scripts/verify_anonymous_bundle_manifest.py
# or
make bundle-verify
```

To test the full reviewer path by building a temporary archive, extracting it,
and verifying the extracted files, run:

```bash
python3 scripts/smoke_test_anonymous_bundle_archive.py
# or
make bundle-smoke
```

The smoke test rebuilds the archive twice and checks byte-identical output. It
also rejects path traversal, Git metadata, symlink/hardlink entries, non-regular
archive members, and archive files that are not accounted for by the manifest.
It runs an extracted anonymous bundle preflight once, using a recursion guard so
the nested smoke test does not call itself indefinitely. When a TeX tool is
available, the extracted-bundle preflight is compile-required.

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
- framing guardrails for adjacency, component baselines, external validation, fold_07, and implementation details
- duplicate labels and unresolved refs
- required labels
- citation coverage in `references.bib`
- figure file availability
- LaTeX table/figure/equation balance

To compile on a machine with a TeX distribution:

```bash
python3 scripts/run_publication_preflight.py --compile
# or
make compile
```

This local machine may not have a TeX compiler installed. In that case, use the preflight audit plus a TeX-enabled machine for final PDF compilation.

## Repository Layout

```text
models/
  __init__.py
  bnt_encoder.py
  regraph_vlm.py

scripts/
  run_regraph_vlm_fold.py
  audit_manuscript_publication_claims.py
  audit_aaai_publication_readiness.py
  audit_ci_workflow.py
  audit_makefile_targets.py
  audit_result_artifact_schemas.py
  audit_result_value_ranges.py
  audit_reviewer_response_readiness.py
  generate_publication_evidence_manifest.py
  make_anonymous_submission_bundle.py
  materialize_publication_readiness_artifacts.py
  run_publication_preflight.py
  smoke_test_anonymous_bundle_archive.py
  summarize_laion_fmri_external_results.py
  export_laion_fmri_visual_roi_scalar4.py
  verify_anonymous_bundle_manifest.py
  shanghai_*.sbatch

reports/neurips_report/
  may30.tex
  references.bib
  neurips_2025.sty
  BUILD.md
  figures/

DATASET_CARD.md
MODEL_CARD.md
REVIEWER_RESPONSE.md

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
