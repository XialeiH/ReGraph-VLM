# Dataset Card

This card summarizes the publication-facing datasets and split artifacts used by
ReGraph-VLM. It is intentionally lightweight: raw fMRI data, beta volumes,
generated tensors, checkpoints, and Slurm logs are not included in this
repository or in the anonymous submission bundle.

## Primary Dataset

The primary dataset is the Natural Scenes Dataset (NSD), represented as
trial-level fixed-order HCP-MMP ROI graphs.

Each graph has:

- 180 fixed-order cortical ROI tokens.
- 4 scalar ROI beta-summary features per token: mean beta, standard deviation
  beta, 90th-percentile beta, and positive fraction.
- Strict three-repeat image sequences for the main repetition/cross-subject
  analyses.

The main task is cross-subject same-image retrieval. Positives pair responses
from different subjects viewing the same image; negatives pair different images
across subjects. The held-out-subject evaluation uses 8 folds and 3 random
seeds.

## Split Accounting

The committed split-accounting artifact is:

```text
preproc_v0/repetition_familiarity/results/final_tables/split_accounting.csv
```

It is cited by the manuscript as Table `tab:split_accounting`.

| Fold | Test subject | Validation subject | Train seq | Val seq | Test seq | Train pairs | Val pairs | Test pairs | Test images |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold_01 | subj01 | subj08 | 3975 | 515 | 766 | 23850 | 3090 | 4596 | 766 |
| fold_02 | subj02 | subj08 | 3975 | 515 | 766 | 23850 | 3090 | 4596 | 766 |
| fold_03 | subj03 | subj08 | 4160 | 515 | 581 | 24960 | 3090 | 3486 | 581 |
| fold_04 | subj04 | subj08 | 4226 | 515 | 515 | 25356 | 3090 | 3090 | 515 |
| fold_05 | subj05 | subj08 | 3975 | 515 | 766 | 23850 | 3090 | 4596 | 766 |
| fold_06 | subj06 | subj08 | 4160 | 515 | 581 | 24960 | 3090 | 3486 | 581 |
| fold_07 | subj07 | subj08 | 3975 | 515 | 766 | 23850 | 3090 | 4596 | 766 |
| fold_08 | subj08 | subj07 | 3975 | 766 | 515 | 23850 | 4596 | 3090 | 515 |

These counts are fold-level training/validation/test construction counts, not
unique global subject/image totals.

## Known Fold Difficulty

The committed fold-difficulty artifact is:

```text
preproc_v0/repetition_familiarity/results/final_tables/fold_difficulty_qc.csv
```

It is cited by the manuscript as Table `tab:fold_difficulty`.

Fold `fold_07` remains the clearest difficult held-out-subject case: it has the
lowest raw AUROC, the lowest raw same-image gap, and the lowest model AUROC and
brain retrieval among the listed folds. The manuscript treats this as an
unresolved robustness limitation rather than a solved confound.

## Session/Order Control Artifact

The committed session/order pair QC artifact is:

```text
preproc_v0/repetition_familiarity/results/final_tables/session_order_pair_qc.csv
```

It is cited by the manuscript as Table `tab:session_order_pair_qc`. This artifact
checks exact anchor-side repeat/session matching for the constructed pair sets.
It does not prove that all session/order confounds are fully eliminated; the
manuscript keeps session/order control as a limitation and reports additional
single-reference controls.

## External Validation Scope

External publication-facing smoke validations are summarized in:

```text
external_validation/summary/external_visual_roi_all4_summary.md
preproc_v0/repetition_familiarity/results/final_tables/table_external_visual_roi_smoke.csv
```

The external probes use public visual-ROI summaries or derivatives from:

- BOLD5000
- CNeuroMod-THINGS
- THINGS-fMRI
- LAION-fMRI

These are useful feasibility checks for above-chance cross-subject signal
outside NSD, but they are not full HCP-MMP 180-ROI external replications. They
should not be presented as reproducing the full NSD result.

## Large-Data Policy

Large fMRI datasets should be downloaded and processed directly on remote HPC
scratch storage, not into a local laptop checkout. In short, raw data belongs on
remote HPC scratch storage. The Git repository and
anonymous bundle include source code, manuscript files, lightweight CSV/Markdown
summaries, figures, and audit artifacts only.

Excluded from Git and from the anonymous bundle:

- Raw NSD data and raw external fMRI data.
- Beta volumes and large neuroimaging derivatives.
- Generated `.pt`, `.npy`, and `.npz` tensors.
- Model checkpoints.
- Slurm logs and scratch-only intermediate files.

## Reviewer-Facing Evidence Index

For a compact map from claims to committed artifacts, see:

```text
preproc_v0/repetition_familiarity/results/final_tables/publication_evidence_manifest.md
```

For anonymous packaging and per-file source checksums, see:

```text
ANONYMIZATION.md
preproc_v0/repetition_familiarity/results/final_tables/anonymous_bundle_manifest.csv
```
