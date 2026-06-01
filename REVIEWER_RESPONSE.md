# Reviewer Response Memo

This memo converts likely reviewer concerns into the manuscript changes and
committed artifacts that answer them. It is not a rebuttal letter for a specific
submission; it is a publication-readiness checklist for the current anonymous
paper package.

## Central Framing

The paper should be framed as a fixed-order anatomical ROI-token
Transformer-VLM with gated ROI-preserving readout and image alignment. The
supported claim is that fixed anatomical ROI-token modeling, gated readout, and
CLIP alignment improve cross-subject retrieval over ROI-MLP+CLIP.

The paper should not claim a performance gain from explicit fixed adjacency.
The no-adjacency gated ROI Transformer and adjacency-based Gated ReGraph/BNT
model are statistically tied, and static adjacency perturbations do not show a
clear adjacency-specific benefit.

Primary evidence:

- `preproc_v0/repetition_familiarity/results/final_tables/table_adjacency_ablation.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/table_roi_token_controls.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/table_adjacency_perturbation.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/table_edge_bias_followup.csv`
- `MODEL_CARD.md`

## Reviewer Concern: Dataset Scale And Accounting

Response: The manuscript now includes fold-level split accounting, pair counts,
session/order QC, and fold difficulty diagnostics. The dataset remains small
with eight subjects, so external generalization is treated as a limitation.

Primary evidence:

- `DATASET_CARD.md`
- `preproc_v0/repetition_familiarity/results/final_tables/split_accounting.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/session_order_pair_qc.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/fold_difficulty_qc.csv`

## Reviewer Concern: Session/Order Confounds

Response: The pair construction includes exact anchor-side session/order QC, and
single-reference controls are reported both as eval-only and retrained settings.
These controls reduce but do not fully eliminate all session/order concerns, so
the limitation remains explicit.

Primary evidence:

- `preproc_v0/repetition_familiarity/results/final_tables/session_order_pair_qc.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/single_ref_matched_summary.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/single_ref_matched_allseed_summary.csv`

## Reviewer Concern: Graph-Adjacency Novelty

Response: The manuscript should state that explicit fixed adjacency is not the
source of the gain. The stronger mechanism claim is fixed ROI-token layout plus
gated ROI-preserving readout. Learned edge bias remains a future direction
because it is competitive but not better than no-adjacency in the current
results.

Primary evidence:

- `preproc_v0/repetition_familiarity/results/final_tables/final_adjacency_ablation_tests.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/table_edge_bias_followup.csv`
- `MODEL_CARD.md`

## Reviewer Concern: Mathematical And Implementation Detail

Response: The manuscript includes explicit BCE, repeat InfoNCE, and CLIP
alignment definitions, embedding normalization, batch positives/negatives,
adjacency construction and leakage controls, architecture settings, optimizer,
folds, seeds, early stopping, metric definitions, and parameter counts.

Primary evidence:

- `reports/neurips_report/may30.tex`
- `preproc_v0/repetition_familiarity/results/final_tables/model_parameter_counts.csv`
- `scripts/verify_model_parameter_counts.py`

## Reviewer Concern: Statistical Reporting

Response: Primary comparisons have paired fold-by-seed tests with bootstrap
confidence intervals. Claims should distinguish statistically supported
comparisons from descriptive stress-test means.

Primary evidence:

- `preproc_v0/repetition_familiarity/results/final_tables/publication_paired_stats.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/manuscript_stat_claims_audit.csv`

## Reviewer Concern: SOTA-Style Baselines

Response: The paper should call these task-matched component baselines, not full
image-reconstruction system reproductions. The comparison tests shared mapping,
subject conditioning, and subject-adversarial ideas under the same retrieval
task.

Primary evidence:

- `preproc_v0/repetition_familiarity/results/final_tables/table_phase2_sota_graph_baselines.csv`
- `MODEL_CARD.md`

## Reviewer Concern: Semantic Alignment

Response: The manuscript separates pair discrimination from semantic
image/brain retrieval. Random image embeddings can remain competitive for pair
discrimination but collapse on image/brain retrieval, while real CLIP semantics
support retrieval to unseen images.

Primary evidence:

- `preproc_v0/repetition_familiarity/results/final_tables/table_heldout_image.csv`

## Reviewer Concern: External Validation

Response: External public visual-ROI checks are presented as feasibility smoke
tests, not full HCP-MMP external replications. The paper should keep broader
external validation as future work.

Primary evidence:

- `external_validation/summary/external_visual_roi_all4_summary.md`
- `preproc_v0/repetition_familiarity/results/final_tables/table_external_visual_roi_smoke.csv`
- `DATASET_CARD.md`

## Reviewer Concern: Fold 07

Response: Fold `fold_07` remains the hardest held-out-subject case. The paper
diagnoses it with fold-level QC but does not over-explain it. It should remain
an unresolved robustness limitation.

Primary evidence:

- `preproc_v0/repetition_familiarity/results/final_tables/fold_difficulty_qc.csv`

## Single Evidence Index

The compact cross-reference from claims to artifacts is:

```text
preproc_v0/repetition_familiarity/results/final_tables/publication_evidence_manifest.md
```

The machine-checkable reviewer-response audit is:

```text
preproc_v0/repetition_familiarity/results/final_tables/reviewer_response_readiness_audit.csv
```
