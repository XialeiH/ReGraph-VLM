# Model Card

This card summarizes the publication-facing model scope for ReGraph-VLM. It is
intended for reviewers and future maintainers who need to understand what the
model supports, what it does not claim, and which committed artifacts verify the
paper-facing statements.

## Model Summary

ReGraph-VLM is a fixed-order anatomical ROI-token graph/vision model for
cross-subject natural-image fMRI retrieval. Each brain input is a sequence of
180 HCP-MMP ROI tokens with four scalar ROI beta-summary features per token.
The main model uses:

- A fixed-order ROI-token transformer brain encoder.
- A gated ROI-preserving readout that keeps anatomical ROI identity in the
  flattened representation.
- A frozen CLIP image-embedding branch for brain-image alignment.

The model outputs a brain embedding used for same-image cross-subject
brain-brain matching and brain-image retrieval.

## Intended Use

The intended use is research on cross-subject brain representation learning,
natural-image fMRI retrieval, and model-based analysis of fixed anatomical ROI
tokens. The model is not a clinical, diagnostic, or subject-identification
system.

## Architecture And Objective

The brain branch maps ROI features into ROI-token embeddings, updates them with
transformer-style interactions, applies a learned gate to each ROI token, then
flattens the gated ROI-token sequence before projecting to the final brain
embedding. The image branch uses frozen CLIP embeddings as image anchors.

The training objective combines:

- Pairwise binary cross-entropy for same-image versus different-image matching.
- Repeat InfoNCE for repeated-response alignment.
- CLIP-style brain-image contrastive alignment, with normalized embeddings.

Adjacency-aware variants use train-only ROI relation matrices, but the current
best-supported interpretation is not adjacency-driven.

## Supported Claims

The committed experiments support the following claims:

- Fixed anatomical ROI-token modeling, gated ROI-preserving readout, and image
  alignment improve cross-subject retrieval over ROI-MLP+CLIP.
- The no-adjacency gated ROI Transformer is statistically tied with the
  adjacency-based Gated ReGraph/BNT+CLIP model.
- Explicit fixed adjacency is not the source of the measured performance gain in
  the current implementation.
- ROI-order, gate, and deletion controls support the importance of fixed
  anatomical ROI-token layout and learned gated readout.
- Gate maps are model-level explanations that identify model dependence on ROIs;
  they are not causal neuroscience evidence.

## Non-Claims

The model card intentionally excludes several stronger claims:

- It does not claim a retrieval benefit from fixed Pearson adjacency.
- It does not claim that learned gates are causal neural mechanisms.
- It does not claim clinical, diagnostic, or individual-subject identification
  validity.
- It does not claim a full external HCP-MMP 180-ROI replication outside NSD.
- It does not claim to reproduce full MindEye2, UMBRAE, MindBridge, or MindLink
  image-reconstruction systems; those experiments are task-matched component
  baselines for this retrieval task.

## Validation Evidence

Reviewer-facing evidence is indexed in:

```text
preproc_v0/repetition_familiarity/results/final_tables/publication_evidence_manifest.md
```

Key committed artifacts include:

- `preproc_v0/repetition_familiarity/results/final_tables/table_allfold_final.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/table_adjacency_ablation.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/table_roi_token_controls.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/table_adjacency_perturbation.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/table_edge_bias_followup.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/table_external_visual_roi_smoke.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/model_parameter_counts.csv`
- `preproc_v0/repetition_familiarity/results/final_tables/publication_paired_stats.csv`

## Limitations

The main NSD analysis has eight subjects and uses atlas-level ROI summaries
rather than full voxel patterns. Fold `fold_07` remains a difficult held-out
subject case, external validation is limited to public visual-ROI smoke checks,
and repetition/session/order confounds are mitigated but not fully eliminated.
Learned edge-bias results are competitive but do not yet establish a stronger
edge-specific contribution than the no-adjacency gated ROI Transformer.

## Reproducibility

Use the publication preflight before submitting or sharing the anonymous bundle:

```bash
python3 scripts/run_publication_preflight.py
```

For data accounting, see `DATASET_CARD.md`. For environment and result
provenance, see `REPRODUCIBILITY.md`. For double-blind packaging, see
`ANONYMIZATION.md`.
