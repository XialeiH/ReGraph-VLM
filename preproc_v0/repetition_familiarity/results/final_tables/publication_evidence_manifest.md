# Publication Evidence Manifest

This manifest gives reviewers a single index from each major claim or concern to the committed manuscript/result artifact that supports it.

Active manuscript: `reports/neurips_report/may30.tex`

## Main Evidence Map

| Claim or concern | Primary manuscript location | Committed evidence artifact | Evidence summary |
| --- | --- | --- | --- |
| Main cross-subject result | Table `tab:cross_subject_main` | `table_allfold_final.csv` | 8 held-out-subject folds x 3 seeds; 3 rows; n sum=72 |
| Explicit adjacency is not the source of the gain | Tables `tab:adjacency_ablation`, `tab:roi_token_controls`, `tab:adjacency_perturbation`, `tab:edge_bias_followup` | `table_adjacency_ablation.csv`, `table_roi_token_controls.csv`, `final_adjacency_ablation_tests.csv` | no-adj and adjacency variants are statistically tied; ROI-order/gate controls drive the interpretation |
| Session/order confound control | Tables `tab:session_order_pair_qc`, `tab:single_ref_matched`, `tab:single_ref_retrained` | `session_order_pair_qc.csv`, `single_ref_matched_summary.csv`, `single_ref_matched_allseed_summary.csv` | exact anchor-side QC plus eval-only and retrained single-reference controls |
| Implementation reproducibility | Table `tab:implementation_details` | `model_parameter_counts.csv`, `manuscript_publication_claims_audit.csv` | loss, normalization, adjacency construction, architecture, parameter counts, optimizer, folds, and seeds are audited |
| Statistical reporting | Results text and statistical claims | `publication_paired_stats.csv`, `manuscript_stat_claims_audit.csv` | paired fold x seed tests with bootstrap CIs; 119 rows; n sum=2744 |
| Component baseline framing | Table `tab:sota_baselines` | `table_phase2_sota_graph_baselines.csv` | task-matched component baselines, not full image-reconstruction system claims |
| Semantic alignment control | Table `tab:heldout` | `table_heldout_image.csv` | separates pair discrimination from image/brain retrieval under real CLIP versus random embeddings |
| External validation limits | Table `tab:external_visual_roi_smoke` | `table_external_visual_roi_smoke.csv`, `external_visual_roi_all4_summary.md` | four public visual-ROI smoke checks; explicitly not full HCP-MMP external replications |
| Fold_07 robustness | Table `tab:fold_difficulty` | `fold_difficulty_qc.csv` | fold_07 is diagnosed as difficult but left as an unresolved robustness case |
| Reviewer-response coverage | Preflight artifact | `reviewer_response_readiness_audit.csv` | ready=11 |
| Double-blind code sharing | `ANONYMIZATION.md` | `scripts/make_anonymous_submission_bundle.py` | Git-history-free anonymous archive workflow; do not submit public GitHub metadata |

## Audit Artifacts

| Artifact | Status | Purpose |
| --- | --- | --- |
| `preproc_v0/repetition_familiarity/results/final_tables/aaai_publication_readiness_audit.csv` | present | ready=33 |
| `preproc_v0/repetition_familiarity/results/final_tables/publication_artifact_provenance_audit.csv` | present | ready=20 |
| `preproc_v0/repetition_familiarity/results/final_tables/manuscript_publication_claims_audit.csv` | present | ready=55 |
| `preproc_v0/repetition_familiarity/results/final_tables/publication_docs_audit.csv` | present | ready=21 |
| `preproc_v0/repetition_familiarity/results/final_tables/reviewer_response_readiness_audit.csv` | present | ready=11 |
| `preproc_v0/repetition_familiarity/results/final_tables/manuscript_table_values_audit.csv` | present | ready=24 |
| `preproc_v0/repetition_familiarity/results/final_tables/manuscript_stat_claims_audit.csv` | present | ready=22 |
| `external_validation/summary/external_visual_roi_all4_summary.md` | present | external smoke-validation summary |
