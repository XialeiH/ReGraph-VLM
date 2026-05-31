# Manuscript Publication Claims Audit

This audit checks manuscript/result consistency for the publication-facing ReGraph-VLM story.

Status counts: {'ready': 48}

| Item | Status | Evidence |
| --- | --- | --- |
| manuscript exists | ready | reports/neurips_report/may30.tex: 834 lines |
| anonymous author block | ready | Anonymous Author(s) present |
| deanonymizing strings | ready | none found |
| fixed-adjacency overclaims | ready | none found |
| duplicate labels | ready | none |
| unresolved refs | ready | none |
| required publication labels | ready | all present |
| citation bibliography coverage | ready | 18 citation keys covered by 1 bibliography file(s) |
| figure file availability | ready | 5 checked, all present |
| figure files tracked by Git | ready | 5 checked, all tracked |
| TeX group brace balance | ready | all unescaped group braces balanced |
| TeX math dollar balance | ready | single=766, double=0 |
| table environment balance | ready | begin=21, end=21 |
| figure environment balance | ready | begin=5, end=5 |
| equation environment balance | ready | begin=6, end=6 |
| tabular environment balance | ready | begin=19, end=19 |
| tabularx environment balance | ready | begin=2, end=2 |
| LaTeX environment nesting | ready | 106 begin/end tokens nested correctly |
| result file: split_accounting.csv | ready | 8 rows, support n=8, expected at least 8 |
| result file: session_order_pair_qc.csv | ready | 4 rows, support n=4, expected at least 1 |
| result file: table_within_subject.csv | ready | 6 rows, support n=6, expected at least 1 |
| result file: table_allfold_final.csv | ready | 3 rows, support n=72, expected at least 24 |
| result file: table_hard_negative_allfold.csv | ready | 3 rows, support n=72, expected at least 24 |
| result file: table_heldout_image.csv | ready | 4 rows, support n=96, expected at least 24 |
| result file: table_phase2_sota_graph_baselines.csv | ready | 5 rows, support n=104, expected at least 100 |
| result file: table_graph_only.csv | ready | 2 rows, support n=48, expected at least 48 |
| result file: table_adjacency_ablation.csv | ready | 3 rows, support n=72, expected at least 1 |
| result file: table_roi_token_controls.csv | ready | 5 rows, support n=120, expected at least 1 |
| result file: table_adjacency_perturbation.csv | ready | 5 rows, support n=5, expected at least 1 |
| result file: table_edge_bias_followup.csv | ready | 3 rows, support n=38, expected at least 1 |
| result file: single_ref_matched_summary.csv | ready | 3 rows, support n=72, expected at least 72 |
| result file: single_ref_matched_allseed_summary.csv | ready | 3 rows, support n=72, expected at least 72 |
| result file: table_lowshot_calibration.csv | ready | 5 rows, support n=120, expected at least 120 |
| result file: table_external_visual_roi_smoke.csv | ready | 8 rows, support n=150, expected at least 100 |
| result file: table_gate_confound.csv | ready | 3 rows, support n=3, expected at least 3 |
| result file: table_matched_deletion.csv | ready | 6 rows, support n=6, expected at least 6 |
| result file: fold_difficulty_qc.csv | ready | 8 rows, support n=8, expected at least 8 |
| result file: publication_paired_stats.csv | ready | 119 rows, support n=2744, expected at least 1 |
| paired stats: main_allfold / Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP | ready | 7 metric rows, expected at least 3 |
| paired stats: main_allfold / Gated ReGraph/BNT+CLIP - Flat ReGraph+CLIP | ready | 7 metric rows, expected at least 3 |
| paired stats: hard_negative / Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP | ready | 7 metric rows, expected at least 3 |
| paired stats: heldout_real_vs_random_available_raw / Gated ReGraph/BNT+CLIP - Gated random embedding | ready | 7 metric rows, expected at least 3 |
| paired stats: component_baselines / Gated ReGraph/BNT+CLIP - MindLink-style subject-adversarial ROI-MLP | ready | 7 metric rows, expected at least 3 |
| paired stats: single_ref_eval_existing / Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP | ready | 7 metric rows, expected at least 3 |
| paired stats: single_ref_eval_existing / Gated ReGraph/BNT+CLIP - No-adj gated ROI Transformer+CLIP | ready | 7 metric rows, expected at least 3 |
| paired stats: single_ref_retrained / Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP | ready | 7 metric rows, expected at least 3 |
| paired stats: single_ref_retrained / Gated ReGraph/BNT+CLIP - No-adj gated ROI Transformer+CLIP | ready | 7 metric rows, expected at least 3 |
| paired stats: single_ref_retrained / No-adj gated ROI Transformer+CLIP - ROI-MLP+CLIP | ready | 7 metric rows, expected at least 3 |
