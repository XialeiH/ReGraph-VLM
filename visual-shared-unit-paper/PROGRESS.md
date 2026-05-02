# PROGRESS.md

This log is intentionally non-linear. Each version records completed stages, open loops, and possible REFINE/PIVOT return points.

## Version map

- `v1`: proposal framing and kickoff report
- `v2`: MVE implementation specification and repository skeleton
- `v3`: NSD subset acquisition, preprocessing, and dataset-view definition
- `v4`: baseline stabilization, nested validation, and prototype main protocol lock
- `v5`: Stage 2 main analysis (usage, sharedness, selectivity preview)
- `v6`: Stage 2 robustness (`all8_ge3_515`) and Stage 2B interpretation (recruitment, ROI attribution, montage)
- `v7`: current handoff version, ready for Stage 3A light interaction modeling

## Completed stages summary

### v1 — Research definition
- Problem narrowed from broad dynamic cross-subject brain graph idea to a staged program.
- Phase 1 fixed on shared-unit discovery before full graph reasoning.
- Main questions defined: shared representation, interaction, generalization.

### v2 — Experiment-ready specification
- Main view conceptually fixed on cross-subject shared image subset.
- Baselines and failure diagnostics specified.
- Shared-unit module defined as prototype-based latent bottleneck.

### v3 — Data and preprocessing
- Minimal NSD subset downloaded and validated.
- Dataset views computed:
  - `all8_ge1_907`
  - `all8_ge2_766` (chosen main view)
  - `all8_ge3_515` (robustness)
  - `full1000_4subj` (sensitivity)
- Preprocessing artifacts built: train-only normalization and fold-wise shared PCA-512.

### v4 — Baselines and prototype
- `B1` and `B4` implemented and run.
- Validation protocol weakness diagnosed.
- `B4` upgraded to nested validation-subject sweep.
- Prototype main protocol locked: top-3 validation subjects ranked by `best_val_top5`, mean-probability aggregation.
- Clean confirmation on `all8_ge2_766` kept prototype top-1 ahead of `B4`.

### v5 — Stage 2 main analysis
- No major prototype collapse.
- Prototype same-vs-different cross-subject gap more consistently positive than `B4 hidden` on the main view.
- Weak selectivity evidence observed; no strong semantic claim permitted.

### v6 — Stage 2 robustness and interpretation
- Robustness on `all8_ge3_515` preserved top-1 direction but not uniform dominance over `B4`.
- Recruitment analysis showed distributed soft recruitment rather than one-hot routing.
- Post-hoc ROI attribution showed predominantly mixed units with weak ROI preference.
- Montage preview generated for weak qualitative interpretability.

## Open loops and pending decisions

### Open loop A — Stage 3A light interaction pilot
Status: READY TO START

Required deliverables:
- `plans/stage3a_light_interaction_plan.md`
- `docs/stage3_light_interaction_spec.md`
- smoke-test results on at least 2 folds
- full 8-fold comparison if smoke tests are healthy

Potential REFINE return points:
- If light interaction underperforms prototype main substantially, return to Stage 2B recruitment x ROI link analysis.
- If graph edges appear noisy or degenerate, refine edge construction before scaling.

### Open loop B — Optional deeper Stage 2B analysis
Status: OPTIONAL SIDE LOOP

Candidates:
- recruitment x ROI linkage refinement
- stronger montage with semantic tags
- conservative cross-fold prototype alignment exploration (analysis only, not claim-making)

## Current scientific stance (must remain consistent)

1. Shared units currently mean **joint visual-cortical latent prototypes** over `V1/V2/V3/hV4`.
2. Sharedness is established statistically, not via strict cross-subject unit identity preservation.
3. ROI attribution is weak-to-moderate and mostly mixed, not anatomically one-to-one.
4. Stage 3 should test **whether interactions among shared units add value**, not jump directly to dynamic graph claims.

## Next actions for the next agent/researcher

1. Read all files in `docs/`.
2. Read `RESTRICTS.yaml` before any new experiment.
3. Write a detailed plan file for Stage 3A before coding.
4. Build a static coactivation-based unit graph as the first interaction candidate.
5. Compare `prototype + interaction` against `prototype main` and `B4 main` under matched protocol.
