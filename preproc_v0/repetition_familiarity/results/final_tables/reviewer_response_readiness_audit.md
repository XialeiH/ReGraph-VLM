# Reviewer Response Readiness Audit

This audit maps likely reviewer concerns to manuscript text and committed result artifacts.

Status counts: {'ready': 11}

| Item | Status | Evidence |
| --- | --- | --- |
| reviewer-response source artifacts | ready | all reviewer-response source artifacts exist |
| dataset accounting response | ready | 8 folds; unique held-out subjects; pair counts equal strict T=3 sequences x 6 |
| session/order confound response | ready | anchor-side QC is exact and both eval-only/retrained single-reference controls are present |
| graph-adjacency limitation response | ready | no-adj, adjacency, and ROI-MLP rows plus main no-adj-vs-adj paired tests support non-adjacency framing |
| ROI-token/gate mechanism response | ready | ROI-order shuffle and gate controls are present and described as mechanism evidence |
| math/implementation detail response | ready | loss definitions and architecture/training table fragments present |
| statistical-reporting response | ready | paired tests with bootstrap CIs cover main, flat, hard-negative, component, and single-reference comparisons |
| component-baseline framing response | ready | component baselines are framed as task-matched, not full-system SOTA comparisons |
| semantic-alignment response | ready | held-out-image CLIP/random controls separate pair discrimination from image/brain retrieval |
| external-validation response | ready | four public visual-ROI smoke checks are present and explicitly limited as feasibility evidence |
| fold_07 robustness response | ready | fold_07 has a QC row and remains framed as an unresolved robustness case |
