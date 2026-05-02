# Visual Shared-Unit Paper Framework

This package adapts the 25-stage research-paper pipeline to our project on **cross-subject shared-unit modeling for visual cortical fMRI**, with a staged path toward **light interaction graphs** and later dynamic graph reasoning.

It is designed as a handoff package: another researcher or agent should be able to pick this up and continue the project without guessing the current scientific state, locked protocols, or next-stage priorities.

## What is already established

The framework assumes the following results are already part of the project state and should be treated as frozen evidence unless recomputed:

- Main dataset view: `all8_ge2_766` (8 subjects, 766 shared images, at least 2 repeats per subject-image).
- Robustness dataset view: `all8_ge3_515` (8 subjects, 515 shared images, strict 3-repeat view).
- Main prototype protocol: top-3 validation subjects ranked by `best_val_top5`, with mean-probability aggregation.
- Strong non-graph baseline (`B4`) uses nested validation-subject sweep with aggregate selection.
- Prototype clean result on `all8_ge2_766`: mean top-1 `0.004406` vs `B4` `0.004080`; mean top-5 `0.015666` vs `0.015666`.
- Robustness on `all8_ge3_515`: prototype mean top-1 `0.00534` vs `B4` `0.00485`; prototype mean top-5 `0.01723` vs `B4` `0.02160`.
- Sharedness finding: prototype same-vs-different cross-subject similarity gap is more consistently positive than `B4 hidden` on the main view; on the robustness view, the gap remains positive but is no longer clearly superior.
- Recruitment finding: shared units behave like **distributed soft recruitment**, not one-hot routing.
- ROI attribution finding: most units are **mixed** joint visual-cortical patterns over `V1/V2/V3/hV4`, with weak ROI preference rather than one-to-one anatomical localization.

## What this package contains

- `MEGA_PROMPT.md`: project-specific master instruction that reuses the user's 25-stage framework, but binds it to our topic, current state, and next-stage scientific decisions.
- `RESTRICTS.yaml`: hard constraints, red lines, and project-specific frozen protocol rules.
- `PROJECT_CONFIG.yaml`: configurable metadata such as conference, title, deadline, and model policy.
- `PROGRESS.md`: non-linear project log with versions, completed stages, open loops, and next-stage checkpoints.
- `docs/`: project-specific paper docs that downstream agents must read before acting.
- `paper/mypaper/`: LaTeX skeleton for drafting inside a conference template.
- `plans/`: stage-plan templates and the current Stage 3A plan.

## How to use this handoff package

1. Read `PROJECT_CONFIG.yaml` and fill any remaining administrative TODO fields.
2. Read `MEGA_PROMPT.md` end to end.
3. Read `RESTRICTS.yaml` and keep it open during execution.
4. Read all files in `docs/` before starting any new experiment or writing stage.
5. Update `PROGRESS.md` at the end of every stage and every loop iteration.
6. Use **GPT-5.4 Pro** as the default reasoning/writing/review model for high-impact stages.

## Immediate next stage

The package is currently positioned to start **Stage 3A: light interaction modeling**:

- Node = learned shared units.
- Node feature = unit activation or activation-weighted prototype embedding.
- Edge = static coactivation-based graph (top-k or thresholded support-aware coactivation).
- Comparison targets = `prototype main` and `B4 main`.
- Explicit non-goals for this stage: no dynamic graph, no temporal graph, no cross-fold unit identity alignment claims.

## Model policy

Default model for core stages: **GPT-5.4 Pro**.

Use smaller/cheaper models only for low-risk repetitive extraction tasks, and never for final scientific judgment, protocol locking, paper claims, or review synthesis.
