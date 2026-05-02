# Stage 3A Light Interaction Plan

## Version tag
v7

## Goal
Test whether explicit interaction among shared units improves over independent-unit prototype readout.

## Inputs
- `RESTRICTS.yaml`
- `docs/current_results_snapshot.md`
- `docs/stage3_light_interaction_spec.md`
- existing coactivation outputs from Stage 2B
- prototype main protocol and clean summaries

## Outputs
- `light_interaction_gnn_spec.md` or update to `docs/stage3_light_interaction_spec.md`
- graph-construction summary file
- smoke-test metrics on `fold_01` and `fold_04`
- comparison table against `prototype main` and `B4 main`
- decision note: PROCEED / REFINE / PIVOT

## Proposed model
- static coactivation-based unit graph
- 1 graph layer first
- compare activation-only node feature vs activation-weighted prototype embedding if feasible

## Required ablations
- identity/no-interaction graph
- coactivation top-k graph
- optional dense learnable graph only if smoke test is stable

## Risks
- graph adds parameters without real structural value
- edge construction may overfit coactivation noise
- performance gain may be absent even if edge patterns appear plausible

## Gate criteria
### PROCEED
- training stable on both smoke-test folds
- no major degradation vs prototype main
- at least one positive sign: top-1 gain or interpretable edge structure

### REFINE
- training stable but graph seems too dense/noisy or no-interaction control performs similarly

### PIVOT
- repeated underperformance and no interpretable interaction signal

## Logging requirements
At stage end, update `PROGRESS.md` with:
- graph definition used
- folds run
- metrics vs prototype main
- decision and rationale
