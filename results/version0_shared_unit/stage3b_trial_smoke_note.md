# Stage 3B Trial-Level Smoke

## Objective

Test whether the relative position of `prototype main` and `light interaction` changes when the setting moves from averaged responses to trial-level responses.

## Scope

- dataset view: `all8_trial_ge1_907`
- folds: `fold_01`, `fold_04`
- lines:
  - `prototype main`
  - `light interaction` (`v3`-style graph core)

## Go / No-Go Rule

Expand beyond 2 folds only if at least one of the following holds:

1. `light interaction` is not materially worse than `prototype main` on both smoke folds, and at least one fold matches or exceeds it.
2. `light interaction` remains below `prototype main`, but is clearly closer than in the averaged setting.
3. `light interaction` improves training stability or another predeclared auxiliary signal relative to the averaged setting.

Do not expand if:

- `prototype main` itself is unstable on trial-level artifacts
- `light interaction` remains clearly behind with no directional improvement
- trial-level results do not materially change the averaged-stage conclusion

## Artifacts

- prototype:
  - `prototype_trial_fold01_metrics.json`
  - `prototype_trial_fold04_metrics.json`
  - `prototype_trial_smoke_summary.csv`
- interaction:
  - `light_interaction_trial_fold01_metrics.json`
  - `light_interaction_trial_fold04_metrics.json`
  - `light_interaction_trial_smoke_summary.csv`
- comparison:
  - `stage3b_trial_smoke_summary.csv`

## Results

Pending.

## Decision

Pending.
