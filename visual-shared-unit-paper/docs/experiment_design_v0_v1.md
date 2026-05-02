# Experiment Design and Locked Protocols (V0/V1)

## Purpose of this document
This document tells future executors exactly what has already been done, what is frozen, and what the next method stage is.

## Completed V0/V1 pipeline

### Data views
- `all8_ge2_766`: main view, 8 subjects, 766 shared images, at least 2 repeats.
- `all8_ge3_515`: robustness view, 8 subjects, 515 shared images, strict 3 repeats.
- `full1000_4subj`: sensitivity-only view.

### Frozen preprocessing
- ROI sources: `V1`, `V2`, `V3`, `hV4`.
- ROI-block padded canonical concatenation.
- Training-only feature-wise normalization.
- Fold-wise pooled shared PCA basis, 512 dimensions.

### Frozen baselines
- `B1`: linear baseline on PCA features.
- `B4`: strong shared encoder with nested validation-subject sweep and aggregate selection.
- `prototype main`: top-3 validation subjects ranked by `best_val_top5`, mean-probability aggregation.

## Main completed results

### Main view (`all8_ge2_766`)
- `B4` mean top-1: `0.004080`
- `prototype` mean top-1: `0.004406`
- `B4` mean top-5: `0.015666`
- `prototype` mean top-5: `0.015666`

### Robustness view (`all8_ge3_515`)
- `B4` mean top-1: `0.00485`
- `prototype` mean top-1: `0.00534`
- `B4` mean top-5: `0.02160`
- `prototype` mean top-5: `0.01723`

## Completed Stage 2 analyses

### Sharedness
- On the main view, prototype same-vs-different cross-subject similarity gap is more consistently positive than `B4 hidden`.
- On the robustness view, same-vs-different gaps remain positive but the prototype no longer clearly dominates `B4 hidden`.

### Recruitment
- Soft distributed recruitment, not near one-hot routing.
- Same-image cross-subject overlap exists but is partial rather than identity-preserving.

### ROI attribution
- Predominantly mixed units.
- Weak but real ROI preference.
- Mild early/mid-visual bias.

## Current next-stage experiment design

### Stage 3A goal
Test whether explicit interaction among shared units improves over the independent-unit prototype readout.

### Required first model
- Static interaction graph over shared units.
- Node feature = unit activation or activation-weighted prototype embedding.
- Edge = coactivation-derived static graph, preferably support-aware.
- 1-2 graph layers maximum.

### Required comparisons
- `B4 main`
- `prototype main`
- `prototype + light interaction`

### Required first-stage execution order
1. Write `plans/stage3a_light_interaction_plan.md`.
2. Build graph construction rule from existing coactivation outputs.
3. Run 2-fold smoke test.
4. If healthy, scale to 8 folds.
5. Add minimal ablations.

### Explicit non-goals
- No dynamic graph.
- No time-varying graph.
- No cross-fold unit matching claim.
- No anatomical localization claim beyond current post-hoc attribution.
