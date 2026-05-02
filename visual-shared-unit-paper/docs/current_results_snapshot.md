# Current Results Snapshot

## Dataset views

### Main
- `all8_ge2_766`
- 8 subjects
- 766 shared images
- at least 2 repeats per subject-image

### Robustness
- `all8_ge3_515`
- 8 subjects
- 515 shared images
- strict 3-repeat view

### Sensitivity only
- `full1000_4subj`

## Main protocol numbers

### Main view (`all8_ge2_766`)
- `B1` mean top-1: `0.00163`
- chance level: `0.001305`
- `B4` mean top-1: `0.004080`
- `B4` mean top-5: `0.015666`
- `prototype` clean mean top-1: `0.004406`
- `prototype` clean mean top-5: `0.015666`

### Robustness view (`all8_ge3_515`)
- `B4` mean top-1: `0.00485`
- `B4` mean top-5: `0.02160`
- `prototype` mean top-1: `0.00534`
- `prototype` mean top-5: `0.01723`

## Sharedness findings

### Main view
- Prototype same-vs-different cross-subject gap is more consistently positive than `B4 hidden`.
- Prototype: positive same-vs-different gap in `8/8` folds.
- `B4 hidden`: positive same-vs-different gap in `5/8` folds.

### Robustness view
- Same-vs-different gap remains positive for both representations.
- Prototype no longer clearly dominates `B4 hidden` on the gap metric.

## Recruitment findings
- Mean entropy: `4.0458`
- Mean max activation: `0.0438`
- Mean number of units above `1/K`: `26.35`
- Mean number of units above `0.05`: `0.25`
- Same-image cross-subject top-1 match rate: `0.2247`
- Same-image cross-subject top-3 Jaccard: `0.2066`
- Same-image cross-subject top-5 Jaccard: `0.2266`

Interpretation: the system behaves like soft distributed recruitment, not near one-hot routing.

## ROI attribution findings
- Dominant ROI counts:
  - mixed: `452`
  - V2: `31`
  - V1: `18`
  - V3: `6`
  - hV4: `5`
- Mean concentration: `0.3078`
- ROI-only retained ratio: `V2` slightly highest.
- ROI-drop effect: `V1` largest average drop.

Interpretation: shared units are mostly mixed joint visual-cortical patterns with weak ROI preference and mild early/mid-visual bias.

## Strong caveats
1. Fold-specific prototype banks are independently trained; `unit_id` cannot be treated as aligned across folds.
2. Units are not currently anatomically localized single-ROI detectors.
3. Semantic selectivity evidence is only weak and qualitative.
4. Current project evidence supports moving to light interaction modeling, not full dynamic graph claims.
