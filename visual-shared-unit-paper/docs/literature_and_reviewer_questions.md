# Literature and Reviewer-Defense Notes

## Why this project is not "just another encoder"
Our key distinction is not merely a stronger classifier. The project is staged around an explicit **shared-unit bottleneck** whose scientific role is to define latent units that are more cross-subject aligned than a strong shared hidden baseline.

## Key nearby literatures to cover

### 1. Cross-subject latent spaces for brain data
Reviewers will ask whether we are simply rebranding subject alignment. The paper must explain that our bottleneck is explicit and unitized, not only a hidden shared space.

### 2. Prototype / slot-like / bottleneck latent structures
We need to connect our units to broader prototype or latent-slot ideas, while being honest that our units are not semantically clean object slots.

### 3. Brain graph modeling
We must position our work as a pre-graph or graph-preparatory stage: we first establish nodes before modeling edges.

### 4. Visual cortical encoding and natural scene datasets
Need to justify why V1/V2/V3/hV4 is a legitimate constrained setting for a first paper stage.

## Reviewer questions to pre-answer

### Q1. Why not go directly to dynamic GNNs?
Answer: because explicit shared units must first be established and shown to be meaningfully shared; otherwise graph nodes are not scientifically grounded.

### Q2. Why are the gains small?
Answer: the task is extremely hard, chance is tiny, and we evaluate against strong baselines under strict LOSO cross-subject conditions. The main claim is not huge accuracy gains, but a combination of modest top-1 improvement and stronger cross-subject sharedness.

### Q3. Are these units anatomically localized?
Answer: no strong one-to-one anatomical claim is made. Current units are latent patterns over joint V1/V2/V3/hV4 input, with weak post-hoc ROI preference.

### Q4. Are these units semantically interpretable?
Answer: only weakly and qualitatively at present. We explicitly avoid strong semantic claims.

### Q5. Why only early/mid visual ROIs?
Answer: this is a staged, controlled setting that supports a clean first paper. It reduces confounds and lets us study shared latent structure before scaling.

### Q6. Why should a graph help?
Answer: because recruitment and coactivation results suggest that shared units are not independent, so a light interaction layer is the next falsifiable test.

## What not to claim to reviewers
- Do not claim to have solved dynamic brain graphs.
- Do not claim unit identity stability across folds.
- Do not claim each unit corresponds to a single cortical area.
- Do not claim strong semantic selectivity without external labels or stronger evidence.
