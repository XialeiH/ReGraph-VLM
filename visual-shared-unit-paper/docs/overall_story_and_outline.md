# Overall Story and Paper Outline

## One-sentence story
We first establish that an explicit shared-unit bottleneck over joint visual-cortical input improves cross-subject organization beyond a strong shared hidden baseline; only then do we ask whether interactions among those units further help.

## Paper story arc

### Stage 1 story
- Build a shared latent representation over V1/V2/V3/hV4.
- Show that a prototype/shared-unit bottleneck modestly outperforms `B4` on top-1.
- Lock clean protocols and fair baselines.

### Stage 2 story
- Show the learned units are not collapsing.
- Show they are more consistently cross-subject shared than `B4 hidden` on the main view.
- Characterize them as distributed soft recruitment patterns with weak but real ROI preference.
- Show the main findings survive in a stricter robustness view, albeit more weakly.

### Stage 3 story
- Test whether explicit interaction among shared units improves over the independent-unit prototype bottleneck.
- Keep this stage light, static, and interpretable.

## Recommended paper outline

1. **Introduction**
   - Why cross-subject fMRI modeling needs explicit shared structure.
   - Why hidden shared encoders are not enough for interpretability or graph reasoning.
   - Our staged answer: shared units first, interaction second.

2. **Related Work**
   - Cross-subject alignment and shared latent spaces.
   - Visual cortical encoding and NSD-style analyses.
   - Brain graphs and latent interaction models.
   - Prototype bottlenecks / slot-like latent units.

3. **Method**
   - Input construction over V1/V2/V3/hV4.
   - Shared PCA space and frozen preprocessing.
   - `B4` baseline.
   - Prototype/shared-unit main model.
   - Stage 3A light interaction extension.

4. **Experiments**
   - Dataset views: `all8_ge2_766` main, `all8_ge3_515` robustness.
   - Baselines and fair tuning.
   - Main metrics: top-1, top-5, cross-subject same-vs-different gap.
   - Analysis suite: usage, recruitment, ROI attribution, montage.

5. **Results**
   - Main `prototype vs B4` results.
   - Robustness on `ge3`.
   - Sharedness evidence.
   - Behavior and interpretation analyses.
   - Stage 3A interaction results (to be added).

6. **Discussion**
   - What shared units are and are not.
   - Why the current units should be interpreted as joint visual-cortical latent patterns.
   - Why interaction is the correct next step.

7. **Limitations**
   - Small metric magnitudes.
   - Restricted ROI set.
   - No dynamic graph evidence yet.
   - No strict cross-fold unit identity matching.

## Figure 1 concept
Figure 1 should show the staged logic:
1. ROI inputs from V1/V2/V3/hV4.
2. Shared PCA space.
3. Prototype/shared-unit bottleneck.
4. Evidence panel: top-1 gain and same-vs-different sharedness.
5. Next-stage light interaction graph over units.

Do NOT let Figure 1 imply that a dynamic temporal graph has already been demonstrated.
