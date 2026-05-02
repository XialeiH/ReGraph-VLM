# Stage 2B ROI Interpretability Note

Primary Torch output directory:
- `/scratch/xh2906/final_project_nsd/v0_shared_unit/preproc_v0/all8_ge2_766/analysis_stage`

Core files:
- `unit_roi_ablation_summary.csv`
- `unit_roi_ablation_foldwise.csv`
- `unit_roi_drop_summary.csv`
- `unit_roi_recruitment_link.csv`
- `unit_montage_index.csv`
- `unit_montage_examples.pdf`
- `stage2b_roi_interpretability_note.md`

Main conclusions:

1. Support-conditioned ROI attribution is more informative than raw all-sample mean activation.
- Because prototype assignments are softmax-normalized and approximately balanced on average, raw all-sample unit means are close to uniform and are not a useful attribution signal.
- The Stage 2B analysis therefore uses unit-specific support sets: samples where a unit appears in the top-3 recruited units.

2. Most shared units are still mixed rather than strongly single-ROI dominated.
- Dominant ROI counts at support-based concentration threshold `0.35`:
  - `mixed: 452`
  - `V2: 31`
  - `V1: 18`
  - `V3: 6`
  - `hV4: 5`
- Mean ROI concentration score: `0.3078`
- So the dominant pattern is still joint visual-cortical dependence rather than clean one-ROI specialization.

3. There are weak but nontrivial ROI preferences.
- Mean support-conditioned retained ratios under ROI-only input:
  - `V1: 0.6617`
  - `V2: 0.7317`
  - `V3: 0.6429`
  - `hV4: 0.5657`
- Mean support-conditioned drop ratios under ROI removal:
  - `V1: 0.1808`
  - `V2: 0.1280`
  - `V3: 0.1089`
  - `hV4: 0.0787`
- These numbers suggest that some units show relative preference structure, with early visual blocks contributing slightly more on average than `hV4`, but not in a way that supports strong one-region claims for most units.

4. Recruitment and ROI preference can now be linked at the fold-unit level.
- Mean top-1 recruitment fraction across fold-units: `0.0156`
- Mean top-3 recruitment fraction across fold-units: `0.0469`
- This is now available in `unit_roi_recruitment_link.csv`, which can support later analyses such as whether more ROI-focused units also show sharper recruitment.

5. Weak interpretability is now available as a qualitative preview.
- `unit_montage_examples.pdf` is intentionally a preview, not semantic proof.
- At this stage we can inspect whether top-activating samples show coarse visual commonality, but we should not yet label units as object-specific or category-specific.

Current claim boundary:
- We can now say that shared units are distributed soft recruitment patterns over a joint visual-cortical latent space, with partial ROI preference and measurable recruitment structure.
- We cannot yet say that most units map cleanly to a single ROI or a single semantic category.
