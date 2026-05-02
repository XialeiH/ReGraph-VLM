# Analysis Stage Note

Status: `Week 1 core analysis completed`

Primary output location on Torch:
- `/scratch/xh2906/final_project_nsd/v0_shared_unit/preproc_v0/all8_ge2_766/analysis_stage/unit_usage_summary.csv`
- `/scratch/xh2906/final_project_nsd/v0_shared_unit/preproc_v0/all8_ge2_766/analysis_stage/unit_usage_foldwise.csv`
- `/scratch/xh2906/final_project_nsd/v0_shared_unit/preproc_v0/all8_ge2_766/analysis_stage/cross_subject_unit_consistency.csv`
- `/scratch/xh2906/final_project_nsd/v0_shared_unit/preproc_v0/all8_ge2_766/analysis_stage/same_vs_diff_image_similarity.csv`
- `/scratch/xh2906/final_project_nsd/v0_shared_unit/preproc_v0/all8_ge2_766/analysis_stage/unit_selectivity_summary.csv`
- `/scratch/xh2906/final_project_nsd/v0_shared_unit/preproc_v0/all8_ge2_766/analysis_stage/unit_top_images_examples.csv`

Current answers to the main Week 1 questions:

1. Prototype/shared units do not show obvious collapse.
- `64` units total.
- `1` dead unit by `top5_count == 0`.
- `4` units never appear as `top1`, but only `1` is fully dead at `top5`.
- The most-used unit has `usage_fraction = 0.116351`, far from a single-unit monopoly.
- Mean `top1` usage fraction is `0.015625`, consistent with a long-tail but still distributed usage pattern.

2. Cross-subject same-vs-diff consistency is present for prototype activations.
- Prototype cosine same-vs-diff gap mean: `0.001042`.
- Prototype correlation same-vs-diff gap mean: `0.005206`.
- Both cosine and correlation gaps are positive in `8/8` held-out folds.

3. Prototype is more consistently shared across subjects than the current `B4 hidden` representation.
- `B4 hidden` cosine same-vs-diff gap mean: `0.000707`.
- `B4 hidden` correlation same-vs-diff gap mean: `0.001435`.
- `B4 hidden` gap is positive in only `5/8` held-out folds for both metrics.
- So the prototype representation is not just slightly better as a classifier; it is also more consistently aligned across subjects for the same stimulus.

4. Foldwise usage can be reported, but raw cross-fold unit identity should be interpreted carefully.
- Each LOSO fold trains its own prototype bank.
- So `unit_id = k` in one fold is not guaranteed to match `unit_id = k` in another fold.
- `unit_usage_foldwise.csv` is therefore useful for within-fold health checks, but not for claiming direct fold-to-fold unit identity stability without an explicit unit matching step.

Immediate implication:
- The analysis now supports the stronger scientific claim that the learned prototypes behave more like shared units than generic hidden features.
- The next step should focus on selectivity and qualitative recurrence rather than more score chasing.

Preliminary selectivity readout:
- `unit_selectivity_summary.csv` and `unit_top_images_examples.csv` are now available.
- At this first pass, unit top-image recurrence is modest rather than dramatic:
  - mean `n_unique_top_images = 19.891` out of top-20
  - mean `cross_subject_recurrence_score = 0.1257`
  - mean `subject_coverage_top_images = 2.359`
- This means the top-activating image sets are currently fairly diverse and often limited to a small subset of subjects.
- That does not negate the shared-unit claim from the consistency analysis, but it does mean the current interpretability evidence is still preliminary.
- The right next step is not to overclaim selectivity yet, but to add image-level montage inspection or semantic labels before making stronger category-level statements.

## Robustness on `all8_ge3_515`

- A stricter robustness run has now been completed on `/scratch/xh2906/final_project_nsd/v0_shared_unit/preproc_v0/all8_ge3_515/analysis_stage/robustness_ge3_summary.csv`.
- On `all8_ge2_766`, the main finding was: prototype achieved a small but clean top-1 advantage over `B4`, while also showing more consistent cross-subject same-vs-diff separation.
- On stricter `all8_ge3_515`, the direction of the top-1 effect remained:
  - prototype mean top-1 = `0.00534`
  - `B4` mean top-1 = `0.00485`
- So the shared-unit signal did not flip or disappear under the 3-repeat-only view.

- The robustness result is mixed rather than uniformly stronger:
  - prototype mean top-5 = `0.01723`
  - `B4` mean top-5 = `0.02160`
- This means prototype does not dominate `B4` on every metric in the strict view.
- The most defensible interpretation is therefore: prototype gains are real but limited, with the clearest benefit still concentrated in top-1 decision quality rather than broad ranking improvements.

- Cross-subject same-vs-diff separation also remains present in the strict view.
  - prototype correlation gap = `0.00614`
  - `B4 hidden` correlation gap = `0.00640`
- So the sharedness phenomenon itself survives the stricter dataset view.
- However, prototype no longer cleanly separates from `B4 hidden` on this consistency-gap metric.

- This robustness pattern is scientifically useful:
  - it argues against the ge2 result being a loose-view artifact,
  - but it also prevents overclaiming that prototype is categorically better than `B4` in every respect.
- The current Stage 2 conclusion should therefore be written as:
  - prototype/shared-unit provides a stable but modest top-1 benefit,
  - has healthy non-collapsed usage,
  - and supports a credible shared latent organization claim,
  - but is not yet a wholesale replacement for a strong non-graph shared encoder.

## Stage 2B Recruitment Preview

- Recruitment analysis has now been started on the main `all8_ge2_766` view using saved prototype models replayed over all fold samples in a common within-fold unit space.
- Output files on Torch:
  - `/scratch/xh2906/final_project_nsd/v0_shared_unit/preproc_v0/all8_ge2_766/analysis_stage/unit_recruitment_summary.csv`
  - `/scratch/xh2906/final_project_nsd/v0_shared_unit/preproc_v0/all8_ge2_766/analysis_stage/unit_coactivation_summary.csv`
  - `/scratch/xh2906/final_project_nsd/v0_shared_unit/preproc_v0/all8_ge2_766/analysis_stage/topk_unit_overlap_across_subjects.csv`

- First-pass recruitment pattern:
  - mean assignment entropy = `4.0458`
  - mean max activation = `0.0438`
  - mean active-unit count at `assignment >= 1/K` = `26.35`
  - mean active-unit count at `assignment >= 0.05` = `0.25`
- This indicates that the current prototype layer behaves more like a soft recruitment system than a near one-hot routing system.
- In other words, most samples distribute moderate mass across many units, while only a small minority of units receive very strong activation in any single sample.

- Same-image cross-subject top-k overlap is also now measurable in a fold-consistent unit space:
  - same-image top-1 match rate = `0.2247`
  - same-image top-3 Jaccard mean = `0.2066`
  - same-image top-5 Jaccard mean = `0.2266`
- These numbers are not high enough to claim near-discrete unit identity preservation across subjects.
- But they are high enough to support the weaker and more realistic claim that the same image tends to recruit partially overlapping top-k unit sets across subjects.

- Coactivation structure is present but should be interpreted with count thresholds.
- The raw highest-lift pairs can come from very rare events, so robust interpretation should prioritize pairs with both:
  - non-trivial `pair_count`, and
  - above-baseline `lift`.
- For example, some of the strongest stable pairs in the current pass combine both substantial support and lift above `2-4`, suggesting that recruitment is not independent across units.

- This recruitment result is exactly the kind of bridge needed before interaction modeling:
  - units are not dead,
  - recruitment is not random,
  - and there is now measurable within-fold coactivation structure to analyze.
