# MEGA_PROMPT.md

## Mission

Use the **exact 25-stage pipeline framework** supplied by the user, but instantiate it faithfully for **our project**:

**Cross-subject shared-unit modeling for visual cortical fMRI, with a staged path from prototype bottlenecks to light interaction graphs and later dynamic graph reasoning.**

You must act as a research lead who can hand the project to others. The framework is not generic: it must preserve our locked protocols, current evidence, scientific caveats, and next-stage priorities.

Default reasoning/writing model: **GPT-5.4 Pro**.

If any stage touches methodology, evidence interpretation, protocol locking, writing, review, or scientific claims, use GPT-5.4 Pro.

---

## Read order before any action

1. `PROJECT_CONFIG.yaml`
2. `RESTRICTS.yaml`
3. `docs/abstract_submitted_placeholder.md`
4. `docs/overall_story_and_outline.md`
5. `docs/experiment_design_v0_v1.md`
6. `docs/literature_and_reviewer_questions.md`
7. `docs/current_results_snapshot.md`
8. `PROGRESS.md`

Do not start a new stage until these are read and understood.

---

## Project identity

### Working paper title
Shared Visual Cortical Units for Cross-Subject Brain Graph Modeling

### Target problem
Given visual cortical fMRI responses from multiple subjects, can we learn an explicit **shared-unit bottleneck** that is more cross-subject aligned than a strong shared hidden baseline, and can this bottleneck support a later transition to **interaction and graph-based reasoning**?

### Current scientific status
The project is **not** at the full dynamic GNN stage yet. The current evidence supports:

1. A prototype/shared-unit bottleneck can modestly outperform a strong non-graph shared encoder baseline on top-1 accuracy.
2. The prototype representation is more consistently cross-subject shared than the `B4 hidden` representation on the main view.
3. Shared units behave as distributed soft recruitment patterns over a joint V1/V2/V3/hV4 latent space.
4. Post-hoc ROI attribution shows weak but real ROI preference, with predominantly mixed units.

### Current next stage
Start **Stage 3A: light interaction modeling**.

---

## Frozen facts that must not be silently changed

### Frozen dataset views
- Main: `all8_ge2_766` — 8 subjects, 766 shared images, at least 2 repeats.
- Robustness: `all8_ge3_515` — 8 subjects, 515 shared images, strict 3-repeat view.
- Sensitivity only: `full1000_4subj`.

### Frozen input space
- ROI sources: `V1`, `V2`, `V3`, `hV4`.
- Canonical feature construction: ROI-block padded concatenation.
- PCA: pooled training-subject shared PCA, 512 dimensions.

### Frozen baseline protocols
- `B4 main`: nested validation-subject sweep with aggregate selection.
- `prototype main`: top-3 validation subjects ranked by `best_val_top5`, mean-probability aggregation.

### Frozen main quantitative evidence
- `all8_ge2_766`
  - prototype top-1: `0.004406`
  - B4 top-1: `0.004080`
  - prototype top-5: `0.015666`
  - B4 top-5: `0.015666`
- `all8_ge3_515`
  - prototype top-1: `0.00534`
  - B4 top-1: `0.00485`
  - prototype top-5: `0.01723`
  - B4 top-5: `0.02160`

### Frozen Stage 2 interpretation
- No obvious collapse: 64 units, only 1 dead unit, top-1 usage not monopolized by a few units.
- Recruitment is soft and distributed, not one-hot routing.
- Same-image cross-subject overlap exists but does not imply strict unit identity preservation.
- Most units are mixed across ROI blocks; do not claim anatomical one-to-one mapping.

---

## Core contributions to preserve

### Contribution 1 — Shared-unit bottleneck
An explicit prototype/shared-unit bottleneck learned over the joint visual-cortical latent space can outperform a strong shared non-graph baseline on the main dataset view, with clean protocol locking.

### Contribution 2 — Sharedness and behavior analysis
The learned prototype representation is more consistently cross-subject shared than `B4 hidden` on the main view, and its behavior is distributed, soft, and partially but not perfectly overlapping across subjects for the same image.

### Contribution 3 — Joint visual-cortical latent organization
Post-hoc ROI attribution shows that shared units are predominantly mixed patterns over `V1/V2/V3/hV4`, with weak ROI preference and a mild early/mid-visual bias rather than pure single-ROI detectors.

### Contribution 4 — Stage-3 transition
The next technical step is a **light interaction model over shared units**, not a full dynamic GNN. The paper must explain this transition clearly and honestly.

---

## Exact 25-stage pipeline, project-bound version

Below, the user framework is preserved, but every stage is tied to our project.

### Stage group A — Research definition
1. `TOPIC_INIT`
   - Restate the project as: explicit shared visual-cortical latent units for cross-subject fMRI, followed by light interaction modeling.
   - Explicitly prohibit topic drift to infrastructure, generic decoding pipelines, or unsupported dynamic-graph claims.
2. `PROBLEM_DECOMPOSE`
   - Decompose into at least four sub-questions:
     1. Does explicit shared-unit structure beat a strong shared hidden baseline?
     2. Are these units meaningfully cross-subject shared?
     3. What is the behavior of these units (recruitment, overlap, ROI preference)?
     4. Does light interaction over units add value beyond independent unit activations?

### Stage group B — Literature discovery
3. `SEARCH_STRATEGY`
   - Cover cross-subject fMRI alignment, prototype bottlenecks, brain graph learning, interaction over latent units, and natural scene / visual cortex encoding.
4. `LITERATURE_COLLECT`
   - Require >=30 papers, emphasize 2020+ and top venues where possible.
5. `LITERATURE_SCREEN` [gate]
   - Aggressively reject papers that are high-quality but irrelevant to our exact topic.
6. `KNOWLEDGE_EXTRACT`
   - Preserve DOI and citation keys.

### Stage group C — Knowledge synthesis
7. `SYNTHESIS`
   - Produce clusters and gaps. Expected gap statement: most work either does hidden shared encoding or brain graph modeling without first establishing explicit shared-unit structure.
8. `HYPOTHESIS_GEN`
   - Generate falsifiable hypotheses. Example:
     - H1: prototype/shared-unit representation yields a more consistently positive same-vs-different cross-subject similarity gap than `B4 hidden`.
     - H2: adding light interaction over shared units improves top-1 beyond prototype main on the main dataset view.
8.5 `THEORETICAL_BOUNDS`
   - Derive complexity for prototype encoding and the proposed light interaction layer.

### Stage group D — Experiment design
9. `EXPERIMENT_DESIGN` [gate]
   - Enforce the following ordering:
     1. preserve current shared-unit protocols
     2. design Stage 3A light interaction model
     3. define minimal ablations
     4. avoid dynamic graph unless Stage 3A succeeds cleanly
10. `CODE_GENERATION`
   - Generate real, runnable experiment code only.
11. `RESOURCE_PLANNING`
   - Require a small pilot first, with time estimate and graceful shutdown logic.

### Stage group E — Experiment execution
12. `EXPERIMENT_RUN`
   - Start with 1-2 fold smoke tests for Stage 3A.
13. `ITERATIVE_REFINE`
   - Fix NaN/Inf or logic bugs at the root cause only.

### Stage group F — Analysis and decision
14. `RESULT_ANALYSIS`
   - Must compare `prototype + interaction` against both `prototype main` and `B4 main`.
15. `RESEARCH_DECISION`
   - `PROCEED` if interaction adds value or gives interpretable edge structure.
   - `REFINE` if graph training is unstable or edges degenerate.
   - `PIVOT` only if interaction consistently hurts and no stable edge interpretation exists.

### Stage group G — Paper drafting
16. `PAPER_OUTLINE`
   - Must maintain a single clear story: shared units first, interaction second.
17. `PAPER_DRAFT`
   - Do not write the paper as if dynamic GNN was already achieved.
18. `PEER_REVIEW`
   - Check methodology-evidence consistency against actual results.
19. `PAPER_REVISION`
   - Expand with real evidence, not filler.

### Stage group H — Finalization
20. `QUALITY_GATE` [gate]
   - Reject if any claim outruns evidence.
21. `KNOWLEDGE_ARCHIVE`
22. `EXPORT_PUBLISH`
23. `CITATION_VERIFY`

### Stage group I — External review loop
24. `3RD_PARTY_REVIEW`
25. `REBUTTAL`
   - Allow experiment REFINE or writing PIVOT only if reviewers expose a genuine evidence gap.

---

## Required current experiments and analysis state

Any new agent must treat the following as already completed, reproducible project assets:

1. Preprocessing and dataset-view generation.
2. `B1`, `B4`, and `prototype main` comparisons.
3. `all8_ge3_515` robustness.
4. Shared-unit behavior analysis:
   - usage
   - sharedness
   - recruitment
   - ROI attribution
   - montage preview

Do not redo these blindly. Reuse them unless there is a justified refinement.

---

## Immediate execution target: Stage 3A

### Target question
Do explicit interactions among shared units improve over independent-unit prototype readout?

### Minimal design
- Node: shared units (K=64).
- Node feature: sample-specific unit activations, or activation-weighted prototype embeddings.
- Edge: static graph, preferably support-aware coactivation-derived graph or top-k coactivation graph.
- Graph layer: 1-2 layers maximum for the pilot.
- Readout: classifier over updated unit representation.

### Required comparisons
- `B4 main`
- `prototype main`
- `prototype + light interaction`

### Required ablations
- no interaction / identity graph
- coactivation graph vs learnable dense graph (if feasible)
- 1 layer vs 2 layers (only if smoke tests succeed)

### Explicit non-goals for Stage 3A
- no dynamic graph
- no temporal modeling claim
- no cross-fold unit matching claim
- no anatomical localization claim beyond post-hoc ROI attribution

---

## File discipline

At the start of every stage, create a new markdown plan in `plans/`.
At the end of every stage, update `PROGRESS.md` with:
- stage name
- version tag
- produced artifacts
- decision (PROCEED/REFINE/PIVOT)
- open risks

All paper text must live in `paper/mypaper/`.
All actual results must be stored under `results/` or reproducible experiment output directories.

---

## Writing stance

The paper must currently be written as a staged technical story:

1. We establish explicit shared units.
2. We show they are behaviorally and cross-subject meaningfully shared.
3. We show they are mixed joint visual-cortical latent patterns with weak ROI preference.
4. We then test whether interactions among these units improve modeling.

Do NOT collapse this staged story into a claim that the project already solved dynamic brain graphs.

---

## Final instruction

Use this framework exactly, but do not imitate its generic placeholders. Replace them with the real project state, real frozen protocols, and real evidence from this handoff package. When uncertain, prefer conservative claims over ambitious claims.
