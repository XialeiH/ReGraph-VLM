# Stage 3A Result Narrative

## Position In The Project

Stage 3A was designed as a light interaction pilot.

The question was not whether shared units exist. That had already been established in Stage 2. The Stage 3A question was narrower:

Can a static unit-interaction layer add value on top of the non-interaction `prototype main` model?

This was intentionally a weak graph setting:

- fold-specific shared units as nodes
- static coactivation graph as edges
- no dynamic adjacency
- no temporal modeling
- no cross-fold unit matching

## What Was Tested

Three pilot variants were run on the same two smoke folds, `fold_01` and `fold_04`.

- `v1`: activation-weighted prototype embeddings with static message passing and pooled graph readout
- `v2`: switch node features to raw assignments and add a direct residual path from assignment summary to the classifier
- `v3`: keep the `v2` design and add gated residual message passing so interaction starts weak and must earn influence during training

The comparison target was fixed:

- `prototype main` as the direct non-interaction shared-unit baseline
- `B4 main` as the strong non-graph shared encoder baseline

## Main Empirical Pattern

The Stage 3A pilot produced a consistent pattern across `v1 / v2 / v3`.

First, static interaction is trainable.

- all three variants completed cleanly
- graph construction was stable
- no numerical failure or graph-collapse behavior appeared

Second, the earliest graph variant was too aggressive.

- `v1` dropped to near-chance behavior on both smoke folds
- this showed that simply adding static message passing on top of shared units was not enough

Third, the more conservative variants improved stability.

- `v2` recovered a meaningful amount of performance relative to `v1`
- `v3` further stabilized the interaction path by reducing early message-passing dominance
- this established that the interaction layer itself is not fundamentally untrainable

Fourth, the core target was still not met.

- none of `v1 / v2 / v3` surpassed `prototype main`
- the gains from making interaction more conservative were not enough to produce a positive pilot result against the non-interaction prototype baseline

## Interpretation

The correct reading of Stage 3A is not that graph interaction failed completely.

The stronger reading is:

Static interaction over the current shared-unit space is feasible and trainable, but the present Stage 3A formulation does not yet show stable evidence that interaction adds value beyond the non-interaction prototype model.

This matters for two reasons.

First, it rules out one bad explanation:

- the negative result is not caused by trivial instability alone

Second, it narrows the likely bottleneck:

- the remaining weakness is more plausibly in how graph-updated node information is read out and supervised than in whether message passing can be optimized at all

## Decision

Stage 3A should be treated as complete.

The right label is:

`a negative but informative pilot`

This means:

- do not continue incremental gate-centric variants
- do not expand Stage 3A to full 8-fold evaluation in its current form
- preserve the result as evidence that weak static interaction is not yet sufficient under the current readout design

## Condition For Any v4

A `v4` should be run only if it tests one new method hypothesis, stated clearly in one sentence, in a 2-fold smoke setting.

The strongest current candidate is:

The Stage 3A bottleneck is in graph readout rather than interaction stability, so graph-aware pooling over updated node states may recover value that simple pooled readout fails to expose.

If that hypothesis is not adopted cleanly and pre-specified, the graph line should be frozen here and resumed later in a better-posed setting.
