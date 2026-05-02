# Stage 3A Light Interaction Specification

## Objective
Test whether explicit interaction among learned shared units improves over the independent-unit prototype main model.

## Scientific question
Given that shared units are now established as meaningful latent visual-cortical patterns, does modeling **unit-unit interaction** help either predictive performance or representation quality beyond the prototype bottleneck alone?

## Node definition
- Node = one learned shared unit (prototype-derived unit).
- Number of nodes = current prototype count (default: 64).

## Candidate node features
Preferred order:
1. sample-specific unit activation vector as scalar node signal
2. activation-weighted prototype embedding
3. concatenation of prototype embedding and scalar activation (optional later)

## Candidate edge construction
Preferred order:
1. support-aware coactivation graph built from existing coactivation statistics
2. top-k coactivation graph per unit
3. learnable dense graph only as later ablation, not as the first pilot

## First pilot architecture
- 1 graph layer first
- static graph
- shallow readout
- no temporal modeling
- no dynamic adjacency

## Required comparisons
- `B4 main`
- `prototype main`
- `prototype + light interaction`

## Required metrics
Primary:
- top-1 accuracy
Secondary:
- top-5 accuracy
- same-vs-different cross-subject similarity gap (optional if extraction is straightforward)

## Required ablations
- no interaction / identity graph
- coactivation graph vs simple dense learnable graph (only if pilot is healthy)
- 1 layer vs 2 layers (only if pilot is healthy)

## Smoke-test plan
Run first on 2 folds:
- `fold_01`
- `fold_04`

Rationale:
- one typical fold
- one historically sensitive fold

## Success criteria
- No major instability.
- No collapse to performance far below `prototype main`.
- Any small but repeatable improvement in top-1 or representation diagnostics is sufficient to justify scaling.

## Failure criteria
- Strong consistent underperformance relative to `prototype main`.
- Degenerate edge usage or training instability.
- No interpretable edge structure and no predictive gain.
