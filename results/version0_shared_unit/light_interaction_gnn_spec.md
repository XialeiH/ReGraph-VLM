# Stage 3A Light Interaction Pilot

## Goal

Test one narrow question:

Given that shared units are already learned, does adding an explicit static unit-unit interaction layer improve held-out-subject identification beyond the prototype main model?

This is a light interaction pilot, not a full graph program.

## Dataset And Protocol

- Dataset view: `all8_ge2_766`
- Outer evaluation: the same 8-fold LOSO protocol used in Stage 2
- Pilot scope: smoke test on `fold_01` and `fold_04`
- Primary metrics: held-out-subject `top-1` and `top-5` accuracy
- Comparisons:
  - `B4 main`: shared encoder without prototype bank
  - `prototype main`: shared-unit model without explicit interaction
  - `light interaction`: shared-unit model plus static unit graph mixing

## Node Definition

The graph nodes are the fold-specific shared units, with `K = 64`.

For this pilot, the node space is defined by a single canonical prototype subrun inside each fold:

- start from the fixed `prototype main` protocol
- look inside `prototype_valsweep_summary.csv`
- restrict to the selected validation subjects from the main protocol
- choose the highest-ranked subrun by `best_val_top5`

This restriction is deliberate. Validation-subject sweep runs do not share an aligned unit identity, so Stage 3A does not attempt cross-subrun unit matching.

## Node Features

For each sample:

- let `a in R^K` be the canonical prototype assignment vector
- let `P in R^(K x d_p)` be the canonical prototype embedding matrix from the same subrun
- define node features as activation-weighted prototype embeddings

`H0[i] = a[i] * P[i]`

This keeps the interaction layer tied to the learned shared-unit geometry instead of treating units as unlabeled bins.

## Edge Definition

The pilot uses a static fold-specific unit graph built from canonical training assignments.

Construction:

- use training samples only
- take each sample's top-3 recruited units
- count pairwise coactivation frequency
- convert pair counts into pair fractions
- keep only edges with sufficient support
- retain the top-`k` neighbors per node
- symmetrize the graph and add self-loops
- use normalized adjacency for message passing

Main pilot setting:

- coactivation source: canonical `fit` split only
- coactivation top-k: `3`
- graph neighbors per node: `8`
- minimum pair count: `12`
- graph is fixed during training
- graph is symmetric

## Graph Layer

The pilot uses a small residual graph mixer, not a full GNN stack.

Per layer:

`M = A_hat H`

`H_next = LayerNorm(H + Dropout(GELU(W_self H + W_neigh M)))`

Main pilot setting:

- layers: `2`
- hidden dim: `256`
- dropout: `0.10`

## Readout

After graph mixing:

- assignment-weighted sum pool
- nodewise mean pool
- nodewise max pool

Concatenate the three pooled vectors and feed them to a small classifier head.

## Training

The pilot reuses the canonical prototype subrun split:

- train on the `fit` subjects
- validate on the canonical validation subject
- test on the held-out subject

Main pilot defaults:

- seeds: `701,702,703`
- epochs: `120`
- patience: `20`
- batch size: `256`
- optimizer: `AdamW`
- learning rate: `1e-3`
- weight decay: `1e-4`

## What Stage 3A Does Not Do

- no dynamic graph
- no temporal graph
- no sample-dependent adjacency
- no multi-view graph
- no cross-fold unit matching
- no cross-subrun unit matching
- no full 8-fold hyperparameter sweep before smoke validation

## Success Criteria For Smoke

The smoke test is considered healthy if:

- training is stable on both `fold_01` and `fold_04`
- no NaN or degenerate adjacency appears
- the interaction model does not collapse well below `prototype main`
- there is at least a small sign of gain on one of the two folds

If smoke is healthy, the next step is an all-fold Stage 3A run under the same fixed protocol.
