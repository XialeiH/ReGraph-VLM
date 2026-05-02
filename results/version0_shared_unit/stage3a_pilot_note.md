# Stage 3A Pilot Note

## 1. Stage 3A Objective

在 shared-unit 已经成立的前提下，测试静态 unit interaction 是否能在 `prototype main` 之上再带来增益。

## 2. Structural Differences Across v1 / v2 / v3

### Shared setup across all three pilots

- Dataset view: `all8_ge2_766`
- Smoke folds: `fold_01` and `fold_04`
- Node space: fold-specific shared units, `K=64`
- Canonical subrun: within each fold, choose the highest-ranked selected validation subject subrun by `best_val_top5`
- Edge construction:
  - training `fit` samples only
  - top-3 recruited units per sample
  - pairwise coactivation counts
  - keep top-8 neighbors per unit
  - minimum pair count `12`
  - symmetrized normalized adjacency with self-loops
- Classifier supervision: same held-out-subject identification task as Stage 2

### v1

- Node feature: activation-weighted prototype embedding
- Graph core: 2-layer static message passing
- Residual: plain residual update
- Gating: none
- Readout: simple pooled readout from updated node states only
  - assignment-weighted sum
  - node mean
  - node max
- Direct access to raw assignment summary: none

### v2

- Node feature changed to `assignment_only`
- Graph core unchanged from v1
- Residual: plain residual update
- Gating: none
- Readout changed:
  - keep graph pooled readout
  - add a direct residual path from raw assignment vector to the classifier

### v3

- Node feature: same as v2, `assignment_only`
- Edge construction: unchanged
- Readout: unchanged from v2
- Graph core changed:
  - keep the same 2-layer static message passing
  - add gated residual message passing
  - initialize gate at `-2.0` so the interaction effect starts weak and must justify itself during training

## 3. Smoke Result Table

Source summaries on Torch:

- `preproc_v0/all8_ge2_766/light_interaction_smoke/light_interaction_smoke_summary.csv`
- `preproc_v0/all8_ge2_766/light_interaction_v2_smoke/light_interaction_smoke_summary.csv`
- `preproc_v0/all8_ge2_766/light_interaction_v3_smoke/light_interaction_smoke_summary.csv`

| Fold | Model | Top-1 | Top-5 | Delta vs Prototype Top-1 | Delta vs Prototype Top-5 | Delta vs B4 Top-1 | Delta vs B4 Top-5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fold_01` | `prototype main` | 0.002611 | 0.006527 | 0.000000 | 0.000000 | -0.001305 | -0.007833 |
| `fold_01` | `B4 main` | 0.003916 | 0.014360 | +0.001305 | +0.007833 | 0.000000 | 0.000000 |
| `fold_01` | `Stage 3A v1` | 0.001305 | 0.006527 | -0.001305 | 0.000000 | -0.002611 | -0.007833 |
| `fold_01` | `Stage 3A v2` | 0.001305 | 0.007833 | -0.001305 | +0.001305 | -0.002611 | -0.006527 |
| `fold_01` | `Stage 3A v3` | 0.001741 | 0.006527 | -0.000870 | 0.000000 | -0.002176 | -0.007833 |
| `fold_04` | `prototype main` | 0.006527 | 0.014360 | 0.000000 | 0.000000 | +0.003916 | +0.002611 |
| `fold_04` | `B4 main` | 0.002611 | 0.011749 | -0.003916 | -0.002611 | 0.000000 | 0.000000 |
| `fold_04` | `Stage 3A v1` | 0.001305 | 0.006527 | -0.005222 | -0.007833 | -0.001305 | -0.005222 |
| `fold_04` | `Stage 3A v2` | 0.002176 | 0.014360 | -0.004352 | 0.000000 | -0.000435 | +0.002611 |
| `fold_04` | `Stage 3A v3` | 0.002176 | 0.012620 | -0.004352 | -0.001741 | -0.000435 | +0.000870 |

## 4. What Worked

- `v3` is materially more stable than `v1`.
  - `v1` showed a clear collapse toward chance on both smoke folds.
  - `v3` no longer shows that pattern.
- `v3` is more conservative than `v2`.
  - the gated residual update reduced the chance that message passing overwhelms the base assignment signal
- Static interaction is trainable.
  - all three variants completed cleanly
  - graph construction was stable
  - no NaN, graph failure, or catastrophic optimization failure appeared
- The interaction layer itself is therefore not inherently impossible to optimize in this setup.

## 5. What Did Not Work

- None of `v1 / v2 / v3` beat `prototype main` on the two smoke folds.
- There is currently no evidence that static interaction, under the present readout and supervision design, provides stable extra value on top of the non-interaction prototype model.
- `v2` and `v3` improved over `v1`, but the improvement was not sufficient to cross the prototype baseline.
- Continued gate-only tuning is unlikely to be informative.
  - `v1 -> v2 -> v3` already established the main pattern
  - the remaining bottleneck is unlikely to be resolved by more gate-size or initialization sweeps alone

## 6. Decision Rule For v4

Continue to `v4` only if it introduces a new, interpretable method hypothesis.

Not allowed for `v4`:

- slightly larger gate
- slightly smaller gate
- one more layer of the same interaction block
- re-running the same graph core with different random initialization only

Recommended `v4` hypothesis:

The current Stage 3A bottleneck is in graph readout, not in interaction stability itself.

Recommended `v4` design:

- keep the current fold-specific static coactivation graph
- keep the `v3` gated residual message-passing core
- do not change the basic supervision protocol
- change only the readout to a graph-aware pooling mechanism
  - for example attention pooling or gated pooling over updated node states
  - optionally concatenate the pooled graph embedding with a compact raw activation summary if this is written into the protocol in advance

Recommended `v4` smoke scope:

- `fold_01`
- `fold_04`

`v4` go / no-go rule:

- must not underperform `v3` on either smoke fold
- must match or beat `prototype main` on at least one smoke fold
- must not introduce new optimization instability

If these conditions are not met, Stage 3A should be recorded as:

`a negative but informative pilot: static interaction alone is trainable but not yet sufficient to outperform the non-interaction prototype model`
