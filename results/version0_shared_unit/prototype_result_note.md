# Prototype Result Note

1. Prototype gain currently comes from explicit shared-unit structure plus a more stable inner-run aggregation rule: top-3 by `best_val_top5` with mean probabilities.

2. This gain is exploratory but credible because it beats a strong `B4 val-sweep` baseline without changing the dataset, outer folds, or test protocol, and because the ranking signal is still inner-validation-only.

3. This rule is strong enough to serve as the prototype main protocol candidate. It should now be treated as fixed rather than extended with more heuristic search.

4. The correct next step is a single clean confirmation rerun under the frozen protocol, followed by a clean comparison table and report-ready summary.
