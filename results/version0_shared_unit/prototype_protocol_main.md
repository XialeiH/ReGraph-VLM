# Prototype Main Protocol

Dataset view: `all8_ge2_766`

Outer evaluation:
- `8-fold` leave-one-subject-out (`LOSO`) over `subj01` to `subj08`
- one held-out subject per fold
- fixed image universe: the `766` shared images in `all8_ge2_766`

Inner validation sweep:
- for each outer fold, rerun the prototype model once per training subject, using that subject as the inner validation subject
- each inner run keeps the same model, seeds, data view, and training hyperparameters

Prototype model:
- input: `PCA-512` fold features
- shared encoder: `512 -> 256 -> 256`
- prototype bank: `64`
- assignment: cosine similarity + softmax
- classifier head on prototype activations

Main aggregation rule:
- rank inner runs by `best_val_top5`
- keep the top `3` validation-subject runs
- aggregate by mean class probabilities, not raw logits

Why `best_val_top5`:
- the task is extremely low-chance and noisy at `top-1`
- `top-5` is a more stable inner-model-selection signal for this Version 0 stage
- it still uses only inner validation information, not test labels

Why top-3 instead of top-1:
- top-1 selection was too brittle across folds because multiple inner runs often tied or nearly tied on validation
- top-3 keeps the ensemble small while reducing single-validation-subject variance

Why mean probabilities:
- raw-logit averaging was unstable because prototype submodels had inconsistent logit scales
- probability averaging produced the most stable cross-fold behavior in the fixed-sweep comparison

Protocol freeze point:
- this rule was fixed on `2026-04-03` after the exploratory sweep outputs had already been generated and compared
- after this freeze, no further aggregation heuristic search is part of the main protocol
- the next step is a single clean confirmation rerun using exactly this rule
