# code/

Place all experiment code here in a structured way.

Recommended layout:

- `preprocess/` — dataset view construction, normalization, PCA artifacts
- `baselines/` — B1, B4, prototype main
- `analysis/` — Stage 2 and Stage 2B analyses
- `interaction/` — Stage 3A light interaction model and graph builders
- `common/` — shared utilities, config readers, metrics

Every experiment script must:
- read explicit config
- log random seeds
- save machine-readable outputs
- support smoke-test mode
- avoid hidden protocol drift from the frozen main settings
