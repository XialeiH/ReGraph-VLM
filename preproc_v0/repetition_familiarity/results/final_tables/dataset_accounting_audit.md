# Dataset Accounting Audit

Status counts: {'ready': 13}

| Item | Status | Evidence |
| --- | --- | --- |
| manuscript exists | ready | reports/neurips_report/may30.tex |
| split fold rows | ready | 8 ordered held-out folds |
| held-out subject assignment | ready | each subject is held out once and validation subject differs from test subject |
| strict T=3 sequence partition | ready | all folds partition 5256 strict T=3 sequences |
| strict T=3 pair counts | ready | all pair counts equal sequence counts x 6 |
| test image counts | ready | test image counts equal held-out strict T=3 sequence counts |
| session/order QC rows | ready | Train, Val, Test, and All rows present |
| session/order positive-negative balance | ready | positive and negative counts are balanced with complete groups |
| session/order anchor matching | ready | all splits have zero problem groups and 100% anchor matching |
| session/order totals match split accounting | ready | session/order pair totals equal split-accounting sums |
| dataset accounting source columns | ready | split and session QC rows cite manuscript table labels |
| manuscript dataset table values | ready | split and session/order table values are present in manuscript text |
| manuscript dataset-accounting claims | ready | required split/session accounting caveats are stated |
