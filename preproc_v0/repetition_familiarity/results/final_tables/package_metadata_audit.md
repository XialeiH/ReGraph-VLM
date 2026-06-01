# Package Metadata Audit

Status counts: {'ready': 11}

| Item | Status | Evidence |
| --- | --- | --- |
| pyproject parses | ready | pyproject.toml |
| project identity | ready | name=regraph-vlm; requires-python=>=3.9 |
| core dependencies | ready | core deps present: numpy, pandas, scipy, scikit-learn, torch, tqdm |
| publication extra | ready | publication extra includes pandas |
| neuroimaging extra | ready | neuro extra present: h5py, matplotlib, nibabel, nilearn, Pillow, requests |
| CLIP extra | ready | clip extra includes open_clip_torch |
| legacy graph extra | ready | legacy-graph extra preserves bridgegen dependencies |
| dev extra | ready | dev extra includes pytest and legacy graph deps |
| package directories | ready | package-dir={'bridgegen': 'src/bridgegen', 'models': 'models'} |
| package discovery | ready | find={'where': ['src', '.'], 'include': ['bridgegen*', 'models*']} |
| model package files | ready | models package exposes ReGraph-VLM source files |
