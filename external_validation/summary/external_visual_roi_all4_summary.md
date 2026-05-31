# External Visual-ROI Smoke Summary

Source: reports/neurips_report/may30.tex: Table tab:external_visual_roi_smoke.

These checks use public visual-ROI summaries or public beta-map derivatives from CNeuroMod-THINGS, BOLD5000, THINGS-fMRI, and LAION-fMRI. They are not full HCP-MMP 180-ROI external validations.

| Dataset | Model | n | AUROC | AUPRC | R@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BOLD5000 visual ROI | ROI-MLP | 18 | 0.6561 +/- 0.0315 | 0.0094 +/- 0.0013 | 0.0539 +/- 0.0143 | 0.0533 +/- 0.0084 |
| BOLD5000 visual ROI | Gated ROI Transformer | 18 | 0.6240 +/- 0.0611 | 0.0085 +/- 0.0020 | 0.0561 +/- 0.0139 | 0.0513 +/- 0.0103 |
| CNeuroMod visual ROI | ROI-MLP | 18 | 0.6248 +/- 0.0212 | 0.0164 +/- 0.0023 | 0.1058 +/- 0.0147 | 0.0879 +/- 0.0102 |
| CNeuroMod visual ROI | Gated ROI Transformer | 18 | 0.6071 +/- 0.0423 | 0.0159 +/- 0.0034 | 0.0979 +/- 0.0184 | 0.0827 +/- 0.0111 |
| THINGS-fMRI visual ROI | ROI-MLP | 9 | 0.5777 +/- 0.0352 | 0.0067 +/- 0.0011 | 0.0506 +/- 0.0159 | 0.0456 +/- 0.0092 |
| THINGS-fMRI visual ROI | Gated ROI Transformer | 9 | 0.5291 +/- 0.0377 | 0.0057 +/- 0.0008 | 0.0308 +/- 0.0132 | 0.0335 +/- 0.0076 |
| LAION-fMRI visual ROI | ROI-MLP | 30 | 0.5315 +/- 0.0213 | 0.0284 +/- 0.0022 | 0.1481 +/- 0.0185 | 0.1215 +/- 0.0103 |
| LAION-fMRI visual ROI | Gated ROI Transformer | 30 | 0.5296 +/- 0.0221 | 0.0284 +/- 0.0023 | 0.1574 +/- 0.0283 | 0.1252 +/- 0.0126 |

Interpretation: these datasets support above-chance cross-subject same-image signal outside NSD, but they do not reproduce the main NSD model ordering. They should be presented as external feasibility checks, not as full external validation of the HCP-MMP ReGraph-VLM result.
