# LAION-fMRI External Visual-ROI Validation

Trial-wise public LAION-fMRI beta maps were summarized with public visual ROI masks, padded to the same 180-token interface, and evaluated with cross-subject same-image retrieval. Values are mean ± std over subject-pair × seed runs.

| model | n | test_AUROC_mean | test_AUROC_std | test_AUPRC_mean | test_AUPRC_std | test_R@5_mean | test_R@5_std | test_MRR_mean | test_MRR_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roi_mlp | 30 | 0.5315 | 0.0213 | 0.0284 | 0.0022 | 0.1481 | 0.0185 | 0.1215 | 0.0103 |
| roi_transformer_gated | 30 | 0.5296 | 0.0221 | 0.0284 | 0.0023 | 0.1574 | 0.0283 | 0.1252 | 0.0126 |

Paired tests compare Gated ROI Transformer against ROI-MLP over matched subject-pair × seed runs.

| comparison | metric | n | mean_diff | std_diff | ci95_low | ci95_high | paired_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gated ROI Transformer - ROI-MLP | test_AUROC | 30 | -0.0019 | 0.0109 | -0.0059 | 0.0022 | 0.3558 |
| Gated ROI Transformer - ROI-MLP | test_AUPRC | 30 | 0.0000 | 0.0010 | -0.0004 | 0.0004 | 0.8779 |
| Gated ROI Transformer - ROI-MLP | test_R@5 | 30 | 0.0093 | 0.0258 | -0.0003 | 0.0189 | 0.0575 |
| Gated ROI Transformer - ROI-MLP | test_MRR | 30 | 0.0036 | 0.0117 | -0.0007 | 0.0080 | 0.0989 |

LaTeX rows are available in `laion_fmri_visual_roi_latex.txt`.
