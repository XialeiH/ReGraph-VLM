# Figure Asset Audit

Status counts: {'ready': 9}

| Item | Status | Evidence |
| --- | --- | --- |
| manuscript exists | ready | reports/neurips_report/may30.tex |
| figure dependencies parsed | ready | 5 IfFileExists guards, 5 includegraphics calls, 5 unique assets |
| includegraphics calls guarded | ready | all includegraphics assets are guarded by IfFileExists |
| figure paths are portable | ready | all figure paths are relative to the manuscript directory |
| figure asset figures/gate_interpretability.pdf | ready | reports/neurips_report/figures/gate_interpretability.pdf; present; nonempty; bundle-allowlisted |
| figure asset figures/model_overview.pdf | ready | reports/neurips_report/figures/model_overview.pdf; present; nonempty; bundle-allowlisted |
| figure asset figures/natural_scene_two_stimuli_repeat_maps_left_lateral_large.png | ready | reports/neurips_report/figures/natural_scene_two_stimuli_repeat_maps_left_lateral_large.png; present; nonempty; bundle-allowlisted |
| figure asset figures/neuroscience_summary.pdf | ready | reports/neurips_report/figures/neuroscience_summary.pdf; present; nonempty; bundle-allowlisted |
| figure asset figures/per_subject_performance.pdf | ready | reports/neurips_report/figures/per_subject_performance.pdf; present; nonempty; bundle-allowlisted |
