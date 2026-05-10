#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"

j0=$(sbatch --parsable scripts/shanghai_freeze_current_results_wave.sbatch)

j1=$(sbatch --parsable scripts/shanghai_gated_allfold_seed11_array.sbatch)
j1s=$(sbatch --parsable --dependency=afterok:${j1} scripts/shanghai_gated_allfold_seed11_summary.sbatch)

j2=$(sbatch --parsable scripts/shanghai_gated_graphonly_array.sbatch)
j2s=$(sbatch --parsable --dependency=afterok:${j2} scripts/shanghai_gated_graphonly_summary.sbatch)

j3=$(sbatch --parsable scripts/shanghai_gated_clip_control_smoke_array.sbatch)
j3s=$(sbatch --parsable --dependency=afterok:${j3} scripts/shanghai_gated_clip_control_smoke_summary.sbatch)

j7=$(sbatch --parsable scripts/shanghai_build_heldout_image.sbatch)
j8=$(sbatch --parsable --dependency=afterok:${j7} scripts/shanghai_heldout_image_smoke_array.sbatch)
j8s=$(sbatch --parsable --dependency=afterok:${j8} scripts/shanghai_heldout_image_smoke_summary.sbatch)

j9=$(sbatch --parsable scripts/shanghai_build_cross_subject_hardneg.sbatch)
j10=$(sbatch --parsable --dependency=afterok:${j9} scripts/shanghai_cross_subject_hardneg_array.sbatch)
j10s=$(sbatch --parsable --dependency=afterok:${j10} scripts/shanghai_cross_subject_hardneg_summary.sbatch)

j11=$(sbatch --parsable scripts/shanghai_cross_subject_cross_infonce_array.sbatch)
j11s=$(sbatch --parsable --dependency=afterok:${j11} scripts/shanghai_cross_subject_cross_infonce_summary.sbatch)

cat <<EOF
freeze_qc=${j0}
gated_allfold_seed11=${j1}
gated_allfold_seed11_summary=${j1s}
gated_graphonly=${j2}
gated_graphonly_summary=${j2s}
gated_clip_controls=${j3}
gated_clip_controls_summary=${j3s}
heldout_build=${j7}
heldout_smoke=${j8}
heldout_smoke_summary=${j8s}
hardneg_build=${j9}
hardneg_train=${j10}
hardneg_summary=${j10s}
cross_infonce=${j11}
cross_infonce_summary=${j11s}
EOF
