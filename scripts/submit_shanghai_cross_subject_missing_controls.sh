#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"

jid_rand_build=$(sbatch scripts/shanghai_build_random_embedding_control.sbatch | awk '{print $4}')
jid_rand_train=$(sbatch --dependency=afterok:${jid_rand_build} scripts/shanghai_cross_subject_random_embedding_array.sbatch | awk '{print $4}')
jid_rand_sum=$(sbatch --dependency=afterok:${jid_rand_train} scripts/shanghai_cross_subject_random_embedding_summary.sbatch | awk '{print $4}')
jid_raw=$(sbatch scripts/shanghai_cross_subject_raw_allfold.sbatch | awk '{print $4}')

cat <<EOF
random_embedding_build=${jid_rand_build}
random_embedding_train_array=${jid_rand_train}
random_embedding_summary=${jid_rand_sum}
raw_allfold=${jid_raw}
EOF
