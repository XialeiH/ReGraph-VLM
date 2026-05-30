#!/usr/bin/env bash
set -euo pipefail

dependency="${1:-}"
dependency_arg=()
if [[ -n "$dependency" ]]; then
  dependency_arg=(--dependency="afterany:${dependency}")
fi

eval_job=$(sbatch --parsable "${dependency_arg[@]}" scripts/shanghai_single_ref_matched_bnt_eval_checkpoints_array.sbatch)
summary_job=$(sbatch --parsable --dependency=afterok:${eval_job} scripts/shanghai_single_ref_matched_allseed_summary.sbatch)
stats_job=$(sbatch --parsable --dependency=afterok:${summary_job} scripts/shanghai_publication_stats_summary.sbatch)
audit_job=$(sbatch --parsable --dependency=afterok:${stats_job} scripts/shanghai_aaai_publication_readiness_audit.sbatch)

printf 'single_ref_bnt_eval=%s\n' "$eval_job"
printf 'single_ref_summary=%s\n' "$summary_job"
printf 'publication_stats=%s\n' "$stats_job"
printf 'aaai_readiness_audit=%s\n' "$audit_job"
