#!/usr/bin/env bash
set -euo pipefail

array_job=$(sbatch --parsable scripts/shanghai_single_ref_matched_bnt_allseed_array.sbatch)
summary_job=$(sbatch --parsable --dependency=afterok:${array_job} scripts/shanghai_single_ref_matched_allseed_summary.sbatch)
stats_job=$(sbatch --parsable --dependency=afterok:${summary_job} scripts/shanghai_publication_stats_summary.sbatch)
audit_job=$(sbatch --parsable --dependency=afterok:${stats_job} scripts/shanghai_aaai_publication_readiness_audit.sbatch)

printf 'single_ref_bnt_array=%s\n' "$array_job"
printf 'single_ref_summary=%s\n' "$summary_job"
printf 'publication_stats=%s\n' "$stats_job"
printf 'aaai_readiness_audit=%s\n' "$audit_job"
