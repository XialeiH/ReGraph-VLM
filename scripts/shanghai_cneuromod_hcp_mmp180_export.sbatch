#!/bin/bash
#SBATCH --job-name=cnm_hcp_export
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/cnm_hcp_export_%j.out
#SBATCH --error=slurm_logs/cnm_hcp_export_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:$PWD/scripts:${PYTHONPATH:-}"

python scripts/export_cneuromod_things_roi_scalar4.py \
  --things-root external_validation/cneuromod_things/metadata_repo/THINGS \
  --conp-base-url https://132.206.37.91/users/cneuromod/cneuromod.all \
  --conp-host-header sftp.conp.ca \
  --atlas external_validation/atlases/hcp_mmp1_fsl/MNI_Glasser_HCP_v1.0.nii.gz \
  --out-dir external_validation/cneuromod_things/hcp_mmp180_scalar4_all4 \
  --subjects sub-01 sub-02 sub-03 sub-06 \
  --space T1w \
  --atlas-to-beta-space mni2009c_to_t1w \
  --max-shared-images 0 \
  --progress-every 250
