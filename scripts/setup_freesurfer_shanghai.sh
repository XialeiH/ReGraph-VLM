#!/bin/bash
export FREESURFER_HOME=/gpfsnyu/scratch/xh2906/software/freesurfer/7.4.1
export SUBJECTS_DIR=${FREESURFER_HOME}/subjects

# FreeSurfer setup references optional unset variables under strict shells.
_fs_had_nounset=0
case $- in *u*) _fs_had_nounset=1; set +u;; esac
if [ -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ]; then
  source "${FREESURFER_HOME}/SetUpFreeSurfer.sh" || true
fi
if [ "${_fs_had_nounset}" = "1" ]; then set -u; fi
unset _fs_had_nounset
