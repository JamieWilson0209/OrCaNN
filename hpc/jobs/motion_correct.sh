#!/bin/bash
# =============================================================================
# Motion correction (CaImAn NoRMCorre), ONE recording per array task, in the
# CAIMAN env: data/raw -> data/pre_processed (<stem>_mc.tif). Submit as an array:
#   bash hpc/submit.sh motion_correct [config.yaml]
# (counts raw recordings and runs qsub -t 1-N; N=1 is a single recording).
# NoRMCorre memmaps go to job-local $TMPDIR, so scratch quota is untouched.
# =============================================================================
#$ -N orcann_mc
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=04:00:00
#$ -pe sharedmem 4
#$ -l h_rss=16G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"

if [ ! -d "${CAIMAN_ENV}" ]; then
    echo "ERROR: caiman env not found: ${CAIMAN_ENV}"
    echo "  Create it once with: bash hpc/setup.sh caiman"
    echo "  (or point CAIMAN_ENV=/abs/path/to/an/existing/caiman/env via qsub -v)"
    exit 1
fi
set +u; source activate "${CAIMAN_ENV}"; set -u

CONFIG="${CONFIG:-config.yaml}"
echo "motion_correct task ${SGE_TASK_ID}: $(date)"
orcann motion_correction --config "${CONFIG}" --task-id "${SGE_TASK_ID}"
echo "task ${SGE_TASK_ID} finished: $(date)"
