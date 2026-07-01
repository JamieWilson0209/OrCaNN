#!/bin/bash
# =============================================================================
# Temporal detection, ONE spatial output per array task (CPU): results/spatial ->
# results/transients. The temporal model runs on CPU. Submit AFTER the
# segment array has finished (this indexes the spatial outputs):
#   bash hpc/submit.sh detect_transients [config.yaml]
# (counts spatial outputs and runs qsub -t 1-N; N=1 is a single recording).
# =============================================================================
#$ -N orcann_detect_transients
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=02:00:00
#$ -pe sharedmem 4
#$ -l h_rss=8G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ENV_PREFIX}"; set -u

CONFIG="${CONFIG:-config.yaml}"
echo "detect_transients task ${SGE_TASK_ID}: $(date)"
orcann detect_transients --config "${CONFIG}" --task-id "${SGE_TASK_ID}"
echo "task ${SGE_TASK_ID} finished: $(date)"
