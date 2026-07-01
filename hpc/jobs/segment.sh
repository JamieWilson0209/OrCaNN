#!/bin/bash
# =============================================================================
# Segmentation + extraction, ONE recording per array task (CPU, no model):
# results/infer (cached prob) + movie -> results/spatial/<recording_id>/.
# This is the cheap stage you re-run while tuning threshold / min_radius. Submit
# AFTER the infer array has finished (this indexes the cached prob maps):
#   bash hpc/submit.sh segment [config.yaml]
#
# To preview a parameter grid instead of extracting (overlay montage + table per
# recording), qsub this script directly with a SWEEP variable, e.g.:
#   qsub -t 1-N -v CONFIG=config.yaml,SWEEP='threshold=0.5:0.6:0.7;min_radius=0:2' hpc/jobs/segment.sh
# Semicolons separate axes; values are colon-separated. Use COLONS, not commas:
# qsub -v splits its variable list on commas, so a comma inside SWEEP would be
# eaten and collapse the sweep to its first value. (On the CLI, --sweep
# threshold=0.5,0.6,0.7 with commas is fine.)
# =============================================================================
#$ -N orcann_segment
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=02:00:00
#$ -pe sharedmem 4
#$ -l h_rss=16G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ENV_PREFIX}"; set -u

CONFIG="${CONFIG:-config.yaml}"
SWEEP_ARGS=()
if [ -n "${SWEEP:-}" ]; then
    IFS=';' read -ra _axes <<< "${SWEEP}"
    for ax in "${_axes[@]}"; do SWEEP_ARGS+=(--sweep "${ax}"); done
fi
echo "segment task ${SGE_TASK_ID}: $(date)  ${SWEEP:+(sweep: ${SWEEP})}"
orcann segment --config "${CONFIG}" --task-id "${SGE_TASK_ID}" "${SWEEP_ARGS[@]}"
echo "task ${SGE_TASK_ID} finished: $(date)"
