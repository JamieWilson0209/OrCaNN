#!/bin/bash
# =============================================================================
# Inference, ONE recording per array task (GPU): data/pre_processed ->
# results/infer/<recording_id>/{prob,max_projection}.npy. Parameter-independent:
# runs the segmenter once and caches the probability map, so tuning (segment)
# never re-runs the GPU. Submit:
#   bash hpc/submit.sh infer [config.yaml]
# (counts corrected movies and runs qsub -t 1-N; N=1 is a single recording).
# =============================================================================
#$ -N orcann_infer
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=04:00:00
#$ -q gpu
#$ -l gpu=1
#$ -l h_rss=32G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
module load "${CUDA_MODULE}" 2>/dev/null || echo "note: '${CUDA_MODULE}' unavailable; using torch's bundled CUDA"
set +u; source activate "${ENV_PREFIX}"; set -u

CONFIG="${CONFIG:-config.yaml}"
echo "infer task ${SGE_TASK_ID}: $(date)"
orcann infer --config "${CONFIG}" --task-id "${SGE_TASK_ID}"
echo "task ${SGE_TASK_ID} finished: $(date)"
