#!/bin/bash
# =============================================================================
# Train the spatial segmenter. Inputs/outputs/knobs come from the train_spatial
# section of the YAML config (movies, masks, out, radii, ...). Submit from root:
#   qsub -v CONFIG=config.yaml hpc/jobs/train_spatial.sh
#   qsub -v CONFIG=config.yaml,SET="train_spatial.epochs=40 train_spatial.holdout=false" \
#        hpc/jobs/train_spatial.sh
# =============================================================================
#$ -N orcann_train_spatial
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=08:00:00
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
SETARGS=(--config "${CONFIG}")
for kv in ${SET:-}; do SETARGS+=(--set "${kv}"); done

orcann train_spatial "${SETARGS[@]}"
