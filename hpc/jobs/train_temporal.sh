#!/bin/bash
# =============================================================================
# Train / evaluate the temporal rate head (leave-one-indicator-out by default;
# set train_temporal.save_final to instead fit one model on all data and save it).
# CPU work. Inputs/knobs come from the train_temporal config section. Submit:
#   qsub -v CONFIG=config.yaml hpc/jobs/train_temporal.sh
#   qsub -v CONFIG=config.yaml,SET="train_temporal.save_final=models/temporal/rate_model.pt" \
#        hpc/jobs/train_temporal.sh
# =============================================================================
#$ -N orcann_train_temporal
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=04:00:00
#$ -pe sharedmem 4
#$ -l h_rss=8G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ENV_PREFIX}"; set -u

CONFIG="${CONFIG:-config.yaml}"
SETARGS=(--config "${CONFIG}")
for kv in ${SET:-}; do SETARGS+=(--set "${kv}"); done

orcann train_temporal "${SETARGS[@]}"
