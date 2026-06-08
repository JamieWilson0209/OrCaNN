#!/bin/bash
# =============================================================================
# Temporal rate head: leave-one-indicator-out training on public ground truth
# (CPU — the 2 Hz temporal kernels are small). Submit from the repo root:
#     source eddie/config.sh
#     qsub eddie/jobs/train_temporal_loio.sh
# =============================================================================
#$ -N orcann_loio
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=04:00:00
#$ -pe sharedmem 4
#$ -l h_rss=8G

set -euo pipefail
source eddie/config.sh

. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
source activate "${ENV_PREFIX}"

export OMP_NUM_THREADS="${NSLOTS:-4}"

# --indicator-map maps each .mat filename to an indicator label; group by
# indicator AND cell class (e.g. GCaMP6f_exc vs GCaMP6f_pv) — see README.
python scripts/run_train_temporal_loio.py \
    --gt-dir        "${PUBLIC_GT_DIR}" \
    --indicator-map "${PUBLIC_GT_DIR}/indicator_map.json" \
    --target-fs     2.0 \
    --report        "${RESULTS_DIR}/loio/report.json"
