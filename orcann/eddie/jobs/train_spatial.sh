#!/bin/bash
# =============================================================================
# Train the spatial scattering detector (GPU). Submit from the repo root:
#     source eddie/config.sh
#     qsub eddie/jobs/train_spatial.sh
# GPU request follows Eddie's current scheduler: -q gpu + -l gpu=1 (+ a100=true
# to pin A100s; drop it to accept an L40S, or use -l gpu-mig=1 for a MIG slice).
# h_rss is system RAM per CPU core (default 1 core); the streaming trainer needs
# only ~1-2 GB system RAM — the 80 GB A100 GPU RAM is separate and automatic.
# =============================================================================
#$ -N orcann_spatial
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=08:00:00
#$ -q gpu
#$ -l gpu=1
#$ -l a100=true
#$ -l h_rss=32G

set -euo pipefail
source eddie/config.sh

. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
module load "${CUDA_MODULE}"
source activate "${ENV_PREFIX}"

# Channels: structural (all somata) + max (ROI substrate) + variance (activity);
# add ,coherence to test the off-diagonal faint-cell channel.
python scripts/run_train_spatial.py \
    --movies   "${ANNOTATED_DIR}/movies" \
    --rois     "${ANNOTATED_DIR}/rois" \
    --channels structural,max,variance \
    --epochs   60 \
    --out      "${MODELS_DIR}/spatial" \
    --report   "${RESULTS_DIR}/spatial_eval/report.json"
