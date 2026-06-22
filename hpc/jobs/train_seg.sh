#!/bin/bash
# =============================================================================
# Train the spatial SEGMENTER (per-pixel soma probability, GPU). Submit from
# the repo root:
#     source hpc/config.sh
#     qsub hpc/jobs/train_seg.sh
# Per the current Eddie GPU docs (the gpu-a100 PE was removed): request the GPU
# as a RESOURCE in the gpu queue, never as a parallel environment. This is the
# docs' literal one-GPU example: -q gpu + -l gpu=1 + -l h_rss=32G (one CPU core,
# 32 GB system RAM). -l gpu=1 takes an A100 OR L40S, whichever is free (widest
# availability); add -l a100=true / -l l40s=true to pin a type, or -l gpu-mig=1
# for a 20 GB MIG slice. Only add "-pe sharedmem N -l h_rss=<per-core>" if you
# need multiple CPU cores; it is not required and can fail to place on the few
# nodes open to the free project. The trainer streams one movie at a time.
#
# Expects instance-label masks in ${ANNOTATED_DIR}/masks (<stem>.npy from
# scripts/rasterize_rois.py), paired by filename stem to ${ANNOTATED_DIR}/movies.
# =============================================================================
#$ -N orcann_seg
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
# torch ships its own CUDA runtime; only load a system cuda module if one exists,
# and never let a missing module name abort the job under `set -e`.
module load "${CUDA_MODULE}" 2>/dev/null || echo "note: '${CUDA_MODULE}' module unavailable; using torch's bundled CUDA"
set +u
source activate "${ENV_PREFIX}"
set -u

# Knobs, overridable per submission with qsub -v (which DOES reach the batch job,
# unlike a plain shell export). Defaults reproduce the held-out selection run.
#   held-out (default):  qsub hpc/jobs/train_seg.sh
#   final, all data:     qsub -v NO_HOLDOUT=1,EPOCHS=100,OUT_SUB=seg_final hpc/jobs/train_seg.sh
EPOCHS="${EPOCHS:-60}"
OUT_SUB="${OUT_SUB:-seg}"
RADII="${RADII:-3:3.7:4.5:5.5:6.7:8.2:10}"
EXTRA=""
if [ "${NO_HOLDOUT:-0}" = "1" ]; then EXTRA="--no-holdout"; fi
if [ -n "${PIXEL_UM:-}" ]; then EXTRA="${EXTRA} --pixel-um ${PIXEL_UM}"; fi
if [ -n "${MIN_CELL_AREA:-}" ]; then EXTRA="${EXTRA} --min-cell-area ${MIN_CELL_AREA}"; fi

# radii match cell size at 512 (somata ~3-8 px); channels: structural (all
# somata) + max (ROI substrate) + variance (activity).
python scripts/run_train_seg.py \
    --movies   "${ANNOTATED_DIR}/movies" \
    --masks    "${ANNOTATED_DIR}/masks" \
    --channels structural,max,variance \
    --radii    "${RADII}" \
    --patch    128 \
    --epochs   "${EPOCHS}" \
    ${EXTRA} \
    --out      "${MODELS_DIR}/${OUT_SUB}" \
    --report   "${RESULTS_DIR}/spatial_eval/${OUT_SUB}_report.json"
