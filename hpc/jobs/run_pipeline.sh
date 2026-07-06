#!/bin/bash
# =============================================================================
# Full pipeline in ONE job (all recordings, serial): motion correction (if
# needed) -> infer -> segment -> activity. Runs in the TORCH env, so
# motion correction must already be done (data/pre_processed populated);
# run_pipeline detects this and skips the caiman step. If pre_processed is empty
# it will try to import caiman and fail — motion-correct first.
#
# This is the convenience path for a single recording or a quick end-to-end run.
# For many recordings in PARALLEL, use the per-recording arrays instead, in order:
#   bash hpc/submit.sh motion_correct  config.yaml   # then wait for it to finish
#   bash hpc/submit.sh infer           config.yaml   # then wait for it to finish
#   bash hpc/submit.sh segment         config.yaml   # then wait for it to finish
#   bash hpc/submit.sh activity config.yaml
# (run_pipeline is not itself arrayed: it chains three differently-indexed stages
# with one task id, and the transients step re-indexes the spatial outputs the
# array would still be writing — racy. The separate arrays avoid that.)
#
# Submit:  qsub -v CONFIG=config.yaml hpc/jobs/run_pipeline.sh
# =============================================================================
#$ -N orcann_pipeline
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=06:00:00
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
orcann run_pipeline --config "${CONFIG}"
