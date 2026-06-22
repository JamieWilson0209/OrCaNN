#!/bin/bash
# =============================================================================
# Run the trained spatial SEGMENTER on new recordings (.nd2 / .tif). Submit from
# the repo root, passing the model and the input/output dirs with qsub -v:
#
#   qsub -v MODEL=models/seg_final/segmenter.pt,MOVIES=/path/to/nd2,OUT=results/seg_infer,THRESH=0.55 \
#        hpc/jobs/seg_infer.sh
#
# GPU request follows the current Eddie docs (gpu queue + gpu resource; the
# gpu-a100 PE was removed). Inference is light, so this schedules and runs fast.
# =============================================================================
#$ -N orcann_seg_infer
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
module load "${CUDA_MODULE}" 2>/dev/null || echo "note: '${CUDA_MODULE}' module unavailable; using torch's bundled CUDA"
set +u
source activate "${ENV_PREFIX}"
set -u

MODEL="${MODEL:-${MODELS_DIR}/seg_final/segmenter.pt}"
OUT="${OUT:-${RESULTS_DIR}/seg_infer}"
THRESH="${THRESH:-0.55}"
RESIZE="${RESIZE:-0}"          # fallback frame size when pixel metadata is absent
: "${MOVIES:?set MOVIES=/path/to/recordings via qsub -v}"

if [ ! -f "${MODEL}" ]; then
    echo "ERROR: model not found: ${MODEL}"
    echo "  Pass an ABSOLUTE path (MODEL=/exports/.../models/seg_final/segmenter.pt)"
    echo "  or omit MODEL to use the default \${MODELS_DIR}/seg_final/segmenter.pt."
    echo "  Relative paths do not work: the job runs from the repo, not the model dir."
    exit 1
fi

TRAIN_UM_ARG=""; [ -n "${TRAIN_UM:-}" ] && TRAIN_UM_ARG="--train-um-per-px ${TRAIN_UM}"
WS_ARG="";       [ "${WATERSHED:-0}" = "1" ] && WS_ARG="--watershed"
MC_ARG="";       [ "${MOTION_CORRECT:-0}" = "1" ] && MC_ARG="--motion-correct --mc-mode ${MC_MODE:-auto}"

python scripts/run_seg_infer.py \
    --model       "${MODEL}" \
    --movies      "${MOVIES}" \
    --out         "${OUT}" \
    --threshold   "${THRESH}" \
    --min-area    "${MIN_AREA:-4}" \
    --min-radius  "${MIN_RADIUS:-0}" \
    ${TRAIN_UM_ARG} ${WS_ARG} ${MC_ARG} \
    --resize-to   "${RESIZE}"
