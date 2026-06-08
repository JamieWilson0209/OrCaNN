#!/bin/bash
# =============================================================================
# Full two-stage inference over many recordings (array job). One task per
# recording listed in a manifest. Build the manifest, then submit with a range
# matching its length:
#     source eddie/config.sh
#     ls "${RAW_DIR}"/*.nd2 > "${RAW_DIR}/manifest.txt"
#     qsub -t 1-$(wc -l < "${RAW_DIR}/manifest.txt") eddie/jobs/infer_array.sh
# =============================================================================
#$ -N orcann_infer
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=02:00:00
#$ -pe sharedmem 4
#$ -l h_rss=16G

set -euo pipefail
source eddie/config.sh

. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
source activate "${ENV_PREFIX}"
export OMP_NUM_THREADS="${NSLOTS:-4}"

MANIFEST="${MANIFEST:-${RAW_DIR}/manifest.txt}"
MOVIE="$(sed -n "${SGE_TASK_ID}p" "${MANIFEST}")"
NAME="$(basename "${MOVIE}")"; NAME="${NAME%.*}"

python scripts/run_infer.py \
    --movie          "${MOVIE}" \
    --spatial-model  "${MODELS_DIR}/spatial/detector.pt" \
    --temporal-model "${MODELS_DIR}/temporal/rate_model.pt" \
    --frame-rate     2.0 \
    --out            "${RESULTS_DIR}/inference/${NAME}.npz"
