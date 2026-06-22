#!/bin/bash
# =============================================================================
# Motion-correct recordings (CaImAn NoRMCorre) BEFORE segmentation. Runs in the
# CAIMAN env, not the segmenter env. Submit from the repo root:
#
#   qsub -v MOVIES=/path/to/nd2,OUT=results/mc_movies hpc/jobs/motion_correct.sh
#
# Then segment the corrected movies:
#   qsub -v MOVIES=results/mc_movies,OUT=results/seg_infer hpc/jobs/seg_infer.sh
#
# This is CPU work. NoRMCorre memmaps go to the job-local $TMPDIR (the module
# picks that automatically), so it does not eat your scratch quota.
# =============================================================================
#$ -N orcann_mc
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=08:00:00
#$ -pe sharedmem 8
#$ -l h_rss=16G

set -euo pipefail
source hpc/config.sh

. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u                       # conda/Intel-MPI activate hooks reference unbound vars
source activate "${CAIMAN_ENV}"
set -u

OUT="${OUT:-${RESULTS_DIR}/mc_movies}"
MODE="${MODE:-auto}"
MAX_SHIFT="${MAX_SHIFT:-20}"
NPROC="${NPROC:-8}"
RESIZE="${RESIZE:-0}"        # downscale to NxN before correcting (e.g. 512); big speedup
NITER="${NITER:-2}"         # rigid iterations; 1 is faster
: "${MOVIES:?set MOVIES=/path/to/recordings via qsub -v}"

if [ ! -d "${CAIMAN_ENV}" ]; then
    echo "ERROR: caiman env not found: ${CAIMAN_ENV}"
    echo "  Set CAIMAN_ENV=/abs/path/to/your/caiman/env via qsub -v, or create one."
    exit 1
fi

python scripts/run_motion_correct.py \
    --movies      "${MOVIES}" \
    --out         "${OUT}" \
    --mode        "${MODE}" \
    --max-shift   "${MAX_SHIFT}" \
    --resize-to   "${RESIZE}" \
    --niter-rig   "${NITER}" \
    --n-processes "${NPROC}"
