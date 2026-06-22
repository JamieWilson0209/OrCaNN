#!/bin/bash
# =============================================================================
# Motion correction as an ARRAY job: one recording per task, run in parallel
# (this is how the original pipeline did it). Build a manifest, then submit with
# -t 1-N where N is the number of recordings:
#
#   ls /abs/path/to/nd2/*.nd2 > "$RESULTS_DIR/mc_manifest.txt"
#   qsub -t 1-$(wc -l < "$RESULTS_DIR/mc_manifest.txt") \
#        -v MANIFEST=$RESULTS_DIR/mc_manifest.txt,OUT=$RESULTS_DIR/mc_movies,RESIZE=512,MODE=rigid \
#        hpc/jobs/motion_correct_array.sh
#
# Each task corrects one movie with its own cores, so all recordings process
# concurrently subject to cluster availability. Then segment the OUT dir as usual.
# =============================================================================
#$ -N orcann_mc
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=04:00:00
#$ -pe sharedmem 4
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
NPROC="${NPROC:-4}"
RESIZE="${RESIZE:-0}"        # downscale to NxN before correcting (e.g. 512)
NITER="${NITER:-2}"
: "${MANIFEST:?set MANIFEST=/path/to/manifest.txt via qsub -v (one recording per line)}"

if [ ! -d "${CAIMAN_ENV}" ]; then
    echo "ERROR: caiman env not found: ${CAIMAN_ENV}"; exit 1
fi

MOVIE="$(sed -n "${SGE_TASK_ID}p" "${MANIFEST}")"
[ -n "${MOVIE}" ] || { echo "no recording at line ${SGE_TASK_ID} of ${MANIFEST}"; exit 1; }
if [ ! -f "${MOVIE}" ]; then
    echo "ERROR: not a file: '${MOVIE}'"
    echo "  The manifest must contain ABSOLUTE paths, one per line. Rebuild with:"
    echo "    ls -d /abs/path/to/nd2/*.nd2 > \"\${MANIFEST}\""
    exit 1
fi
echo "task ${SGE_TASK_ID}: ${MOVIE}"

python scripts/run_motion_correct.py \
    --movie       "${MOVIE}" \
    --out         "${OUT}" \
    --mode        "${MODE}" \
    --max-shift   "${MAX_SHIFT}" \
    --resize-to   "${RESIZE}" \
    --niter-rig   "${NITER}" \
    --n-processes "${NPROC}"
