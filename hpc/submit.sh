#!/bin/bash
# =============================================================================
# Submit a per-recording stage as an SGE array — one task per recording. Run on
# a login node, from anywhere:
#
#   bash hpc/submit.sh <stage> [config.yaml]
#
# stages:  motion_correct | infer | segment | activity
#
# It writes the run's config snapshot (results/runs/<run.key>/config.yaml), counts
# the recordings the stage will process — using OrCaNN's own listing, so the task
# indices line up exactly with what the worker picks per task — and runs:
#   qsub -t 1-N -v CONFIG=<the snapshot> hpc/jobs/<stage>.sh
#
# The jobs read the snapshot rather than config.yaml, so editing the config while
# the array sits in the queue cannot change what the queued tasks compute.
#
# For the whole pipeline in one command, submitting only the stages with work to
# do, use hpc/run_all.sh. This script submits one stage.
#
# A single recording is just N=1 (a one-task array); nothing special is needed.
# Run the stages in order (each indexes the previous stage's outputs):
#   motion_correct -> infer -> segment -> activity
# To preview a parameter grid before extracting, qsub segment.sh with a SWEEP
# variable directly (see hpc/jobs/segment.sh); submit.sh runs the real extract.
#
# Not handled here (these are not per-recording, submit them directly):
#   qsub -v CONFIG=config.yaml hpc/jobs/train_spatial.sh    # trains one model
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
cd "${REPO_ROOT}"                 # -cwd jobs run from here; the count is repo-relative
source "${HERE}/config.sh"

STAGE="${1:?usage: bash hpc/submit.sh <stage> [config.yaml]   stage: motion_correct|infer|segment|activity}"
CONFIG="${2:-config.yaml}"

case "${STAGE}" in
    motion_correct|infer|segment|activity) ;;
    *) echo "unknown stage '${STAGE}'. Use: motion_correct | infer | segment | activity" >&2
       echo "(train_* are not per-recording arrays; qsub them directly. For a full" >&2
        echo " chained run use: bash hpc/run_all.sh)" >&2
       exit 2 ;;
esac
[ -f "${CONFIG}" ] || { echo "config not found: ${CONFIG}" >&2; exit 2; }

# Count with OrCaNN's own enumeration (identical to the worker's list_* calls), in
# the torch env — this needs only orcann + yaml, no GPU and no caiman.
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ENV_PREFIX}"; set -u

# The snapshot first: the count and every task must read one file that cannot
# change under them once the array is queued.
SNAPSHOT="$(orcann snapshot --config "${CONFIG}")"
echo "config      : ${CONFIG} -> ${SNAPSHOT}"

N=$(python - "${STAGE}" "${SNAPSHOT}" <<'PY'
import sys
from orcann.configLoader import Config
from orcann.pipeline.provenance import ProvenanceError, stage_inputs
stage, cfgpath = sys.argv[1], sys.argv[2]
cfg = Config.load(cfgpath).resolve_paths()
# stdout carries the count and nothing else -- the caller compares it as an
# integer, and a line of explanation mixed in would make that test fail open and
# submit an array sized from an error message. The reason goes to stderr.
try:
    n = len(stage_inputs(cfg, stage))
except ProvenanceError as e:
    print(e, file=sys.stderr)
    n = 0
print(n)
PY
)

if [ "${N}" -lt 1 ]; then
    echo "no recordings to process for '${STAGE}' (config: ${CONFIG})." >&2
    echo "  the stage's input directory is empty — run the previous stage first." >&2
    exit 1
fi

trap 'rm -rf "${ORCANN_CRLF_TMPDIR}"' EXIT

echo "submitting '${STAGE}' as array 1-${N}  (config: ${SNAPSHOT})"
qsub -t 1-"${N}" -v CONFIG="${SNAPSHOT}" "$(crlf_safe_job "hpc/jobs/${STAGE}.sh")"
