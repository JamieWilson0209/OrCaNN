#!/bin/bash
# =============================================================================
# Launch a full segment -> activity -> analysis run in one command, with every
# output collected under a single timestamped directory. Run on a login node:
#
#   bash hpc/run_chain.sh [--label NAME] [--config config.yaml]
#
# It creates  results_<timestamp>[_<label>]/  containing:
#   config.yaml   the exact config the run used (a snapshot, see below)
#   logs/         stdout/stderr for every task of every stage
#   spatial/      segment output, per recording
#   activity/     activity output, per recording
#   analysis/     group figures and tables
#   jobs.txt      the SGE job IDs, so the run can be cancelled or chased
#
# Infer is deliberately NOT part of this chain. It is the expensive GPU stage
# and its output is parameter-independent, so it is run once per dataset and
# reused across every parameter experiment:
#
#   bash hpc/submit.sh infer            # once, beforehand
#   bash hpc/run_chain.sh --label t04   # as many times as you like
#
# The chain therefore reads the SHARED paths.infer cache and writes only the
# three downstream stages into the run directory.
#
# Staging: activity holds on segment and analysis holds on activity, using
# whole-array -hold_jid. Do not switch these to -hold_jid_ad: the per-task
# variant would start segment task k while infer/segment is still running, and
# the stage listings are index-into-a-sorted-directory, so a partially written
# directory would make index k mean different things at different times.
#
# Failure policy is continue-and-report. -hold_jid waits for completion, not
# success, so a failed task does not stall the chain; the analysis job counts
# what actually arrived and prints a WARNING naming any missing recordings.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
cd "${REPO_ROOT}"                 # -cwd jobs run from here
source "${HERE}/config.sh"

LABEL=""
CONFIG="config.yaml"
while [ $# -gt 0 ]; do
    case "$1" in
        -l|--label)  LABEL="${2:?--label needs a value}"; shift 2 ;;
        -c|--config) CONFIG="${2:?--config needs a value}"; shift 2 ;;
        -h|--help)
            echo "usage: bash hpc/run_chain.sh [--label NAME] [--config config.yaml]"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2
           echo "usage: bash hpc/run_chain.sh [--label NAME] [--config config.yaml]" >&2
           exit 2 ;;
    esac
done

[ -f "${CONFIG}" ] || { echo "config not found: ${CONFIG}" >&2; exit 2; }

# Label is used in a directory name: keep it to safe characters.
if [ -n "${LABEL}" ] && ! [[ "${LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: --label may contain only letters, digits, dot, dash, underscore." >&2
    exit 2
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)${LABEL:+_${LABEL}}"
RUN_DIR="${REPO_ROOT}/results_${RUN_ID}"

# A timestamp to the second should never collide, so an existing directory means
# two launches in the same second (or a hand-made directory). Refuse rather than
# silently merge two runs' outputs.
if [ -e "${RUN_DIR}" ]; then
    echo "ERROR: ${RUN_DIR} already exists." >&2
    echo "  Two runs launched in the same second, or the directory was created by hand." >&2
    echo "  Re-run with a distinct label, e.g.:  bash hpc/run_chain.sh --label ${LABEL:-myrun}2" >&2
    exit 1
fi

. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ENV_PREFIX}"; set -u

# activity runs in the caiman env; check it now rather than after segment has
# spent an hour on the queue.
if [ ! -d "${CAIMAN_ENV}" ]; then
    echo "ERROR: caiman env not found at ${CAIMAN_ENV}" >&2
    echo "  The activity stage needs it. Build it with:  bash hpc/setup.sh caiman" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}/logs"
SNAPSHOT="${RUN_DIR}/config.yaml"

# Snapshot the config into the run directory and count the work.
#
# The snapshot is written with every path already made ABSOLUTE. That matters
# twice over. Relative paths in a config resolve against that config file's own
# directory, so a naive copy into the run directory would re-root data/ and
# models/ underneath it; and resolving now means the jobs are immune to
# config.yaml being edited between launch and the jobs actually starting, which
# is the normal way of working when sweeping parameters.
N=$(python - "${CONFIG}" "${RUN_DIR}" "${SNAPSHOT}" <<'PY'
import os, sys
from orcann.configLoader import Config
from orcann.pipeline.cli import list_infer_recordings

src, run_dir, snapshot = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = Config.load(src).resolve_paths()

# Only the three downstream stages are redirected. paths.infer stays pointed at
# the shared cache so every run reuses the same probability maps.
cfg.paths.spatial = os.path.join(run_dir, "spatial")
cfg.paths.activity = os.path.join(run_dir, "activity")
cfg.paths.analysis = os.path.join(run_dir, "analysis")
cfg.dump(snapshot)

print(len(list_infer_recordings(cfg.paths.infer)))
PY
)

if [ "${N}" -lt 1 ]; then
    echo "ERROR: no cached probability maps found, so there is nothing to segment." >&2
    echo "  Run the infer stage first:  bash hpc/submit.sh infer ${CONFIG}" >&2
    rm -rf "${RUN_DIR}"
    exit 1
fi

QSUB_COMMON=(-v CONFIG="${SNAPSHOT}" -o "${RUN_DIR}/logs" -e "${RUN_DIR}/logs")

echo "run id      : ${RUN_ID}"
echo "run dir     : ${RUN_DIR}"
echo "config      : ${CONFIG} -> ${SNAPSHOT}"
echo "recordings  : ${N}"
echo

# -terse prints the bare job id. For an array job that is "12345.1-N:1", so trim
# at the first dot to get the id -hold_jid wants.
SEG_ID=$(qsub -terse -t 1-"${N}" "${QSUB_COMMON[@]}" hpc/jobs/segment.sh)
SEG_ID="${SEG_ID%%.*}"
echo "segment     : ${SEG_ID}  (array 1-${N})"

ACT_ID=$(qsub -terse -t 1-"${N}" -hold_jid "${SEG_ID}" "${QSUB_COMMON[@]}" hpc/jobs/activity.sh)
ACT_ID="${ACT_ID%%.*}"
echo "activity    : ${ACT_ID}  (array 1-${N}, holds on ${SEG_ID})"

# activity's own task count cannot be known at launch (its input is segment's
# output, which does not exist yet), so it reuses N. Surplus tasks are harmless:
# an out-of-range --task-id yields an empty listing and the stage exits cleanly.
ANA_ID=$(qsub -terse -hold_jid "${ACT_ID}" -v CONFIG="${SNAPSHOT}",EXPECTED_N="${N}" \
              -o "${RUN_DIR}/logs" -e "${RUN_DIR}/logs" hpc/jobs/analysis.sh)
ANA_ID="${ANA_ID%%.*}"
echo "analysis    : ${ANA_ID}  (holds on ${ACT_ID})"

{
    echo "run_id=${RUN_ID}"
    echo "submitted=$(date -Is)"
    echo "config=${SNAPSHOT}"
    echo "recordings=${N}"
    echo "segment=${SEG_ID}"
    echo "activity=${ACT_ID}"
    echo "analysis=${ANA_ID}"
} > "${RUN_DIR}/jobs.txt"

echo
echo "watch:   qstat -u \$USER"
echo "cancel:  qdel ${SEG_ID} ${ACT_ID} ${ANA_ID}"
echo "logs:    ${RUN_DIR}/logs"
