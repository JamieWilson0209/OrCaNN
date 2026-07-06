#!/bin/bash
# =============================================================================
# Submit a per-recording stage as an SGE array — one task per recording. Run on
# a login node, from anywhere:
#
#   bash hpc/submit.sh <stage> [config.yaml]
#
# stages:  motion_correct | infer | segment | activity
#
# It counts the recordings the stage will process — using OrCaNN's own listing,
# so the task indices line up exactly with what the worker picks per task — and
# runs:  qsub -t 1-N hpc/jobs/<stage>.sh
#
# A single recording is just N=1 (a one-task array); nothing special is needed.
# Run the stages in order (each indexes the previous stage's outputs):
#   motion_correct -> infer -> segment -> activity
# To preview a parameter grid before extracting, qsub segment.sh with a SWEEP
# variable directly (see hpc/jobs/segment.sh); submit.sh runs the real extract.
#
# Not handled here (these are not per-recording, submit them directly):
#   qsub -v CONFIG=config.yaml hpc/jobs/run_pipeline.sh     # all recordings, one job
#   qsub -v CONFIG=config.yaml hpc/jobs/train_spatial.sh    # trains one model
#   qsub -v CONFIG=config.yaml hpc/jobs/train_temporal.sh
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
       echo "(run_pipeline and train_* are not per-recording arrays; qsub them directly.)" >&2
       exit 2 ;;
esac
[ -f "${CONFIG}" ] || { echo "config not found: ${CONFIG}" >&2; exit 2; }

# Count with OrCaNN's own enumeration (identical to the worker's list_* calls), in
# the torch env — this needs only orcann + yaml, no GPU and no caiman.
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ENV_PREFIX}"; set -u

N=$(python - "${STAGE}" "${CONFIG}" <<'PY'
import sys
from orcann.configLoader import Config
from orcann.pipeline.cli import (list_recordings, list_infer_recordings,
                                  list_spatial_recordings)
stage, cfgpath = sys.argv[1], sys.argv[2]
cfg = Config.load(cfgpath).resolve_paths()
if stage == "motion_correct":
    n = len(list_recordings(cfg.paths.raw))
elif stage == "infer":
    n = len(list_recordings(cfg.paths.pre_processed))
elif stage == "segment":
    n = len(list_infer_recordings(cfg.paths.infer))
else:  # activity
    n = len(list_spatial_recordings(cfg.paths.spatial))
print(n)
PY
)

if [ "${N}" -lt 1 ]; then
    echo "no recordings to process for '${STAGE}' (config: ${CONFIG})." >&2
    echo "  the stage's input directory is empty — run the previous stage first." >&2
    exit 1
fi

echo "submitting '${STAGE}' as array 1-${N}  (config: ${CONFIG})"
qsub -t 1-"${N}" -v CONFIG="${CONFIG}" "hpc/jobs/${STAGE}.sh"
