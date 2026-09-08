#!/bin/bash
# =============================================================================
# The whole pipeline in one command. Run on a login node:
#
#   bash hpc/run_all.sh [--key NAME] [--config config.yaml] [--dry-run]
#
# It asks the pipeline what is already there (`orcann status --json`), submits
# only the stages with work to do, and chains them with -hold_jid so each starts
# when the one above it has finished:
#
#   motion_correction -> infer -> segment -> activity -> analysis
#
# A stage every recording is already current for is not submitted at all, so a
# re-run after a threshold change queues segment, activity and analysis and
# leaves the GPU alone. A stage whose input does not exist yet is submitted
# regardless: its input will exist by the time it runs, which is the whole point
# of the chain.
#
# Every array is sized from the raw recording count, taken once, so task index k
# means recording k in every stage. A stage that ends up with fewer recordings
# than that -- because one failed upstream -- leaves surplus tasks, and those
# exit cleanly saying the index is past the end.
#
# Staging uses whole-array -hold_jid, never -hold_jid_ad: the per-task variant
# would start segment task k while the stage above is still running, and the
# listings are index-into-a-sorted-directory, so a partially written directory
# would make index k mean different things at different times.
#
# Failure policy is continue-and-report. -hold_jid waits for completion, not
# success, so a failed task does not stall the chain; the analysis job counts
# what arrived and names anything missing.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
cd "${REPO_ROOT}"                 # -cwd jobs run from here
source "${HERE}/config.sh"

RUN_KEY=""
CONFIG="config.yaml"
DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        -k|--key)    RUN_KEY="${2:?--key needs a value}"; shift 2 ;;
        -c|--config) CONFIG="${2:?--config needs a value}"; shift 2 ;;
        -n|--dry-run) DRY=1; shift ;;
        -h|--help)
            echo "usage: bash hpc/run_all.sh [--key NAME] [--config config.yaml] [--dry-run]"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2
           echo "usage: bash hpc/run_all.sh [--key NAME] [--config config.yaml] [--dry-run]" >&2
           exit 2 ;;
    esac
done

[ -f "${CONFIG}" ] || { echo "config not found: ${CONFIG}" >&2; exit 2; }

if [ -r /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
    module load "${ANACONDA_MODULE}"
    set +u; source activate "${ENV_PREFIX}"; set -u
elif [ "${DRY}" -eq 0 ]; then
    # A dry run is worth having anywhere -- it only reads the results tree and
    # says what it would submit. Actually submitting is a login node or nothing.
    echo "ERROR: no module system here, so this is not a cluster login node." >&2
    echo "  run_all.sh submits jobs; --dry-run prints the plan without them." >&2
    exit 1
fi

# motion_correction and activity run in the caiman env. Checked before anything
# is queued rather than after the GPU stage has spent an hour waiting.
if [ "${DRY}" -eq 0 ] && [ ! -d "${CAIMAN_ENV}" ]; then
    echo "ERROR: caiman env not found at ${CAIMAN_ENV}" >&2
    echo "  motion_correction and activity need it. Build it with:  bash hpc/setup.sh caiman" >&2
    exit 1
fi

# ${a[@]+"${a[@]}"} rather than "${a[@]}": under set -u an empty array is an
# unbound variable on bash 3.
SET_KEY=()
if [ -n "${RUN_KEY}" ]; then SET_KEY=(--set "run.key=${RUN_KEY}"); fi
SNAPSHOT="$(orcann snapshot --config "${CONFIG}" ${SET_KEY[@]+"${SET_KEY[@]}"})"
RUN_DIR="$(dirname "${SNAPSHOT}")"
RUN_KEY="$(basename "${RUN_DIR}")"
mkdir -p "${RUN_DIR}/logs"

# One tab-separated line per stage, in pipeline order:
#   <stage>  <n_recordings>  <n_todo>  <blocked reason | ->
# Tabs, not spaces: the reason is a sentence, and a stage name has an underscore
# in it that a space-encoding would have to mangle to survive field splitting.
PLAN=$(orcann status --json --config "${SNAPSHOT}" | python -c '
import json, sys
doc = json.load(sys.stdin)
if "error" in doc:
    print(doc["error"], file=sys.stderr)
    raise SystemExit(1)
for s in doc["stages"]:
    print("\t".join([s["stage"], str(s["n_recordings"]), str(s["n_todo"]),
                     s["blocked"] or "-"]))
')

N=$(printf "%s\n" "${PLAN}" | awk -F"\t" '$1=="motion_correction" {print $2}')
if [ "${N}" -lt 1 ]; then
    echo "ERROR: no recordings to process." >&2
    echo "  Nothing in the raw directory, and no corrected movies to fall back on." >&2
    exit 1
fi

echo "run key     : ${RUN_KEY}"
echo "run dir     : ${RUN_DIR}"
echo "config      : ${CONFIG} -> ${SNAPSHOT}"
echo "recordings  : ${N}"
echo
orcann status --config "${SNAPSHOT}"
echo

trap 'rm -rf "${ORCANN_CRLF_TMPDIR}"' EXIT

# The job script for each stage. motion_correction's is named for the submit.sh
# stage word rather than the record's stage name.
job_for () {
    case "$1" in
        motion_correction) echo "hpc/jobs/motion_correct.sh" ;;
        *)                 echo "hpc/jobs/$1.sh" ;;
    esac
}

HOLD=""
QUEUED=0              # whether anything above analysis is going to run
SUBMITTED=()          # "<stage>=<job id>", in submission order
while IFS=$'\t' read -r STAGE n todo blocked; do
    [ "${STAGE}" = "analysis" ] && continue        # not an array; submitted below
    if [ "${blocked}" = "-" ] && [ "${todo}" -eq 0 ] && [ "${n}" -gt 0 ]; then
        printf "%-18s current for %s recording(s), not submitted\n" "${STAGE}" "${n}"
        continue
    fi
    WHY="${todo} to do"
    [ "${blocked}" != "-" ] && WHY="${blocked}"
    JOB="$(job_for "${STAGE}")"
    QUEUED=1
    if [ "${DRY}" -eq 1 ]; then
        printf "%-18s would submit array 1-%s  (%s)\n" "${STAGE}" "${N}" "${WHY}"
        continue
    fi
    QSUB=(qsub -terse -t 1-"${N}" -v CONFIG="${SNAPSHOT}"
          -o "${RUN_DIR}/logs" -e "${RUN_DIR}/logs")
    [ -n "${HOLD}" ] && QSUB+=(-hold_jid "${HOLD}")
    # -terse prints the bare job id. For an array job that is "12345.1-N:1", so
    # trim at the first dot to get the id -hold_jid wants.
    ID=$("${QSUB[@]}" "$(crlf_safe_job "${JOB}")")
    ID="${ID%%.*}"
    printf "%-18s %s  array 1-%s  (%s)%s\n" "${STAGE}" "${ID}" "${N}" "${WHY}" \
           "${HOLD:+  holds on ${HOLD}}"
    SUBMITTED+=("${STAGE}=${ID}")
    HOLD="${ID}"
done <<PLANEOF
${PLAN}
PLANEOF

# analysis is one aggregate job over the whole cohort, never an array.
IFS=$'\t' read -r _ an atodo ablocked <<<"$(printf "%s\n" "${PLAN}" | awk -F"\t" '$1=="analysis"')"
# Analysis is skipped only when nothing above it is running: any stage that does
# run changes the cohort it aggregates, which is what its record is keyed on.
if [ "${ablocked}" = "-" ] && [ "${atodo}" -eq 0 ] && [ "${an}" -gt 0 ] && [ "${QUEUED}" -eq 0 ]; then
    printf "%-18s current for %s recording(s), not submitted\n" "analysis" "${an}"
elif [ "${DRY}" -eq 1 ]; then
    printf "%-18s would submit\n" "analysis"
else
    QSUB=(qsub -terse -v CONFIG="${SNAPSHOT}",EXPECTED_N="${N}"
          -o "${RUN_DIR}/logs" -e "${RUN_DIR}/logs")
    [ -n "${HOLD}" ] && QSUB+=(-hold_jid "${HOLD}")
    ID=$("${QSUB[@]}" "$(crlf_safe_job "hpc/jobs/analysis.sh")")
    ID="${ID%%.*}"
    printf "%-18s %s%s\n" "analysis" "${ID}" "${HOLD:+  holds on ${HOLD}}"
    SUBMITTED+=("analysis=${ID}")
fi

[ "${DRY}" -eq 1 ] && exit 0

{
    echo "run_key=${RUN_KEY}"
    echo "submitted=$(date -Is)"
    echo "config=${SNAPSHOT}"
    echo "recordings=${N}"
    printf "%s\n" ${SUBMITTED[@]+"${SUBMITTED[@]}"}
} > "${RUN_DIR}/jobs.txt"

IDS=$(printf "%s\n" ${SUBMITTED[@]+"${SUBMITTED[@]}"} | cut -d= -f2 | tr '\n' ' ')
echo
echo "watch:   qstat -u \$USER"
echo "cancel:  qdel ${IDS}"
echo "logs:    ${RUN_DIR}/logs"
echo "state:   orcann status --config ${SNAPSHOT}"
