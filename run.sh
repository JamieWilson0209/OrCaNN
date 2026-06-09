#!/bin/bash
# =============================================================================
# OrCaNN — Unified Entry Point  (batch submission architecture)
# =============================================================================
# Mirrors the proven calcium-pipeline run.sh: a single entry point that
# generates an SGE job script with absolute paths baked in at submit time and
# qsub's it (avoids the "#$ directives can't read shell variables" trap).
#
# Each per-recording job runs the FULL pipeline in one process:
#     movie (.nd2) -> spatial detection -> trace extraction -> transient extraction
# writing one complete result folder per recording. The GROUP analysis across
# recordings is a SEPARATE module, run later via `analyse`.
#
# COMMANDS:
#   single    Full pipeline on ONE recording
#   batch     SGE array: full pipeline on every recording under --data-root
#   analyse   Group analysis across results        (hook; separate module)
#   full      batch + analyse (analysis held until batch completes)
#
# USAGE:
#   bash run.sh batch   --data-root /path/to/nd2_dir
#   bash run.sh single  --movie /path/to/file.nd2
#   bash run.sh analyse --results-dir /path/to/run_XXXX
#   bash run.sh full    --data-root /path/to/nd2_dir
# =============================================================================

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-${SCRIPT_DIR}}"
[ -f "${SCRIPT_DIR}/hpc/config.sh" ] && source "${SCRIPT_DIR}/hpc/config.sh"

SCRATCH_DIR="${WORKSPACE:-$(pwd)}"
OUTPUT_BASE="${OUTPUT_BASE:-${RESULTS_DIR:-${SCRATCH_DIR}/results}}"
LOG_DIR="${LOGS_DIR:-${SCRATCH_DIR}/logs}"
JOB_SCRIPT_DIR="${SCRATCH_DIR}/.job_scripts"

export MPLCONFIGDIR="${SCRATCH_DIR}/.cache/matplotlib"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
mkdir -p "${MPLCONFIGDIR}" 2>/dev/null || true

# Models + per-recording detection params (shell-level; override by flag/env)
SPATIAL_MODEL="${SPATIAL_MODEL:-${MODELS_DIR:-${SCRATCH_DIR}/models}/spatial/detector.pt}"
TEMPORAL_MODEL="${TEMPORAL_MODEL:-${MODELS_DIR:-${SCRATCH_DIR}/models}/temporal/rate_model.pt}"
FRAME_RATE="${FRAME_RATE:-2.0}"
DET_THRESHOLD="${DET_THRESHOLD:-0.5}"
MIN_DISTANCE="${MIN_DISTANCE:-5}"
MIN_PROMINENCE="${MIN_PROMINENCE:-0.5}"
FLOOR_PCT="${FLOOR_PCT:-25}"
MIN_ISI_S="${MIN_ISI_S:-1.0}"
INPUT="${INPUT:-movie}"          # movie = full pipeline (.nd2); traces = temporal-only (interim, on existing temporal_traces.npy)

# SGE resources. Default CPU (device-aware runner; keeps inference off the
# contended GPU queue). For GPU, set e.g. RES="-q gpu -l gpu=1 -l a100=true".
H_RT="${H_RT:-04:00:00}"
N_SLOTS="${N_SLOTS:-4}"
H_RSS="${H_RSS:-8G}"              # per-slot resident memory; total = H_RSS x N_SLOTS
# Optional h_vmem. Current scheduler convention is h_rss-only; the older
# (heavily-tested) config also set h_vmem. Set e.g. H_VMEM=8G to include it if
# your cluster still enforces virtual memory separately. Empty = omit (default).
H_VMEM="${H_VMEM:-}"
_VMEM_LINE=""; [ -n "${H_VMEM}" ] && _VMEM_LINE="\n#\$ -l h_vmem=${H_VMEM}"
RES="${RES:-#\$ -pe sharedmem ${N_SLOTS}\n#\$ -l h_rss=${H_RSS}${_VMEM_LINE}}"

COMMAND="${1:-}"; shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --movie)          MOVIE="$2";          shift 2 ;;
        --data-root)      DATA_ROOT="$2";      shift 2 ;;
        --results-dir)    RESULTS_IN="$2";     shift 2 ;;
        --spatial-model)  SPATIAL_MODEL="$2";  shift 2 ;;
        --temporal-model) TEMPORAL_MODEL="$2"; shift 2 ;;
        --output-base)    OUTPUT_BASE="$2";    shift 2 ;;
        --input)          INPUT="$2";          shift 2 ;;
        --min-prominence) MIN_PROMINENCE="$2"; shift 2 ;;
        --help|-h)        COMMAND="help";      shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ "${INPUT}" = "traces" ]; then FILE_PATTERN="${FILE_PATTERN:-temporal_traces.npy}";
else                               FILE_PATTERN="${FILE_PATTERN:-*.nd2}"; fi

abspath() { [ -n "$1" ] && ( cd "$(dirname "$1")" 2>/dev/null && echo "$(pwd)/$(basename "$1")" ) || echo "$1"; }
absdir()  { [ -n "$1" ] && ( cd "$1" 2>/dev/null && pwd ) || echo "$1"; }
[ -n "${MOVIE:-}" ]      && MOVIE="$(abspath "${MOVIE}")"
[ -n "${DATA_ROOT:-}" ]  && DATA_ROOT="$(absdir "${DATA_ROOT}")"
[ -n "${RESULTS_IN:-}" ] && RESULTS_IN="$(absdir "${RESULTS_IN}")"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}" "${JOB_SCRIPT_DIR}" 2>/dev/null
OUTPUT_BASE="$(absdir "${OUTPUT_BASE}")"

task_cmd() {  # $1 = input file (movie or traces), $2 = out-dir
    if [ "${INPUT}" = "traces" ]; then
        cat << CMD
cd "${CODE_DIR}"
python scripts/run_transients.py \\
    --traces "$1" --model "${TEMPORAL_MODEL}" \\
    --frame-rate ${FRAME_RATE} --min-prominence ${MIN_PROMINENCE} \\
    --floor-pct ${FLOOR_PCT} --min-isi-s ${MIN_ISI_S} \\
    --out-dir "$2"
CMD
    else
        cat << CMD
cd "${CODE_DIR}"
python scripts/run_infer.py \\
    --movie "$1" \\
    --spatial-model "${SPATIAL_MODEL}" --temporal-model "${TEMPORAL_MODEL}" \\
    --frame-rate ${FRAME_RATE} --det-threshold ${DET_THRESHOLD} \\
    --min-distance ${MIN_DISTANCE} --min-prominence ${MIN_PROMINENCE} \\
    --floor-pct ${FLOOR_PCT} --min-isi-s ${MIN_ISI_S} \\
    --out-dir "$2"
CMD
    fi
}

conda_block() {
    cat << CONDA
set +e
source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${ANACONDA_MODULE:-anaconda}" 2>/dev/null || true
source activate "${ENV_PREFIX}" 2>/dev/null || conda activate "${ENV_PREFIX}" 2>/dev/null || true
set -e
export OMP_NUM_THREADS="\${NSLOTS:-${N_SLOTS}}"
echo "Python: \$(which python)  Host: \$(hostname)  Time: \$(date)"
CONDA
}

sge_header() {  # $1 = job name
    printf '#!/bin/bash\n#$ -N %s\n#$ -l h_rt=%s\n' "$1" "${H_RT}"
    printf "${RES}\n"
    printf '#$ -j y\n#$ -o %s\n#$ -V\n' "${LOG_DIR}"
}

show_help() { sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

cmd_single() {
    [ -f "${MOVIE:-}" ] || { echo "ERROR: --movie FILE not found: ${MOVIE:-<unset>}"; exit 1; }
    JOB_SCRIPT="${JOB_SCRIPT_DIR}/single_$(date +%Y%m%d_%H%M%S)_$$.sh"
    { sge_header orcann_single; conda_block; task_cmd "${MOVIE}" "${OUTPUT_BASE}/single"; } > "${JOB_SCRIPT}"
    chmod +x "${JOB_SCRIPT}"
    JID=$(qsub -terse "${JOB_SCRIPT}")
    echo "Submitted single: job ${JID}  ->  ${OUTPUT_BASE}/single/"
}

cmd_batch() {
    [ -d "${DATA_ROOT:-}" ] || { echo "ERROR: --data-root DIR not found: ${DATA_ROOT:-<unset>}"; exit 1; }
    mapfile -t FILES < <(find "${DATA_ROOT}" -name "${FILE_PATTERN}" -type f | sort)
    N_FILES=${#FILES[@]}
    [ "${N_FILES}" -gt 0 ] || { echo "ERROR: no '${FILE_PATTERN}' under ${DATA_ROOT}"; exit 1; }

    echo "============================================================"
    echo "OrCaNN — batch [input=${INPUT}]"
    echo "  data-root:   ${DATA_ROOT}   (pattern: ${FILE_PATTERN})"
    echo "  recordings:  ${N_FILES}"
    [ "${INPUT}" = "traces" ] || echo "  spatial:     ${SPATIAL_MODEL}"
    echo "  temporal:    ${TEMPORAL_MODEL}"
    echo "  output-base: ${OUTPUT_BASE}"
    echo "============================================================"

    JOB_SCRIPT="${JOB_SCRIPT_DIR}/batch_$(date +%Y%m%d_%H%M%S)_$$.sh"
    { sge_header orcann_batch
      cat << CFG

CODE_DIR="${CODE_DIR}"
DATA_ROOT="${DATA_ROOT}"
FILE_PATTERN="${FILE_PATTERN}"
INPUT="${INPUT}"
SPATIAL_MODEL="${SPATIAL_MODEL}"
TEMPORAL_MODEL="${TEMPORAL_MODEL}"
FRAME_RATE="${FRAME_RATE}"
DET_THRESHOLD="${DET_THRESHOLD}"
MIN_DISTANCE="${MIN_DISTANCE}"
MIN_PROMINENCE="${MIN_PROMINENCE}"
FLOOR_PCT="${FLOOR_PCT}"
MIN_ISI_S="${MIN_ISI_S}"
OUTPUT_BASE="${OUTPUT_BASE}"
CFG
      conda_block
      cat << 'BODY'

TASK_ID="${SGE_TASK_ID:-1}"
mapfile -t FILES < <(find "${DATA_ROOT}" -name "${FILE_PATTERN}" -type f | sort)
N_FILES=${#FILES[@]}
if [ "${TASK_ID}" -gt "${N_FILES}" ]; then echo "Task ${TASK_ID} > ${N_FILES}; skip"; exit 0; fi
FILE="${FILES[$((TASK_ID - 1))]}"
RUN_OUT="${OUTPUT_BASE}/run_${JOB_ID:-local}"
echo "Task ${TASK_ID}/${N_FILES}: ${FILE}"
cd "${CODE_DIR}"
if [ "${INPUT}" = "traces" ]; then
    python scripts/run_transients.py \
        --traces "${FILE}" --model "${TEMPORAL_MODEL}" \
        --frame-rate ${FRAME_RATE} --min-prominence ${MIN_PROMINENCE} \
        --floor-pct ${FLOOR_PCT} --min-isi-s ${MIN_ISI_S} \
        --out-dir "${RUN_OUT}"
else
    python scripts/run_infer.py \
        --movie "${FILE}" \
        --spatial-model "${SPATIAL_MODEL}" --temporal-model "${TEMPORAL_MODEL}" \
        --frame-rate ${FRAME_RATE} --det-threshold ${DET_THRESHOLD} \
        --min-distance ${MIN_DISTANCE} --min-prominence ${MIN_PROMINENCE} \
        --floor-pct ${FLOOR_PCT} --min-isi-s ${MIN_ISI_S} \
        --out-dir "${RUN_OUT}"
fi
echo "Task ${TASK_ID} finished: $(date)"
BODY
    } > "${JOB_SCRIPT}"
    chmod +x "${JOB_SCRIPT}"

    BATCH_JOB_ID=$(qsub -t 1-${N_FILES} -terse "${JOB_SCRIPT}")
    BATCH_JOB_ID="${BATCH_JOB_ID%%.*}"
    echo "Submitted batch: job ${BATCH_JOB_ID} (${N_FILES} tasks)"
    echo "  Results:  ${OUTPUT_BASE}/run_${BATCH_JOB_ID}/"
    echo "  Monitor:  qstat -j ${BATCH_JOB_ID}"
    echo "  Progress: ls ${OUTPUT_BASE}/run_${BATCH_JOB_ID}/*/meta.json 2>/dev/null | wc -l"
    export _BATCH_JOB_ID="${BATCH_JOB_ID}"
    export _BATCH_RESULTS="${OUTPUT_BASE}/run_${BATCH_JOB_ID}"
}

cmd_analyse() {
    local RES_DIR="${RESULTS_IN:-${_BATCH_RESULTS:-}}"
    [ -n "${RES_DIR}" ] || { echo "ERROR: --results-dir DIR required"; exit 1; }
    if [ ! -f "${CODE_DIR}/scripts/run_analysis.py" ]; then
        echo "NOTE: scripts/run_analysis.py (group analysis module) not built yet."
        echo "      Per-recording outputs under ${RES_DIR} are its input contract:"
        echo "      spatial_footprints.npz / centroids.npy / temporal_traces.npy /"
        echo "      rates.npy / events.npz / meta.json. Skipping submit."
        return 0
    fi
    JOB_SCRIPT="${JOB_SCRIPT_DIR}/analyse_$(date +%Y%m%d_%H%M%S)_$$.sh"
    { sge_header orcann_analyse; conda_block
      echo "cd \"${CODE_DIR}\""
      echo "python scripts/run_analysis.py --results-dir \"${RES_DIR}\" --output \"${RES_DIR}/analysis\""
    } > "${JOB_SCRIPT}"
    chmod +x "${JOB_SCRIPT}"
    HOLD=""; [ -n "${_BATCH_JOB_ID:-}" ] && HOLD="-hold_jid ${_BATCH_JOB_ID}"
    JID=$(qsub ${HOLD} -terse "${JOB_SCRIPT}")
    echo "Submitted analyse: job ${JID}${_BATCH_JOB_ID:+ (held on ${_BATCH_JOB_ID})}"
}

cmd_full() { cmd_batch; RESULTS_IN="${_BATCH_RESULTS}"; mkdir -p "${RESULTS_IN}"; cmd_analyse; }

case "${COMMAND}" in
    single)          cmd_single  ;;
    batch)           cmd_batch   ;;
    analyse|analyze) cmd_analyse ;;
    full)            cmd_full    ;;
    help|-h|"")      show_help   ;;
    *) echo "Unknown command '${COMMAND}'"; show_help; exit 1 ;;
esac
