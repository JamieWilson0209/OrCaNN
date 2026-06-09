#!/bin/bash
# =============================================================================
# Create the workspace directory tree on scratch.
#     source hpc/config.sh && bash hpc/make_workspace.sh
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/config.sh"

mkdir -p \
    "${CODE_DIR}" \
    "${ANNOTATED_DIR}/movies" \
    "${ANNOTATED_DIR}/rois" \
    "${RAW_DIR}" \
    "${PUBLIC_GT_DIR}" \
    "${MODELS_DIR}/spatial" \
    "${MODELS_DIR}/temporal" \
    "${RESULTS_DIR}/spatial_eval" \
    "${RESULTS_DIR}/loio" \
    "${RESULTS_DIR}/inference" \
    "${LOGS_DIR}"

echo "Workspace tree under ${WORKSPACE}:"
find "${WORKSPACE}" -maxdepth 2 -type d | sed "s|${WORKSPACE}|.|"
echo ""
echo "Next: copy the repo into ${CODE_DIR} (git clone or rsync), then run"
echo "  source hpc/config.sh && bash hpc/setup_hpc.sh"
echo "Place annotated movies in ${ANNOTATED_DIR}/movies and ROIs in .../rois,"
echo "raw recordings in ${RAW_DIR}, and CASCADE .mat files in ${PUBLIC_GT_DIR}."
