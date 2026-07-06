#!/bin/bash
# =============================================================================
# Activity stage, ONE recording per array task (CPU, no model):
# results/spatial + movie -> results/activity/<recording_id>/. Baseline-corrects
# the per-ROI traces to dF/F0, infers spike trains with OASIS, and renders the
# per-recording interactive HTML gallery. Submit AFTER the segment array has
# finished (this indexes the segmented recordings):
#   bash hpc/submit.sh activity [config.yaml]
#
# OASIS is CaImAn's deconvolution, so this stage runs in the CAIMAN env (the same
# env motion correction uses), NOT the torch env. It is torch-free: baseline,
# OASIS, and the HTML gallery need only caiman's numpy/scipy/scikit-image/
# matplotlib stack plus pillow. If run in an env without caiman (e.g. the one-job
# run_pipeline path in the torch env), OASIS import fails and deconvolution falls
# back to the dependency-free `threshold` method with a warning - no crash. Set
# deconvolution.method=threshold to skip OASIS entirely.
# =============================================================================
#$ -N orcann_activity
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=02:00:00
#$ -pe sharedmem 4
#$ -l h_rss=16G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${CAIMAN_ENV}"; set -u

CONFIG="${CONFIG:-config.yaml}"
echo "activity task ${SGE_TASK_ID}: $(date)"
orcann activity --config "${CONFIG}" --task-id "${SGE_TASK_ID}"
echo "task ${SGE_TASK_ID} finished: $(date)"
