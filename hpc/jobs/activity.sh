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
# matplotlib stack plus pillow. Running this stage in an env WITHOUT caiman is a
# hard error when deconvolution.method='oasis': OASIS is CaImAn's
# constrained_foopsi, and silently substituting another detector would change the
# results while still reporting success. Set deconvolution.method='robust' (or
# 'threshold') to run without caiman deliberately.
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
orcann_activate_env "${CAIMAN_ENV}" caiman

CONFIG="${CONFIG:-config.yaml}"
echo "activity task ${SGE_TASK_ID}: $(date)"
orcann activity --config "${CONFIG}" --task-id "${SGE_TASK_ID}"
echo "task ${SGE_TASK_ID} finished: $(date)"
