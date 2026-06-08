# =============================================================================
# Eddie configuration — edit, then `source eddie/config.sh` before submitting.
# Every value can be overridden from the environment (export X=... first).
# =============================================================================

# University login name (UUN). Set this — either edit here or `export UUN=...`
# before sourcing. The default is a placeholder and will not resolve to a real
# scratch path.
export UUN="${UUN:-CHANGE_ME}"

# Project workspace on scratch. NOTE: Eddie scratch is purged after ~1 month of
# no access. Keep the conda env and trained models here only if you use them
# regularly; otherwise point ENV_PREFIX/MODELS at group/DataStore space.
export WORKSPACE="${WORKSPACE:-/exports/eddie/scratch/${UUN}/orcann_workspace}"

export CODE_DIR="${CODE_DIR:-${WORKSPACE}/code}"        # the orcann repo
export ENV_PREFIX="${ENV_PREFIX:-${WORKSPACE}/env/orcann}"   # conda prefix env

# Data
export ANNOTATED_DIR="${ANNOTATED_DIR:-${WORKSPACE}/data/annotated}"   # movies/ + rois/
export RAW_DIR="${RAW_DIR:-${WORKSPACE}/data/raw}"                     # raw recordings
export PUBLIC_GT_DIR="${PUBLIC_GT_DIR:-${WORKSPACE}/data/public_gt}"   # CASCADE .mat

# Outputs
export MODELS_DIR="${MODELS_DIR:-${WORKSPACE}/models}"
export RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE}/results}"
export LOGS_DIR="${LOGS_DIR:-${WORKSPACE}/logs}"

# Module names — CONFIRM against `module avail` on the current cluster.
export ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda}"
export CUDA_MODULE="${CUDA_MODULE:-cuda}"

# GPU request model (Eddie's current scheduler): GPU jobs use the directives
#     #$ -q gpu
#     #$ -l gpu=1        (number of GPUs; up to 4)
#     #$ -l a100=true    (pin A100; or l40s=true; or use -l gpu-mig=1 for a MIG slice)
# The old `gpu-a100` parallel environment has been RETIRED — do not use -pe for
# GPUs. CPU cores are still requested with -pe sharedmem N. These live in the
# job scripts directly (SGE #$ directives can't read shell variables).

# PyTorch CUDA build. cu121 is the conservative, broadly driver-compatible
# choice for the A100/L40S nodes; bump only if a GPU-node check shows a newer
# CUDA (nvidia-smi "CUDA Version", via jobs/check_gpu.sh).
export CUDA_BUILD="${CUDA_BUILD:-cu121}"

# Keep ~/.local user-site packages from leaking into the prefix env.
export PYTHONNOUSERSITE=1
