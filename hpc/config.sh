# =============================================================================
# HPC configuration (SGE / Grid Engine) — edit, then `source hpc/config.sh`
# before submitting. Every value can be overridden from the environment
# (export X=... first).
# =============================================================================

# Project workspace. Defaults to the cluster's scratch space ($SCRATCH) if set,
# else $HOME. Override WORKSPACE to point somewhere specific. NOTE: scratch is
# often purged after a period of inactivity on HPC systems — keep the conda env
# and trained models there only if used regularly, otherwise point
# ENV_PREFIX/MODELS_DIR at longer-term/group storage.
export WORKSPACE="${WORKSPACE:-/exports/eddie/scratch/$USER/calcineps_workspace}"

export CODE_DIR="${CODE_DIR:-${WORKSPACE}/code}"            # the orcann repo
export ENV_PREFIX="${ENV_PREFIX:-${WORKSPACE}/env/calcineps}"  # conda prefix env
# Motion correction (NoRMCorre) needs caiman, which was removed from the env
# above; it lives in a separate env. Activate this one only for the MC step.
export CAIMAN_ENV="${CAIMAN_ENV:-/exports/eddie/scratch/$USER/conda/envs/caiman}"

# Data
export ANNOTATED_DIR="${ANNOTATED_DIR:-${WORKSPACE}/data/annotated}"   # movies/ + rois/
export RAW_DIR="${RAW_DIR:-${WORKSPACE}/data/raw}"                     # raw recordings
export PUBLIC_GT_DIR="${PUBLIC_GT_DIR:-${WORKSPACE}/data/public_gt}"   # CASCADE .mat

# Outputs
export MODELS_DIR="${MODELS_DIR:-${WORKSPACE}/models}"
export RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE}/results}"
export LOGS_DIR="${LOGS_DIR:-${WORKSPACE}/logs}"

# Module names — CONFIRM against `module avail` on your cluster.
export ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda}"
export CUDA_MODULE="${CUDA_MODULE:-cuda}"

# GPU request model. On this SGE setup, GPU jobs use the directives
#     #$ -q gpu
#     #$ -l gpu=1        (number of GPUs)
#     #$ -l a100=true    (pin a GPU type; or l40s=true; or -l gpu-mig=1 for a MIG slice)
# rather than a -pe parallel environment. CPU cores are requested with
# -pe sharedmem N. These live in the job scripts directly (SGE #$ directives
# can't read shell variables). The specific resource labels above are examples —
# confirm the correct queue/resource names for your cluster.

# PyTorch CUDA build. cu121 is a conservative, broadly driver-compatible choice;
# bump only if a GPU-node check shows a newer CUDA (nvidia-smi "CUDA Version",
# via jobs/check_gpu.sh).
export CUDA_BUILD="${CUDA_BUILD:-cu121}"

# Keep ~/.local user-site packages from leaking into the prefix env.
export PYTHONNOUSERSITE=1
