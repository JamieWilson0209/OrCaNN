# =============================================================================
# HPC configuration (SGE / Grid Engine) — edit, then `source hpc/config.sh`
# before submitting. Every value can be overridden from the environment
# (export X=... first).
#
# This file now carries ONLY what the job scripts actually read at run time: the
# two conda env locations, the module names, and the torch CUDA build. The
# workspace layout (raw/pre_processed/results/models) is NOT here — it lives in
# the repo itself and is resolved from config.yaml's location. Point those paths
# at scratch or group storage by editing config.yaml, not this file.
# =============================================================================

# Conda envs. Kept OUTSIDE the repo on purpose: they are large and rebuildable,
# and scratch is often purged after inactivity, so place them somewhere you are
# willing to recreate (or point at longer-term/group storage). Two separate envs:
#   - ENV_PREFIX : torch env for segmentation/detection/training (created by
#                  hpc/setup.sh). The job scripts `source activate` this.
#   - CAIMAN_ENV : caiman env for NoRMCorre motion correction ONLY (created by
#                  hpc/setup.sh caiman); the motion_correct*.sh jobs activate it
#                  themselves. Never needed for training or segmentation.
export ENV_PREFIX="${ENV_PREFIX:-/exports/eddie/scratch/$USER/conda/envs/calcineps}"
export CAIMAN_ENV="${CAIMAN_ENV:-/exports/eddie/scratch/$USER/conda/envs/caiman}"

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
