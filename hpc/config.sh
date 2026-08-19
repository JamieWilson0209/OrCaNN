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

# CRLF-safe submission, used by submit.sh and run_chain.sh. A Windows checkout
# (git's default core.autocrlf=true) leaves every line ending in \r, and SGE
# spools the job file verbatim — the job then dies on the node with
# `set -euo pipefail\r` -> "invalid option name", after queueing, which is a slow
# and confusing way to find out. This echoes a path safe to hand to qsub: the
# original when it is clean, or a stripped copy under TMPDIR when it is not. The
# repo file is left untouched (`bash hpc/setup.sh` fixes those in place).
# Warnings go to stderr so command substitution captures only the path.
#
# Defined here rather than duplicated in each submitter; it is a function
# definition, so job scripts that source this file pay nothing for it.
export ORCANN_CRLF_TMPDIR="${TMPDIR:-/tmp}/orcann-crlf-$$"
crlf_safe_job() {
    local job="$1" out
    if grep -q $'\r' "${job}" 2>/dev/null; then
        mkdir -p "${ORCANN_CRLF_TMPDIR}"
        out="${ORCANN_CRLF_TMPDIR}/$(basename "${job}")"
        tr -d '\r' < "${job}" > "${out}"
        echo "note: ${job} has Windows line endings; submitting a stripped copy." >&2
        echo "      run 'bash hpc/setup.sh' to fix the checkout permanently." >&2
        printf '%s\n' "${out}"
    else
        printf '%s\n' "${job}"
    fi
}

# Package caches. The cluster's base pkgs dir is READ ONLY, so conda falls back
# to ~/.conda/pkgs, which sits on a quota-limited home directory (~10 GB on
# Eddie). A caiman solve writes ~200 MB of conda-forge repodata before it
# downloads anything, so a nearly-full home aborts the solve with:
#   OSError: [Errno 122] Disk quota exceeded
# and conda reports it as "An unexpected error has occurred" plus a suggestion
# to disable plugins, which is a red herring. Pointing every cache at scratch
# avoids this. pip's cache is redirected for the same reason: the CUDA torch
# wheels are several GB.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/exports/eddie/scratch/$USER/conda/pkgs}"
export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-/exports/eddie/scratch/$USER/conda/envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/exports/eddie/scratch/$USER/pip-cache}"
export TMPDIR="${TMPDIR:-/exports/eddie/scratch/$USER/tmp}"

# XDG_CACHE_HOME catches what the three variables above miss, because conda does
# NOT keep all of its caches under pkgs_dirs. Channel notices in particular live
# in ~/.cache/conda/notices, and a truncated (0-byte) notices file — what a full
# home quota leaves behind — makes every later conda invocation die on
#     json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
# from conda/notices/cache.py, before it does any work. That is a separate
# failure from the quota error itself, and it outlives the quota being fixed.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/exports/eddie/scratch/$USER/cache}"

# Channel notices are the channel's own announcement banners: no value in a batch
# run, and a fetch + cache write per invocation. Disabling removes the failure
# mode above at the source rather than relocating it.
export CONDA_NUMBER_CHANNEL_NOTICES="${CONDA_NUMBER_CHANNEL_NOTICES:-0}"

# The last write conda makes to home that nothing above relocates. `conda create
# --prefix` registers the new env in ~/.conda/environments.txt, and that path is
# hardcoded (join(userhome, ".conda", "environments.txt")) — CONDA_ENVS_DIRS does
# not move it. It matters because register_env() warns-and-continues only for
# EACCES/EROFS/ENOENT and RE-RAISES anything else, and a full quota is EDQUOT
# (errno 122): the env creation fails outright. The registry is only a
# convenience list for `conda env list`, and envs here are addressed by absolute
# prefix, so switching it off costs nothing — and with it off, setup writes
# nothing to home at all.
export CONDA_REGISTER_ENVS="${CONDA_REGISTER_ENVS:-false}"
