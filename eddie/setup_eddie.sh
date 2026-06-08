#!/bin/bash
# =============================================================================
# One-time environment setup on Eddie. Run from the repo root on a login node:
#     source eddie/config.sh
#     bash eddie/setup_eddie.sh
# A login node is fine for installs; do NOT run training here.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/config.sh"

. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"

echo "Creating conda env at ${ENV_PREFIX}"
mkdir -p "$(dirname "${ENV_PREFIX}")"
if [ ! -d "${ENV_PREFIX}" ]; then
    conda create --yes --prefix "${ENV_PREFIX}" python=3.11
fi
# `source activate` is the robust form for prefix envs in non-interactive shells
source activate "${ENV_PREFIX}"

# PyTorch: install the CUDA build matching the cluster (CUDA_BUILD in config.sh).
# cu121 is the safe default for the A100 nodes; verify with jobs/check_gpu.sh and
# bump if the node reports a newer CUDA. PYTHONNOUSERSITE (from config) stops
# ~/.local leaking into the env.
python -m pip install --upgrade pip
python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_BUILD}"

# Install the project (editable) + its dependencies from pyproject.toml.
python -m pip install -e "${CODE_DIR}"

echo "Done. Verify:"
python -c "import torch, orcann; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
