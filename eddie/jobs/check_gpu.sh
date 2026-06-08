#!/bin/bash
# =============================================================================
# Verify CUDA on a GPU compute node BEFORE spending a real training slot.
#     source eddie/config.sh && qsub eddie/jobs/check_gpu.sh
# Read logs/orcann_gpucheck.o* — it should print an A100 and "cuda True".
# If it prints "cuda False" on the GPU node, the torch CUDA build is newer than
# the driver: set CUDA_BUILD in config.sh to match nvidia-smi's "CUDA Version"
# and reinstall (setup_eddie.sh).
# For a faster-scheduling test on a MIG slice, swap the GPU request for:
#     #$ -l gpu-mig=1     (instead of -l gpu=1 -l a100=true)
# =============================================================================
#$ -N orcann_gpucheck
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=00:10:00
#$ -q gpu
#$ -l gpu=1
#$ -l a100=true
#$ -l h_rss=16G

set -euo pipefail
source eddie/config.sh

. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
module load "${CUDA_MODULE}"
source activate "${ENV_PREFIX}"

nvidia-smi
python -c "import torch; ok=torch.cuda.is_available(); print('cuda', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO GPU VISIBLE')"
