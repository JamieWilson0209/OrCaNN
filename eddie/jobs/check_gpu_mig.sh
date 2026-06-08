#!/bin/bash
# =============================================================================
# Like check_gpu.sh but requests a MIG slice (20 GB A100 partition) — schedules
# far faster than a full A100 when the GPU nodes are busy. Good for quick CUDA
# verification and light testing.   qsub eddie/jobs/check_gpu_mig.sh
# =============================================================================
#$ -N orcann_gpucheck_mig
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=00:10:00
#$ -q gpu
#$ -l gpu-mig=1
#$ -l h_rss=16G

set -euo pipefail
source eddie/config.sh

. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
module load "${CUDA_MODULE}"
source activate "${ENV_PREFIX}"

nvidia-smi
python -c "import torch; ok=torch.cuda.is_available(); print('cuda', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO GPU VISIBLE')"
