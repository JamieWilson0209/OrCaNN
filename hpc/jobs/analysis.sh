#!/bin/bash
# =============================================================================
# Group analysis over the transient outputs: within-recording frequency and
# timescale distributions, genotype comparison, and longitudinal-by-day trends.
# A single aggregate job (not a per-recording array); reads results/transients
# and writes results/analysis. CPU work. Submit AFTER detect_transients has run
# for every recording:
#   qsub -v CONFIG=config.yaml hpc/jobs/analysis.sh
#   qsub -v CONFIG=config.yaml,SET="analysis.control_prefix=3" hpc/jobs/analysis.sh
# =============================================================================
#$ -N orcann_analysis
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=01:00:00
#$ -pe sharedmem 2
#$ -l h_rss=8G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ENV_PREFIX}"; set -u

CONFIG="${CONFIG:-config.yaml}"
SETARGS=(--config "${CONFIG}")
for kv in ${SET:-}; do SETARGS+=(--set "${kv}"); done

orcann analysis "${SETARGS[@]}"
