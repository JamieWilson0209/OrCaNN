#!/bin/bash
# =============================================================================
# Group analysis over the activity outputs: within-recording frequency and
# timescale distributions, genotype comparison, and longitudinal-by-day trends.
# A single aggregate job (not a per-recording array); reads results/activity
# and writes results/analysis. CPU work. Submit AFTER activity has run
# for every recording:
#   qsub -v CONFIG=config.yaml hpc/jobs/analysis.sh
#
# Normally submitted for you by hpc/run_all.sh, which holds it behind the
# activity array and passes EXPECTED_N so a shortfall gets reported.
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
orcann_activate_env "${ENV_PREFIX}" main

CONFIG="${CONFIG:-config.yaml}"
SETARGS=(--config "${CONFIG}")
for kv in ${SET:-}; do SETARGS+=(--set "${kv}"); done

# When launched by hpc/run_all.sh, EXPECTED_N is the recording count taken at
# launch. Upstream array tasks that fail exit without writing their output, and
# the later stages simply list a shorter directory, so a shortfall is otherwise
# silent. This is the one place in the chain that sees every stage's result, so
# report it here before the analysis itself runs.
if [ -n "${EXPECTED_N:-}" ]; then
    python - "${CONFIG}" "${EXPECTED_N}" <<'PY' || true
import os, sys
from orcann.configLoader import Config
from orcann.pipeline import provenance as prov

cfg = Config.load(sys.argv[1]).resolve_paths()
expected_n = int(sys.argv[2])

wanted = prov.stage_inputs(cfg, "activity")
act = prov.store_dir(cfg.paths.activity, prov.chain(cfg).activity)
got = set(prov.list_dir_records(act, prov.RUN_INFO, prov.STAGE_ACTIVITY))

if len(got) < expected_n:
    missing = [r for r in wanted if r not in got]
    print(f"WARNING: expected {expected_n} recording(s), found {len(got)}.")
    if missing:
        print("WARNING: no activity output for: " + ", ".join(missing))
    print("WARNING: check the segment/activity logs for these; "
          "analysis continues on what is present.")
else:
    print(f"all {len(got)} recording(s) present.")
PY
fi

orcann analysis "${SETARGS[@]}"
