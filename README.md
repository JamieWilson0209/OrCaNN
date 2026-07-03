# OrCaNN

A calcium-imaging analysis pipeline for organoid recordings. It detects somata in
a movie, extracts a fluorescence trace per cell, and estimates a per-bin event
rate plus discrete transients, then a separate module aggregates across
recordings for group analysis.

Spatial and temporal detection are the **same** derivative-of-Gaussian operator
(∇²G) applied in two domains: a Laplacian-of-Gaussian blob detector in 2-D space,
and a Ricker wavelet in 1-D time. One idea, two axes. The per-module `*.md` files
under `src/orcann/` document the math and the honesty ledger for each stage.

The pipeline runs as discrete, restartable stages that hand off on disk:

```
data/raw/<rec>.nd2 (.tif/.npy)
  → motion_correct      (caiman env)        → data/pre_processed/<rec>_mc.tif
  → infer               (GPU model, once)   → results/infer/<rec>/      cached probability map
  → segment             (CPU, threshold+extract) → results/spatial/<rec>/   labels, centroids, traces
  → detect_transients   (CPU, temporal model)    → results/transients/<rec>/ rates, events
  → analysis            (CPU, all recordings)     → results/analysis/         group figures + tables
```

`infer` is split from `segment` on purpose: the GPU model runs once and the prob
map is cached, so tuning the threshold only re-runs the cheap CPU `segment` stage.

---

## 1. Install (one time)

**On an SGE / Grid Engine cluster (e.g. Eddie).** Edit `hpc/config.sh` first to
point at your env locations and confirm module names (`ENV_PREFIX`, `CAIMAN_ENV`,
`ANACONDA_MODULE`, `CUDA_MODULE`, `CUDA_BUILD`). Then, on a login node:

```bash
bash hpc/setup.sh all       # builds both conda envs (safe to re-run)
```

This creates two envs, kept separate so caiman's pinned stack never constrains
torch:

- **main** (`ENV_PREFIX`): python 3.11 + CUDA torch + `orcann`. Used by every
  stage except motion correction, and by training. `bash hpc/setup.sh` alone
  builds just this one.
- **caiman** (`CAIMAN_ENV`): caiman + nd2 + tifffile + `orcann`. Used **only** by
  `motion_correction`. Skip it (`bash hpc/setup.sh main`) if your recordings are
  already motion-corrected.

**Locally (no cluster).** `pip install -e ".[torch]"` (or `pip install -e .` and
bring your own torch build). The non-GPU stages and `--synthetic` self-tests run
anywhere; `infer` will fall back to CPU if no GPU is present.

---

## 2. Where your data goes

The repo **is** the workspace. It ships the directory tree the pipeline reads and
writes (each dir ships empty with a `.gitkeep`; contents stay local and never
travel with the repo):

| Put here | What |
|---|---|
| `data/raw/` | your recordings to process (`.nd2` / `.tif` / `.npy`) |
| `data/pre_processed/` | motion-corrected movies (output of `motion_correct`; or drop pre-corrected movies here and start at `infer`) |
| `data/annotated/movies/`, `data/annotated/masks/` | ImageJ-annotated recordings, to **train** the spatial model |
| `data/public_gt/` | public CASCADE ground truth, to **train** the temporal model |
| `models/seg_final/`, `models/temporal/` | trained weights land here |
| `results/` | stage outputs (`infer/`, `spatial/`, `transients/`) |
| `logs/` | SGE job logs |

So the minimum to analyse data: drop your recordings in `data/raw/`. If they are
already motion-corrected, drop them in `data/pre_processed/` instead and skip the
first stage.

Paths resolve relative to the **config file's own location**, so the tree is
found no matter where the repo lives and regardless of the current directory. Any
path in the config may be absolute instead, to point `data/` or `models/` at
scratch or group storage while the code lives elsewhere.

---

## 3. Make a config

Everything (paths, model files, tuning) lives in one YAML file. Generate a
commented starting point and edit it:

```bash
orcann run_pipeline --dump-config config.yaml
```

The sections you will touch: `paths` (only if you move data off the default
tree), `models.spatial` / `models.temporal` (the `.pt` weight files), and
`spatial` (threshold, min_radius, watershed) for detection tuning.

---

## 4. Run on the cluster: the qsub path

Source the config once per login shell (loads env locations for the submit
wrapper), then submit each stage as a per-recording array. `hpc/submit.sh` counts
the recordings the stage will process and runs `qsub -t 1-N hpc/jobs/<stage>.sh`,
one task per recording:

```bash
source hpc/config.sh

bash hpc/submit.sh motion_correct    config.yaml   # data/raw → data/pre_processed   (caiman env)
bash hpc/submit.sh infer             config.yaml   # data/pre_processed → results/infer   (GPU)
bash hpc/submit.sh segment           config.yaml   # results/infer + movie → results/spatial   (CPU)
bash hpc/submit.sh detect_transients config.yaml   # results/spatial → results/transients   (CPU)
```

**Run them in order, waiting for each array to finish before the next**, because
each stage indexes the previous stage's outputs. Monitor with `qstat`; logs are
in `logs/`. A single recording is just `N=1` (a one-task array), nothing special.
If your recordings are already motion-corrected (dropped in `data/pre_processed/`),
skip the first line and start at `infer`.

**One-job alternative (small batches / a single recording).** Chains
infer → segment → detect_transients in one serial job (motion correction must
already be done, since this runs in the torch env):

```bash
qsub -v CONFIG=config.yaml hpc/jobs/run_pipeline.sh
```

---

## 5. Tune segmentation and curate (optional, between infer and segment)

Because `infer` cached the prob map, tuning never touches the GPU. To choose the
threshold / min_radius, **preview a grid** before extracting (writes
`results/spatial/<rec>/sweep_montage.png` and `sweep_table.csv` per recording, no
extraction). Submit `segment.sh` directly with a `SWEEP` variable (`N` is the same
recording count `submit.sh` reports for `segment`; semicolons separate axes, and
values are colon-separated because `qsub -v` would split on commas):

```bash
qsub -t 1-N -v CONFIG=config.yaml,SWEEP='threshold=0.5:0.6:0.7;min_radius=0:2' hpc/jobs/segment.sh
```

Pull the montages, pick the winning values, set them in `config.yaml`
(`spatial.threshold`, `spatial.min_radius`), then run the real `segment`.

**Curate without a GUI.** Read the bad ROI numbers off
`results/spatial/<rec>/figures/overlay.png` and drop them with a per-recording
`results/spatial/<rec>/curate.json`:

```json
{"exclude_rois": [5, 37, 112], "exclude_boxes": [[0, 0, 40, 512]]}
```

ids are the overlay numbers; `exclude_boxes` are `[r0, c0, r1, c1]` regions that
drop any ROI whose centroid is inside. Re-run `segment` with `--force` for that
recording and the ROIs are removed as a post-extraction subset (no
re-segmentation), then run `detect_transients`.

---

## 6. Results

Each stage writes a canonical `<results>/<recording_id>/` folder (filenames
defined once in `src/orcann/pipeline/inference.py`):

```
results/infer/<rec>/        prob.npy  max_projection.npy  meta.json
                            prob_overlay.png   (gamma-stretched prob over max proj; QC)
results/spatial/<rec>/      data/    labels.npy  centroids.npy  traces.npy
                                     max_projection.npy  meta.json
                            figures/ overlay.png   (outlines + numbered centroids)
results/transients/<rec>/   data/    traces.npy  rates.npy  events.npz  meta.json
                            figures/ roi_<i>.png  (trace + scalogram + rate)
results/analysis/           recording_metrics.csv  summary.json
                            within_recording_distributions.png  genotype_comparison.png
                            longitudinal_by_day.png
```

`results/transients/<rec>/` is the final per-recording output. Row `i` of
`traces` / `rates` is label `i+1` in `labels` and centroid `i` in `centroids`;
`events.npz` is long-format (`roi, time_s, duration_s, amplitude`) and its `roi`
column indexes that same axis. The `recording_id` is stable across stages (the
`_mc` suffix from motion correction is normalised away).

Cross-recording **group analysis** reads every `results/transients/<rec>/` and
writes `results/analysis/`. It is a single aggregate job (not a per-recording
array), so `qsub` it directly once `detect_transients` has run for everything:

```bash
qsub -v CONFIG=config.yaml hpc/jobs/analysis.sh
```

It computes, per recording, the within-recording distributions of event frequency
(events/min per ROI) and timescale (event duration), plus the active fraction,
then compares those summaries between genotype (Control vs Mutant, with a
Mann-Whitney p-value) and across developmental day. Genotype and day are parsed
from the `recording_id` by the regexes in the `analysis` config section; **check
`recording_metrics.csv` after the first run** and adjust `analysis.day_regex`,
`analysis.line_regex`, or `analysis.control_prefix` if the parsed columns are
wrong for your id convention.

---

## Training (separate path)

Not per-recording arrays, so `qsub` them directly. Spatial training reads
`data/annotated/`; temporal training reads `data/public_gt/`:

```bash
qsub -v CONFIG=config.yaml hpc/jobs/train_spatial.sh    # fits the segmenter → models/seg_final/
qsub -v CONFIG=config.yaml hpc/jobs/train_temporal.sh   # LOIO validation table + saves the final model
```

Pre-flight an ImageJ annotation against its movie before training:

```bash
python scripts/check_annotation.py --movie rec.nd2 --annotation RoiSet.zip --out check.png
```

---

## Run locally / self-test

Every subcommand also runs without a cluster. The same stages work on a worktree
with `data/raw/` populated:

```bash
orcann run_pipeline --config config.yaml            # full chain locally
orcann segment --config config.yaml --set spatial.threshold=0.55   # one-off override

orcann train_spatial  --synthetic                   # self-tests: no data, no GPU
orcann train_temporal --synthetic
python scripts/visualize_transients.py --config config.yaml
```

`hpc/README_HPC.md` has the SGE/GPU specifics (resource directives, env details).

---

## Status and caveats

- **Temporal stage**: validated cross-indicator on CASCADE (LOIO). Transfers
  well to mainstream GECIs/dyes and GCaMP8; weak on SST/VIP interneurons.
- **Spatial stage**: implemented and self-tested on synthetic data; awaiting
  annotated recordings to train. ImageJ `RoiSet.zip` intake is wired.
- **Standing caveat**: the temporal model is trained on in-vivo 2-photon data,
  so the absolute rate scale on Fluo-4 organoid recordings is uncalibrated.
  Prefer continuous rate measures over absolute event counts for cross-domain
  analysis.
