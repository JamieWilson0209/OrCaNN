# OrCaNN

A calcium-imaging analysis pipeline for organoid recordings. It motion-corrects a
movie, detects somata with a trained segmenter, extracts a fluorescence trace per
cell, infers spike trains, and then aggregates across recordings for group
analysis.

This build merges two codebases. Spatial detection (`infer` + `segment`) is
OrCaNN's learned soma segmenter. Motion correction, the interactive HTML gallery,
and the group analysis are carried over from the earlier calcium pipeline. The
old temporal-model transient detector has been removed: activity now comes from
OASIS deconvolution of the segmenter's traces, which is what the gallery and the
group analysis consume.

The pipeline runs as discrete, restartable stages that hand off on disk:

```
data/raw/<rec>.nd2 (.tif/.npy)
  -> motion_correction  (caiman env)          -> data/pre_processed/<rec>_mc.tif
  -> infer              (torch env, GPU)       -> results/infer/<rec>/      cached probability map
  -> segment            (torch env, CPU)       -> results/spatial/<rec>/    labels, centroids, traces
  -> activity           (caiman env, CPU)      -> results/activity/<rec>/   dF/F0, spikes, footprints, gallery
  -> analysis           (torch env, CPU)       -> results/analysis/         group figures + tables
```

`infer` is split from `segment` on purpose: the GPU model runs once and the
probability map is cached, so tuning the threshold only re-runs the cheap CPU
`segment` stage.

`activity` is the bridge. It reads the segmenter's per-ROI traces, baseline-
corrects them to dF/F0, runs OASIS spike inference (CaImAn's `constrained_foopsi`,
which is why this stage shares the caiman env with motion correction), builds the
sparse spatial footprints, and writes the calcium-format per-recording folder that
the group analysis reads, plus the interactive `gallery.html`. If caiman is not on
the path, deconvolution falls back to a dependency-free threshold method with a
warning rather than failing.

---

## 1. Install (one time)

Two conda envs are used, kept separate so caiman's pinned numpy/scipy stack never
constrains torch:

- **main** (`ENV_PREFIX`): python 3.11 + CUDA torch + `orcann`. Runs `infer`,
  `segment`, `analysis`, and `train_spatial`.
- **caiman** (`CAIMAN_ENV`): caiman + nd2 + tifffile + pillow + `orcann`. Runs
  `motion_correction` and `activity` (OASIS deconvolution and the HTML gallery).

**On an SGE / Grid Engine cluster (e.g. Eddie).** Edit `hpc/config.sh` first to
point at your env locations and confirm module names. Then, on a login node:

```bash
bash hpc/setup.sh all       # builds both envs (safe to re-run)
```

`bash hpc/setup.sh` alone builds just the main env; `bash hpc/setup.sh caiman`
builds just the caiman env (needed for motion correction and OASIS).

**Locally (no cluster).** `pip install -e ".[torch]"` gives you every stage except
OASIS deconvolution; for real OASIS you need caiman on the path (install it from
conda-forge, never `pip install caiman`). Without caiman, set
`deconvolution.method: threshold`. The `--synthetic` self-tests run anywhere.

---

## 2. Where your data goes

The repo is the workspace. It ships the directory tree the pipeline reads and
writes (each dir ships empty with a `.gitkeep`; contents stay local):

| Put here | What |
|---|---|
| `data/raw/` | your recordings to process (`.nd2` / `.tif` / `.npy`) |
| `data/pre_processed/` | motion-corrected movies (output of `motion_correction`; or drop pre-corrected movies here and start at `infer`) |
| `data/annotated/movies/`, `data/annotated/masks/` | ImageJ-annotated recordings, to **train** the spatial model |
| `models/seg_final/` | the trained segmenter lands here |
| `results/` | stage outputs (`infer/`, `spatial/`, `activity/`, `analysis/`) |
| `logs/` | SGE job logs |

Paths resolve relative to the config file's own location, so the tree is found no
matter where the repo lives. Any path in the config may be absolute instead, to
point `data/` or `models/` at scratch or group storage while the code lives
elsewhere.

---

## 3. Make a config

Everything (paths, model, tuning) lives in one YAML file. Generate a commented
starting point and edit it:

```bash
orcann run_pipeline --dump-config config.yaml
```

The sections you will touch most: `imaging` (frame rate and indicator, which sets
the OASIS decay time), `spatial` (threshold, watershed) for detection tuning,
`baseline` / `deconvolution` for the activity stage, and `analysis` for the group
comparison thresholds.

---

## 4. Run on the cluster

Source the config once per login shell, then submit each stage as a per-recording
array. `hpc/submit.sh` counts the recordings and runs `qsub -t 1-N`, one task per
recording:

```bash
source hpc/config.sh

bash hpc/submit.sh motion_correct config.yaml   # caiman env: data/raw -> data/pre_processed
bash hpc/submit.sh infer          config.yaml   # torch env, GPU: -> results/infer
bash hpc/submit.sh segment        config.yaml   # torch env, CPU: -> results/spatial
bash hpc/submit.sh activity       config.yaml   # caiman env, CPU: -> results/activity
```

**Run them in order, waiting for each array to finish before the next**, because
each stage indexes the previous stage's outputs. If your recordings are already
motion-corrected (dropped in `data/pre_processed/`), skip the first line and start
at `infer`.

**Then run the group analysis.** This is a separate step, not part of `activity`
and not a per-recording array: it is one aggregate job over the whole batch, so
run it only once every recording has been through `activity`. It uses `qsub`
directly rather than `hpc/submit.sh` (there is nothing to spread over an array):

```bash
qsub -v CONFIG=config.yaml hpc/jobs/analysis.sh   # torch env, CPU: results/activity -> results/analysis
```

Nothing triggers the analysis automatically; the `activity` stage finishes with a
`now run: analysis` reminder, and you submit it yourself. See section 6 for what it
produces. To make the analysis start automatically when the last `activity` task
finishes, add an SGE dependency: `qsub -hold_jid orcann_activity -v
CONFIG=config.yaml hpc/jobs/analysis.sh`.

**One-job alternative (small batches / a single recording).** Chains
infer -> segment -> activity in one serial job (analysis is still separate):

```bash
qsub -v CONFIG=config.yaml hpc/jobs/run_pipeline.sh
```

This job runs in the torch env, so its `activity` step uses the threshold
deconvolution fallback (no caiman there). For real OASIS, run the `activity` array
in the caiman env as above.

---

## 5. Tune segmentation and curate (optional, between infer and segment)

Because `infer` cached the prob map, tuning never touches the GPU. To choose the
threshold / min_radius, **preview a grid** before extracting (writes
`sweep_montage.png` + `sweep_table.csv` per recording, no extraction):

```bash
qsub -t 1-N -v CONFIG=config.yaml,SWEEP='threshold=0.5:0.6:0.7;min_radius=0:2' hpc/jobs/segment.sh
```

Values are colon-separated because `qsub -v` splits on commas. Pull the montages,
set the winning values in `config.yaml` (`spatial.threshold`, `spatial.min_radius`),
then run the real `segment`.

**Curate without a GUI.** Read the bad ROI numbers off
`results/spatial/<rec>/figures/overlay.png` and drop them with a per-recording
`results/spatial/<rec>/curate.json`:

```json
{"exclude_rois": [5, 37, 112], "exclude_boxes": [[0, 0, 40, 512]]}
```

Re-run `segment --force` for that recording, then run `activity`.

---

## 6. Results

Each stage writes a canonical `<results>/<recording_id>/` folder (filenames
defined once in `src/orcann/pipeline/inference.py` for the spatial stage):

```
results/infer/<rec>/        prob.npy  max_projection.npy  meta.json
                            prob_overlay.png   (gamma-stretched prob over max proj; QC)
results/spatial/<rec>/      data/    labels.npy  centroids.npy  traces.npy
                                     max_projection.npy  meta.json
                            figures/ overlay.png   (outlines + numbered centroids)
results/activity/<rec>/     data/    temporal_traces.npy       (dF/F0, N x T)
                                     temporal_traces_raw.npy   (raw fluorescence)
                                     traces_denoised.npy       (OASIS denoised)
                                     spike_trains.npy          (inferred spikes)
                                     deconv_noise.npy          (per-trace noise)
                                     spatial_footprints.npz    (sparse d1*d2 x N)
                                     max_projection.npy  mean_projection.npy
                            run_info.json   (dims, frame rate, decay time)
                            gallery.html    (interactive per-ROI viewer)
results/analysis/           data/     analysis_results.json  (all stats: tests, p, effect sizes)
                                      dataset_features.csv   (per-recording feature matrix + genotype/day)
                                      selected_rois.csv      (per-ROI selection + quality)
                                      quality_gating.json    (QC decisions: motion, drift, activity)
                                      outlier_report.txt
                            figures/  9 groups: Main Results, Metrics, Genotype
                                      Comparison, Activity Analysis, Full Overview,
                                      Correlation Graphs, Results by Dataset,
                                      Selected Traces, Temporal Visualisations
```

Row `i` of every per-recording array is label `i+1` in `labels` and centroid `i`
in `centroids`; the `recording_id` is stable across stages (the `_mc` suffix from
motion correction is normalised away).

Cross-recording **group analysis** reads every `results/activity/<rec>/` and writes
`results/analysis/`. It is a single aggregate job (not a per-recording array):

```bash
qsub -v CONFIG=config.yaml hpc/jobs/analysis.sh
```

It applies per-recording QC (motion, using the shift stats the motion_correction
stage carries into each activity folder, plus baseline drift and activity gating),
scores neuron quality, deduplicates ROIs, and computes per-recording functional
summaries (spike rate, amplitude, pairwise correlation, synchrony, network bursts,
active fraction), then compares them between genotype (Control vs Mutant, with a
statistical test) and across developmental day. Genotype and day are parsed from
each recording folder name; **check `dataset_features.csv` after the first run**
to confirm the parsed columns match your id convention.

---

## Training (spatial only)

Not a per-recording array, so `qsub` it directly. Spatial training reads
`data/annotated/`:

```bash
qsub -v CONFIG=config.yaml hpc/jobs/train_spatial.sh    # fits the segmenter -> models/seg_final/
```

Pre-flight an ImageJ annotation against its movie before training:

```bash
python scripts/check_annotation.py --movie rec.nd2 --annotation RoiSet.zip --out check.png
```

---

## Run locally / self-test

Every subcommand also runs without a cluster:

```bash
orcann run_pipeline --config config.yaml            # full chain locally (through activity)
orcann segment --config config.yaml --set spatial.threshold=0.55   # one-off override
orcann activity --config config.yaml --set deconvolution.method=threshold  # no caiman
orcann analysis --config config.yaml                # group analysis, after activity

orcann train_spatial --synthetic                    # self-test: no data, no GPU
```

`hpc/README_HPC.md` has the SGE/GPU specifics (resource directives, env details).

---

## Status and caveats

- **Spatial stage**: OrCaNN's learned segmenter; ImageJ `RoiSet.zip` intake is
  wired for training.
- **Activity stage**: OASIS decay time is set from `imaging.indicator`. The
  absolute spike-rate scale on Fluo-4 organoid recordings is uncalibrated; prefer
  relative and distributional measures for cross-domain comparison.
- **Group analysis**: genotype and developmental day are parsed from the recording
  id, so confirm `dataset_features.csv` after the first run and adjust the naming
  or the parser if the parsed columns are wrong.
- **Envs**: `activity` runs in the caiman env for real OASIS; run elsewhere it
  falls back to the threshold method. Keep the two envs separate so caiman's
  numpy pin never constrains torch.

---

## References

- **CaImAn NoRMCorre**: Pnevmatikakis & Giovannucci, J. Neurosci. Methods 2017
- **OASIS**: Friedrich et al., PLoS Comp Biol 2017
- **LoG blob detection**: Lindeberg, IJCV 1998
