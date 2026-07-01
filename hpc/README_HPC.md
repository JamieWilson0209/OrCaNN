# Running OrCaNN on an HPC cluster (SGE / Grid Engine)

Written for an SGE / Grid Engine cluster with GPU nodes. The spatial stage trains
on GPU; the temporal stage is small enough for CPU.

The repo **is** the workspace: it ships the `data/ models/ results/ logs/` tree
(empty), and `config.yaml` resolves its paths relative to its own location, so
everything works wherever you put the repo with no build step. Cluster-specific
names (conda envs, modules, CUDA build) live in `hpc/config.sh`; the data layout
lives in `config.yaml`. Note that **scratch is often purged after inactivity** —
put the conda envs (and anything you want to keep) on storage you are willing to
recreate, or point `config.yaml` paths and `hpc/config.sh` env locations at
group/long-term storage.

## 1. One-time setup (login node)

```bash
# Put the repo somewhere on the cluster (git clone, or rsync the extracted repo).
# Wherever it lands is the workspace — no separate src/ copy, no tree to build.
cd /path/to/OrCaNN

source hpc/config.sh     # edit conda-env / module names first if needed
bash   hpc/setup.sh      # main env: conda + torch + `pip install -e .`
```

`hpc/setup.sh` builds the conda env(s) and prints a per-env import check at the
end. Installs run fine on a login node; **never train on the login node** —
always `qsub`.

Motion correction is optional and runs in a **separate** caiman env (it is never
needed for training or segmentation). Only if you intend to motion-correct raw
recordings, build that env too — the same script handles it, and it does not
touch the main env:

```bash
bash hpc/setup.sh all       # main env AND caiman env
bash hpc/setup.sh caiman    # just the caiman env (if the main one already exists)
```

> Reminder: scratch is often purged after inactivity. To persist the conda envs,
> set `ENV_PREFIX` / `CAIMAN_ENV` in `hpc/config.sh` to long-term or group
> storage; to persist trained models, point the `models:` paths in `config.yaml`
> there (an absolute path is used as-is).

## 2. Stage data

```
data/raw/                 raw recordings (motion_correction input)
data/pre_processed/       motion-corrected movies (infer input)
data/annotated/movies/    annotated recordings for segmenter training
data/annotated/masks/     matching instance masks (.npy) or ImageJ ROI sets (same stem)
data/public_gt/           CASCADE ground-truth .mat + indicator_map.json
```

## 3. Submit jobs (from the repo root)

Per-recording stages run as SGE arrays, one task per recording. `hpc/submit.sh`
counts the recordings and submits `qsub -t 1-N` for you (N=1 for a single
recording is just a one-task array — nothing special needed):

```bash
# per-recording arrays — run in order, waiting for each to finish before the next
bash hpc/submit.sh motion_correct   config.yaml   # caiman env: data/raw -> data/pre_processed
bash hpc/submit.sh infer            config.yaml   # GPU: data/pre_processed -> results/infer (cache prob maps)
bash hpc/submit.sh segment          config.yaml   # CPU: results/infer + movie -> results/spatial
bash hpc/submit.sh detect_transients config.yaml  # CPU: results/spatial -> results/transients

# training (not per-recording; submit directly)
source hpc/config.sh
qsub hpc/jobs/train_spatial.sh                     # GPU; spatial segmenter
qsub hpc/jobs/train_temporal.sh                    # CPU; cross-indicator table (LOIO)

# single recording / quick end-to-end in one job (not an array)
qsub -v CONFIG=config.yaml hpc/jobs/run_pipeline.sh
```

**Order matters between arrays.** Each stage indexes the previous stage's outputs,
so run them in order (`motion_correct -> infer -> segment -> detect_transients`),
waiting for each array to finish before submitting the next — otherwise some tasks
would index a dir still being written. The arrays are separate submissions for
exactly this reason; `run_pipeline` chains them in one serial job instead (which
is why it is not itself arrayed).

**Inference is split from segmentation on purpose.** `infer` runs the GPU model
once and caches the probability map; `segment` (CPU, no model) thresholds and
extracts. So tuning `spatial.threshold` / `spatial.min_radius` only re-runs the
cheap `segment` array — the GPU pass is done once. To choose values, preview a
grid without extracting (writes `results/spatial/<rec>/sweep_montage.png` + a
`sweep_table.csv` per recording):

```bash
qsub -t 1-N -v CONFIG=config.yaml,SWEEP='threshold=0.5:0.6:0.7;min_radius=0:2' hpc/jobs/segment.sh
```

Then set the winning values in `config.yaml` and run `segment` normally.

**Curation without a GUI.** Read the bad ROI numbers off `results/spatial/<rec>/
figures/overlay.png` and drop them with a per-recording
`results/spatial/<rec>/curate.json`:

```json
{"exclude_rois": [5, 37, 112], "exclude_boxes": [[0, 0, 40, 512]]}
```

`segment` applies it as a post-extraction subset (ids are the overlay numbers;
boxes are `[r0, c0, r1, c1]` in the overlay's resolution and drop any ROI whose
centroid is inside). Re-run `segment --force` for that recording to apply.

**Staged, restartable.** Each stage skips recordings whose output already exists,
so a partially-failed array can be resubmitted and only the missing recordings
are redone. Tuning lives in `config.yaml` — edit it rather than passing per-job
overrides. `infer` requests GPU (`-q gpu -l gpu=1`) for the segmenter and
auto-detects the device; `segment` and `detect_transients` are CPU-only. All paths,
models, and tuning come from `config.yaml` (the `paths`, `models`, `spatial`, and
`temporal` sections; see the commented file from `orcann run_pipeline
--dump-config`).

If your job needs an L40S instead of an A100, change `-l a100=true` to
`-l l40s=true`; for a fast-scheduling test slice use `-l gpu-mig=1` (a 20 GB MIG
partition). Memory is system RAM per CPU core via `-l h_rss` (default 1 core,
32 GB max per core); the GPU's 80 GB is separate and automatic.

Each runner has a `--synthetic` self-test that needs no data, e.g.
`orcann train_spatial --synthetic` — useful to
confirm the environment before the real data lands.

**Interactive GPU testing.** First validate CUDA cheaply with
`qsub hpc/jobs/check_gpu.sh` (prints `nvidia-smi` and `cuda True`). For an
interactive session — e.g. the one-recording sanity load on the first real
`.nd2` — request a GPU and, on some SGE setups, **source the scheduler
environment script**, or CUDA may not be visible (a qlogin quirk where
`CUDA_VISIBLE_DEVICES` isn't set automatically; check your cluster's docs for
the exact script):

```bash
qlogin -q gpu -l gpu=1 -l h_rt=02:00:00
# if needed on your cluster: source the site's qlogin/scheduler environment script
module load anaconda cuda && source activate "$ENV_PREFIX"
orcann train_spatial --synthetic --set train_spatial.epochs=1
```

## 4. Outputs

```
models/seg_final/segmenter.pt       trained spatial segmenter
models/temporal/rate_model.pt       trained temporal rate head
results/spatial_eval/report.json    held-out IoU (train_spatial report)
results/loio/report.json            leave-one-indicator-out transfer table
results/spatial/<rec>/              per recording: labels, centroids, traces,
                                    max_projection, overlay.png
results/transients/<rec>/           per recording: traces, rates, events,
                                    roi_<i>.png
```

## Still to confirm at data intake

- **ROI format & pairing** — label image / centroids / ImageJ ROI set; row-col
  vs x-y; and that `movies/<stem>` ↔ `rois/<stem>`. Wire `load_recording` to match.
- **Pixel size (µm/px)** — to seed `--radii` from real cell diameters.
- **indicator_map.json** — `{"file.mat": "GCaMP6f_exc", ...}`, grouped by
  indicator *and* cell class.
- **Max-projection substrate** — which summary image the annotators drew on
  (raw max / smoothed max / percentile), so the structural channel can match it.
