# Running OrCaNN on Eddie

SGE cluster, NVIDIA A100 GPU nodes. The spatial stage trains on GPU; the
temporal stage is small enough for CPU. All paths and cluster-specific names
live in `eddie/config.sh` — edit it once.

## 1. One-time setup (login node)

```bash
# get the repo into place, e.g.
mkdir -p /exports/eddie/scratch/$USER/orcann_workspace/code
rsync -a ./ /exports/eddie/scratch/$USER/orcann_workspace/code/      # or git clone
cd /exports/eddie/scratch/$USER/orcann_workspace/code

source eddie/config.sh        # edit UUN / paths first if needed
bash  eddie/make_workspace.sh # create the directory tree
bash  eddie/setup_eddie.sh    # conda env + torch + `pip install -e .`
```

`setup_eddie.sh` prints a torch/CUDA check at the end. Installs run fine on a
login node; **never train on the login node** — always `qsub`.

> Scratch is purged after ~1 month of no access. If you want the conda env and
> trained models to persist, point `ENV_PREFIX` and `MODELS_DIR` at group or
> DataStore space in `config.sh`.

## 2. Stage data

```
data/annotated/movies/   manually annotated recordings (motion-corrected)
data/annotated/rois/      matching ROI annotations (same filename stem)
data/raw/                 raw recordings for inference (+ measure noise SD)
data/public_gt/           CASCADE ground-truth .mat + indicator_map.json
```

## 3. Submit jobs (from the repo root)

```bash
source eddie/config.sh
qsub eddie/jobs/train_spatial.sh                    # GPU; spatial detector
qsub eddie/jobs/train_temporal_loio.sh              # CPU; cross-indicator table

ls "$RAW_DIR"/*.nd2 > "$RAW_DIR/manifest.txt"
qsub -t 1-$(wc -l < "$RAW_DIR/manifest.txt") eddie/jobs/infer_array.sh
```

**Batch pipeline (`run.sh`).** One job per recording runs the whole pipeline —
spatial detection → trace extraction → transient extraction — through a single
entry point that generates the SGE array script with absolute paths baked in and
submits it (the proven calcium-pipeline pattern). Transient extraction is *inside*
each per-recording job; only the cross-recording group analysis is separate.

```bash
# needs a trained spatial detector + temporal model
bash run.sh batch --data-root "$RAW_DIR" \
    --spatial-model "$MODELS_DIR/spatial/detector.pt" \
    --temporal-model "$MODELS_DIR/temporal/rate_model.pt" \
    --min-prominence 0.5
```

`run.sh` commands: `single` (one `.nd2`), `batch` (SGE array over every `.nd2`
under `--data-root`), `analyse` (hook for the separate group-analysis module),
`full` (batch + held analyse). Each recording writes a complete result folder
to `$RESULTS_DIR/run_<JOBID>/<recording>/`:
`spatial_footprints.npz`, `centroids.npy`, `temporal_traces.npy`, `rates.npy`,
`events.npz` (per-transient: roi, time_s, duration_s, amplitude), `meta.json`.
Default resources are CPU (`sharedmem`); set `RES="-q gpu -l gpu=1 -l a100=true"`
to run the spatial stage on GPU. Set `--min-prominence` to the visualizer-tuned value.

If your job needs an L40S instead of an A100, change `-l a100=true` to
`-l l40s=true`; for a fast-scheduling test slice use `-l gpu-mig=1` (a 20 GB MIG
partition). Memory is system RAM per CPU core via `-l h_rss` (default 1 core,
32 GB max per core); the GPU's 80 GB is separate and automatic.

Each runner has a `--synthetic` self-test that needs no data, e.g.
`python scripts/run_train_spatial.py --synthetic --out /tmp/m` — useful to
confirm the environment before the real data lands.

**Interactive GPU testing.** First validate CUDA cheaply with
`qsub eddie/jobs/check_gpu.sh` (prints `nvidia-smi` and `cuda True`). For an
interactive session — e.g. the one-recording sanity load on the first real
`.nd2` — request a GPU and **source the scheduler environment**, or CUDA won't
be visible (a qlogin quirk: `CUDA_VISIBLE_DEVICES` isn't set automatically):

```bash
qlogin -q gpu -l gpu=1 -l h_rt=02:00:00
source /exports/applications/support/set_qlogin_environment.sh
module load anaconda cuda && source activate "$ENV_PREFIX"
python scripts/run_train_spatial.py --movies "$ANNOTATED_DIR/movies" \
    --rois "$ANNOTATED_DIR/rois" --epochs 1 --n-patch 2 --out /tmp/sanity
```

## 4. Outputs

```
models/spatial/detector.pt          trained spatial detector
models/temporal/rate_model.pt       trained temporal rate head
results/spatial_eval/report.json    held-out precision/recall/F1 (incl. faint)
results/loio/report.json            leave-one-indicator-out transfer table
results/inference/<rec>.npz         per recording: centroids, footprints,
                                    traces, rates, durations
```

## Still to confirm at data intake

- **ROI format & pairing** — label image / centroids / ImageJ ROI set; row-col
  vs x-y; and that `movies/<stem>` ↔ `rois/<stem>`. Wire `load_recording` to match.
- **Pixel size (µm/px)** — to seed `--radii` from real cell diameters.
- **indicator_map.json** — `{"file.mat": "GCaMP6f_exc", ...}`, grouped by
  indicator *and* cell class.
- **Max-projection substrate** — which summary image the annotators drew on
  (raw max / smoothed max / percentile), so the structural channel can match it.
