# Running OrCaNN on an HPC cluster (SGE / Grid Engine)

Written for an SGE / Grid Engine cluster with GPU nodes. The spatial segmenter
trains and infers on GPU; segmentation, the activity stage (dF/F0 + OASIS +
gallery), and group analysis are CPU work. Motion correction and the activity
stage run in the caiman env; every other stage runs in the torch env.

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
bash   hpc/setup.sh      # BOTH envs (default) — see the stage/env split below
```

`hpc/setup.sh` builds the conda env(s) and prints a per-env import check at the
end. Installs run fine on a login node; **never train on the login node** —
always `qsub`.

OrCaNN needs **two** conda envs, and which stage runs where is not guessable:

| env | stages |
|---|---|
| `ENV_PREFIX` (torch) | `infer`, `segment`, `analysis`, `train_spatial` |
| `CAIMAN_ENV` | `motion_correction`, `activity` |

`activity` is in the caiman env because OASIS deconvolution *is* CaImAn's
`constrained_foopsi`. So the caiman env is not an optional extra for people who
motion-correct — every run that produces spike trains needs it. `bash
hpc/setup.sh` builds both by default and prints which are usable when it
finishes. To build just one:

```bash
bash hpc/setup.sh main      # torch env only
bash hpc/setup.sh caiman    # caiman env only
```

> Building only the torch env leaves a setup that looks complete and then fails
> at the first stage. Until recently it could be worse than a failure: with
> caiman absent the OASIS import raised inside a `try`, so deconvolution fell
> back to the threshold method and the run completed with different numbers.
> `activity` now refuses to run OASIS without CaImAn. **Results produced before
> that change are not self-describing** — `run_info.json` records the
> *configured* method, not the one that ran — so check those job logs for
> `OASIS failed:` / `Falling back to threshold deconvolution` before trusting
> them.

### If conda fails with "An unexpected error has occurred"

The traceback ends in `OSError: [Errno 122] Disk quota exceeded` and conda
suggests disabling plugins. Ignore that suggestion: the real cause is a full
**home** directory. The cluster's base package cache is read only, so conda
falls back to `~/.conda/pkgs`, and a caiman solve writes roughly 200 MB of
conda-forge repodata before downloading a single package.

`hpc/config.sh` now points `CONDA_PKGS_DIRS`, `CONDA_ENVS_DIRS` and
`PIP_CACHE_DIR` at scratch, which prevents this on a fresh setup. If home is
already full from an earlier run, reclaim the space once:

```bash
quota -s
du -sh ~/.conda ~/.cache
conda clean --all --yes
```

Then re-run `bash hpc/setup.sh all`.

`hpc/config.sh` also redirects `TMPDIR` and `XDG_CACHE_HOME`, and sets
`CONDA_REGISTER_ENVS=false`. That last one matters more than it looks: `conda
create --prefix` registers the new env in `~/.conda/environments.txt`, a
hardcoded path that `CONDA_ENVS_DIRS` does not move, and `register_env()`
warns-and-continues only for `EACCES/EROFS/ENOENT` while re-raising anything
else — a full quota is `EDQUOT` (errno 122), so the create fails outright. With
it off, setup writes **nothing** to home, and a full home no longer blocks an
install at all.

### `JSONDecodeError` from conda, after a quota failure

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
  ... conda/notices/cache.py, line 100, in get_notice_response_from_cache
```

A quota failure leaves damage behind. Conda keeps its channel-notices cache in
`~/.cache/conda/notices`, outside `pkgs_dirs`, and a write truncated by a full
quota leaves a 0-byte file there. Every later `conda` call then dies reading it.
This is not a plugin problem, so `--no-plugins` does not help.

`bash hpc/setup.sh` deletes empty or unparseable notices files during preflight,
and `CONDA_NUMBER_CHANNEL_NOTICES=0` in `hpc/config.sh` stops them being written
at all. To clear it by hand:

```bash
python -c "from conda.notices.cache import get_notices_cache_dir; print(get_notices_cache_dir())"
rm -rf ~/.cache/conda     # pure cache; conda rebuilds it
conda config --set number_channel_notices 0
```

### Scratch purge left a half-empty env

Purges delete files by age from *inside* a tree, so an env can keep its directory
and lose its interpreter. Setup used to skip creation in that state (the
directory existed) and fail much later inside pip. It now tests the interpreter
and stops with:

```
ERROR: the main env at <prefix> exists but its interpreter does not run.
```

The script never deletes an env itself; clear it and re-run:

```bash
rm -rf /exports/eddie/scratch/$USER/conda/envs/calcineps
bash hpc/setup.sh
```

### Windows line endings (`pipefail` errors)

```
hpc/setup.sh: line 26: set: pipefail: invalid option name
```

or, via a shebang, `bad interpreter: /bin/bash^M`. The repository stores LF, but
Git for Windows defaults to `core.autocrlf=true` and rewrites every line ending
to CRLF **on checkout**; those bytes reach the cluster unchanged.

`.gitattributes` (`* text=auto eol=lf`) fixes this at the source, but only for
clones made after it was committed. Two runtime guards catch the rest:

- `bash hpc/setup.sh` strips CR from `hpc/*.sh` and `hpc/jobs/*.sh` in preflight.
- `hpc/submit.sh` and `hpc/run_all.sh` submit a CR-stripped copy of the job
  script, so a Windows checkout cannot queue a job that dies on the node.

Neither can rescue *itself*: a script with CRLF fails while bash is still parsing
it. If `hpc/setup.sh` is the broken file, bootstrap by typing the fix at the
shell:

```bash
sed -i 's/\r$//' hpc/*.sh hpc/jobs/*.sh
```

On Windows, also `git config --global core.autocrlf false`.

### Diagnosing

```bash
bash hpc/setup.sh doctor
```

Installs nothing. Prints where every cache resolves, whether notices and env
registration are disabled, whether `$HOME` is writable, the quota if the cluster
reports one, and whether each env is `ok` / `ABSENT` / `BROKEN`. Run this first
when a job reports an environment problem.

> Reminder: scratch is often purged after inactivity. To persist the conda envs,
> set `ENV_PREFIX` / `CAIMAN_ENV` in `hpc/config.sh` to long-term or group
> storage; to persist trained models, point the `models:` paths in `config.yaml`
> there (an absolute path is used as-is).

## 2. Stage data

```
data/raw/                 raw recordings (motion_correction input)
data/pre_processed/       motion-corrected movies, one store per settings choice
                          (infer input; movies corrected elsewhere go in the root)
data/annotated/movies/    annotated recordings for segmenter training
data/annotated/masks/     matching instance masks (.npy) or ImageJ ROI sets (same stem)
```

## 3. Submit jobs (from the repo root)

Per-recording stages run as SGE arrays, one task per recording. `hpc/submit.sh`
counts the recordings and submits `qsub -t 1-N` for you (N=1 for a single
recording is just a one-task array — nothing special needed):

```bash
# per-recording arrays — run in order, waiting for each to finish before the next
bash hpc/submit.sh motion_correct   config.yaml   # caiman env: data/raw -> data/pre_processed
bash hpc/submit.sh infer            config.yaml   # torch env, GPU: data/pre_processed -> results/infer (cache prob maps)
bash hpc/submit.sh segment          config.yaml   # torch env, CPU: results/infer + movie -> results/spatial
bash hpc/submit.sh activity         config.yaml   # caiman env, CPU: results/spatial -> results/activity (dF/F0, OASIS, gallery)

# group analysis (not per-recording; submit directly, after activity)
source hpc/config.sh
qsub -v CONFIG=config.yaml hpc/jobs/analysis.sh    # torch env, CPU: results/activity -> results/analysis

# training (not per-recording; submit directly)
qsub hpc/jobs/train_spatial.sh                     # GPU; spatial segmenter

# chained submission (segment -> activity -> analysis, via -hold_jid)
bash hpc/run_all.sh --config config.yaml
```

`activity` runs in the **caiman env** (OASIS is CaImAn's `constrained_foopsi`),
the same env as motion correction; every other stage runs in the torch env.
Running `activity` without caiman on the path is a hard error when
`deconvolution.method='oasis'` — it used to fall back to the `threshold` method
with only a warning, which changed the results while still reporting success.

**Order matters between arrays.** Each stage indexes the previous stage's outputs,
so run them in order (`motion_correct -> infer -> segment -> activity`),
waiting for each array to finish before submitting the next — otherwise some tasks
would index a dir still being written. The arrays are separate submissions for
exactly this reason; `hpc/run_all.sh` submits them as a dependency chain
(`-hold_jid`) so the ordering is enforced by the scheduler rather than by you.

**Inference is split from segmentation on purpose.** `infer` runs the GPU model
once and caches the probability map; `segment` (CPU, no model) thresholds and
extracts. So tuning `spatial.threshold` / `spatial.min_radius` only re-runs the
cheap `segment` array — the GPU pass is done once. To choose values, preview a
grid without extracting (writes `sweep_montage.png` + a `sweep_table.csv` per
recording, into the segment store the current settings name):

```bash
qsub -t 1-N -v CONFIG=config.yaml,SWEEP='threshold=0.5:0.6:0.7;min_radius=0:2' hpc/jobs/segment.sh
```

Then set the winning values in `config.yaml` and run `segment` normally.

**Staged, restartable.** Each stage skips recordings whose output already exists,
so a partially-failed array can be resubmitted and only the missing recordings
are redone. Tuning lives in `config.yaml` — edit it rather than passing per-job
overrides. `infer` requests GPU (`-q gpu -l gpu=1`) for the segmenter and
auto-detects the device; `segment`, `activity`, and `analysis` are CPU-only. All
paths, the model, and tuning come from `config.yaml` (the `paths`, `models`,
`spatial`, `imaging`, `baseline`, `deconvolution`, and `analysis` sections; see
the commented file from `orcann segment --dump-config`).

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
models/trained/<identity>/          models training produced, awaiting promotion;
                                    train_report.json (held-out IoU) sits beside each
models/in_use/<identity>/           promoted models; models.spatial selects one, or 'latest'
results/spatial/<store>/<rec>/      per recording: labels, centroids, traces,
                                    max_projection, overlay.png
results/activity/<store>/<rec>/     per recording (calcium-format): temporal_traces,
                                    temporal_traces_raw, traces_denoised, spike_trains,
                                    spatial_footprints.npz, max/mean_projection,
                                    run_info.json, gallery.html
results/analysis/<store>/           group figures + tables (genotype + longitudinal)

Each `<store>` is named for the settings that stage ran under and the store it
read, so a stage re-run with the same settings skips and a changed setting writes
beside the old output instead of over it. `orcann status --config config.yaml`
prints which stores a config names and which recordings are already current in
them; it runs nothing, so it is safe on the login node.
```

## Still to confirm at data intake

- **ROI format & pairing** — label image / centroids / ImageJ ROI set; row-col
  vs x-y; and that `movies/<stem>` ↔ `masks/<stem>`. Wire `load_recording` to match.
- **Pixel size (µm/px)** — to seed the training `radii` from real cell diameters.
- **Indicator & frame rate** — set `imaging.indicator` / `imaging.frame_rate` so
  the OASIS decay time and dF/F0 windows are right for the recording.
- **Max-projection substrate** — which summary image the annotators drew on
  (raw max / smoothed max / percentile), so the structural channel can match it.
