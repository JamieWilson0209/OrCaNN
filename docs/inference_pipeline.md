# Running new recordings: motion correction, then segmentation

Two separate steps in two environments. NoRMCorre needs caiman, which lives in
its own env; the segmenter runs in the torch (`calcineps`) env. Step 1 writes
corrected movies to disk, step 2 reads them. They never share a process.

All commands are run from the repo root after `source hpc/config.sh`.

## One-time check (caiman env)

```bash
source activate "$CAIMAN_ENV"      # default /exports/eddie/scratch/$USER/conda/envs/caiman
python -c "import caiman, numpy, nd2, tifffile; print('caiman env ready')"
```

If it reports a missing `nd2` or `tifffile`, add only those: `pip install nd2 tifffile`.
You do NOT install the `orcann` package or torch into this env; the script puts
the repo on its path itself. Nothing is added to the `calcineps` env.

## Step 1 — motion correction (caiman env)

```bash
qsub -v MOVIES=/abs/path/to/nd2_dir,OUT=$RESULTS_DIR/mc_movies hpc/jobs/motion_correct.sh
```

Writes per recording into `OUT`:
- `<stem>_mc.tif`  corrected movie (T,H,W float32)
- `<stem>_mc.json` shift summary (max/mean dy,dx, border crop, seconds)

Knobs via `-v`: `MODE` (rigid|piecewise_rigid|auto, default auto), `MAX_SHIFT`
(px, default 20), `NPROC` (cores, default 4), `CAIMAN_ENV` (override env path).
NoRMCorre scratch goes to the job-local `$TMPDIR` automatically.

## Step 2 — segmentation (calcineps env)

Point `MOVIES` at the corrected dir from step 1:

```bash
qsub -v MOVIES=$RESULTS_DIR/mc_movies,OUT=$RESULTS_DIR/seg_infer,THRESH=0.55 hpc/jobs/seg_infer.sh
```

Writes per recording into `OUT`: `<stem>_mc_labels.npy`, `<stem>_mc_traces.npy`,
`<stem>_mc_centroids.npy`, `<stem>_mc_overlay.png`.

Knobs via `-v`: `MODEL` (default `$MODELS_DIR/seg_final/segmenter.pt`, use an
ABSOLUTE path if overriding), `THRESH` (default 0.55), `MIN_AREA` (drop specks
smaller than N px, default 4), `WATERSHED=1` (split touching cells; default
merges them), `RESIZE` (force NxN if a movie has no scale metadata).

Scale is automatic: the segmenter resamples each movie to the frame size it was
trained at. Corrected movies are `.tif` and carry no µm/px, so the frame-size
match is used, which is correct for same-magnification data.

## Notes

- Run step 2 only after step 1 finishes (`qstat` shows it gone). Step 2 errors
  immediately if the model path is missing.
- To skip motion correction, give step 2 the original `.nd2` dir as `MOVIES`.
- The integrated `MOTION_CORRECT=1` flag on `seg_infer.sh` fuses both steps into
  one job but requires caiman inside the `calcineps` env; the two-step path above
  is preferred and needs no environment change.
