# OrCaNN

A calcium-imaging analysis pipeline built around a single operator. Spatial
neuron detection and temporal transient detection are the **same** derivative-of-Gaussian
operator (∇²G) applied in two domains: a Laplacian-of-Gaussian blob detector in
2-D space, and a Ricker (mexican-hat) wavelet in 1-D time. One idea, two axes.

The pipeline detects somata in a recording, extracts a fluorescence trace per
cell, and estimates a per-bin event rate plus discrete transients — then a
separate module aggregates across recordings for group analysis.

## The idea

- **Space.** `ParametricLoG2d` is a learnable bank of scale-normalised LoG
  kernels. A small head fuses temporal-moment channels (structural / max /
  variance) of the LoG-filtered movie into a per-pixel *cellness* map — the
  learned replacement for intensity thresholding.
- **Time.** `ParametricDoGWavelet1d` is the 1-D counterpart (ψ = cosθ·Ricker +
  sinθ·∇G, with a learned rise/decay asymmetry θ). A head maps its multi-scale
  response to a non-negative firing rate, trained on public CASCADE ground truth
  to invert the calcium indicator (trace → rate).

## Pipeline

```
movie (.nd2/.tif/.npy)
  → spatial detector      → cellness → centroids, soft footprints
  → trace extraction      → per-ROI ΔF/F
  → temporal model + gate → rate, discrete transients, durations
```

Each per-recording job runs the whole chain in one process; only the
cross-recording group analysis is separate.

## Install

```bash
pip install -e .            # torch installed separately (see hpc/setup_hpc.sh)
pip install -e ".[torch]"   # or pull a CPU/CUDA torch build yourself
```

On an SGE / Grid Engine HPC cluster, see `hpc/README_HPC.md` for the
prefix-env setup and GPU/SGE specifics.

## Usage

```bash
# train the temporal model on CASCADE (leave-one-indicator-out validation)
python scripts/run_train_temporal_loio.py --gt-dir <CASCADE> \
    --indicator-map <map.json> --exclude spinal-cord --report report.json
# save one deployable temporal model
python scripts/run_train_temporal_loio.py --gt-dir <CASCADE> \
    --indicator-map <map.json> --exclude spinal-cord \
    --save-final models/temporal/rate_model.pt

# train the spatial detector on annotated recordings
python scripts/run_train_spatial.py --recordings <pairs> \
    --out models/spatial/detector.pt

# batch the full pipeline over a directory of .nd2 (one SGE job per recording)
bash run.sh batch --data-root <nd2_dir> \
    --spatial-model models/spatial/detector.pt \
    --temporal-model models/temporal/rate_model.pt

# interim: transient detection on existing traces (no spatial model needed)
bash run.sh batch --input traces --data-root <old_results_dir> \
    --temporal-model models/temporal/rate_model.pt

# inspect / pre-flight
python scripts/visualize_transients.py --traces traces.npy --model rate_model.pt --out fig.png
python scripts/check_annotation.py --movie rec.nd2 --annotation RoiSet.zip --out check.png
```

## Per-recording outputs

`run_<JOBID>/<recording>/`

```
data/    spatial_footprints.npz  centroids.npy  temporal_traces.npy
         rates.npy  events.npz  max_projection.npy  meta.json
figures/ roi_<i>.png (trace + scalogram + rate)   max_projection_detections.png
```

`events.npz` is long-format (`roi, time_s, duration_s, amplitude`). See
`orcann/DOCUMENTATION.md` for the full schema, math, and the honesty ledger.

## Status

- **Temporal stage** — validated cross-indicator on CASCADE (LOIO). Transfers
  well to mainstream GECIs/dyes and GCaMP8; weak on SST/VIP interneurons; spinal
  cord excluded as a distinct domain.
- **Spatial stage** — implemented and self-tested on synthetic data; awaiting
  annotated recordings to train. ImageJ `RoiSet.zip` intake is wired
  (`scripts/check_annotation.py` is the orientation pre-flight).
- **Standing caveat** — the temporal model is trained on in-vivo 2-photon data;
  absolute rate scale on Fluo-4 organoid recordings is uncalibrated. Prefer
  continuous rate measures over absolute event counts for cross-domain analysis.

## Tests

```bash
pytest -q            # or: python tests/test_smoke.py
```

Synthetic smoke/integration tests covering every module; no real data or GPU
required.
