"""Composition layer: the single seam where the spatial and temporal stages meet.
A recording flows movie -> segment -> labels -> extract -> traces -> detect ->
rates, events. This module owns that composition, the per-recording output
contract (the filename constants below are the single source of truth), and the
writer both runners share, so they never drift.

Row ``i`` of traces/rates is label ``i+1`` in labels and centroid ``i`` in
centroids; ``events.npz`` rows index that axis via their ``roi`` column. See
docs/README.md for the full output contract.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np


# CANONICAL FILENAMES — the single source of truth for the contract.
# Every read or write of a per-recording artifact goes through these names;
# nothing in the codebase should spell them as string literals.

DATA_DIRNAME = "data"
FIGURES_DIRNAME = "figures"

LABELS_NPY = "labels.npy"            # (H,W) int32 instance labels (0 = bg)
CENTROIDS_NPY = "centroids.npy"      # (N,2) float32 (row, col)
TRACES_NPY = "traces.npy"            # (N,T) float32 per-ROI fluorescence
RATES_NPY = "rates.npy"             # (N,T) float32 per-bin rate
EVENTS_NPZ = "events.npz"           # long-format transient table
MAXPROJ_NPY = "max_projection.npy"   # (H,W) float32 max over time
PROB_NPY = "prob.npy"              # (H,W) float32 soma probability (cached by infer)
META_JSON = "meta.json"

OVERLAY_PNG = "overlay.png"          # spatial QC: instance outlines
ROI_FIG_FMT = "roi_{:03d}.png"       # temporal QC: one panel per ROI


# RECORDING ID — robust to every input shape we feed the runners

# stems that are generic dump names rather than a recording id; when we see one
# we climb to the enclosing recording directory instead.
_GENERIC_STEMS = {"traces", "temporal_traces", "trace", "rates"}
# trailing tokens we strip from a movie/trace stem to recover the recording id.
_STRIP_SUFFIXES = ("_traces", "_mc")
_STRIP_INFIXES = (" - Denoised",)


def recording_id(path: str) -> str:
    """A clean, collision-free recording id from a movie OR trace path.

    Handles every input the runners see:
      ``.../REC.nd2``                     -> ``REC``      (a movie)
      ``.../REC_traces.npy``              -> ``REC``      (a trace file)
      ``.../REC_mc.tif``                  -> ``REC``      (motion-corrected movie)
      ``.../REC/data/traces.npy``         -> ``REC``      (canonical contract)
      ``.../REC - Denoised/data/...npy``  -> ``REC``      (legacy layout)

    The id is the file stem with known suffixes stripped and spaces folded to
    underscores; when the stem is a generic dump name we use the enclosing
    recording directory instead of a fixed parent depth (the bug in the old
    fixed-depth parser, which mislabelled flat seg outputs).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in _GENERIC_STEMS:
        d = os.path.dirname(path)
        if os.path.basename(d) == DATA_DIRNAME:        # .../REC/data/traces.npy
            d = os.path.dirname(d)
        stem = os.path.basename(d) or stem
    for infix in _STRIP_INFIXES:
        stem = stem.replace(infix, "")
    for suf in _STRIP_SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem.strip().replace(" ", "_")


# SPATIAL — movie -> probability -> labels -> traces

def nd2_um_per_px(path: str) -> Optional[float]:
    """Microns/pixel from ND2 (xy) metadata, or None if unavailable."""
    try:
        import nd2
        with nd2.ND2File(path) as f:
            vs = f.voxel_size()                        # (x, y, z) microns
        return float(vs.x)
    except Exception:
        return None


def resample_for_model(model, movie: np.ndarray, in_um: Optional[float] = None,
                       resize_to: int = 0, train_um_override: Optional[float] = None
                       ) -> np.ndarray:
    """Resample a movie to the scale the segmenter was trained at.

    Physical match (µm/px) is preferred when both the recording's pixel size and
    the model's training pixel size are known; otherwise a frame-size match to
    the model's ``train_hw`` is used, with ``resize_to`` as an explicit override
    for inputs that carry no scale metadata at all.
    """
    from scipy.ndimage import zoom
    cfg = getattr(model, "config", {})
    target_um = train_um_override or cfg.get("pixel_um")
    target_hw = cfg.get("train_hw")
    T, H, W = movie.shape
    if target_um and in_um:                            # physical scale match
        f = in_um / target_um
        return zoom(movie, (1, f, f), order=1).astype(np.float32)
    if resize_to:                                      # explicit override
        return zoom(movie, (1, resize_to / H, resize_to / W), order=1).astype(np.float32)
    if target_hw and tuple((H, W)) != tuple(target_hw):  # frame-size match
        return zoom(movie, (1, target_hw[0] / H, target_hw[1] / W),
                    order=1).astype(np.float32)
    return movie.astype(np.float32)


def traces_from_labels(movie: np.ndarray, labels: np.ndarray,
                       weights: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted-average trace and centroid per instance label.

    ``weights`` (H,W) defaults to uniform (a plain mean inside each label); pass
    the soma probability to down-weight uncertain edge pixels. Returns
    ``(traces (N,T), centroids (N,2) as row,col)`` ordered by ascending label.
    """
    T = movie.shape[0]
    ids = np.unique(labels); ids = ids[ids != 0]
    traces = np.zeros((len(ids), T), np.float32)
    cents = np.zeros((len(ids), 2), np.float32)
    flat = movie.reshape(T, -1)
    for i, k in enumerate(ids):
        m = labels == k
        w = (np.ones(int(m.sum()), np.float32) if weights is None
             else weights[m].astype(np.float32))
        s = float(w.sum())
        if s <= 0:                                     # all-zero weights -> mean
            w = np.ones_like(w); s = float(w.sum())
        traces[i] = (flat[:, m.ravel()] * w[None, :]).sum(1) / s
        ys, xs = np.where(m); cents[i] = (ys.mean(), xs.mean())
    return traces, cents


# WRITE — the canonical per-recording folder (single writer, both runners)

def write_recording(out_root: str, rec_id: str, *,
                    traces: np.ndarray, frame_rate: float,
                    detection: Dict, stage: str, models: Dict[str, str],
                    rates: Optional[np.ndarray] = None,
                    events: Optional[Dict[str, np.ndarray]] = None,
                    labels: Optional[np.ndarray] = None,
                    centroids: Optional[np.ndarray] = None,
                    max_projection: Optional[np.ndarray] = None,
                    prob: Optional[np.ndarray] = None,
                    source: Optional[str] = None,
                    extra_meta: Optional[Dict] = None) -> str:
    """Write ``<out_root>/<rec_id>/data/`` to the contract and return the dir.

    Spatial arrays (``labels``, ``centroids``, ``max_projection``, ``prob``) are
    optional: the full runner passes them, the trace-only runner omits them. The
    ROI axis of ``traces``/``rates``/``events`` is always written.
    """
    out = os.path.join(out_root, rec_id)
    data = os.path.join(out, DATA_DIRNAME)
    os.makedirs(data, exist_ok=True)

    np.save(os.path.join(data, TRACES_NPY), traces.astype(np.float32))
    if rates is not None:
        np.save(os.path.join(data, RATES_NPY), rates.astype(np.float32))
    if events is not None:
        np.savez_compressed(os.path.join(data, EVENTS_NPZ), **events)
    if labels is not None:
        np.save(os.path.join(data, LABELS_NPY), labels.astype(np.int32))
    if centroids is not None:
        np.save(os.path.join(data, CENTROIDS_NPY),
                np.asarray(centroids, np.float32).reshape(-1, 2))
    if max_projection is not None:
        np.save(os.path.join(data, MAXPROJ_NPY), max_projection.astype(np.float32))
    if prob is not None:
        np.save(os.path.join(data, PROB_NPY), prob.astype(np.float32))

    meta = {
        "recording_id": rec_id,
        "stage": stage,
        "frame_rate": float(frame_rate),
        "n_roi": int(traces.shape[0]),
        "n_frames": int(traces.shape[1]),
        "n_events": int(len(events["roi"])) if events is not None else 0,
        "detection": detection,
        "models": models,
        "source": os.path.abspath(source) if source else None,
        "contract": {
            "data": sorted(os.listdir(data)),
            "roi_axis": "row i of traces/rates == label i+1 == centroid i; "
                        "events.npz 'roi' indexes this axis",
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(data, META_JSON), "w") as f:
        json.dump(meta, f, indent=2)
    return out


def write_figures(out: str, model, traces: np.ndarray, rates: np.ndarray,
                  frame_rate: float, detection: Dict, *,
                  max_projection: Optional[np.ndarray] = None,
                  labels: Optional[np.ndarray] = None,
                  max_roi_figures: int = 0) -> int:
    """Render the spatial QC overlay into ``<out>/figures/overlay.png``.

    The segment stage writes only the instance overlay (unlabelled outlines);
    the interactive per-ROI trace view is the activity stage's HTML gallery, so
    this no longer renders per-ROI temporal panels. The unused
    ``model``/``rates``/``detection`` parameters are kept for call-site
    compatibility with the shared writer signature.
    """
    fig_dir = os.path.join(out, FIGURES_DIRNAME)
    os.makedirs(fig_dir, exist_ok=True)
    if max_projection is not None and labels is not None:
        write_overlay(os.path.join(fig_dir, OVERLAY_PNG), max_projection, labels)
    return 0


def write_overlay(path: str, max_proj: np.ndarray, labels: np.ndarray) -> None:
    """Instance outlines on the max projection.

    Boundaries come straight from the label image (instance-aware). ROI indices
    are not drawn; use the activity stage's HTML gallery to identify individual
    ROIs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    mp = max_proj.astype(float)
    mp = mp / max(np.percentile(mp, 99.5), 1e-6)
    b = find_boundaries(labels, mode="outer")
    H, W = labels.shape
    fig, ax = plt.subplots(figsize=(8, 8 * H / max(W, 1)))
    ax.imshow(np.clip(mp, 0, 1), cmap="gray")
    ax.imshow(np.ma.masked_where(~b, b), cmap="autumn", alpha=0.9)
    ax.set_title(f"{int(labels.max())} ROIs", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# FULL CHAIN — movie -> everything, in one process (the fused runner's core)

def infer_prob(movie_path: str, spatial_model, *, resize_to: int = 0,
               train_um_override: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parameter-independent half of detection (the ``infer`` stage): load ->
    resample to the model's scale -> predict the soma-probability map. Returns
    ``(prob, max_projection, n_frames)`` at the model's working resolution. No
    threshold, no extraction — the result is cached so tuning never re-runs this.
    """
    from orcann.pipeline.extraction import _load_movie
    from orcann.spatial.detection.segmenter import predict_prob
    movie = _load_movie(movie_path)
    in_um = nd2_um_per_px(movie_path) if movie_path.endswith(".nd2") else None
    movie = resample_for_model(spatial_model, movie, in_um=in_um,
                               resize_to=resize_to, train_um_override=train_um_override)
    prob = predict_prob(spatial_model, movie).astype(np.float32)
    return prob, movie.max(axis=0).astype(np.float32), int(movie.shape[0])


def resample_to_shape(movie: np.ndarray, hw) -> np.ndarray:
    """Spatially resample a movie to a target (H, W) — the model-free counterpart
    of ``resample_for_model``, used by the ``segment`` stage to match the movie to
    a cached prob map without loading the model. Identical zoom (order=1) to the
    same output shape, so traces match what the welded path would extract.
    """
    from scipy.ndimage import zoom
    T, H, W = movie.shape
    Ht, Wt = int(hw[0]), int(hw[1])
    if (H, W) == (Ht, Wt):
        return movie.astype(np.float32)
    return zoom(movie, (1, Ht / H, Wt / W), order=1).astype(np.float32)
