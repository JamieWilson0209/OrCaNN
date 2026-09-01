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

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from orcann.run_info import write_record


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
      ``.../REC_mc.tif``                  -> ``REC``      (a movie corrected elsewhere)
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

def resample_plan(model, hw, in_um: Optional[float] = None, resize_to: int = 0,
                  train_um_override: Optional[float] = None) -> Dict[str, object]:
    """Which rule ``resample_for_model`` will apply to a frame of shape ``hw``.

    The decision and the zoom factors live here rather than in the resampler, so
    what a stage records is what the stage did. ``branch`` is one of
    ``physical`` (µm/px matched), ``resize_to`` (explicit override), ``train_hw``
    (frame-size matched to the model) or ``none`` (the movie is already at the
    model's size, or the model records no size).
    """
    cfg = getattr(model, "config", {})
    target_um = train_um_override or cfg.get("pixel_um")
    target_hw = cfg.get("train_hw")
    H, W = int(hw[0]), int(hw[1])
    plan: Dict[str, object] = {
        "source_hw": [H, W], "in_um_per_px": in_um, "target_um_per_px": target_um,
        "train_hw": [int(v) for v in target_hw] if target_hw else None,
        "resize_to": int(resize_to)}
    if target_um and in_um:                            # physical scale match
        f = in_um / target_um
        plan.update(branch="physical", zoom_y=float(f), zoom_x=float(f))
    elif resize_to:                                    # explicit override
        plan.update(branch="resize_to", zoom_y=resize_to / H, zoom_x=resize_to / W)
    elif target_hw and (H, W) != tuple(target_hw):     # frame-size match
        plan.update(branch="train_hw",
                    zoom_y=target_hw[0] / H, zoom_x=target_hw[1] / W)
    else:
        plan.update(branch="none", zoom_y=1.0, zoom_x=1.0)
    return plan


def resample_for_model(model, movie: np.ndarray, in_um: Optional[float] = None,
                       resize_to: int = 0, train_um_override: Optional[float] = None
                       ) -> np.ndarray:
    """Resample a movie to the scale the segmenter was trained at.

    Physical match (µm/px) is preferred when both the recording's pixel size and
    the model's training pixel size are known; otherwise a frame-size match to
    the model's ``train_hw`` is used, with ``resize_to`` as an explicit override
    for inputs that carry no scale metadata at all. ``resample_plan`` holds the
    choice between those rules.
    """
    from scipy.ndimage import zoom
    plan = resample_plan(model, movie.shape[1:], in_um, resize_to, train_um_override)
    if plan["branch"] == "none":
        return movie.astype(np.float32)
    return zoom(movie, (1, plan["zoom_y"], plan["zoom_x"]), order=1).astype(np.float32)


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
                    provenance: Optional[Dict] = None,
                    source: Optional[str] = None) -> str:
    """Write ``<out_root>/<rec_id>/data/`` to the contract and return the dir.

    Spatial arrays (``labels``, ``centroids``, ``max_projection``, ``prob``) are
    optional: the full runner passes them, the trace-only runner omits them. The
    ROI axis of ``traces``/``rates``/``events`` is always written.

    ``provenance`` carries the store this output belongs to and the store it was
    made from. The record is written after the arrays and is what marks the
    recording done, so a run killed partway leaves arrays no later run will read.
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
        "frame_rate": float(frame_rate),
        "n_roi": int(traces.shape[0]),
        "n_frames": int(traces.shape[1]),
        "n_events": int(len(events["roi"])) if events is not None else 0,
        "detection": detection,
        "models": models,
        **(provenance or {}),
        "source": os.path.abspath(source) if source else None,
        "contract": {
            "data": sorted(os.listdir(data)),
            "roi_axis": "row i of traces/rates == label i+1 == centroid i; "
                        "events.npz 'roi' indexes this axis",
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_record(os.path.join(data, META_JSON), stage, meta)
    return out


def write_figures(out: str, model, traces: np.ndarray, rates: np.ndarray,
                  frame_rate: float, detection: Dict, *,
                  max_projection: Optional[np.ndarray] = None,
                  labels: Optional[np.ndarray] = None,
                  max_roi_figures: int = 0) -> int:
    """Render the spatial QC overlay into ``<out>/figures/overlay.png``.

    The segment stage writes only the instance overlay (unlabelled outlines);
    the interactive per-ROI trace view is the activity stage's HTML gallery, so
    per-ROI temporal panels are not rendered here. The unused
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
               train_um_override: Optional[float] = None
               ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Parameter-independent half of detection (the ``infer`` stage): load ->
    resample to the model's scale -> predict the soma-probability map. Returns
    ``(prob, max_projection, provenance)`` at the model's working resolution. No
    threshold, no extraction — the result is cached so tuning never re-runs this.

    ``provenance`` is what the recording said about itself
    (``RecordingMeta.as_record()``) plus the ``resample`` plan that was applied,
    so which scaling rule ran, and by how much, is answerable from the stage's
    record instead of being unrecoverable from the arrays afterwards.
    """
    import torch

    from orcann.pipeline.extraction import open_recording
    from orcann.spatial.detection.segmenter import predict_prob

    with open_recording(movie_path) as rec:
        n_frames = rec.meta.n_frames
        meta_record = rec.meta.as_record()
        # The recording already told us its pixel size when it was opened; reading it
        # here rather than re-opening the file is the only change to what
        # resample_for_model is given. The .nd2 restriction is kept deliberately: a
        # declared TIFF now carries a micron size too, and passing it would arm the
        # physical-scale branch the moment a checkpoint records a training scale.
        # That switch is a decision, not a side effect -- see dev/segment-stage-review.md.
        in_um = rec.meta.um_per_px if movie_path.endswith(".nd2") else None

        # Only the frames the energy front-end will actually pool. It subsamples on an
        # even stride when handed more than n_energy_frames, so selecting the same
        # frames here and handing it exactly those leaves the result unchanged while
        # keeping the rest of the movie off the device.
        n_pool = getattr(spatial_model.front, "n_energy_frames", None)
        if n_pool and n_frames > n_pool:
            idx = torch.linspace(0, n_frames - 1, n_pool).round().long().tolist()
        else:
            idx = list(range(n_frames))
        plan = resample_plan(spatial_model, (rec.meta.height, rec.meta.width),
                             in_um=in_um, resize_to=resize_to,
                             train_um_override=train_um_override)
        pooled = resample_for_model(
            spatial_model, rec.frames(idx, require_finite=True), in_um=in_um,
            resize_to=resize_to, train_um_override=train_um_override)
        prob = predict_prob(spatial_model, pooled).astype(np.float32)

        # The projection is over every frame, so it is streamed rather than
        # materialised: max is associative, and the spatial resample is applied per
        # block, which is identical to resampling the whole movie because the time
        # factor is exactly 1.
        maxproj = None
        for block in rec.stream(32, require_finite=True):
            block = resample_for_model(spatial_model, block, in_um=in_um,
                                       resize_to=resize_to,
                                       train_um_override=train_um_override)
            block_max = block.max(axis=0)
            maxproj = block_max if maxproj is None else np.maximum(maxproj, block_max)

    provenance = dict(meta_record, resample=plan)
    return prob, maxproj.astype(np.float32), provenance


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
