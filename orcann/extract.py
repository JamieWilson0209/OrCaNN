"""Non-learned data plumbing between the two ∇²G stages.

Everything here is deterministic array work — no learnable parameters. It sits
between the spatial detector (which produces a cellness map) and the temporal
model (which consumes per-ROI traces):

    movie (T, H, W)                          <- _load_movie  (.npy / .tif / .nd2)
      + cellness (H, W) + centroids (N, 2)   -> soft_footprints  -> (N, H, W)
      + movie                                -> extract_traces   -> (N, T)

The footprint step is the learned-head replacement for Otsu (irregular
boundaries come straight from the cellness map), with optional activity-gating
carried over from the old contour step. Trace extraction is a weighted average.
"""
from __future__ import annotations

import warnings

import numpy as np


# =============================================================================
# MOVIE I/O
# =============================================================================

def _load_movie(path: str) -> np.ndarray:
    guessed_order = True
    if path.endswith(".npy"):
        mv = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    elif path.endswith((".tif", ".tiff")):
        import tifffile
        mv = tifffile.imread(path).astype(np.float32)
    elif path.endswith(".nd2"):
        mv = _load_nd2(path)            # axes assigned by name (T, Y, X)
        guessed_order = False
    else:
        raise ValueError(f"unsupported movie format: {path}")
    if mv.ndim != 3:
        raise ValueError(f"expected a (T, H, W) movie, got shape {mv.shape} from {path}")
    if guessed_order and mv.shape[0] < min(mv.shape[1], mv.shape[2]):
        warnings.warn(f"{path}: first axis ({mv.shape[0]}) smaller than spatial "
                      f"dims {mv.shape[1:]}; expected (T, H, W) — check orientation.")
    return mv


def _load_nd2(path: str) -> np.ndarray:
    """Read a Nikon .nd2 recording to a (T, H, W) float32 array.

    Uses the named axes (``ND2File.sizes``) rather than guessing order: spatial
    axes Y,X are kept, T is time; any extra axis (C channel, Z plane, P point)
    is squeezed if singleton, else index 0 is taken with a warning. Single-plane
    single-channel 2 Hz calcium recordings collapse cleanly to (T, Y, X).
    DATA-INTAKE: if your .nd2 has real multiple channels/planes/points, confirm
    which to use — see the warning, and we can select explicitly.
    """
    import nd2
    with nd2.ND2File(path) as f:
        axes = list(f.sizes.keys())          # array axis order matches this
        sizes = dict(f.sizes)
        arr = f.asarray()
    keep = {"T", "Y", "X"}
    index, kept = [], []
    for ax in axes:
        if ax in keep:
            index.append(slice(None)); kept.append(ax)
        else:
            if sizes[ax] > 1:
                warnings.warn(f"{path}: nd2 axis '{ax}' size {sizes[ax]} > 1; "
                              f"taking index 0. Confirm this is intended.")
            index.append(0)
    arr = arr[tuple(index)]
    for need in ("T", "Y", "X"):
        if need not in kept:
            raise ValueError(f"{path}: nd2 missing a '{need}' axis (axes={axes}); "
                             f"cannot form (T, H, W).")
    arr = np.transpose(arr, [kept.index(a) for a in ("T", "Y", "X")])
    return np.ascontiguousarray(arr, dtype=np.float32)


# =============================================================================
# FOOTPRINTS  (learned-head replacement for Otsu + activity-gating)
# =============================================================================

def soft_footprints(
    cellness: np.ndarray,
    centroids: np.ndarray,
    movie: "np.ndarray | None" = None,
    threshold: float = 0.4,
    activity_gate_frac: float = 0.1,
) -> np.ndarray:
    """Assign foreground cellness to nearest peak -> per-instance soft masks.

    Irregular boundaries come straight from the learned cellness map (the Otsu
    role). If ``movie`` is given, each footprint is sharpened by its own
    peak-activity frames: the top ``activity_gate_frac`` of frames by in-mask
    mean intensity build a local activity map that multiplicatively weights the
    footprint — the activity-gating that made the old contour step effective,
    and the natural seam where the temporal stage can later drive frame
    selection.
    """
    H, W = cellness.shape
    N = len(centroids)
    if N == 0:
        return np.zeros((0, H, W), dtype=np.float32)

    fg = cellness >= threshold
    ys, xs = np.where(fg)
    pts = np.stack([ys, xs], axis=1)
    d2 = ((pts[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)   # (P, N)
    owner = d2.argmin(axis=1)

    fps = np.zeros((N, H, W), dtype=np.float32)
    for p, (y, x) in enumerate(pts):
        fps[owner[p], y, x] = cellness[y, x]

    if movie is not None and activity_gate_frac > 0:
        T = movie.shape[0]
        k = max(1, int(T * activity_gate_frac))
        for i in range(N):
            m = fps[i] > 0
            if not m.any():
                continue
            trace = movie[:, m].mean(axis=1)
            top = np.argsort(trace)[-k:]
            act = movie[top].max(axis=0)
            act = (act - act.min()) / (np.ptp(act) + 1e-9)
            fps[i] *= act
        # renormalise so each footprint still sums sensibly for averaging
        s = fps.reshape(N, -1).sum(axis=1)
        fps[s > 0] /= s[s > 0][:, None, None]
        fps[s > 0] *= (cellness >= threshold).sum() / max(N, 1)  # rough scale
    return fps


# =============================================================================
# TRACES
# =============================================================================

def extract_traces(movie: np.ndarray, footprints: np.ndarray) -> np.ndarray:
    """Weighted-average trace per footprint: C_i(t) = Σ_x A_i(x) Y(x,t) / Σ A_i."""
    T = movie.shape[0]
    N = footprints.shape[0]
    A = footprints.reshape(N, -1).astype(np.float32)
    w = A.sum(axis=1, keepdims=True)
    w[w == 0] = 1e-9
    Y = movie.reshape(T, -1).astype(np.float32)          # (T, P)
    return (A @ Y.T) / w                                  # (N, T)
