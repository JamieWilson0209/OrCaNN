"""Movie I/O for the pipeline: load a recording to a ``(T, H, W)`` float32 array.

Deterministic array work, no learnable parameters. ``_load_movie`` dispatches on
extension (.npy / .tif / .nd2); ND2 axes are assigned by name rather than guessed.
Trace extraction lives in ``inference.traces_from_labels`` (label-weighted means),
so this module is just intake.
"""
from __future__ import annotations

import warnings

import numpy as np


# MOVIE I/O

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

