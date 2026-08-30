"""Movie intake: open a recording, report what it says about itself, read what is asked.

This module reports; it does not decide. It reads the axis meaning, pixel size and
frame interval the file states, and refuses a file it cannot honestly interpret. It
does not guess which axis is time, does not choose a channel, and does not resample:
scale reaches ``inference.resample_for_model`` as a recorded fact, and the decision
about what to do with it stays there.

``open_recording`` reads headers only, so a caller can ask for the frames it needs
rather than the whole movie — ``frames`` for a subset, ``stream`` for a reduction over
every frame at bounded memory, ``array`` for the lot. Trace extraction lives in
``inference.traces_from_labels``, so this module is just intake.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_EXTENSIONS = (".npy", ".tif", ".tiff", ".nd2")

# A TIFF names its axes; the last two must be the image plane. 'T' is a declaration,
# 'Q' and 'I' mean the writer said nothing (a plain tifffile.imwrite produces 'QYX',
# which is every _mc.tif this pipeline has written). Any other letter is a positive
# statement that the file is not a time series.
_DECLARED_LEADING = {"T"}
_UNDECLARED_LEADING = {"Q", "I"}


class IntakeError(ValueError):
    """A recording that cannot be read as a time series, refused by name."""


@dataclass(frozen=True)
class RecordingMeta:
    """What the file said about itself. Every field is read, never inferred.

    ``um_per_px`` of None means the file did not say — never a default and never a
    config value. ``axes_source`` is "declared" when the file named its axes and
    "assumed" when it did not and (T, H, W) was taken by convention.
    """
    source: str
    n_frames: int
    height: int
    width: int
    dtype_in: str
    axes: str = "TYX"
    axes_source: str = "assumed"
    um_per_px: Optional[float] = None
    um_per_px_y: Optional[float] = None
    frame_interval_s: Optional[float] = None
    selected: Dict[str, int] = field(default_factory=dict)

    def as_record(self) -> Dict[str, object]:
        """The provenance dict for a stage record. Keys are stable; None stays None."""
        return {"source": self.source, "n_frames": self.n_frames,
                "source_hw": [self.height, self.width], "dtype_in": self.dtype_in,
                "axes": self.axes, "axes_source": self.axes_source,
                "um_per_px": self.um_per_px, "um_per_px_y": self.um_per_px_y,
                "frame_interval_s": self.frame_interval_s,
                "selected": dict(self.selected)}


def _check_finite(arr: np.ndarray, path: str) -> np.ndarray:
    """Refuse an array holding NaN or Inf, naming the file and the first offender."""
    bad = ~np.isfinite(arr)
    if bad.any():
        n = int(bad.sum())
        first = np.argwhere(bad)[0].tolist()
        raise IntakeError(
            f"{path}: {n} non-finite value(s) in the frames read, first at index "
            f"{first}. A recording with NaN or Inf reads as an absence of signal "
            f"rather than a failure; correct it upstream or exclude it.")
    return arr


class Recording:
    """An opened recording. Header read; pixel data on request.

    ``frames``/``stream``/``array`` all return float32 in the source's own intensity
    units — no normalisation, scaling or background subtraction — with the frame axis
    first. ``require_finite`` is a per-call argument rather than a separate pass, so a
    caller reading 64 frames does not pay for a scan of all 601.
    """

    def __init__(self, meta: RecordingMeta, reader, closer=None) -> None:
        self.meta = meta
        self._read = reader
        self._close = closer

    def close(self) -> None:
        """Release the file handle. Idempotent.

        The lazy readers hold the file open between calls, so a stage that opens
        many recordings must close each one or exhaust its descriptors partway
        through an array job.
        """
        if self._close is not None:
            self._close()
            self._close = None

    def __enter__(self) -> "Recording":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):                     # a caller that forgets still releases it
        try:
            self.close()
        except Exception:
            pass

    def frames(self, indices: Sequence[int], *, require_finite: bool = False) -> np.ndarray:
        """(len(indices), H, W) float32, in the order requested. 0-based."""
        idx = [int(i) for i in indices]
        n = self.meta.n_frames
        bad = [i for i in idx if not 0 <= i < n]
        if bad:
            raise IntakeError(f"{self.meta.source}: frame index {bad[0]} outside "
                              f"[0, {n}); the recording has {n} frames.")
        out = self._read(idx)
        return _check_finite(out, self.meta.source) if require_finite else out

    def stream(self, chunk: int = 32, *, require_finite: bool = False) -> Iterator[np.ndarray]:
        """Consecutive (c, H, W) float32 blocks covering every frame once, in order."""
        for start in range(0, self.meta.n_frames, chunk):
            stop = min(start + chunk, self.meta.n_frames)
            block = self._read(list(range(start, stop)))
            yield _check_finite(block, self.meta.source) if require_finite else block

    def array(self, *, require_finite: bool = False) -> np.ndarray:
        """The whole movie, (T, H, W) float32, C-contiguous and resident."""
        out = self._read(list(range(self.meta.n_frames)))
        return _check_finite(out, self.meta.source) if require_finite else out


# READERS  — one per format; each returns (meta, reader, closer), where
# reader(indices) -> (n, H, W) float32 and closer() releases the file handle.


def _open_npy(path: str):
    mm = np.load(path, mmap_mode="r")
    if mm.ndim != 3:
        raise IntakeError(f"{path}: expected a 3-D (T, H, W) movie, got shape {mm.shape}.")
    t, h, w = mm.shape
    meta = RecordingMeta(source=os.path.abspath(path), n_frames=int(t), height=int(h),
                         width=int(w), dtype_in=str(mm.dtype), axes_source="assumed")

    def read(idx):
        return np.ascontiguousarray(np.asarray(mm[idx]), dtype=np.float32)

    def close():
        handle = getattr(mm, "_mmap", None)
        if handle is not None:
            handle.close()

    return meta, read, close


def _tiff_scale(tf) -> tuple:
    """(um_per_px_x, um_per_px_y, frame_interval_s) from the tags, or Nones."""
    ux = uy = interval = None
    try:
        ij = tf.imagej_metadata or {}
        unit = str(ij.get("unit", "")).lower()
        if unit in ("um", "micron", "microns", "\\u00b5m", "µm"):
            tags = tf.pages[0].tags
            for name, slot in (("XResolution", "x"), ("YResolution", "y")):
                if name in tags:
                    num, den = tags[name].value
                    if num:
                        val = float(den) / float(num)
                        if slot == "x":
                            ux = val
                        else:
                            uy = val
        fi = ij.get("finterval")
        if fi:
            interval = float(fi)
    except Exception:                      # tags absent or malformed -> stays unknown
        pass
    return ux, uy, interval


def _open_tiff(path: str):
    import tifffile
    tf = tifffile.TiffFile(path)
    series = tf.series[0]
    axes, shape = series.axes, series.shape
    if len(shape) != 3:
        tf.close()
        raise IntakeError(f"{path}: expected a 3-D (T, H, W) movie, got shape {shape} "
                          f"with axes {axes!r}.")
    if axes[-2:] != "YX":
        tf.close()
        raise IntakeError(f"{path}: axes {axes!r} do not end in YX, so the last two "
                          f"dimensions are not an image plane.")
    lead = axes[0]
    if lead in _DECLARED_LEADING:
        source = "declared"
    elif lead in _UNDECLARED_LEADING:
        source = "assumed"
        logger.info("  %s: axes %r name no time axis; reading as (T, H, W)",
                    os.path.basename(path), axes)
    else:
        tf.close()
        raise IntakeError(
            f"{path}: axes {axes!r} declare a '{lead}' first axis, not time. This "
            f"stage analyses one time series per file; a z-stack or a channel stack "
            f"is not one. Export the time series, or exclude the file.")
    ux, uy, interval = _tiff_scale(tf)
    t, h, w = shape
    meta = RecordingMeta(source=os.path.abspath(path), n_frames=int(t), height=int(h),
                         width=int(w), dtype_in=str(series.dtype), axes_source=source,
                         um_per_px=ux, um_per_px_y=uy, frame_interval_s=interval)

    def read(idx):
        arr = tf.asarray(key=idx)
        if arr.ndim == 2:                  # a single key collapses the frame axis
            arr = arr[None]
        return np.ascontiguousarray(arr, dtype=np.float32)

    return meta, read, tf.close


def _open_nd2(path: str):
    import nd2
    f = nd2.ND2File(path)
    sizes = dict(f.sizes)
    extra = {k: v for k, v in sizes.items() if k not in ("T", "Y", "X")}
    if extra:
        f.close()
        named = ", ".join(f"{k}={v}" for k, v in extra.items())
        raise IntakeError(
            f"{path}: nd2 carries {named} beyond the time series. Index 0 is not "
            f"chosen for you — a multi-channel, multi-plane or multi-point file is "
            f"more than one recording. Export the series to analyse, or exclude it.")
    for need in ("T", "Y", "X"):
        if need not in sizes:
            f.close()
            have = ", ".join(f"{k}={v}" for k, v in sizes.items())
            extra_hint = (" A single-frame acquisition has no time axis."
                          if need == "T" else "")
            raise IntakeError(f"{path}: nd2 has no '{need}' axis (present: {have}); "
                              f"cannot form (T, H, W).{extra_hint}")
    ux = uy = None
    try:
        v = f.voxel_size()
        ux, uy = float(v.x), float(v.y)
    except Exception:                      # metadata absent -> stays unknown
        pass
    meta = RecordingMeta(source=os.path.abspath(path), n_frames=int(sizes["T"]),
                         height=int(sizes["Y"]), width=int(sizes["X"]),
                         dtype_in=str(f.dtype), axes_source="declared",
                         um_per_px=ux, um_per_px_y=uy)

    def read(idx):
        out = np.empty((len(idx), meta.height, meta.width), np.float32)
        for j, i in enumerate(idx):
            out[j] = f.read_frame(i)
        return out

    return meta, read, f.close


def open_recording(path: str) -> Recording:
    """Open a recording and read its metadata. Does not read pixel data.

    Raises ``IntakeError``, naming the file and the reason, for an unsupported
    extension, a shape that is not 3-D, or axes that state the file is not a time
    series. Dispatch is on the literal, case-sensitive suffix, matching the globs in
    ``cli.list_recordings``.
    """
    if path.endswith(".npy"):
        meta, reader, closer = _open_npy(path)
    elif path.endswith((".tif", ".tiff")):
        meta, reader, closer = _open_tiff(path)
    elif path.endswith(".nd2"):
        meta, reader, closer = _open_nd2(path)
    else:
        raise IntakeError(f"unsupported movie format: {path} "
                          f"(expected one of {', '.join(_EXTENSIONS)})")
    return Recording(meta, reader, closer)


def _load_movie(path: str) -> np.ndarray:
    """The whole movie as (T, H, W) float32.

    Kept for the callers that want every frame and no provenance. A caller that needs
    the pixel size, or only some of the frames, should use ``open_recording``.
    """
    with open_recording(path) as rec:
        return rec.array()
