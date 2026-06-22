"""Annotation and ROI loading utilities.

Parsers that turn a manual annotation (ImageJ ROI set, embedded-overlay TIFF,
flattened RGB overlay, label image, or centroid table) into either an integer
label image or a (centroids, radii) pair. Used by the segmenter's data loader
(:func:`orcann.spatial_seg.load_seg_recording`) and the annotation QA tool
(``scripts/check_annotation.py``). Heavy optional deps (skimage, roifile,
tifffile, scipy) are imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from orcann.spatial_log import centroids_from_masks


def _rasterize_rois(rois, shape: Tuple[int, int]) -> np.ndarray:
    """A sequence of ImageJ ROI objects -> integer label image (H, W).

    ImageJ stores ROI coordinates as (x, y) = (column, row); image arrays are
    (row, col). x->col, y->row is mapped explicitly here: getting it backwards
    is a SILENT transpose (cells land mirrored). Run scripts/check_annotation.py
    on the first real recording to eyeball the overlay before training.
    """
    from skimage.draw import polygon as sk_polygon
    H, W = shape
    label = np.zeros((H, W), dtype=np.int32)
    n = 0
    for roi in rois:
        xy = np.asarray(roi.coordinates(), dtype=float)     # (N, 2) as (x, y)
        if xy.ndim != 2 or len(xy) < 3:
            continue
        n += 1
        cols, rows = xy[:, 0], xy[:, 1]                      # x->col, y->row
        rr, cc = sk_polygon(rows, cols, shape=(H, W))
        label[rr, cc] = n
    return label


def _rasterize_imagej_rois(path: str, shape: Tuple[int, int]) -> np.ndarray:
    """ImageJ/FIJI ROI set (.zip or single .roi) -> integer label image (H, W)."""
    from roifile import roiread
    rois = roiread(path)
    if not isinstance(rois, (list, tuple)):
        rois = [rois]
    return _rasterize_rois(rois, shape)


def _imagej_rois_from_tiff(path: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """ROIs embedded in a TIFF as an ImageJ Overlay -> label image, or None.

    This is the "max projection with ROIs drawn over, saved as TIFF" case.
    ImageJ serialises the drawn ROIs into the ImageJ metadata ('Overlays' for
    many, 'ROI' for one); they are recovered as vector polygons and rasterised,
    NOT read from pixel intensities. Returns None if the TIFF carries no
    embedded ROIs, leaving the caller to treat it as a label image or reject it.
    """
    import tifffile
    from roifile import ImagejRoi
    with tifffile.TiffFile(path) as tf:
        meta = tf.imagej_metadata or {}
    ov = meta.get("Overlays", meta.get("ROI", None))
    if ov is None:
        return None
    if isinstance(ov, (bytes, bytearray)):
        ov = [ov]
    rois = [ImagejRoi.frombytes(bytes(b)) for b in ov]
    return _rasterize_rois(rois, shape)


def _label_image_to_centroids(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A (H, W) image that is meant to be labels/mask -> (centroids, radii).

    Guards the catastrophic misread of a flattened grayscale max projection as
    labels. Accepts the image as labels only if it is integer-valued, has a
    plausible cell count (<=1000 distinct nonzero values), and has a real
    background (a meaningful fraction of exact-zero pixels). A binary or
    single-value mask is connected-component relabelled so each soma becomes one
    instance. Anything else raises with a fix-it message.
    """
    from scipy.ndimage import label as cc_label
    a = np.asarray(arr)
    finite = a[np.isfinite(a)]
    is_int = finite.size > 0 and np.allclose(finite, np.round(finite))
    nz = np.unique(a[a != 0])
    zero_frac = float((a == 0).mean())
    if (not is_int) or len(nz) > 1000 or zero_frac < 0.2:
        raise ValueError(
            "annotation TIFF has no embedded ImageJ ROIs and does not look like a "
            "label image (continuous values, too many distinct values, or no "
            "background). This is most likely a raw max projection with ROI "
            "outlines burned into the pixels. In ImageJ, select the cells in the "
            "ROI Manager and use More > Save to write a RoiSet.zip, then point the "
            "loader at that .zip instead."
        )
    if len(nz) <= 2:                                   # binary / single-value mask
        lab, _ = cc_label(a > 0)
        return centroids_from_masks(lab.astype(np.int32))
    return centroids_from_masks(a.astype(np.int32))


def _colored_outline_to_centroids(rgb: np.ndarray, spread_thresh: int = 30,
                                  min_area: int = 6
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Recover annotations from a FLATTENED ImageJ overlay (RGB) image.

    The "max projection with ROIs drawn over, saved/flattened to RGB TIFF" case:
    the vector ROIs are gone, but the drawn outlines survive as coloured pixels
    over a grayscale (R==G==B) background. Coloured pixels (large channel spread,
    so this works for any overlay colour, not just yellow) are the outlines; each
    closed outline is filled to a blob and its centroid taken.

    LOSSY: touching cells can merge and very faint or broken outlines can be
    missed. ALWAYS eyeball the result with scripts/check_annotation.py before
    training on a recovered recording.
    """
    from scipy.ndimage import (binary_fill_holes, binary_closing,
                               label as cc_label, generate_binary_structure)
    a = np.asarray(rgb)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:   # (3,H,W) -> (H,W,3)
        a = np.moveaxis(a, 0, -1)
    a = a[..., :3].astype(np.int32)
    outline = (a.max(-1) - a.min(-1)) > spread_thresh          # coloured = drawn
    st = generate_binary_structure(2, 2)
    filled = binary_fill_holes(binary_closing(outline, st, iterations=1))
    lab, n = cc_label(filled, st)
    if n == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
    areas = np.array([(lab == i).sum() for i in range(1, n + 1)])
    keep = np.where(areas >= min_area)[0] + 1
    relab = np.zeros_like(lab)
    for j, i in enumerate(keep, 1):
        relab[lab == i] = j
    return centroids_from_masks(relab.astype(np.int32))


def _load_annotation(path: str, shape: Tuple[int, int], default_radius: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    if path.endswith((".zip", ".roi")):                     # ImageJ/FIJI ROI set
        return centroids_from_masks(_rasterize_imagej_rois(path, shape))
    if path.endswith(".csv"):
        arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2).astype(np.float32)
        cents = arr[:, :2]
        radii = arr[:, 2] if arr.shape[1] >= 3 else np.full(len(arr), default_radius, np.float32)
        return cents, radii
    if path.endswith((".tif", ".tiff")):
        import tifffile
        roi_label = _imagej_rois_from_tiff(path, shape)     # embedded Overlay first
        if roi_label is not None:
            return centroids_from_masks(roi_label)
        img = tifffile.imread(path)
        if img.ndim == 3 and (img.shape[-1] == 3 or img.shape[0] == 3):
            return _colored_outline_to_centroids(img)       # flattened RGB overlay
        return _label_image_to_centroids(img)               # 2D label/mask, guarded
    if path.endswith(".npy"):
        a = np.load(path)
        if a.ndim == 2 and a.shape != (len(a), 2):       # label image
            return _label_image_to_centroids(a)
        cents = a[:, :2].astype(np.float32)              # centroids
        radii = (a[:, 2] if a.shape[1] >= 3 else
                 np.full(len(a), default_radius, np.float32)).astype(np.float32)
        return cents, radii
    raise ValueError(f"unsupported annotation format: {path}")
