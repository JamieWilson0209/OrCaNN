"""Annotation intake: ImageJ/FIJI ROI sets (RoiSet.zip, or a single .roi) -> a
label image and ROI centroids, for the segmenter loader and the annotation QA
pre-flight. ROI sets are the only accepted format. See docs/spatial/training.md.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from orcann.spatial.detection.laplacian import centroids_from_masks


def _rasterize_rois(rois, shape: Tuple[int, int]) -> np.ndarray:
    """ImageJ ROI objects -> integer label image (H, W). ImageJ stores (x, y) =
    (col, row); the x->col, y->row mapping is explicit (backwards = silent
    transpose). See docs/spatial/training.md."""
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


def _load_annotation(path: str, shape: Tuple[int, int]
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """ImageJ/FIJI ROI set -> (centroids, radii). ROI sets are the only accepted
    annotation format. See docs/spatial/training.md."""
    if path.endswith((".zip", ".roi")):
        return centroids_from_masks(_rasterize_imagej_rois(path, shape))
    raise ValueError(
        f"unsupported annotation format: {path}. Annotations must be an ImageJ "
        "ROI set (.zip or .roi): in ImageJ, select the cells in the ROI Manager "
        "and use More > Save to write a RoiSet.zip, then point the loader at it."
    )
