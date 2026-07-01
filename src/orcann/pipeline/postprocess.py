"""Probability-map post-processing: turn a soma-probability map into instances.

The parameter-dependent, model-free half of spatial detection: thresholding the
cached probability map into instance labels, size-filtering, and (post-extraction)
dropping curated ROIs by id or box. Connected components need only numpy + scipy;
the watershed path lazily imports the torch-backed detection operators, so it
costs nothing unless ``watershed=True`` is requested.

The ``segment`` stage (pipeline.run_segment) calls ``labels_from_prob`` here on
the cached probability map, so thresholding and curation live in one place.
"""
from __future__ import annotations

import numpy as np


def drop_small_labels(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove instances below ``min_area`` px and relabel contiguous 1..N."""
    sizes = np.bincount(labels.ravel())
    drop = np.where(sizes < min_area)[0]
    drop = drop[drop != 0]
    if drop.size:
        labels = labels.copy()
        labels[np.isin(labels, drop)] = 0
    keep = np.unique(labels); keep = keep[keep != 0]
    remap = np.zeros(int(labels.max()) + 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    return remap[labels]


def labels_from_prob(prob: np.ndarray, threshold: float = 0.5,
                     watershed: bool = False, min_distance: int = 4,
                     min_area: int = 4, min_radius: float = 0.0) -> np.ndarray:
    """Reduce a soma-probability map to an instance label image.

    Connected components by default (touching cells merge); ``watershed`` splits
    them with peak-seeded basins (this branch needs torch). Regions below
    ``min_area`` px (or the area implied by ``min_radius``, pi*r^2) are dropped,
    then labels are relabelled contiguous 1..N.
    """
    from scipy.ndimage import label as cc_label

    if watershed:
        # torch-backed helpers, imported only when watershed is requested
        from orcann.spatial.detection.laplacian import extract_instances
        from orcann.spatial.detection.segmenter import segment_instances
        seeds, _ = extract_instances(prob, min_distance=min_distance,
                                     threshold=threshold)
        labels = segment_instances(prob, seeds, threshold=threshold)
    else:
        labels = cc_label(prob >= threshold)[0].astype(np.int32)

    area = min_area
    if min_radius > 0:
        area = max(area, int(round(np.pi * min_radius ** 2)))
    if area > 0:
        labels = drop_small_labels(labels, area)
    return labels.astype(np.int32)


def subset_rois(labels, centroids, traces, exclude_rois=None, exclude_boxes=None):
    """Drop curated ROIs and relabel; returns ``(labels, centroids, traces)``.

    Curation is a post-extraction subset (no re-segmentation): ``exclude_rois`` is
    a list of 1-based ids exactly as numbered on the overlay (row i of traces is
    label i+1 is centroid i); ``exclude_boxes`` is a list of ``[r0, c0, r1, c1]``
    pixel boxes (in the working/overlay resolution) and drops any ROI whose
    centroid lies inside. The label image is relabelled contiguous 1..K so the
    kept axis stays consistent across labels / centroids / traces.
    """
    n = int(traces.shape[0])
    keep = np.ones(n, bool)
    for r in (exclude_rois or []):
        if 1 <= int(r) <= n:
            keep[int(r) - 1] = False
    cen = np.asarray(centroids, np.float32).reshape(-1, 2)
    for box in (exclude_boxes or []):
        r0, c0, r1, c1 = box
        rlo, rhi = sorted((r0, r1)); clo, chi = sorted((c0, c1))
        inside = ((cen[:, 0] >= rlo) & (cen[:, 0] <= rhi) &
                  (cen[:, 1] >= clo) & (cen[:, 1] <= chi))
        keep[inside] = False
    if keep.all():
        return labels, cen, traces

    old_ids = np.nonzero(keep)[0] + 1                    # kept labels (1-based)
    remap = np.zeros(int(labels.max()) + 1, np.int32)
    remap[old_ids] = np.arange(1, len(old_ids) + 1, dtype=np.int32)
    return remap[labels].astype(np.int32), cen[keep], traces[keep]
