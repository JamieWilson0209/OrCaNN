"""Present an OrCaNN label image the way the gallery expects.

The calcium-pipeline gallery was written around a contour-based detector: it
reads a ``seeds`` object (per-ROI centre, radius, boundary polygon, circularity)
and a ``projections`` object (max / mean / std / correlation images). OrCaNN's
``segment`` stage instead produces an instance-label image plus per-ROI traces.
This module derives the gallery's inputs from that label image, so the exact
same gallery renders over segmenter output with no changes to it:

  - ``build_seed_view``    labels -> a duck-typed seeds object (one entry per label)
  - ``build_projections``  movie  -> max / mean / std / correlation images
  - ``footprints_from_labels`` labels -> sparse (d1*d2, N) footprints for analysis

Every derived quantity is read straight off the label geometry (region area,
boundary, edge contact), so nothing is invented: contour polygons are the true
label outlines, and ``contour_success`` is True for every ROI because the
outline is exact rather than fitted. Row ``i`` of every array is label ``i+1``,
matching ``inference.write_recording``'s ROI axis so ids line up across stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# --- one ROI's derived geometry (the gallery reads .contour/.circularity/.solidity)


@dataclass
class _Contour:
    contour: np.ndarray          # (K, 2) boundary points as (x, y) = (col, row)
    circularity: float
    solidity: float


@dataclass
class SeedView:
    """Duck-typed stand-in for the calcium pipeline's ContourSeedResult.

    Exposes only the attributes the gallery touches. ``centers`` are (row, col)
    to match the gallery's ``y, x = centers[i]`` unpacking; ``contours[i].contour``
    is (x, y) to match its canvas drawing. All arrays are ordered by ascending
    label, so index ``i`` is label ``i+1``.
    """
    n_seeds: int
    centers: np.ndarray                      # (N, 2) float (row, col)
    radii: np.ndarray                        # (N,) float, equivalent-area radius
    intensities: np.ndarray                  # (N,) float, mean projection value in ROI
    contour_success: np.ndarray              # (N,) bool, always True here (exact outlines)
    boundary_touching: np.ndarray            # (N,) bool, ROI touches the frame edge
    source_projection: List[str]             # (N,) provenance label, "segmenter"
    contours: List[Optional[_Contour]] = field(default_factory=list)


@dataclass
class ProjectionSet:
    """Duck-typed stand-in for the calcium pipeline's ProjectionSet."""
    max_proj: np.ndarray
    mean_proj: np.ndarray
    std_proj: np.ndarray
    correlation: np.ndarray


def _region_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Longest boundary polygon of a binary mask as (x, y) points, or None."""
    from skimage import measure
    conts = measure.find_contours(mask.astype(float), 0.5)
    if not conts:
        return None
    c = max(conts, key=len)                  # (row, col) float pairs
    return np.stack([c[:, 1], c[:, 0]], axis=1)  # -> (x, y) = (col, row)


def build_seed_view(labels: np.ndarray,
                    max_projection: Optional[np.ndarray] = None,
                    centroids: Optional[np.ndarray] = None) -> SeedView:
    """Derive a gallery-ready seeds view from an instance-label image.

    ``max_projection`` (if given) supplies the per-ROI intensity readout;
    ``centroids`` (if given) is used verbatim, else centres are recomputed from
    the labels. Circularity/solidity come from region props; the radius is the
    equivalent-area radius ``sqrt(area / pi)``.
    """
    from skimage import measure

    ids = np.unique(labels)
    ids = ids[ids != 0]
    n = len(ids)
    H, W = labels.shape
    mp = None if max_projection is None else np.asarray(max_projection, float)

    centers = np.zeros((n, 2), np.float32)
    radii = np.zeros(n, np.float32)
    intens = np.zeros(n, np.float32)
    success = np.ones(n, bool)
    edge = np.zeros(n, bool)
    contours: List[Optional[_Contour]] = []

    props = {p.label: p for p in measure.regionprops(labels)}
    for i, k in enumerate(ids):
        m = labels == k
        ys, xs = np.where(m)
        if centroids is not None and i < len(centroids):
            centers[i] = np.asarray(centroids[i], np.float32).ravel()[:2]
        else:
            centers[i] = (ys.mean(), xs.mean())
        area = float(m.sum())
        radii[i] = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
        intens[i] = float(mp[m].mean()) if mp is not None else 0.0
        edge[i] = bool(ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1)

        p = props.get(int(k))
        perim = float(getattr(p, "perimeter", 0.0)) if p is not None else 0.0
        circ = float(4.0 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0
        circ = min(circ, 1.0)
        conv = 0.0
        if p is not None:
            conv = float(getattr(p, "area_convex", None) or getattr(p, "convex_area", 0.0))
        sol = float(area / conv) if conv > 0 else 1.0
        poly = _region_contour(m)
        contours.append(_Contour(poly, circ, sol) if poly is not None else None)
        if poly is None:
            success[i] = False

    return SeedView(
        n_seeds=n, centers=centers, radii=radii, intensities=intens,
        contour_success=success, boundary_touching=edge,
        source_projection=["segmenter"] * n, contours=contours,
    )


def build_projections(movie: np.ndarray,
                      max_projection: Optional[np.ndarray] = None) -> ProjectionSet:
    """max / mean / std / local-correlation images for the gallery backgrounds.

    ``max_projection`` may be passed in to reuse the one ``segment`` cached (it is
    computed at the model's working resolution); mean/std/correlation are cheap
    to recompute from the movie. The correlation image is the mean 8-neighbour
    Pearson correlation, the standard calcium-imaging local-correlation view.
    """
    mv = np.asarray(movie, np.float32)
    mean_p = mv.mean(axis=0)
    std_p = mv.std(axis=0)
    max_p = (np.asarray(max_projection, np.float32)
             if max_projection is not None else mv.max(axis=0))
    return ProjectionSet(max_proj=max_p, mean_proj=mean_p, std_proj=std_p,
                         correlation=_correlation_image(mv))


def _correlation_image(movie: np.ndarray) -> np.ndarray:
    """Mean Pearson correlation of each pixel with its 8 neighbours (H, W)."""
    T, H, W = movie.shape
    x = movie - movie.mean(axis=0, keepdims=True)
    sd = np.sqrt((x ** 2).sum(axis=0))
    sd[sd == 0] = 1e-12
    xn = x / sd
    corr = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.float32)
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)):
        ys0, ys1 = max(0, dy), H + min(0, dy)
        xs0, xs1 = max(0, dx), W + min(0, dx)
        yn0, yn1 = max(0, -dy), H + min(0, -dy)
        xn0, xn1 = max(0, -dx), W + min(0, -dx)
        prod = (xn[:, ys0:ys1, xs0:xs1] * xn[:, yn0:yn1, xn0:xn1]).sum(axis=0)
        corr[ys0:ys1, xs0:xs1] += prod
        count[ys0:ys1, xs0:xs1] += 1.0
    count[count == 0] = 1.0
    return corr / count


def footprints_from_labels(labels: np.ndarray):
    """Sparse (d1*d2, N) binary footprints, column ``i`` = label ``i+1``.

    C-order flattening to match the analysis loader's ``fp.reshape(d1, d2)`` and
    ``run_info.json['dims'] = [d1, d2]``. Returned as CSC for the overlap and
    crop reads in ``orcann.analysis``.
    """
    from scipy.sparse import csc_matrix

    ids = np.unique(labels)
    ids = ids[ids != 0]
    H, W = labels.shape
    npix, n = H * W, len(ids)
    rows, cols = [], []
    for i, k in enumerate(ids):
        idx = np.flatnonzero((labels == k).ravel(order="C"))
        rows.append(idx)
        cols.append(np.full(idx.size, i))
    if n:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.ones(rows.size, np.float32)
    else:
        rows = cols = data = np.array([], np.float32)
    return csc_matrix((data, (rows, cols)), shape=(npix, n))
