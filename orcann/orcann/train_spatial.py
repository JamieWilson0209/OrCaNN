"""
Spatial training harness — train the scattering detector on annotated
recordings, evaluated on held-out recordings.
=============================================================

The mask-side counterpart to ``train_loio``. It answers the first questions
that need real numbers:

  * precision / recall / F1 of the scattering detector on held-out
    recordings, against manual annotations;
  * recall stratified by cell activity, since the var-only detector is
    expected to miss faint cells — this is where the coherence channel
    either earns its place or doesn't;
  * the variance-only vs variance+coherence A/B, decided on data rather
    than on the synthetic sweep.

Everything runs through one interface, ``AnnotatedRecording``. A synthetic
bank exercises the harness today; ``load_recording`` is the real adapter and
dispatches on file type. Two things about the real data are NOT yet known and
are marked DATA-INTAKE: the annotation format, and — more importantly — the
*definition* of an annotated cell (active units only, or all visible somata).
See the module note in DOCUMENTATION.md §9; the second one decides whether a
variance/coherence (activity) detector is even the right target model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from orcann.spatial_scatter import SpatialScatterDetector
from orcann.extract import _load_movie
from orcann.spatial_log import (
    cellness_target, centroids_from_masks, bce_dice_loss, extract_instances,
)


# =============================================================================
# DATASET INTERFACE
# =============================================================================

@dataclass
class AnnotatedRecording:
    movie: np.ndarray                 # (T, H, W)
    centroids: np.ndarray             # (N, 2) in (row, col)
    radii: np.ndarray                 # (N,) px; from masks, or a default
    recording_id: str = ""
    pixel_size_um: Optional[float] = None


# =============================================================================
# REAL ADAPTER  (DATA-INTAKE: verify against the actual files)
# =============================================================================

def load_recording(movie_path: str, annotation_path: str,
                   recording_id: str = "", default_radius: float = 6.0,
                   pixel_size_um: Optional[float] = None) -> AnnotatedRecording:
    """Load one annotated recording. Dispatches on file extension.

    movie:      .npy (T,H,W) | .tif/.tiff stack
    annotation: .zip/.roi -> ImageJ/FIJI ROI set (rasterised; x,y -> col,row);
                .npy  -> a (H,W) integer label image, or an (N,2) centroid
                         array (optionally (N,3) with a radius col);
                .tif  -> (H,W) integer label image;
                .csv  -> columns row,col[,radius].
    DATA-INTAKE: confirm (a) row/col vs x/y ordering, (b) label vs centroid,
    (c) whether annotations mark active cells or all somata.
    """
    movie = _load_movie(movie_path)
    H, W = movie.shape[1:]
    cents, radii = _load_annotation(annotation_path, (H, W), default_radius)
    return AnnotatedRecording(movie, cents, radii, recording_id, pixel_size_um)


def _rasterize_imagej_rois(path: str, shape: Tuple[int, int]) -> np.ndarray:
    """ImageJ/FIJI ROI set (.zip or single .roi) -> integer label image (H, W).

    ImageJ stores ROI coordinates as (x, y) = (column, row); image arrays are
    (row, col). This function maps x->col, y->row explicitly — getting that
    backwards is a SILENT failure (cells land transposed). Run
    scripts/check_annotation.py after the first real recording to eyeball the
    overlay before training on it.
    """
    from roifile import roiread
    from skimage.draw import polygon as sk_polygon
    H, W = shape
    rois = roiread(path)
    if not isinstance(rois, (list, tuple)):
        rois = [rois]
    label = np.zeros((H, W), dtype=np.int32)
    for i, roi in enumerate(rois, start=1):
        xy = np.asarray(roi.coordinates(), dtype=float)     # (N, 2) as (x, y)
        if xy.ndim != 2 or len(xy) < 3:
            continue
        cols, rows = xy[:, 0], xy[:, 1]                      # x->col, y->row
        rr, cc = sk_polygon(rows, cols, shape=(H, W))
        label[rr, cc] = i
    return label


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
        return centroids_from_masks(tifffile.imread(path))
    if path.endswith(".npy"):
        a = np.load(path)
        if a.ndim == 2 and a.shape != (len(a), 2):       # label image
            return centroids_from_masks(a)
        cents = a[:, :2].astype(np.float32)              # centroids
        radii = (a[:, 2] if a.shape[1] >= 3 else
                 np.full(len(a), default_radius, np.float32)).astype(np.float32)
        return cents, radii
    raise ValueError(f"unsupported annotation format: {path}")


# =============================================================================
# SYNTHETIC STAND-IN  (runs the harness today)
# =============================================================================

def synthetic_annotated_bank(n_recordings: int = 4, fs_irrelevant=None,
                             H: int = 64, W: int = 64, T: int = 60,
                             cells: int = 9, noise_sd: float = 0.12,
                             silent_frac: float = 0.33,
                             seed: int = 0) -> List[AnnotatedRecording]:
    """Synthetic recordings where every cell has baseline brightness (dye
    loading, so it is visible in the mean image) and a fraction are SILENT
    (never fire) — the silent-but-visible case the manual annotations include.
    """
    rng = np.random.default_rng(seed)
    out = []
    kern = np.exp(-np.arange(0, 10) / 3.0).astype(np.float32)
    for rec in range(n_recordings):
        cents = rng.uniform(8, H - 8, size=(cells, 2)).astype(np.float32)
        radii = rng.uniform(3, 6, size=cells).astype(np.float32)
        baseline = rng.uniform(0.3, 1.0, size=cells).astype(np.float32)
        silent = rng.random(cells) < silent_frac
        amp = rng.uniform(0.2, 1.5, size=cells).astype(np.float32)
        nfire = rng.integers(4, 20, size=cells)
        nfire[silent] = 0
        yy, xx = np.mgrid[0:H, 0:W]
        movie = rng.normal(0.2, noise_sd, size=(T, H, W)).astype(np.float32)
        for i, ((cy, cx), r) in enumerate(zip(cents, radii)):
            fp = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r / 2) ** 2))
            movie += (baseline[i] * fp[None]).astype(np.float32)       # dye loading
            if nfire[i] > 0:
                sp = np.zeros(T, np.float32)
                sp[rng.integers(0, T, size=nfire[i])] = amp[i]
                movie += (fp[None] * np.convolve(sp, kern)[:T][:, None, None]).astype(np.float32)
        out.append(AnnotatedRecording(movie, cents, radii, f"synthetic_{rec}"))
    return out


# =============================================================================
# ACTIVITY SNR  (faintness stratification; works on real data, no labels)
# =============================================================================

def radius_bank_from_recordings(recordings: List[AnnotatedRecording], k: int = 5,
                                q_lo: float = 0.1, q_hi: float = 0.9
                                ) -> Tuple[float, ...]:
    """Seed the ∇²G radius bank directly from the annotation radii.

    The ROIs carry the real cell-size distribution, so the scale bank can be
    read from the data — no pixel-size metadata needed. Log-spaced between the
    lower and upper radius quantiles.
    """
    radii = np.concatenate([r.radii for r in recordings if len(r.radii)])
    lo, hi = np.quantile(radii, [q_lo, q_hi])
    lo = max(float(lo), 1.5)
    hi = max(float(hi), lo * 1.5)
    return tuple(np.round(np.geomspace(lo, hi, k), 2))


def cell_activity_snr(rec: AnnotatedRecording, disk: float = 2.0) -> np.ndarray:
    """Per-cell transient SNR: peak Δ over a small disk vs the local noise.

    A label-free proxy for how detectable each annotated cell is, used to
    stratify recall. Robust noise = MAD of frame-to-frame differences.
    """
    T, H, W = rec.movie.shape
    yy, xx = np.mgrid[0:H, 0:W]
    snr = np.zeros(len(rec.centroids), np.float32)
    for i, (cy, cx) in enumerate(rec.centroids):
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= disk ** 2
        tr = rec.movie[:, m].mean(axis=1)
        noise = np.median(np.abs(np.diff(tr) - np.median(np.diff(tr)))) / 0.6745
        snr[i] = (tr.max() - np.percentile(tr, 10)) / (noise + 1e-6)
    return snr


# =============================================================================
# PATCH SAMPLING + METRICS
# =============================================================================

def _patches(rec: AnnotatedRecording, patch: int, n: int,
             rng: np.random.Generator):
    T, H, W = rec.movie.shape
    if patch >= H or patch >= W:
        yield rec.movie, rec.centroids, rec.radii
        return
    for _ in range(n):
        y0 = rng.integers(0, H - patch); x0 = rng.integers(0, W - patch)
        sub = rec.movie[:, y0:y0 + patch, x0:x0 + patch]
        c = rec.centroids - np.array([y0, x0])
        keep = ((c[:, 0] >= 0) & (c[:, 0] < patch) &
                (c[:, 1] >= 0) & (c[:, 1] < patch))
        yield sub, c[keep], rec.radii[keep]


def match_centroids(true: np.ndarray, pred: np.ndarray, tol_px: float = 4.0):
    """Greedy nearest matching -> (hit mask over true, n_false_pos)."""
    if len(true) == 0:
        return np.zeros(0, bool), len(pred)
    if len(pred) == 0:
        return np.zeros(len(true), bool), 0
    d = np.sqrt(((true[:, None, :] - pred[None, :, :]) ** 2).sum(-1))
    hit = np.zeros(len(true), bool); used = set()
    for i in np.argsort(d.min(axis=1)):
        j = int(np.argmin([d[i, k] if k not in used else np.inf
                           for k in range(len(pred))]))
        if d[i, j] <= tol_px:
            hit[i] = True; used.add(j)
    return hit, len(pred) - len(used)


# =============================================================================
# TRAIN / EVAL
# =============================================================================

Source = object  # AnnotatedRecording, or (movie_path, roi_path) tuple loaded lazily


def _materialize(src, loader):
    """Return an AnnotatedRecording from an in-memory object or a path pair."""
    if isinstance(src, AnnotatedRecording):
        return src
    if loader is None:
        raise ValueError("path sources require a loader")
    return loader(*src)


def _pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_detector(sources: List[Source], channels: Dict[str, bool],
                   radii_px=(3, 5, 7), patch: int = 48, n_patch: int = 4,
                   epochs: int = 40, lr: float = 6e-3,
                   n_energy_frames: int = 48, seed: int = 0,
                   loader=None, checkpoint_path: Optional[str] = None,
                   device: Optional[torch.device] = None
                   ) -> SpatialScatterDetector:
    """Train the detector. Streams recordings from disk one at a time (bounded
    memory at real dataset scale) and runs on GPU when available.

    ``sources`` may be in-memory AnnotatedRecording objects (synthetic / small)
    or (movie_path, roi_path) pairs materialised via ``loader`` and freed after
    each use. A checkpoint is written every epoch so a queue-killed job is not
    a total loss.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = device or _pick_device()
    model = SpatialScatterDetector(radii_px=radii_px, n_energy_frames=n_energy_frames,
                                   corr_radius=2, corr_dirs=4, **channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    order = list(range(len(sources)))
    for ep in range(epochs):
        rng.shuffle(order)
        for i in order:
            rec = _materialize(sources[i], loader)
            for sub, c, r in _patches(rec, patch, n_patch, rng):   # one patch held at a time
                x = torch.from_numpy(sub.astype(np.float32)).to(device)[None]
                y = torch.from_numpy(cellness_target(sub.shape[1:], c, r)).to(device)[None, None]
                opt.zero_grad()
                loss = bce_dice_loss(model(x), y)
                loss.backward(); opt.step()
            del rec                                                 # free movie before next
        if checkpoint_path:
            torch.save(model, checkpoint_path)                      # resume-from / safety net
    return model.eval()


@torch.no_grad()
def evaluate(model: SpatialScatterDetector, sources: List[Source],
             threshold: float = 0.5, min_distance: int = 4, tol_px: float = 4.0,
             loader=None) -> Dict[str, float]:
    device = next(model.parameters()).device
    hits, snrs, n_fp, n_pred, n_true = [], [], 0, 0, 0
    for src in sources:
        rec = _materialize(src, loader)
        x = torch.from_numpy(rec.movie.astype(np.float32)).to(device)[None]
        cm = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        pred, _ = extract_instances(cm, min_distance, threshold)
        hit, fp = match_centroids(rec.centroids, pred, tol_px)
        hits.append(hit); snrs.append(cell_activity_snr(rec))
        n_fp += fp; n_pred += len(pred); n_true += len(rec.centroids)
        del rec
    hit = np.concatenate(hits); snr = np.concatenate(snrs)
    recall = hit.mean() if len(hit) else 0.0
    precision = (n_pred - n_fp) / max(n_pred, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    q = np.quantile(snr, [1 / 3, 2 / 3]) if len(snr) else [0, 0]
    faint = snr <= q[0]; bright = snr > q[1]
    return dict(recall=round(recall, 3), precision=round(precision, 3),
                f1=round(f1, 3),
                recall_faint=round(hit[faint].mean(), 3) if faint.any() else float("nan"),
                recall_bright=round(hit[bright].mean(), 3) if bright.any() else float("nan"),
                n_true=n_true, n_pred=n_pred, false_pos=n_fp)


def run_configs(sources: List[Source],
                configs: List[Tuple[str, Dict[str, bool]]],
                n_test: int = 1, loader=None, **kw) -> Dict[str, Dict]:
    """Recording-level split; train+evaluate each channel configuration."""
    train, test = sources[:-n_test], sources[-n_test:]
    out = {}
    for label, channels in configs:
        model = train_detector(train, channels=channels, loader=loader, **kw)
        out[label] = evaluate(model, test, loader=loader)
    return out
