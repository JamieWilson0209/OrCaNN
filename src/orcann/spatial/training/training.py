"""Spatial segmenter training: annotated-recording loading, patch sampling,
IoU metrics, the training loop, and a synthetic source generator.

Mirrors ``orcann.temporal.training``; the model is in ``orcann.spatial.detection.segmenter``.
See docs/spatial/training.md.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from orcann.spatial.detection.segmenter import SpatialSegmenter, focal_dice_loss


class SegRecording:
    """Movie (T,H,W) float32 + instance label image (H,W) int (0 = background)."""
    def __init__(self, movie: np.ndarray, label: np.ndarray, rid: str = ""):
        assert movie.shape[1:] == label.shape, "movie/label shape mismatch"
        self.movie = movie.astype(np.float32)
        self.label = label.astype(np.int32)
        self.rid = rid

    @property
    def centroids(self) -> np.ndarray:
        labs = np.unique(self.label); labs = labs[labs != 0]
        cents = []
        for l in labs:
            ys, xs = np.where(self.label == l)
            cents.append((ys.mean(), xs.mean()))
        return np.asarray(cents, np.float32).reshape(-1, 2)


def load_seg_recording(movie_path: str, label_path: str,
                       min_area: int = 0) -> SegRecording:
    from orcann.pipeline.extraction import _load_movie
    movie = _load_movie(movie_path)
    if label_path.endswith(".npy"):
        label = np.load(label_path)
    else:                                                   # ImageJ ROI set
        from orcann.spatial.training.annotations import _rasterize_imagej_rois
        label = _rasterize_imagej_rois(label_path, movie.shape[1:])
    label = np.asarray(label, np.int32)
    if min_area > 0:                                        # drop tiny ROIs (noise)
        sizes = np.bincount(label.ravel())
        drop = np.where(sizes < min_area)[0]
        drop = drop[drop != 0]
        if drop.size:
            label[np.isin(label, drop)] = 0
        keep = np.unique(label); keep = keep[keep != 0]
        remap = np.zeros(int(label.max()) + 1, np.int32)
        remap[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
        label = remap[label]
    return SegRecording(movie, label, rid=movie_path)


def _seg_patches(rec: SegRecording, patch: int, n: int, rng: np.random.Generator,
                 fg_frac: float = 0.75):
    """Yield (movie_patch, mask_patch); a fraction centred on a random cell so
    sparse foreground is seen. See docs/spatial/training.md."""
    T, H, W = rec.movie.shape
    mask = (rec.label > 0).astype(np.float32)
    if patch >= H or patch >= W:
        yield rec.movie, mask
        return
    cents = rec.centroids
    for i in range(n):
        if len(cents) and rng.random() < fg_frac:
            cy, cx = cents[rng.integers(len(cents))]
            y0 = int(np.clip(cy - patch // 2, 0, H - patch))
            x0 = int(np.clip(cx - patch // 2, 0, W - patch))
        else:
            y0 = int(rng.integers(0, H - patch)); x0 = int(rng.integers(0, W - patch))
        sub = rec.movie[:, y0:y0 + patch, x0:x0 + patch]
        m = mask[y0:y0 + patch, x0:x0 + patch]
        yield sub, m


def _materialize(src, loader):
    if isinstance(src, SegRecording):
        return src
    return loader(*src)


def soft_iou(prob: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> float:
    p = prob >= thr
    inter = (p & (mask > 0)).sum()
    union = (p | (mask > 0)).sum()
    return float(inter / (union + 1e-6))


def best_iou(prob: np.ndarray, mask: np.ndarray,
             thresholds: Optional[Sequence[float]] = None) -> Tuple[float, float]:
    """IoU at the best threshold over a sweep -> (iou, thr); 0.5 is arbitrary for
    an imbalanced soft map. See docs/spatial/training.md."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    best_v, best_t = 0.0, 0.5
    for t in thresholds:
        v = soft_iou(prob, mask, float(t))
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_v, best_t


def train_segmenter(sources, channels: Dict[str, bool],
                    radii_px=(4, 6, 9, 13, 18), patch: int = 128, n_patch: int = 6,
                    epochs: int = 30, lr: float = 3e-3, hidden: int = 24,
                    n_energy_frames: int = 64, seed: int = 0, pixel_um=None,
                    loader=None, checkpoint_path: Optional[str] = None,
                    device: Optional[torch.device] = None) -> SpatialSegmenter:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialSegmenter(radii_px=radii_px, hidden=hidden, pixel_um=pixel_um,
                             n_energy_frames=n_energy_frames, **channels).to(device)
    # record the training frame size so inference can auto-rescale new recordings
    rec0 = _materialize(sources[0], loader)
    model.train_hw = tuple(int(x) for x in rec0.movie.shape[1:])
    model.config["train_hw"] = list(model.train_hw)
    del rec0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    order = list(range(len(sources)))
    for ep in range(epochs):
        rng.shuffle(order)
        tot = 0.0; nb = 0
        for i in order:
            rec = _materialize(sources[i], loader)
            for sub, m in _seg_patches(rec, patch, n_patch, rng):
                x = torch.from_numpy(sub.astype(np.float32)).to(device)[None]
                y = torch.from_numpy(m).to(device)[None, None]
                opt.zero_grad()
                loss = focal_dice_loss(model(x), y)
                loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1
            del rec
        sched.step()
        if checkpoint_path:
            from orcann.pipeline.model_io import save_model
            save_model(model, checkpoint_path)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  loss {tot / max(nb,1):.4f}")
    return model


def synthetic_sources(n_rec: int = 4, hw: int = 128, n_cells: int = 25,
                      T: int = 40, seed: int = 0) -> List[SegRecording]:
    rng = np.random.default_rng(seed)
    recs = []
    yy, xx = np.mgrid[0:hw, 0:hw]
    for r in range(n_rec):
        label = np.zeros((hw, hw), np.int32)
        movie = rng.normal(0.1, 0.03, (T, hw, hw)).astype(np.float32)
        placed = 0
        for _ in range(n_cells * 3):
            if placed >= n_cells:
                break
            cy, cx = rng.integers(12, hw - 12, size=2)
            a, b = rng.uniform(3, 6), rng.uniform(3, 6)       # ellipse semi-axes
            th = rng.uniform(0, np.pi)
            ct, st = np.cos(th), np.sin(th)
            xr = (xx - cx) * ct + (yy - cy) * st
            yr = -(xx - cx) * st + (yy - cy) * ct
            ell = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
            if (label[ell] != 0).any():
                continue
            placed += 1
            label[ell] = placed
            fires = rng.random(T) < 0.1
            movie[fires] += ell[None] * rng.uniform(0.5, 1.5)
        recs.append(SegRecording(movie, label, rid=f"syn_{r}"))
    return recs
