"""
Stage 1 (segmentation formulation) — per-pixel P(inside a soma).
================================================================

The scatter detector (``spatial_scatter.py``) localises cell *centres*: its
target is a round Gaussian at each centroid and its output is reduced to peaks.
This module keeps the same physically-grounded front-end — the per-frame ∇²G
band-pass pooled into temporal moments (``SpatialScatterDetector.energy``) — but
replaces the detection head and target with a segmentation head and the real
rasterised footprints. The network now predicts, per pixel, the probability that
the pixel lies inside an annotated soma; thresholding yields a mask, and a
centroid-seeded watershed splits touching cells into per-instance footprints.

Why keep ``energy`` and not segment the raw movie: the structural / max /
variance / coherence moments are exactly the evidence that separates an active
soma from neuropil, computed in bounded memory over the whole recording. The
U-Net only has to turn that evidence into a boundary, which is a far easier
learning problem than reconstructing the moments from raw frames.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from orcann.spatial_scatter import SpatialScatterDetector


# =============================================================================
# MODEL
# =============================================================================

def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
    )


class SpatialSegmenter(nn.Module):
    """∇²G temporal-moment front-end + a small U-Net -> per-pixel soma logits."""

    KIND = "spatial_seg"

    def __init__(
        self,
        radii_px: Sequence[float] = (3, 3.7, 4.5, 5.5, 6.7, 8.2, 10.0),
        hidden: int = 24,
        n_energy_frames: Optional[int] = 64,
        use_structural: bool = True,
        use_max: bool = True,
        use_variance: bool = True,
        use_correlation: bool = False,
        learnable_scales: bool = False,
        corr_radius: int = 2,
        corr_dirs: int = 4,
        pixel_um: Optional[float] = None,
        train_hw: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.config = {"radii_px": list(radii_px), "hidden": hidden,
                       "n_energy_frames": n_energy_frames,
                       "use_structural": use_structural, "use_max": use_max,
                       "use_variance": use_variance, "use_correlation": use_correlation,
                       "learnable_scales": learnable_scales,
                       "corr_radius": corr_radius, "corr_dirs": corr_dirs,
                       "pixel_um": pixel_um,
                       "train_hw": list(train_hw) if train_hw else None}
        self.pixel_um = pixel_um            # microns/px the model was trained at
        self.train_hw = tuple(train_hw) if train_hw else None   # training frame (H,W)
        # Use the detector purely as the energy() feature extractor; it has no
        # detection head, so it contributes only the LoG scale parameters.
        self.front = SpatialScatterDetector(
            radii_px=radii_px, n_energy_frames=n_energy_frames,
            use_structural=use_structural, use_max=use_max,
            use_variance=use_variance, use_correlation=use_correlation,
            learnable_scales=learnable_scales,
            corr_radius=corr_radius, corr_dirs=corr_dirs)

        k = len(radii_px)
        n_groups = use_structural + use_max + use_variance + use_correlation
        n_ch = k * n_groups
        h = hidden
        self.enc1 = _block(n_ch, h)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = _block(h, 2 * h)
        self.reduce = nn.Conv2d(2 * h, h, 1)
        self.dec = _block(2 * h, h)
        self.out = nn.Conv2d(h, 1, 1)

    def forward(self, movie: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W) -> (B, 1, H, W) soma logits."""
        feats = self.front.energy(movie)
        rms = feats.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
        feats = feats / (rms + 1e-6)
        e1 = self.enc1(feats)
        e2 = self.enc2(self.pool(e1))
        e2u = self.reduce(F.interpolate(e2, size=e1.shape[-2:],
                                        mode="bilinear", align_corners=False))
        d = self.dec(torch.cat([e1, e2u], dim=1))
        return self.out(d)


# =============================================================================
# LOSS  (foreground is 0.6–19% of the frame -> needs to be imbalance-robust)
# =============================================================================

def focal_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                    gamma: float = 2.0, dice_w: float = 1.0) -> torch.Tensor:
    """Focal BCE + soft Dice. Dice is invariant to the foreground fraction;
    the focal term down-weights the easy background so sparse frames still
    learn. Reduces to plain BCE+Dice at gamma=0."""
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    focal = ((1 - pt).pow(gamma) * bce).mean()
    inter = (p * target).sum(dim=(-2, -1))
    denom = p.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + 1e-6
    dice = 1.0 - (2.0 * inter / denom).mean()
    return focal + dice_w * dice


# =============================================================================
# DATA
# =============================================================================

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
    from orcann.extract import _load_movie
    movie = _load_movie(movie_path)
    if label_path.endswith(".npy"):
        label = np.load(label_path)
    else:                                                   # ImageJ ROI set
        from orcann.annotations import _rasterize_imagej_rois
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
    """Yield (movie_patch, mask_patch). A fraction of patches are centred on a
    random cell so sparse foreground is actually seen; the rest are uniform."""
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


# =============================================================================
# TRAIN / EVAL
# =============================================================================

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
    """IoU at the best threshold (0.5 is arbitrary for an imbalanced soft map,
    and for 3-8px cells the cut point moves IoU a lot). Returns (iou, thr)."""
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
            from orcann.io import save_model
            save_model(model, checkpoint_path)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  loss {tot / max(nb,1):.4f}")
    return model


@torch.no_grad()
def predict_prob(model: SpatialSegmenter, movie: np.ndarray,
                 device: Optional[torch.device] = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.from_numpy(movie.astype(np.float32)).to(device)[None]
    return torch.sigmoid(model(x))[0, 0].cpu().numpy()


def segment_instances(prob: np.ndarray, centroids: np.ndarray,
                      threshold: float = 0.5) -> np.ndarray:
    """Threshold -> foreground; centroid-seeded watershed -> instance labels.

    Touching cells (a fifth of cells in the dense recordings) merge under plain
    connected components; seeding the watershed with the known centroids splits
    them back into one basin per annotated soma.
    """
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    fg = prob >= threshold
    markers = np.zeros(prob.shape, np.int32)
    for i, (cy, cx) in enumerate(centroids, 1):
        yy, xx = int(round(cy)), int(round(cx))
        if 0 <= yy < prob.shape[0] and 0 <= xx < prob.shape[1]:
            markers[yy, xx] = i
    if markers.max() == 0:
        return ndi.label(fg)[0]
    return watershed(-prob, markers, mask=fg)


# =============================================================================
# SYNTHETIC SELF-TEST  (non-round cells, so segmentation has shape to learn)
# =============================================================================

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
