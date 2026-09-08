"""Spatial segmenter: ∇²G moment front-end + small U-Net -> per-pixel soma
probability. The model, its loss, and inference helpers.

Training and data loading live in ``orcann.spatial.training``. See
docs/spatial/detector.md.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from orcann.spatial.detection.scattering import SpatialScatterDetector


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
        train_hw: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.config = {"radii_px": list(radii_px), "hidden": hidden,
                       "n_energy_frames": n_energy_frames,
                       "use_structural": use_structural, "use_max": use_max,
                       "use_variance": use_variance, "use_correlation": use_correlation,
                       "learnable_scales": learnable_scales,
                       "corr_radius": corr_radius, "corr_dirs": corr_dirs,
                       "train_hw": list(train_hw) if train_hw else None}
        # The frame size the LoG bank was fitted at. A movie of a different size
        # presents cells at a different number of pixels across, so infer brings
        # it here before the forward pass.
        self.train_hw = tuple(train_hw) if train_hw else None
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

    def _feature_rms(self, feats: torch.Tensor) -> torch.Tensor:
        """Per-channel RMS over the pixels the ∇²G bank could read whole.

        Within ``_half`` px of the frame edge the convolution reads zero padding
        instead of tissue, and a demeaned kernel answers that step about as
        strongly as it answers a soma. Drawing the divisor from the whole frame
        therefore scales it with the recording's baseline brightness and with
        its frame size, neither of which is a property of the signal: the same
        cell would reach the encoder weaker in a brighter recording. The border
        is excluded rather than extrapolated, as ``motion_correction`` crops the
        band its shifts moved on and off the frame.
        """
        m = self.front.log._half
        H, W = feats.shape[-2:]
        if H <= 2 * m or W <= 2 * m:
            raise ValueError(
                f"frame is {H}x{W}, which leaves no interior once the {m}px "
                f"border the ∇²G bank cannot read is excluded; train_spatial.patch "
                f"and every recording must exceed {2 * m}px")
        interior = feats[..., m:H - m, m:W - m]
        return interior.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()

    def forward(self, movie: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W) -> (B, 1, H, W) soma logits."""
        feats = self.front.energy(movie)
        rms = self._feature_rms(feats)
        feats = feats / (rms + 1e-6)
        e1 = self.enc1(feats)
        e2 = self.enc2(self.pool(e1))
        e2u = self.reduce(F.interpolate(e2, size=e1.shape[-2:],
                                        mode="bilinear", align_corners=False))
        d = self.dec(torch.cat([e1, e2u], dim=1))
        return self.out(d)


def focal_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                    gamma: float = 2.0, dice_w: float = 1.0) -> torch.Tensor:
    """Focal BCE + soft Dice (imbalance-robust). See docs/spatial/detector.md."""
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    focal = ((1 - pt).pow(gamma) * bce).mean()
    inter = (p * target).sum(dim=(-2, -1))
    denom = p.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + 1e-6
    dice = 1.0 - (2.0 * inter / denom).mean()
    return focal + dice_w * dice


@torch.no_grad()
def predict_prob(model: SpatialSegmenter, movie: np.ndarray,
                 device: Optional[torch.device] = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.from_numpy(movie.astype(np.float32)).to(device)[None]
    return torch.sigmoid(model(x))[0, 0].cpu().numpy()
