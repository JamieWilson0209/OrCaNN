"""Spatial detector layer 1: a learnable multi-scale ∇²G (Laplacian-of-Gaussian)
filterbank, plus label/centroid helpers. See docs/spatial/detector.md.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# LAYER 1 — learnable scale-normalised ∇²G filterbank

class ParametricLoG2d(nn.Module):
    """Bank of K isotropic ∇²G kernels at learnable scales (stored as log-σ)."""

    def __init__(
        self,
        radii_px: Sequence[float],
        truncate: float = 3.0,
        learnable_scales: bool = True,
    ) -> None:
        super().__init__()
        # A blob of radius r is matched by a LoG of σ = r / √2.
        sigmas = torch.tensor([float(r) / math.sqrt(2.0) for r in radii_px],
                              dtype=torch.float32)
        self.log_sigma = nn.Parameter(torch.log(sigmas),
                                      requires_grad=learnable_scales)
        self.truncate = float(truncate)
        # kernel support sized for the largest scale; smaller kernels zero-pad inside
        max_sigma = float(sigmas.max())
        self._half = int(math.ceil(self.truncate * max_sigma * 1.3))

    @property
    def sigmas(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    @property
    def radii_px(self) -> torch.Tensor:
        return self.sigmas * math.sqrt(2.0)

    def _kernels(self) -> torch.Tensor:
        """Generate the (K, 1, L, L) kernel stack from the current scales."""
        h = self._half
        dev = self.log_sigma.device
        ax = torch.arange(-h, h + 1, device=dev, dtype=torch.float32)
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        r2 = (xx * xx + yy * yy)[None]                 # (1, L, L)
        s = self.sigmas[:, None, None]                 # (K, 1, 1)
        s2 = s * s
        g = torch.exp(-r2 / (2.0 * s2))
        # scale-normalised, sign-flipped LoG: bright blob -> positive centre
        log = s2 * ((2.0 * s2 - r2) / (s2 * s2)) * g    # = -σ²·∇²G
        log = log - log.mean(dim=(-2, -1), keepdim=True)   # DC rejection
        return log[:, None]                            # (K, 1, L, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) detection image -> (B, K, H, W) scale responses."""
        k = self._kernels()
        return F.conv2d(x, k, padding=self._half)


# THE STAGE — input fusion -> LoG bank -> combination head -> cellness logits

def centroids_from_masks(masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(centroids, radii) from a (H, W) integer label image (0 = background)."""
    labels = np.unique(masks)
    labels = labels[labels != 0]
    cents, radii = [], []
    for lab in labels:
        ys, xs = np.where(masks == lab)
        cents.append((ys.mean(), xs.mean()))
        radii.append(math.sqrt(len(ys) / math.pi))
    return np.asarray(cents, dtype=np.float32), np.asarray(radii, dtype=np.float32)


# INFERENCE — cellness map -> instances

def extract_instances(
    cellness: np.ndarray,
    min_distance: int = 6,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Peak-local-max on the cellness map -> (centroids, peak scores)."""
    from scipy.ndimage import maximum_filter

    m = cellness >= threshold
    mx = maximum_filter(cellness, size=2 * min_distance + 1)
    peaks = m & (cellness == mx)
    ys, xs = np.where(peaks)
    cents = np.stack([ys, xs], axis=1).astype(np.float32)
    scores = cellness[ys, xs]
    return cents, scores
