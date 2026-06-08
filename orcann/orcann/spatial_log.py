"""
Stage 1 — Spatial neuron detection as a learnable Laplacian-of-Gaussian.
========================================================================

The classical pipeline runs ``skimage.blob_log``: a bank of Laplacian-of-
Gaussian (LoG) filters at hand-picked scales, followed by hand-tuned
intensity / contrast / local-max rejection and an Otsu contour. This module
keeps the *form* of that — a multi-scale ∇²G filterbank — but makes the two
things that were guessed into the two things that are learned:

    layer 1   a bank of scale-normalised ∇²G kernels, the SCALES learnable
              (this is ``blob_log``, except the σ-bank fits the real neuron
              size distribution instead of being read off a config file);

    layer 2   a thin head that learns how to combine the scale responses
              and clean them into a per-pixel cellness map — replacing the
              fixed intensity/contrast/local-max gates.

The kernels stay exactly ∇²G (generated on the fly from the learnable
log-σ), so the layer remains an interpretable scale-space, not an opaque
conv bank. That faithfulness to a single generating function is the whole
point of the architecture and is shared with the temporal (Ricker) stage —
the same operator, one in 2-D over space, one in 1-D over time.

Supervision is label-form-agnostic. Manual annotations as instance *masks*
or as *centroids* are both reduced to one soft target: a Gaussian cellness
heatmap. So we are not blocked on which form the 85 annotated recordings use.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# LAYER 1 — learnable scale-normalised ∇²G filterbank
# =============================================================================

class ParametricLoG2d(nn.Module):
    """A bank of K isotropic ∇²G kernels whose scales are learnable.

    Each output channel k is the Laplacian-of-Gaussian at scale σ_k:

        LoG(x, y; σ) = σ² · ((x² + y² − 2σ²) / σ⁴) · G_σ(x, y)

    The leading σ² is the conventional scale-normalisation factor (Lindeberg,
    1998). Note: in this implementation the per-scale peak response is *not*
    equalised across scales — empirically it grows with σ rather than peaking at
    the matching blob size, so a single scale does not act as a clean size
    selector. Scale/size information is instead recovered by the learned head,
    which combines all K channels (and by the spatial extent of each response).
    The kernel is sign-flipped so a *bright* blob gives a *positive* response,
    and demeaned so a flat or slowly-varying background contributes nothing (DC
    rejection, exactly as the Ricker kernel does in the temporal stage).

    The scales are stored as log-σ and exponentiated, so σ > 0 always and the
    bank is differentiable end-to-end with respect to the scales themselves.
    """

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
        # Fixed kernel support, sized for the largest scale this bank can grow
        # to; smaller kernels live inside it, zero-padded. Generous headroom so
        # learned σ can increase without re-allocating.
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


# =============================================================================
# THE STAGE — input fusion -> LoG bank -> combination head -> cellness logits
# =============================================================================

def cellness_target(
    shape: Tuple[int, int],
    centroids: np.ndarray,
    radii: Optional[Sequence[float]] = None,
    default_radius: float = 8.0,
) -> np.ndarray:
    """Soft cellness heatmap in [0, 1] from instance centroids.

    Masks and points reduce to the same target: instance masks contribute
    their centroid (and a radius from √(area/π)); manual points contribute a
    fixed-radius blob. A Gaussian (σ = radius/2) is stamped per instance and
    the map is the per-pixel max, so overlapping cells do not sum past 1.
    """
    H, W = shape
    heat = np.zeros((H, W), dtype=np.float32)
    if len(centroids) == 0:
        return heat
    if radii is None:
        radii = [default_radius] * len(centroids)
    yy, xx = np.mgrid[0:H, 0:W]
    for (cy, cx), r in zip(centroids, radii):
        sig = max(1.0, float(r) / 2.0)
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sig * sig))
        np.maximum(heat, g, out=heat)
    return heat


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


# =============================================================================
# LOSS
# =============================================================================

def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                  dice_w: float = 1.0) -> torch.Tensor:
    """Pixelwise BCE + soft Dice on the cellness heatmap."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(-2, -1))
    denom = p.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + 1e-6
    dice = 1.0 - (2.0 * inter / denom).mean()
    return bce + dice_w * dice


# =============================================================================
# INFERENCE — cellness map -> instances
# =============================================================================

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
