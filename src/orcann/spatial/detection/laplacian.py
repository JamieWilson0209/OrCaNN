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

EDGE_CORRECTIONS = ("zeros", "mean_fill", "mean_fill_gain")


class ParametricLoG2d(nn.Module):
    """Bank of K isotropic ∇²G kernels at learnable scales (stored as log-σ)."""

    def __init__(
        self,
        radii_px: Sequence[float],
        truncate: float = 3.0,
        learnable_scales: bool = True,
        edge_correction: str = "zeros",
    ) -> None:
        super().__init__()
        if edge_correction not in EDGE_CORRECTIONS:
            raise ValueError(f"edge_correction {edge_correction!r} is not one of "
                             f"{', '.join(EDGE_CORRECTIONS)}")
        self.edge_correction = edge_correction
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
        """x: (B, 1, H, W) detection image -> (B, K, H, W) scale responses.

        Within ``_half`` px of the edge part of the kernel falls outside the
        recording, and what fills it is the level the tissue is read against:
        the kernel is demeaned, so for a constant fill ``c``,

            Σ_all w·x = Σ_valid w·x + c·Σ_invalid w = Σ_valid w·(x − c)

        since Σ_all w = 0. ``zeros``, torch's default, therefore reads the
        tissue against a cliff the height of the baseline, which a demeaned
        kernel answers about as strongly as it answers a soma -- and answers
        with a sign that flips with depth, so a cell 8px in reads too bright
        and one at 2px too dim. ``mean_fill`` sets ``c`` to the frame's own
        mean, removing the step; a cell near the edge is then read low but
        honestly, part of its kernel having landed on flat fill.
        ``mean_fill_gain`` additionally divides by the fraction of kernel
        energy that reached real pixels, restoring the amplitude.
        """
        k = self._kernels()
        if self.edge_correction == "zeros":
            return F.conv2d(x, k, padding=self._half)
        h = self._half
        # A constant added everywhere is invisible to a demeaned kernel, so
        # subtracting the level and padding with zeros *is* padding with it.
        y = F.conv2d(F.pad(x - x.mean(dim=(-2, -1), keepdim=True), (h,) * 4), k)
        if self.edge_correction == "mean_fill":
            return y
        return y / self._retained_energy(k, x)

    def _retained_energy(self, k: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel share of each kernel's squared weight that landed on real
        pixels. Σw is zero for these kernels, so the weight sum cannot serve as
        the divisor and the energy does. One convolution over a mask against the
        B·T the response itself costs, so it is recomputed, not cached."""
        k2 = k.pow(2)
        inside = F.pad(x.new_ones((1, 1) + tuple(x.shape[-2:])), (self._half,) * 4)
        return F.conv2d(inside, k2) / k2.sum(dim=(-3, -2, -1)).view(1, -1, 1, 1)


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
