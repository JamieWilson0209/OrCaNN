"""Spatial detector energy front-end: per-frame ∇²G then a hierarchy of temporal
moments (structural, max, variance, coherence). See docs/spatial/detector.md.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from orcann.spatial.detection.laplacian import ParametricLoG2d


class SpatialScatterDetector(nn.Module):
    """Per-frame learnable ∇²G energy front-end (no detection head); the segmenter
    consumes :meth:`energy` and adds its own U-Net. ``n_energy_frames`` subsamples
    long movies to bound compute."""

    def __init__(
        self,
        radii_px: Sequence[float] = (4, 6, 9, 13, 18),
        learnable_scales: bool = True,
        n_energy_frames: Optional[int] = 256,
        use_structural: bool = True,
        use_max: bool = True,
        use_variance: bool = True,
        use_correlation: bool = False,
        max_substrate: str = "percentile",
        max_q: float = 0.99,
        corr_radius: int = 2,
        corr_dirs: int = 8,
    ) -> None:
        super().__init__()
        self.config = {"radii_px": list(radii_px),
                       "learnable_scales": learnable_scales, "n_energy_frames": n_energy_frames,
                       "use_structural": use_structural, "use_max": use_max,
                       "use_variance": use_variance, "use_correlation": use_correlation,
                       "max_substrate": max_substrate, "max_q": max_q,
                       "corr_radius": corr_radius, "corr_dirs": corr_dirs}
        self.log = ParametricLoG2d(radii_px, learnable_scales=learnable_scales)
        self.use_structural = use_structural
        self.use_max = use_max
        self.use_variance = use_variance
        self.use_correlation = use_correlation
        self.max_substrate = max_substrate
        self.max_q = max_q
        self._offsets = self._make_offsets(corr_radius, corr_dirs) if use_correlation else []
        n_groups = use_structural + use_max + use_variance + use_correlation
        assert n_groups >= 1, "enable at least one channel"
        self.n_energy_frames = n_energy_frames

    @staticmethod
    def _make_offsets(r: int, dirs: int):
        base = [(r, 0), (0, r), (-r, 0), (0, -r)]
        if dirs >= 8:
            base += [(r, r), (r, -r), (-r, r), (-r, -r)]
        return base

    def energy(self, movie: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W) -> (B, K·n_groups, H, W): the same ∇²G band-pass pooled at
        several temporal statistics, in fixed order [structural, max, variance,
        coherence]. Two chunked passes; the per-frame response is never
        materialised. See docs/spatial/detector.md."""
        B, T, H, W = movie.shape
        if self.n_energy_frames and T > self.n_energy_frames:
            idx = torch.randperm(T, device=movie.device)[:self.n_energy_frames]
            movie = movie[:, idx]
            T = movie.shape[1]

        K = self.log.log_sigma.shape[0]
        s1 = movie.new_zeros((B, K, H, W))                     # Σ_t L
        s2 = movie.new_zeros((B, K, H, W))                     # Σ_t L²
        sx = [movie.new_zeros((B, K, H, W)) for _ in self._offsets]  # Σ_t L·L_δ
        chunk = 32
        for t0 in range(0, T, chunk):
            fr = movie[:, t0:t0 + chunk]
            c = fr.shape[1]
            L = self.log(fr.reshape(B * c, 1, H, W)).reshape(B, c, K, H, W)
            s1 = s1 + L.sum(dim=1)
            s2 = s2 + (L * L).sum(dim=1)
            for i, (dy, dx) in enumerate(self._offsets):
                Lr = torch.roll(L, shifts=(dy, dx), dims=(-2, -1))
                sx[i] = sx[i] + (L * Lr).sum(dim=1)

        mean = s1 / T                                          # = ∇²G · ȳ
        chans = []
        if self.use_structural:
            chans.append(mean)
        if self.use_max:
            # robust max projection (top-k ≈ (1-q) percentile) then the same ∇²G bank
            if self.max_substrate == "percentile":
                k = max(1, int(round((1.0 - self.max_q) * T)))
                proj = movie.topk(k, dim=1).values[:, -1]               # (B, H, W)
            else:
                proj = movie.max(dim=1).values
            chans.append(self.log(proj[:, None]))                       # (B, K, H, W)
        if self.use_variance:
            chans.append(s2 / T - mean * mean)
        if self.use_correlation:
            coh = movie.new_zeros((B, K, H, W))
            for i, (dy, dx) in enumerate(self._offsets):
                mr = torch.roll(mean, shifts=(dy, dx), dims=(-2, -1))
                coh = coh + (sx[i] / T - mean * mr)            # Cov_t(L, L_δ)
            chans.append(coh / len(self._offsets))
        return torch.cat(chans, dim=1)
