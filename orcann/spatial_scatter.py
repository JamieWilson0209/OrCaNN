"""
Stage 1 (scattering formulation) — spatial detection as a first-order
scattering coefficient of the movie.
======================================================================

A projection-based detector would smooth each frame, reduce over time
(max / std / correlation), then run a LoG bank — two spatial scales (a
smoothing σ and a detection σ) straddling a nonlinear projection.

Applying the LoG *per frame*, before the temporal nonlinearity, collapses
that. The LoG's own Gaussian factor is then the per-frame smoothing, so the
two scales become one, and the operator is a textbook first-order scattering
coefficient:

        S(x, σ) = var_t | ∇²G_σ *_x Y(·, t) |

  • band-pass each frame at a cell scale σ  (the learnable ∇²G bank, reused
    verbatim from the projection detector — same generating function);
  • the nonlinearity is the temporal variance, placed exactly where
    scattering theory puts the modulus;
  • variance, not raw energy: ∇²G is zero-mean, so removing the temporal
    mean discards static structure and keeps only *active* blobs — which is
    precisely "active neuron", the detection target.

One scale, no separate smoothing hyper-parameter, no projection step, no max.
This is the same wavelet→nonlinearity→head shape as the temporal stage, so
both halves of the pipeline are now the same operator, one over space-then-
time-energy, one over time.

Trade-off carried forward deliberately: max projection was the most sensitive
channel for rarely-firing cells and is the one statistic that does NOT fold in
(an order statistic, outside the moment algebra). We try the elegant,
integrable form first; if recall on sparse cells suffers, max returns as an
auxiliary channel where the sequential order is simply accepted.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from orcann.spatial_log import ParametricLoG2d


class SpatialScatterDetector(nn.Module):
    """Per-frame learnable LoG energy front-end (no detection head).

    Produces a stack of temporal moments of the same ∇²G-filtered movie
    (order [structural, max, variance, coherence]); see :meth:`energy`. The
    structural (mean) channel responds to all visible somata; the max, variance
    and coherence channels add activity information. The output is the feature
    stack itself: :class:`~orcann.spatial_seg.SpatialSegmenter` consumes
    ``energy()`` directly and supplies its own U-Net.

    Parameters
    ----------
    radii_px : sequence of float
        Initial neuron-radius bank (pixels); the σ-seeds, learnable.
    n_energy_frames : int, optional
        If set and the movie has more frames, a random subset of this many
        frames estimates the moments — bounds compute on long recordings.
    use_structural, use_variance, use_correlation : bool
        Which temporal-moment channels to include (>=1).
    """

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
        """(B, T, H, W) -> (B, K·n_groups, H, W); a hierarchy of temporal moments.

        All channels are the same ∇²G_σ band-pass pooled at a different
        temporal statistic, stacked in the fixed order [structural, max,
        variance, coherence]:
          • structural (0th moment): ∇²G·ȳ, LoG of the mean image — all somata;
          • max (L∞): ∇²G on the (robust) max projection — the substrate the
            ROIs were drawn on; surfaces sparse, low-baseline cells that the
            mean/variance channels cannot see. The order statistic is exactly
            the channel that does not fold into the convolution, so it is
            computed explicitly here;
          • variance (2nd, diagonal): active-blob energy;
          • coherence (2nd, off-diagonal): noise-robust coherent activity.
        Two passes over frame chunks; the full per-frame response is never
        materialised.
        """
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
            # Robust max projection then the SAME ∇²G bank on it. Top-k (k-th
            # largest ≈ the (1-q) percentile) is a partial sort — ~20× faster
            # than torch.quantile and rejects single-frame hotspots just as well.
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
