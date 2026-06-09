"""
Stage 2 — Temporal transient detection as a learnable, shape-parameterised
derivative-of-Gaussian wavelet.
==========================================================================

The classical pipeline correlates each ΔF/F₀ trace against a bank of *fixed,
symmetric* Ricker (Mexican-hat) wavelets and reads a transient from the
best-NCC scale along each ridge. A symmetric Ricker is biased against
asymmetric calcium events: a fast-rise/slow-decay transient matches a symmetric
hat progressively worse as the wavelet broadens, so slow asymmetric events slip
the gate.

This module keeps the single-generating-function idea that unifies the spatial
and temporal stages (both are the same operator across a scale group, ∇²G in
space and time) but fixes that one bias. The mother is a rotation inside the
derivative-of-Gaussian family:

    ψ_θ(t) = cos(θ) · R(t)  +  sin(θ) · D(t)

      R(t) ∝ (1 − (t/a)²) · exp(−t²/2a²)      symmetric, center-positive  (−∇²G)
      D(t) ∝ (t/a)         · exp(−t²/2a²)      antisymmetric              ( ∇G )

  • θ = 0  →  exactly the Ricker wavelet → exactly the 1-D twin of the
    spatial LoG. The unification is preserved at the symmetric setting.
  • θ ≠ 0  →  unequal lobes / a skewed central response: the matched,
    zero-mean detector for an asymmetric transient.
  • Both R and D are zero-mean (derivatives of a Gaussian integrate to 0),
    so every ψ_θ is zero-mean for free — DC rejection and wavelet
    admissibility hold for any θ, exactly as the spatial kernel is demeaned.

What is learned vs. what is geometry
------------------------------------
  layer 1   the mother bank: per-channel log-SCALES and one global
            asymmetry θ. Differentiable; kernels generated on the fly.
  layer 2   a thin head mapping the multi-scale response to a per-bin
            event RATE — the supervised output, trained on public
            ground-truth spike trains resampled to the target rate and
            noise-matched to the real recordings (the CASCADE recipe).

  DURATION is *not* learned from labels (there are none): it is read
  geometrically from the scale at which the layer-1 response concentrates,
  using the now-asymmetric mother, then optionally mapped to a calibrated
  FWHM band. Keeping rate (supervised) and duration (geometry) on separate
  paths means the rate head cannot distort the duration read-out.

Honesty at 2 Hz: the supervised target is a smoothed per-bin rate, never
sub-frame spike times — the input physically cannot support finer timing.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Characteristic Fourier period of a Ricker (DOG m=2) wavelet at scale a, in
# samples (Torrence & Compo 1998). Used to label scale -> timescale in seconds.
# Held fixed at the m=2 value; θ shifts it only slightly and the timescale axis
# is already a "characteristic width", not a decay constant.
_FOURIER_FACTOR = 2.0 * math.pi / math.sqrt(2.5)        # ≈ 3.974


# =============================================================================
# LAYER 1 — learnable, shape-parameterised derivative-of-Gaussian bank
# =============================================================================

class ParametricDoGWavelet1d(nn.Module):
    """Bank of K mother wavelets ψ_θ at learnable scales, one shared asymmetry.

    A single generating function deployed across a scale group — the temporal
    counterpart of ``ParametricLoG2d``. At ``theta = 0`` the bank is the exact
    Ricker CWT the classical pipeline uses.
    """

    def __init__(
        self,
        timescales_s: Sequence[float],
        frame_rate: float,
        truncate: float = 4.0,
        learnable_scales: bool = True,
        learnable_asymmetry: bool = True,
    ) -> None:
        super().__init__()
        self.frame_rate = float(frame_rate)
        ts = torch.tensor(list(timescales_s), dtype=torch.float32)
        scales = ts * self.frame_rate / _FOURIER_FACTOR          # a, in samples
        self.log_scale = nn.Parameter(torch.log(scales.clamp_min(1e-3)),
                                      requires_grad=learnable_scales)
        # asymmetry stored unconstrained; squashed to (-π/2, π/2) so θ=0 stays
        # the natural init (pure Ricker) and the sign is the rise/decay direction.
        self.asym = nn.Parameter(torch.zeros(()),
                                 requires_grad=learnable_asymmetry)
        self.truncate = float(truncate)
        self._half = int(math.ceil(self.truncate * float(scales.max()) * 1.3))

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    @property
    def theta(self) -> torch.Tensor:
        return (math.pi / 2.0) * torch.tanh(self.asym)

    @property
    def timescales_s(self) -> torch.Tensor:
        return self.scales * _FOURIER_FACTOR / self.frame_rate

    def _kernels(self) -> torch.Tensor:
        """(K, 1, L) zero-mean, unit-L2 ψ_θ stack from current scales and θ."""
        h = self._half
        dev = self.log_scale.device
        t = torch.arange(-h, h + 1, device=dev, dtype=torch.float32)[None]   # (1, L)
        a = self.scales[:, None]                                             # (K, 1)
        a2 = a * a
        g = torch.exp(-(t * t) / (2.0 * a2))
        R = (1.0 - (t * t) / a2) * g            # symmetric, center-positive (−∇²G)
        D = (t / a) * g                         # antisymmetric              ( ∇G )
        th = self.theta
        psi = torch.cos(th) * R + torch.sin(th) * D
        psi = psi - psi.mean(dim=-1, keepdim=True)                  # DC rejection
        psi = psi / (psi.norm(dim=-1, keepdim=True) + 1e-12)        # unit L2
        return psi[:, None]                                        # (K, 1, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) mean-subtracted trace -> (B, K, T) wavelet response."""
        return F.conv1d(x, self._kernels(), padding=self._half)


# =============================================================================
# THE STAGE — mother bank -> learned per-bin rate; geometric duration read-out
# =============================================================================

class TemporalRateModel(nn.Module):
    """Two-layer temporal model: ψ_θ scale-space + learned rate head.

    ``forward`` returns the supervised quantity (a non-negative per-bin event
    rate). The multi-scale response used for the geometric duration read-out is
    available via :meth:`response`.
    """

    KIND = "temporal_rate"

    def __init__(
        self,
        timescales_s: Sequence[float],
        frame_rate: float,
        hidden: int = 16,
        scale_dropout: float = 0.0,
        **bank_kw,
    ) -> None:
        super().__init__()
        self.config = {"timescales_s": list(timescales_s), "frame_rate": float(frame_rate),
                       "hidden": hidden, "scale_dropout": scale_dropout, **bank_kw}
        self.bank = ParametricDoGWavelet1d(timescales_s, frame_rate, **bank_kw)
        k = len(timescales_s)
        # Scale-channel dropout: during training, randomly zero whole wavelet-scale
        # channels so the head cannot over-rely on the scale(s) that dominate the
        # training indicators — forcing a scale-robust read-out for transfer.
        # nn.Dropout1d drops entire (B, K, T) channels and is identity at eval, so
        # scale_dropout=0 (default) leaves behaviour unchanged.
        self.scale_drop = nn.Dropout1d(scale_dropout)
        # layer 2: weight across scales (1×1), local temporal refine, softplus.
        self.head = nn.Sequential(
            nn.Conv1d(k, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x[:, None]
        return x - x.mean(dim=-1, keepdim=True)

    def response(self, x: torch.Tensor) -> torch.Tensor:
        """(B, K, T) layer-1 wavelet response (geometry; duration read-out)."""
        return self.bank(self._prep(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) or (B, 1, T) -> (B, T) non-negative per-bin rate."""
        s = self.bank(self._prep(x))
        drop = getattr(self, "scale_drop", None)   # guard models pickled pre-dropout
        if drop is not None:
            s = drop(s)
        return F.softplus(self.head(s)).squeeze(1)


# =============================================================================
# DATA PREP — degrade high-rate ground truth to the target regime
# =============================================================================

def downsample_trace(trace: np.ndarray, factor: int) -> np.ndarray:
    """Anti-aliased decimation: AVERAGE over non-overlapping bins, not pick.

    Averaging approximates the temporal integration of a true low-frame-rate
    acquisition; plain subsampling would alias fast structure into the signal.
    """
    n = (len(trace) // factor) * factor
    return trace[:n].reshape(-1, factor).mean(axis=1)


def match_noise(trace: np.ndarray, target_sd: float,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add white Gaussian noise so the trace reaches a target noise SD.

    A binned-down high-rate trace is cleaner than a native low-rate one
    (averaging suppressed shot noise); CASCADE matches noise the same way.
    ``target_sd`` should be measured from the real organoid recordings.
    """
    rng = rng or np.random.default_rng()
    cur = _robust_sd(trace)
    if target_sd > cur:
        extra = math.sqrt(max(target_sd ** 2 - cur ** 2, 0.0))
        trace = trace + rng.normal(0.0, extra, size=trace.shape)
    return trace


def _robust_sd(x: np.ndarray) -> float:
    """MAD-based noise estimate (median of |Δx|), robust to transients."""
    d = np.diff(np.asarray(x, dtype=np.float64))
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / math.sqrt(2.0))


def standardize_trace(trace: np.ndarray, frame_rate: float,
                      baseline_percentile: float = 10.0,
                      baseline_window_s: float = 20.0) -> np.ndarray:
    """The single trace convention the temporal model sees — in BOTH training
    and inference, so it can never face a shifted input distribution.

    Two steps, well-defined for any input (raw F, ΔF/F, public or organoid):
      1. detrend — subtract a rolling low-percentile baseline, removing slow
         drift / bleaching and flattening the baseline to ~0;
      2. normalise — divide by the robust noise σ (MAD of differences), so the
         trace is in SNR units and the model is invariant to absolute noise
         level (achieving by normalisation what CASCADE achieves by matching
         training noise to each test condition, without needing to measure the
         organoid noise level).

    Training augments by adding noise BEFORE this call, so the model still sees
    transients across a range of SNRs.
    """
    from scipy.ndimage import percentile_filter
    trace = np.asarray(trace, dtype=np.float64)
    w = max(int(baseline_window_s * frame_rate), 5)
    f0 = percentile_filter(trace, int(baseline_percentile), size=w, mode="nearest")
    detrended = trace - f0
    noise = _robust_sd(detrended)
    return (detrended / (noise + 1e-9)).astype(np.float32)


def rate_target_from_spikes(spike_times_s: np.ndarray, n_bins: int,
                            frame_rate: float, smooth_s: float = 1.0
                            ) -> np.ndarray:
    """High-rate spike times -> smoothed per-bin event rate at the target rate.

    Bin spikes to the target frame rate, then Gaussian-smooth (σ = smooth_s).
    This is the supervised target: a rate, never discrete sub-bin spikes.
    """
    from scipy.ndimage import gaussian_filter1d

    counts = np.zeros(n_bins, dtype=np.float32)
    idx = np.floor(np.asarray(spike_times_s) * frame_rate).astype(int)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    np.add.at(counts, idx, 1.0)
    sigma_bins = max(smooth_s * frame_rate, 0.5)
    return gaussian_filter1d(counts, sigma_bins).astype(np.float32)


# =============================================================================
# DURATION — read geometrically from the scale axis (no labels)
# =============================================================================

def read_durations(model: "TemporalRateModel", x: torch.Tensor,
                   event_bins: Sequence[int]) -> np.ndarray:
    """Characteristic timescale (s) at each event, via soft-argmax over scale.

    At each event time the wavelet response across scales is softmax-weighted
    (in |response|) and the timescales are combined in log-space — the smooth
    analogue of the classical "scale = best shape match". Uses the learned,
    now-asymmetric mother, so asymmetric transients are read at their true
    width rather than being pushed to fine scales by a symmetric-hat mismatch.
    """
    with torch.no_grad():
        W = model.response(x)[0].abs()                 # (K, T)
        ts = model.bank.timescales_s                   # (K,)
        out = []
        for b in event_bins:
            col = W[:, int(b)]
            w = torch.softmax(col / (col.max() + 1e-9) * 4.0, dim=0)
            log_ts = (w * torch.log(ts)).sum()
            out.append(float(torch.exp(log_ts)))
    return np.asarray(out)


def detect_transients(model: "TemporalRateModel", trace: np.ndarray,
                      frame_rate: float, min_prominence: float = 0.5,
                      floor_pct: float = 25.0, min_isi_s: float = 1.0) -> dict:
    """Run the temporal model on one ROI trace and extract discrete transients.

    The single source of truth for detection — used by both the batch runner and
    the visualizer so they never drift. Returns the continuous rate plus a
    discrete event list. Detection = a height floor (``floor_pct`` percentile of
    the rate, gating quiet/noise baseline) + a prominence requirement
    (``min_prominence`` in rate units, the s_min analogue: a transient must stand
    out from its local surroundings). Input may be raw F or ΔF/F —
    ``standardize_trace`` detrends and unit-noise-normalises either way.
    """
    from scipy.signal import find_peaks
    x = standardize_trace(np.asarray(trace), frame_rate)
    device = next(model.parameters()).device
    xb = torch.from_numpy(x)[None].to(device)
    with torch.no_grad():
        rate = model(xb)[0].cpu().numpy()
    floor = float(np.percentile(rate, floor_pct))
    peaks, _ = find_peaks(rate, height=floor, prominence=min_prominence,
                          distance=max(int(min_isi_s * frame_rate), 1))
    durs = read_durations(model, xb, peaks) if len(peaks) else np.array([])
    return {
        "standardized": x.astype(np.float32),
        "rate": rate.astype(np.float32),
        "peaks": np.asarray(peaks, dtype=np.int64),
        "times_s": (peaks / frame_rate).astype(np.float32),
        "durations_s": np.asarray(durs, dtype=np.float32),
        "amplitudes": (rate[peaks] if len(peaks) else np.array([])).astype(np.float32),
        "floor": floor,
    }

def rate_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on the smoothed rate + a (1 − correlation) term per trace.

    The correlation term rewards getting the *timing/shape* of activity right
    even where absolute scale is uncertain; MSE anchors the absolute rate.
    """
    mse = F.mse_loss(pred, target)
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    corr = (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1) + 1e-8)
    return mse + (1.0 - corr).mean()


# =============================================================================
# SMOKE TEST — generate asymmetric transients high-rate, degrade to 2 Hz, learn
# =============================================================================

def _calcium_kernel(fs: float, tau_rise: float, tau_decay: float) -> np.ndarray:
    t = np.arange(0, int(8 * tau_decay * fs)) / fs
    k = (1.0 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)
    return (k / k.max()).astype(np.float32)
