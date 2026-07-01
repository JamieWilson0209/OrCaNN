"""Temporal detector: a shape-parameterised derivative-of-Gaussian wavelet whose
multi-scale response feeds a learned per-bin event-rate head; duration is read
geometrically off the scale axis. See docs/temporal/detector.md.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Ricker (DOG m=2) Fourier factor mapping scale a -> timescale in seconds.
_FOURIER_FACTOR = 2.0 * math.pi / math.sqrt(2.5)        # ≈ 3.974


# LAYER 1 — learnable, shape-parameterised derivative-of-Gaussian bank

class ParametricDoGWavelet1d(nn.Module):
    """Bank of K mother wavelets ψ_θ at learnable scales, one shared asymmetry;
    θ=0 is the Ricker CWT. See docs/temporal/detector.md."""

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
        # ψ_θ = cos(θ)·R + sin(θ)·D, both zero-mean (see docs/temporal/detector.md)
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


# THE STAGE — mother bank -> learned per-bin rate; geometric duration read-out

class TemporalRateModel(nn.Module):
    """ψ_θ scale-space + learned rate head. ``forward`` returns the per-bin rate;
    :meth:`response` exposes the layer-1 output for the duration read-out."""

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
        # scale-channel dropout for transfer; identity at eval (docs/temporal/detector.md)
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


# DATA PREP — degrade high-rate ground truth to the target regime

def match_noise(trace: np.ndarray, target_sd: float,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add white noise so the trace reaches a target noise SD. See docs/temporal/training.md."""
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
    """Put a trace in SNR units: detrend by a rolling low-percentile baseline, then
    divide by the robust noise σ. Applied in training and inference. See
    docs/temporal/detector.md."""
    from scipy.ndimage import percentile_filter
    trace = np.asarray(trace, dtype=np.float64)
    w = max(int(baseline_window_s * frame_rate), 5)
    f0 = percentile_filter(trace, int(baseline_percentile), size=w, mode="nearest")
    detrended = trace - f0
    noise = _robust_sd(detrended)
    return (detrended / (noise + 1e-9)).astype(np.float32)


# DURATION — read geometrically from the scale axis (no labels)

def read_durations(model: "TemporalRateModel", x: torch.Tensor,
                   event_bins: Sequence[int]) -> np.ndarray:
    """Characteristic timescale (s) per event via soft-argmax over the scale axis
    (geometry, not learned). See docs/temporal/detector.md."""
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
    """Run the model on one ROI trace -> continuous rate + discrete events (peaks
    above a height floor that clear a prominence requirement). The single
    detection path used by the runner and visualiser. See docs/temporal/detector.md."""
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
    """MSE on the rate (anchors absolute scale) + a (1 − correlation) term per
    trace (rewards timing/shape)."""
    mse = F.mse_loss(pred, target)
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    corr = (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1) + 1e-8)
    return mse + (1.0 - corr).mean()


# SMOKE TEST — generate asymmetric transients high-rate, degrade to 2 Hz, learn

def _calcium_kernel(fs: float, tau_rise: float, tau_decay: float) -> np.ndarray:
    t = np.arange(0, int(8 * tau_decay * fs)) / fs
    k = (1.0 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)
    return (k / k.max()).astype(np.float32)
