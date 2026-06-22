"""
Cross-indicator training harness: CASCADE-style ground truth -> the temporal
stage, evaluated leave-one-indicator-out (LOIO).
====================================================================

This is the early go/no-go on the cross-indicator bet. The plan trains the
temporal rate head on public ground-truth spike trains (different indicators)
and applies it to organoid Fluo-4. Before any organoid sees the model, LOIO
inside the public data answers the only question that matters yet: does a rate
head trained on a *set* of indicators generalise to a *held-out* indicator it
never saw? If transfer is already weak here — at the 2 Hz target rate, with
noise matched to the real recordings — the bet fails early and we fall back to
the synthetic forward model, exactly as agreed. If it holds, the public route
is justified.

Everything runs through one interface, ``GroundTruthNeuron``. Two producers
implement it:

  * ``load_cascade_mat`` — adapter for the real CASCADE ground-truth files
    (``CAttached`` cell array of structs with ``fluo_time`` [s], ``fluo_mean``
    [ΔF/F], ``events_AP`` [AP time tags]). The unit of ``events_AP`` is
    auto-detected against ``fluo_time`` and reported, since it is not crisply
    documented and differs between datasets.
  * ``synthetic_indicator_bank`` — several "indicators" with distinct
    rise/decay kinetics, so the LOIO loop is fully testable without the
    multi-GB download. Swap this for ``load_cascade_mat`` with real CASCADE data on the cluster.

The preprocessing is the honest-transfer recipe from the temporal stage:
anti-aliased resample to the target rate, noise-match to a measured SD, and a
Gaussian-smoothed per-bin rate target (never sub-frame spikes).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from orcann.temporal_dog import (
    TemporalRateModel, rate_loss, match_noise, standardize_trace,
)


# =============================================================================
# DATASET INTERFACE
# =============================================================================

@dataclass
class GroundTruthNeuron:
    """One paired recording: a ΔF/F trace and its true spike times."""
    dff: np.ndarray                  # (T,)  ΔF/F at native frame rate
    frame_rate: float                # Hz
    spike_times_s: np.ndarray        # (n_spikes,) in seconds
    indicator: str                   # e.g. 'GCaMP6f', 'OGB-1', 'jRGECO1a'
    dataset_id: str = ""


# =============================================================================
# REAL CASCADE LOADER  (verify against your files; not exercised in smoke test)
# =============================================================================

def load_cascade_mat(path: str, indicator: str, struct_key: str = "CAttached",
                     dataset_id: str = "") -> List[GroundTruthNeuron]:
    """Load one CASCADE ground-truth .mat into GroundTruthNeuron objects.

    Each entry of the ``CAttached`` cell array is one neuron with ``fluo_time``
    (s), ``fluo_mean`` (ΔF/F) and ``events_AP`` (AP time tags). The frame rate
    is read from ``fluo_time``; ``events_AP`` units are auto-detected against
    the ``fluo_time`` span and converted to seconds.

    Tries ``scipy.io.loadmat`` (v7); falls back to ``h5py`` for v7.3 files.
    Robust to the container shape varying across datasets (CAttached is a scalar
    struct in some, an array of structs in others, nested deeper in a few): it
    flattens to the actual neuron structs and skips anything lacking the three
    required fields, so one odd file cannot abort a multi-dataset run.
    """
    required = ("fluo_time", "fluo_mean", "events_AP")
    try:
        from scipy.io import loadmat
        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
        if struct_key not in raw:
            warnings.warn(f"{dataset_id or path}: no '{struct_key}' variable, skipping")
            return []
        structs = list(_struct_leaves(raw[struct_key], required))
        get = lambda c, f: _field(c, f)
    except (NotImplementedError, ValueError):
        import h5py                                   # v7.3 / HDF5 MATLAB files
        f = h5py.File(path, "r")
        refs = f[struct_key]
        structs = list(np.array(refs).ravel())
        get = lambda r, fld: np.asarray(f[r][fld]).ravel().astype(np.float64)

    out: List[GroundTruthNeuron] = []
    for c in structs:
        try:
            t = get(c, "fluo_time")
            dff = get(c, "fluo_mean")
            ap = get(c, "events_AP")
            n = min(len(t), len(dff))
            if n < 2:
                continue
            t, dff = t[:n], dff[:n]
            fr = 1.0 / float(np.median(np.diff(t)))
            if not np.isfinite(fr) or fr <= 0:
                continue
            ap_s = _events_to_seconds(ap, t)
            out.append(GroundTruthNeuron(dff=dff.astype(np.float32), frame_rate=fr,
                                         spike_times_s=ap_s, indicator=indicator,
                                         dataset_id=dataset_id))
        except (KeyError, AttributeError, TypeError):
            continue                                  # malformed neuron entry; skip
    return out


def _struct_leaves(obj, required):
    """Yield every scipy mat_struct carrying all `required` fields, flattening
    through any (possibly nested) object ndarray. Version-independent: a struct
    is anything exposing `_fieldnames`."""
    fns = getattr(obj, "_fieldnames", None)
    if fns is not None:
        low = {f.lower() for f in fns}
        if all(r.lower() in low for r in required):
            yield obj
        return
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for x in obj.ravel():
            yield from _struct_leaves(x, required)


def _field(c, name):
    """Case-insensitive field access on a mat_struct -> 1-D float64 array."""
    for fn in c._fieldnames:
        if fn.lower() == name.lower():
            return np.asarray(getattr(c, fn)).ravel().astype(np.float64)
    raise KeyError(name)


def _events_to_seconds(ap: np.ndarray, fluo_time: np.ndarray) -> np.ndarray:
    """Coerce events_AP to seconds, auto-detecting the storage scale.

    Compares the AP span to the fluorescence-time span and picks the power-of-
    ten divisor that aligns them. Logs the chosen factor so a wrong guess is
    visible rather than silent.
    """
    if len(ap) == 0:
        return ap.astype(np.float64)
    span = float(fluo_time[-1] - fluo_time[0]) or 1.0
    ap_max = float(np.nanmax(ap))
    factor = 1.0
    if ap_max > span * 5.0:                      # stored scaled (e.g. ×1e4)
        factor = 10.0 ** round(math.log10(ap_max / span))
    import logging
    logging.getLogger(__name__).info(
        "events_AP scale factor %.0f (ap_max=%.1f, time_span=%.1fs)",
        factor, ap_max, span)
    return (ap / factor) + float(fluo_time[0])


# =============================================================================
# SYNTHETIC MULTI-INDICATOR STAND-IN  (runs the harness today)
# =============================================================================

# Each "indicator" is a (tau_rise, tau_decay) kinetics signature — deliberately
# spread so LOIO has to generalise across genuinely different shapes.
_SYNTH_INDICATORS = {
    "fast_GECI":  (0.05, 0.45),
    "med_GECI":   (0.10, 1.20),
    "slow_GECI":  (0.20, 2.50),
    "dye_OGB":    (0.08, 0.80),
}


def _calcium_kernel(fs: float, tau_rise: float, tau_decay: float) -> np.ndarray:
    t = np.arange(0, int(8 * tau_decay * fs)) / fs
    k = (1.0 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)
    return (k / k.max()).astype(np.float32)


def synthetic_indicator_bank(
    indicators: Optional[Sequence[str]] = None,
    n_per_indicator: int = 8,
    fs: float = 30.0,
    dur_s: float = 300.0,
    seed: int = 0,
) -> List[GroundTruthNeuron]:
    """Build high-rate paired neurons across several synthetic indicators."""
    rng = np.random.default_rng(seed)
    inds = indicators or list(_SYNTH_INDICATORS)
    n_hi = int(dur_s * fs)
    out: List[GroundTruthNeuron] = []
    for ind in inds:
        tr, td = _SYNTH_INDICATORS[ind]
        kern = _calcium_kernel(fs, tr, td)
        for j in range(n_per_indicator):
            rate = rng.uniform(0.2, 0.9)
            sp = np.sort(rng.uniform(0, dur_s, size=rng.poisson(rate * dur_s)))
            hi = np.zeros(n_hi, dtype=np.float32)
            idx = (sp * fs).astype(int)
            idx = idx[idx < n_hi]
            hi[idx] = rng.uniform(0.5, 1.5, size=len(idx))
            hi = np.convolve(hi, kern)[:n_hi]
            out.append(GroundTruthNeuron(
                dff=hi, frame_rate=fs, spike_times_s=sp,
                indicator=ind, dataset_id=f"{ind}_{j}"))
    return out


# =============================================================================
# PREPROCESS  —  anti-aliased resample, noise-match, smoothed rate target
# =============================================================================

def _resample_antialias(x: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    """Polyphase (anti-aliased) resample of a trace from src_fs to dst_fs."""
    from scipy.signal import resample_poly
    frac = Fraction(dst_fs / src_fs).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator).astype(np.float32)


def _rate_target(spike_times_s: np.ndarray, n_bins: int, dst_fs: float,
                 smooth_s: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    counts = np.zeros(n_bins, dtype=np.float32)
    sp = np.asarray(spike_times_s, dtype=np.float64)
    sp = sp[np.isfinite(sp)]                       # drop NaN/inf padding (e.g. DS15/16)
    if sp.size:                                    # silent neurons -> all-zero target (kept)
        idx = np.floor(sp * dst_fs).astype(int)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        np.add.at(counts, idx, 1.0)
    return gaussian_filter1d(counts, max(smooth_s * dst_fs, 0.5)).astype(np.float32)


def preprocess(neuron: GroundTruthNeuron, dst_fs: float, target_noise_sd: float,
               smooth_s: float, rng: np.random.Generator
               ) -> Tuple[np.ndarray, np.ndarray]:
    """One neuron -> (standardized_trace_at_target_rate, smoothed_rate_target).

    Resample → add noise (SNR augmentation) → ``standardize_trace`` — the same
    standardization applied at inference, so train/test inputs match.
    """
    lo = _resample_antialias(neuron.dff, neuron.frame_rate, dst_fs)
    lo = match_noise(lo, target_noise_sd, rng)          # augment SNR pre-standardize
    lo = standardize_trace(lo, dst_fs)
    rate = _rate_target(neuron.spike_times_s, len(lo), dst_fs, smooth_s)
    return lo.astype(np.float32), rate


def windowize(pairs: List[Tuple[np.ndarray, np.ndarray]], win: int
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cut (trace, rate) pairs into non-overlapping windows and stack.

    Returns empty (0, win) tensors if nothing yields a full window, so a fold
    with only very short recordings is skippable rather than fatal.
    """
    xs, ys = [], []
    for x, y in pairs:
        n = (len(x) // win) * win
        if n == 0:
            continue
        xs.append(x[:n].reshape(-1, win))
        ys.append(y[:n].reshape(-1, win))
    if not xs:
        return torch.zeros((0, win), dtype=torch.float32), torch.zeros((0, win), dtype=torch.float32)
    X = torch.tensor(np.concatenate(xs), dtype=torch.float32)
    Y = torch.tensor(np.concatenate(ys), dtype=torch.float32)
    return X, Y


# =============================================================================
# LEAVE-ONE-INDICATOR-OUT
# =============================================================================

def leave_one_indicator_out(
    neurons: List[GroundTruthNeuron],
) -> Iterator[Tuple[str, List[GroundTruthNeuron], List[GroundTruthNeuron]]]:
    inds = sorted({n.indicator for n in neurons})
    for held in inds:
        train = [n for n in neurons if n.indicator != held]
        test = [n for n in neurons if n.indicator == held]
        yield held, train, test


def _median_corr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Median Pearson correlation over windows that actually contain firing.

    Correlation is undefined for a constant (all-zero, silent-neuron) target, so
    such windows are excluded from the metric — otherwise silent cells would
    drag the median toward zero in the interneuron folds. Silent windows still
    train the model through the MSE term of rate_loss; they just can't enter a
    correlation."""
    p = pred - pred.mean(-1, keepdim=True)
    t = target - target.mean(-1, keepdim=True)
    tn = t.norm(dim=-1)
    valid = tn > 1e-6
    if int(valid.sum()) == 0:
        return float("nan")
    corr = (p * t).sum(-1) / (p.norm(dim=-1) * tn + 1e-8)
    return float(corr[valid].median())


def run_loio(
    neurons: List[GroundTruthNeuron],
    dst_fs: float = 2.0,
    target_noise_sd: float = 0.05,
    smooth_s: float = 1.0,
    win: int = 120,
    timescales_s: Sequence[float] = tuple(np.geomspace(1.0, 20.0, 20)),
    epochs: int = 25,
    lr: float = 5e-3,
    seed: int = 0,
    scale_dropout: float = 0.0,
) -> List[Dict[str, float]]:
    """Train on all-but-one indicator, test on the held-out one, per split."""
    rng = np.random.default_rng(seed)
    report: List[Dict[str, float]] = []

    for held, train_n, test_n in leave_one_indicator_out(neurons):
        Xtr, Ytr = windowize([preprocess(n, dst_fs, target_noise_sd, smooth_s, rng)
                              for n in train_n], win)
        Xte, Yte = windowize([preprocess(n, dst_fs, target_noise_sd, smooth_s, rng)
                              for n in test_n], win)
        if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
            warnings.warn(f"LOIO fold '{held}' skipped: "
                          f"{Xtr.shape[0]} train / {Xte.shape[0]} test windows "
                          f"(recordings shorter than win={win} samples at {dst_fs} Hz).")
            continue

        torch.manual_seed(seed)
        model = TemporalRateModel(list(timescales_s), frame_rate=dst_fs, scale_dropout=scale_dropout)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = rate_loss(model(Xtr), Ytr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            train_corr = _median_corr(model(Xtr), Ytr)
            test_corr = _median_corr(model(Xte), Yte)
        report.append({
            "held_out": held,
            "n_train_windows": int(Xtr.shape[0]),
            "n_test_windows": int(Xte.shape[0]),
            "within_train_corr": round(train_corr, 3),
            "heldout_corr": round(test_corr, 3),
            "transfer_gap": round(train_corr - test_corr, 3),
            "learned_theta": round(float(model.bank.theta.detach()), 3),
        })
    return report


def train_final(
    neurons: List[GroundTruthNeuron],
    dst_fs: float = 2.0,
    target_noise_sd: float = 0.05,
    smooth_s: float = 1.0,
    win: int = 120,
    timescales_s: Sequence[float] = tuple(np.geomspace(1.0, 20.0, 20)),
    epochs: int = 40,
    lr: float = 5e-3,
    seed: int = 0,
    scale_dropout: float = 0.0,
) -> TemporalRateModel:
    """Train ONE deployable rate model on all supplied neurons (no held-out).

    This is the model used for inference on real recordings — LOIO only built
    per-fold models to estimate transfer; this fits the full kept database.
    """
    rng = np.random.default_rng(seed)
    Xtr, Ytr = windowize([preprocess(n, dst_fs, target_noise_sd, smooth_s, rng)
                          for n in neurons], win)
    if Xtr.shape[0] == 0:
        raise ValueError("no training windows — recordings shorter than the window?")
    torch.manual_seed(seed)
    model = TemporalRateModel(list(timescales_s), frame_rate=dst_fs, scale_dropout=scale_dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = rate_loss(model(Xtr), Ytr)
        loss.backward()
        opt.step()
    return model.eval()
