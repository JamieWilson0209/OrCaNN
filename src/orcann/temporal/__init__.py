"""Temporal stage: detect *when* the cells fire.

A shape-parameterised derivative-of-Gaussian wavelet (`detector`) feeds a learned
per-bin rate head and a geometric duration read-out; `training` fits the rate
head leave-one-indicator-out on public ground truth. See `docs/temporal/detector.md` and `docs/temporal/training.md`.
"""
from orcann.temporal.detector import (
    ParametricDoGWavelet1d, TemporalRateModel,
    standardize_trace, match_noise, read_durations, detect_transients, rate_loss,
)
from orcann.temporal.training import (
    load_cascade_mat, synthetic_indicator_bank, run_loio, train_final,
)

__all__ = [
    "ParametricDoGWavelet1d", "TemporalRateModel",
    "standardize_trace", "match_noise", "read_durations", "detect_transients",
    "rate_loss",
    "load_cascade_mat", "synthetic_indicator_bank", "run_loio", "train_final",
]
