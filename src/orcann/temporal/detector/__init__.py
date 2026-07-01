"""Temporal detector: DoG wavelet front-end + learned per-bin rate head."""
from orcann.temporal.detector.detector import (
    ParametricDoGWavelet1d, TemporalRateModel,
    standardize_trace, match_noise, read_durations, detect_transients, rate_loss)
