"""
Preprocessing — Trace-Level Baseline Correction
================================================

Converts raw fluorescence traces to ΔF/F₀ using a rolling low-percentile
baseline: a per-trace rolling 8th-percentile F₀, the same family of approach as
Suite2p and CaImAn's ``detrend_df_f``.

A per-ROI local F₀, taken from a tissue-masked annulus, would suit this data
better — bright organoid tissue surrounded by dark medium is exactly the case a
whole-trace percentile handles poorly. It is not implemented here. Building it
against the segmenter's own cellness map, rather than an intensity threshold,
is the sensible route.

These operate on extracted traces (N × T), not on the movie itself.
Movie-level preprocessing (motion correction) is handled separately.

Known limitation — F₀ near zero (UNSOLVED)
------------------------------------------
ΔF/F₀ = (F − F₀) / F₀ is undefined as F₀ → 0, and there is no settled answer
in the field for what to do about it. What the rolling baseline here does is
floor F₀ at 1% of the trace's own median, which keeps the division
well-conditioned for dim ROIs and is scale-relative — but it is a guard, not a
solution, and it silently changes the meaning of the output for any trace whose
true resting fluorescence is genuinely near zero.

The three published positions disagree:

- **CaImAn** (``detrend_df_f``) divides by ``Df + Fd`` — the background
  baseline *added back* to the signal baseline — so its denominator is
  structurally non-small and it needs no floor at all. It also offers
  ``detrend_only``, documented for 1p data "where baseline fluorescence cannot
  be determined": subtract, do not divide.
- **Practical guides** subtract background *before* ΔF/F and then add a
  "safety margin chosen by experience… or whatever makes sense for your data" —
  i.e. an offset in the data's own units, not a universal constant.
- **Suite2p** concedes F₀ "may sometimes be estimated to be negative or
  near-zero for high SNR sensors", and its suggested workaround comes with the
  caveat that the choice affects interpretation.
- **Chen et al., "Calcium Imaging and the Curse of Negativity"** recommends no
  floor at all, and z-scoring instead, because a moving-baseline ΔF/F₀ invents
  positive signal when an inhibited neuron returns to rest.

This matters little for OrCaNN today: traces are weighted means of raw
microscopy counts with medians in the hundreds, so the floor is never reached.
It becomes live for normalised, background-subtracted or externally-sourced
traces. Deciding between a detrend-only fallback, a background-inclusive
denominator, and z-scoring is a scientific choice about what the pipeline
reports, and has not been made.
"""

import numpy as np
import logging
from typing import Tuple
from scipy import ndimage

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_window(T: int, fraction: float = 0.25,
                     min_window: int = 50, max_window: int = 500) -> int:
    """Compute an odd rolling-window size from recording length."""
    window = int(T * fraction)
    window = max(min_window, min(max_window, window))
    return window if window % 2 == 1 else window + 1


def _detect_edge_artefacts(
    traces: np.ndarray, T: int,
) -> Tuple[int, int]:
    """
    Detect boundary artefacts at the start/end of traces.

    Returns (trim_start, trim_end) — the number of frames to exclude
    from baseline estimation at each end.  Zero if no artefact detected.
    """
    edge_check = max(5, T // 10)
    trace_medians = np.median(traces, axis=1)
    safe_medians = np.maximum(trace_medians, 1.0)

    start_ratios = np.median(traces[:, :edge_check], axis=1) / safe_medians
    end_ratios = np.median(traces[:, -edge_check:], axis=1) / safe_medians

    trim_start = edge_check if np.mean(start_ratios < 0.90) > 0.15 else 0
    trim_end = edge_check if np.mean(end_ratios < 0.90) > 0.15 else 0

    return trim_start, trim_end


def _rolling_baseline(
    trace: np.ndarray,
    percentile: float,
    window: int,
    trim_start: int = 0,
    trim_end: int = 0,
) -> np.ndarray:
    """
    Compute a rolling-percentile baseline for a single 1-D trace.

    If trim_start/trim_end are nonzero, the baseline is estimated on the
    interior and extrapolated to the edges by repeating the boundary value.
    The baseline is floored at 1% of the trace's own median to prevent
    division-by-near-zero for dim ROIs. The floor is scale-relative: a trace
    measured in raw counts and the same trace scaled to [0, 1] get proportional
    floors. Traces with no positive samples are left unfloored and warned about
    — see the known limitation in the module docstring.

    Parameters
    ----------
    trace : 1-D array (float64)
    percentile : float
    window : int (odd)
    trim_start, trim_end : int
        Frames to exclude from each end.

    Returns
    -------
    baseline : 1-D array (float64), same length as trace
    """
    T = len(trace)

    if trim_start > 0 or trim_end > 0:
        t_s = trim_start
        t_e = T - trim_end if trim_end > 0 else T
        interior = trace[t_s:t_e]
        bl_interior = ndimage.percentile_filter(
            interior, percentile,
            size=min(window, len(interior)), mode='reflect',
        )
        baseline = np.empty(T, dtype=np.float64)
        baseline[t_s:t_e] = bl_interior
        if trim_start > 0:
            baseline[:t_s] = bl_interior[0]
        if trim_end > 0:
            baseline[t_e:] = bl_interior[-1]
    else:
        baseline = ndimage.percentile_filter(
            trace, percentile, size=window, mode='reflect',
        )

    # Floor at 1% of the trace's own median, keeping the ΔF/F₀ division
    # well-conditioned for dim ROIs. Relative rather than absolute, so a trace in
    # raw counts and the same trace scaled to [0, 1] get proportional floors —
    # ΔF/F₀ is a ratio and must not depend on the units of its input.
    # This is a guard, not an answer; see "F₀ near zero" in the module docstring.
    positive = trace[trace > 0]
    if positive.size:
        np.maximum(baseline, float(np.median(positive)) * 0.01, out=baseline)
    else:
        # No positive sample anywhere: there is no resting fluorescence to
        # normalise against, so there is nothing meaningful to floor toward.
        # Left as measured rather than pinned to an invented constant, and
        # reported so the caller is not handed a silent -1.0 trace.
        logger.warning(
            "trace has no positive samples — F0 is undefined and the ΔF/F₀ "
            "returned for it is not a ratio (see the known limitation in "
            "orcann.activity.baseline)")

    return baseline


def _compute_dff_stats(C_dff, baseline_drift_pcts, n_clipped, T):
    """Compute summary statistics shared by both correction methods."""
    if C_dff.size == 0:
        # Every statistic below reduces over C_dff, including one inside a log
        # f-string; on an empty ROI set that raised from the logging line.
        return {'median_dff': 0.0, 'median_baseline_drift_pct': 0.0,
                'n_clipped': int(n_clipped), 'n_traces': 0}
    median_drift = float(np.median(baseline_drift_pcts))
    median_dff = float(np.median(C_dff))

    chunk = max(1, T // 10)
    residual_drifts = (np.mean(C_dff[:, -chunk:], axis=1)
                       - np.mean(C_dff[:, :chunk], axis=1))

    logger.info(f"  Median baseline drift: {median_drift:.1f}%")
    logger.info(f"  ΔF/F₀ range: [{C_dff.min():.4f}, {C_dff.max():.4f}], "
                f"median={median_dff:.4f}")
    if n_clipped > 0:
        logger.info(f"  Clamped {n_clipped} extreme values (|ΔF/F₀| > 50)")
    logger.info(f"  Residual drift: median={np.median(residual_drifts):.4f}, "
                f"p10={np.percentile(residual_drifts, 10):.4f}, "
                f"p90={np.percentile(residual_drifts, 90):.4f}")

    return {
        'median_baseline_drift_pct': median_drift,
        'median_dff': median_dff,
        'n_clipped': n_clipped,
        'residual_drift_median': float(np.median(residual_drifts)),
        'residual_drift_p10': float(np.percentile(residual_drifts, 10)),
        'residual_drift_p90': float(np.percentile(residual_drifts, 90)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API: PER-TRACE ΔF/F₀
# ─────────────────────────────────────────────────────────────────────────────

def compute_dff_traces(
    C_raw: np.ndarray,
    frame_rate: float,
    percentile: float = 8.0,
    window_fraction: float = 0.25,
    min_window: int = 50,
    max_window: int = 500,
    edge_trim: bool = False,
    presmooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert raw fluorescence traces to ΔF/F₀ using per-trace rolling baseline.

    A rolling low-percentile filter estimates F₀(t) for each trace, then
    ΔF/F₀ = (F − F₀) / F₀.  Same family as Suite2p and CaImAn's
    ``detrend_df_f``.

    Parameters
    ----------
    C_raw : array (N, T)
        Raw fluorescence traces.
    frame_rate : float
        Acquisition rate in Hz (logging only).
    percentile : float
        Baseline percentile (default 8.0).
    window_fraction : float
        Rolling window as fraction of T.
    min_window, max_window : int
        Window size bounds in frames.
    edge_trim : bool
        Detect and exclude boundary artefacts from baseline estimation.
    presmooth_sigma : float
        If > 0, Gaussian-smooth each trace (this sigma, in frames) *before*
        estimating F₀, so the low percentile is not dragged below the true
        resting level by per-frame noise.  ΔF/F₀ is still computed from the
        original unsmoothed trace, so transient sharpness is preserved; only the
        baseline is smoothed.  0 disables it (default, unchanged behaviour).

    Returns
    -------
    C_dff : array (N, T)
        ΔF/F₀ traces.
    C_raw : array (N, T)
        Pass-through of input traces.
    info : dict
        Diagnostics.
    """
    N, T = C_raw.shape
    window = _adaptive_window(T, window_fraction, min_window, max_window)

    logger.info(f"Computing per-trace ΔF/F₀: {N} traces, {T} frames, "
                f"percentile={percentile}, window={window}, "
                f"presmooth_sigma={presmooth_sigma}")

    # Edge trim
    trim_start, trim_end = (0, 0)
    if edge_trim:
        trim_start, trim_end = _detect_edge_artefacts(C_raw, T)
        if trim_start or trim_end:
            logger.info(f"  Edge trim: {trim_start} start, {trim_end} end frames")
        else:
            logger.info("  Edge trim enabled but no artefacts detected")

    C_dff = np.zeros_like(C_raw, dtype=np.float32)
    drift_pcts = np.zeros(N)
    n_clipped = 0

    for i in range(N):
        trace = C_raw[i].astype(np.float64)
        # Baseline is estimated on a smoothed copy (if requested) so per-frame
        # noise cannot drag the low percentile below the resting level; ΔF/F₀ is
        # still computed from the original trace.
        est = (ndimage.gaussian_filter1d(trace, presmooth_sigma)
               if presmooth_sigma > 0 else trace)
        baseline = _rolling_baseline(est, percentile, window,
                                     trim_start, trim_end)

        dff = (trace - baseline) / baseline

        if baseline[0] > 0:
            drift_pcts[i] = 100.0 * (baseline[0] - baseline[-1]) / baseline[0]

        # n_clipped counts samples flagged as extreme (|dff|>50);
        # the clip below is more aggressive (cap at 100, floor -1) so
        # the output stays well-conditioned for downstream stats.
        n_clipped += int(np.sum((dff < -1.0) | (dff > 100.0)))
        C_dff[i] = np.clip(dff, -1.0, 100.0).astype(np.float32)

    stats = _compute_dff_stats(C_dff, drift_pcts, n_clipped, T)
    info = {
        'method': 'per_trace_rolling_percentile',
        'percentile': percentile,
        'window_frames': window,
        **stats,
    }
    # A copy, not the caller's array: this is saved as temporal_traces_raw.npy,
    # so returning the input object let an upstream in-place mutation silently
    # rewrite the archived raw traces.
    return C_dff, np.array(C_raw, copy=True), info
