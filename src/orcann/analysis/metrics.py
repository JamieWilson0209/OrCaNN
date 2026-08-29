"""
Metric computation helpers for individual datasets.

All metrics are computed from the SELECTED neurons (filtered by quality at
load time). Data sources:

  C_sel (selected_traces)     - Denoised calcium traces from OASIS
  S_sel (selected_spikes)     - Deconvolved spike trains from OASIS
  R_sel (selected_raw_traces) - Raw fluorescence (for QC only)

Metric source mapping:

  Spike Rate               -> S_sel
  Transient amplitude      -> S_sel
  Pairwise Correlation     -> C_sel
  Synchrony Index          -> C_sel
  IEI (mean / CV)          -> S_sel
  Network bursts           -> S_sel

Rationale for trace-based correlation/synchrony:
At 2 Hz imaging the temporal resolution is too coarse for reliable
spike-train correlations. GCaMP6s decay (~1s) acts as natural temporal
integration, so denoised-trace correlations capture functional coupling
more reliably.
"""

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np

from .loading import DatasetMetrics, FEATURE_NAMES

logger = logging.getLogger(__name__)


# =============================================================================
# PER-NEURON / PER-RECORDING METRIC EXTRACTORS
# =============================================================================

def _get_neuron_rates(ds) -> np.ndarray:
    """Get per-neuron spike rates for ACTIVE neurons only (rate > 0)."""
    if ds.neuron_spike_rates is not None and len(ds.neuron_spike_rates) > 0:
        rates = ds.neuron_spike_rates
        return rates[rates > 0]
    if ds.selected_spikes is None:
        return np.array([])
    dur_s = ds.duration_seconds if ds.duration_seconds > 0 else 1.0
    rates = np.array([np.sum(ds.selected_spikes[j] > 0) / dur_s * 10.0
                     for j in range(ds.selected_spikes.shape[0])])
    return rates[rates > 0]


def _get_neuron_amplitudes(ds) -> np.ndarray:
    """Per-neuron amplitudes for neurons that have a measured transient.

    Drops NaN (never measured) as well as non-positive values, so the result
    is shorter than n_selected and cannot be indexed against the per-neuron
    arrays or zipped with _get_neuron_rates.
    """
    if ds.neuron_spike_amplitudes is not None and len(ds.neuron_spike_amplitudes) > 0:
        amps = ds.neuron_spike_amplitudes
        return amps[np.isfinite(amps) & (amps > 0)]
    if ds.selected_traces is None or ds.selected_spikes is None:
        return np.array([])
    amps = _measure_transient_amplitudes(
        ds.selected_traces, ds.selected_spikes, ds.frame_rate)
    return amps[np.isfinite(amps) & (amps > 0)]


def _recording_metric(ds, attr: str):
    """Get a recording-level metric value, using active-neuron-only means
    for spike rate and amplitude (consistent across all figures).
    
    Returns float or None if no active neurons for rate/amplitude metrics.
    """
    if attr == 'mean_spike_rate':
        rates = _get_neuron_rates(ds)
        return float(np.mean(rates)) if len(rates) > 0 else None
    elif attr == 'mean_spike_amplitude':
        amps = _get_neuron_amplitudes(ds)
        return float(np.mean(amps)) if len(amps) > 0 else None
    elif attr == 'total_events':
        if ds.selected_spikes is not None:
            return float(np.sum(ds.selected_spikes > 0))
        rates = _get_neuron_rates(ds)
        if len(rates) > 0 and ds.duration_seconds > 0:
            return float(np.sum(rates) * ds.duration_seconds / 10.0)
        return 0.0
    else:
        return getattr(ds, attr, None)



# =============================================================================
# POPULATION METRICS
# =============================================================================

def _pairwise_correlations(C: np.ndarray,
                           S: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Compute mean and median pairwise Pearson correlations.

    Uses **denoised calcium traces** (C) rather than spike trains (S).
    
    Rationale: For calcium imaging at low frame rates (2 Hz), the slow 
    GCaMP6s dynamics (~1s decay) provide a natural temporal integration 
    window that makes trace-based correlations a reliable measure of 
    functional co-activation. Spike-train correlations at this frame rate 
    are unreliable because co-firing neurons rarely spike on the exact 
    same frame due to temporal discretization.

    Parameters
    ----------
    C : array (N, T)
        Denoised calcium traces from OASIS deconvolution.
    S : array (N, T), optional
        Deconvolved spike trains (not used — kept for API compatibility).

    Returns
    -------
    mean_r : float
        Mean of all pairwise Pearson r values (upper triangle).
    median_r : float
        Median of all pairwise Pearson r values.
    
    Notes
    -----
    - Input should be the denoised traces (C) from OASIS, NOT raw fluorescence.
    - The denoised trace represents the estimated calcium concentration,
      which directly reflects underlying neural activity.
    """
    N = C.shape[0]
    if N < 2:
        return 0.0, 0.0
    if N < 5:
        logger.debug(f"  _pairwise_correlations: N={N} neurons — correlation "
                     f"from {N*(N-1)//2} pairs, treat with caution")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R = np.corrcoef(C)

    triu_idx = np.triu_indices(N, k=1)
    r_vals = R[triu_idx]
    r_vals = r_vals[np.isfinite(r_vals)]
    if len(r_vals) == 0:
        return 0.0, 0.0
    return float(np.mean(r_vals)), float(np.median(r_vals))


def _synchrony_index(C: np.ndarray, fraction_threshold: float = 0.20,
                     S: Optional[np.ndarray] = None) -> float:
    """Compute population synchrony index from denoised calcium traces.

    Uses two complementary approaches and returns their weighted mean:

    1. **Population coupling** (60% weight) — for each neuron, correlate its 
       denoised trace with the mean of all other traces (population mean-field).
       Average across neurons. High values mean neurons track population activity.

    2. **Co-activation fraction** (40% weight) — fraction of time when a 
       substantial proportion of neurons are simultaneously above baseline.
       Uses denoised traces because the slow calcium dynamics (~1s decay)
       provide an appropriate coincidence window for 2 Hz imaging.

    The result is scaled to [0, 1] where 0 = independent firing,
    1 = perfectly synchronised population.

    Parameters
    ----------
    C : array (N, T)
        Denoised calcium traces from OASIS deconvolution.
    fraction_threshold : float
        Minimum fraction of neurons co-active for a "synchronous" frame.
    S : array (N, T), optional
        Deconvolved spike trains (not used for primary metric).

    Returns
    -------
    float
        Synchrony index in [0, 1].
    
    Notes
    -----
    Using denoised traces (C) rather than spike trains (S) is intentional:
    - At 2 Hz, spike trains have poor temporal resolution for coincidence
    - The ~1s GCaMP6s decay acts as a natural integration window
    - Trace correlations capture functional coupling more reliably
    """
    N, T = C.shape
    if N < 3:
        return 0.0

    # ── 1. Population coupling ───────────────────────────────────────────
    # For each neuron, compute Pearson r with the mean of all OTHER neurons.
    # This measures how strongly each neuron tracks the population.
    pop_r = np.zeros(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N):
            others = np.delete(C, i, axis=0)
            pop_mean = others.mean(axis=0)

            std_i = np.std(C[i])
            std_p = np.std(pop_mean)
            if std_i < 1e-10 or std_p < 1e-10:
                pop_r[i] = 0.0
                continue

            r = np.corrcoef(C[i], pop_mean)[0, 1]
            pop_r[i] = r if np.isfinite(r) else 0.0

    # Clip negatives (anti-correlated neurons don't contribute to synchrony)
    pop_coupling = float(np.mean(np.maximum(pop_r, 0)))

    # ── 2. Co-activation fraction ────────────────────────────────────────
    # Use denoised traces: a neuron is "active" if above baseline + 2σ noise
    diffs = np.diff(C, axis=1)
    noise = np.median(np.abs(diffs), axis=1) / 0.6745
    noise[noise == 0] = 1e-10
    baseline = np.percentile(C, 20, axis=1, keepdims=True)
    active = (C - baseline) > (2.0 * noise[:, np.newaxis])

    frac_per_frame = active.mean(axis=0)
    coactivation = float((frac_per_frame >= fraction_threshold).mean())

    # ── Combine: weighted mean ───────────────────────────────────────────
    # Population coupling is more robust; co-activation captures burst events
    sync_index = 0.6 * pop_coupling + 0.4 * coactivation

    return float(np.clip(sync_index, 0, 1))


def _inter_event_intervals_from_spikes(
    S: np.ndarray, frame_rate: float
) -> Tuple[float, float]:
    """Compute inter-event interval (IEI) statistics from deconvolved spike trains.
    
    This is the PRIMARY IEI computation method, using the discrete spike events
    detected by OASIS deconvolution rather than calcium trace peaks.
    
    Parameters
    ----------
    S : array (N, T)
        Deconvolved spike trains from OASIS. Non-zero values indicate spike events.
        The amplitude of S reflects the estimated spike magnitude (Ca2+ transient size).
    frame_rate : float
        Sampling rate in Hz.
    
    Returns
    -------
    mean_iei : float
        Mean inter-event interval in seconds across all neurons.
    cv_iei : float
        Coefficient of variation (SD/mean) of IEIs. Higher values indicate
        more irregular/bursty firing patterns.
    
    Notes
    -----
    - Uses S > 0 to identify spike frames (OASIS outputs positive values for spikes)
    - Pools IEIs across all neurons to get population-level statistics
    - Requires at least 2 spikes per neuron to compute IEIs for that neuron
    - Returns (0, 0) if fewer than 3 IEIs total across all neurons
    """
    all_ieis = []
    for i in range(S.shape[0]):
        spike_frames = np.where(S[i] > 0)[0]
        if len(spike_frames) >= 2:
            ieis = np.diff(spike_frames) / frame_rate
            all_ieis.extend(ieis.tolist())
    if len(all_ieis) < 3:
        return 0.0, 0.0
    arr = np.array(all_ieis)
    m = float(np.mean(arr))
    return m, float(np.std(arr) / m) if m > 0 else 0.0


def _network_bursts_from_spikes(
    S: np.ndarray, frame_rate: float,
    threshold: float = 0.15, min_gap: float = 1.0,
) -> Tuple[int, float, float]:
    """Detect network bursts from deconvolved spike trains.
    
    This is the PRIMARY burst detection method, using discrete spike events
    from OASIS deconvolution rather than calcium trace thresholding.
    
    A network burst is defined as a time point where the fraction of neurons
    spiking exceeds `threshold`, with bursts separated by at least `min_gap`.
    
    Parameters
    ----------
    S : array (N, T)
        Deconvolved spike trains from OASIS. S[i,t] > 0 indicates neuron i
        fired at frame t.
    frame_rate : float
        Sampling rate in Hz.
    threshold : float
        Minimum fraction of neurons that must spike simultaneously to count
        as a network burst. Default 0.15 (15% of neurons).
    min_gap : float
        Minimum time in seconds between distinct burst peaks.
    
    Returns
    -------
    n_bursts : int
        Number of detected network bursts.
    burst_rate : float
        Burst rate in bursts per 10 seconds.
    mean_participation : float
        Mean fraction of neurons participating in detected bursts.
    
    Notes
    -----
    - Uses binary spike detection (S > 0) to compute population activity
    - Peak detection on population spike rate identifies burst events
    - Lower threshold (0.15) than trace-based method because spike detection
      is more temporally precise
    """
    from scipy.signal import find_peaks
    N, T = S.shape
    dur_s = T / frame_rate

    # Population spike rate per frame (fraction of neurons spiking)
    active = (S > 0).astype(float)
    pop_rate = active.mean(axis=0)

    peaks, _ = find_peaks(
        pop_rate, height=threshold,
        distance=max(1, int(min_gap * frame_rate)),
    )
    n = len(peaks)
    rate = n / dur_s * 10.0 if dur_s > 0 else 0.0
    part = float(np.mean(pop_rate[peaks])) if n > 0 else 0.0
    return n, rate, part


def _measure_transient_amplitudes(
    C: np.ndarray, S: np.ndarray, frame_rate: float,
    baseline_window_s: float = 1.0,
    baseline_offset_s: float = 0.2,
    peak_window_s: float = 0.5,
) -> np.ndarray:
    """
    Measure calcium transient amplitudes as a local ΔF/F₀ increment per event.

    For each detected spike event, computes the amplitude as:
        amplitude = peak - baseline

    on the denoised trace the activity stage wrote, which is already ΔF/F₀.
    The difference is therefore the event's rise above its own pre-event
    baseline, self-referenced and independent of slow drift over the
    recording.  There is one measurement method, so amplitudes are
    comparable across every recording the analysis will load.
    
    Parameters
    ----------
    C : array (N, T)
        Denoised calcium traces (used for spike timing from S, and as
        fallback for amplitude measurement).
    S : array (N, T)
        Deconvolved spike trains from OASIS (used only for spike timing).
    frame_rate : float
        Imaging frame rate in Hz.
    baseline_window_s : float
        Duration of window before spike for baseline estimation (default 1.0s).
    baseline_offset_s : float
        Gap between baseline window end and spike time (default 0.2s).
    peak_window_s : float
        Duration of window after spike to search for peak (default 0.5s).
    
    Returns
    -------
    amplitudes : array (N,)
        Per-neuron mean transient amplitude in ΔF/F₀ units, indexed to match
        the rows of ``C`` and ``S``.  A neuron with no measurable transient is
        ``np.nan``, which is not the same as zero: a neuron can spike and still
        yield no valid event, because events too close to the recording start,
        events on a baseline at or below zero, and events with a non-positive
        amplitude are all skipped.
    """
    N, T = C.shape
    
    
    # Convert time windows to frames
    baseline_frames = max(1, int(baseline_window_s * frame_rate))
    offset_frames = max(1, int(baseline_offset_s * frame_rate))
    peak_frames = max(1, int(peak_window_s * frame_rate))
    
    neuron_amplitudes = np.full(N, np.nan, dtype=float)
    
    for j in range(N):
        trace_for_amp = C[j]
        spikes = S[j]
        
        spike_frames = np.where(spikes > 0)[0]
        if len(spike_frames) == 0:
            continue
        
        transient_amps = []
        
        for t_spike in spike_frames:
            bl_end = t_spike - offset_frames
            bl_start = bl_end - baseline_frames
            pk_start = t_spike
            pk_end = min(T, t_spike + peak_frames)
            
            if bl_start < 0 or pk_end <= pk_start:
                continue
            
            baseline = np.median(trace_for_amp[bl_start:bl_end])
            peak_val = np.max(trace_for_amp[pk_start:pk_end])
            
            # Both terms are already ΔF/F₀, so their difference is the
            # event's rise above its own pre-event baseline.
            amp = peak_val - baseline
            
            if amp > 0:
                transient_amps.append(amp)
        
        if len(transient_amps) > 0:
            neuron_amplitudes[j] = float(np.mean(transient_amps))
    
    return neuron_amplitudes




# =============================================================================
# FEATURE MATRIX
# =============================================================================

def build_feature_matrix(datasets: List[DatasetMetrics]) -> Tuple[np.ndarray, List[str]]:
    """Build (n_datasets, n_features) matrix from dataset metrics."""
    names = [d.name for d in datasets]
    X = np.zeros((len(datasets), len(FEATURE_NAMES)))
    for i, ds in enumerate(datasets):
        for j, (attr, _) in enumerate(FEATURE_NAMES):
            val = _recording_metric(ds, attr)
            X[i, j] = val if val is not None else 0.0

    # Replace NaN/inf with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad]) if (~bad).any() else 0.0

    return X, names

