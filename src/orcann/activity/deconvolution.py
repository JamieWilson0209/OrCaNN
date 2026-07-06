"""
Calcium Trace Deconvolution
============================

Infers spike trains from noisy ΔF/F calcium traces using constrained
deconvolution.

Primary method: CaImAn's OASIS (Online Active Set method to Infer Spikes),
which is fast, parameter-light, and handles the AR(1)/AR(2) calcium
indicator models.

Fallback: a simple threshold-based peak detection if CaImAn is unavailable.

Also provides optional temporal filtering (low-pass Butterworth) as a
preprocessing step before deconvolution.

References:
- Friedrich et al., "Fast online deconvolution of calcium imaging data",
  PLoS Comp Bio 2017.
- Pachitariu et al., "Suite2p", bioRxiv 2017.
"""

import os
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)


# =============================================================================
# TEMPORAL FILTERING
# =============================================================================

def temporal_filter(
    C: np.ndarray,
    frame_rate: float,
    cutoff_hz: float = 2.0,
    order: int = 3,
) -> np.ndarray:
    """
    Low-pass Butterworth filter for calcium traces.

    Parameters
    ----------
    C : array (N, T)
        Trace matrix.
    frame_rate : float
        Sampling rate in Hz.
    cutoff_hz : float
        Cutoff frequency in Hz (default 2.0).  For Fluo-4 (τ≈400ms) at
        2 Hz, real transient content is below ~2 Hz.
    order : int
        Filter order (default 3).

    Returns
    -------
    C_filtered : array (N, T)
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = frame_rate / 2.0
    if cutoff_hz >= nyquist:
        logger.warning(f"Cutoff {cutoff_hz} Hz >= Nyquist {nyquist} Hz — skipping filter")
        return C.copy()

    sos = butter(order, cutoff_hz / nyquist, btype='low', output='sos')

    N, T = C.shape
    C_filt = np.zeros_like(C)

    for i in range(N):
        try:
            C_filt[i] = sosfiltfilt(sos, C[i])
        except Exception:
            C_filt[i] = C[i]

    logger.info(f"  Temporal filter: Butterworth LP, cutoff={cutoff_hz} Hz, "
                f"order={order}, frame_rate={frame_rate} Hz")

    return C_filt


# =============================================================================
# DECONVOLUTION — OASIS (CaImAn)
# =============================================================================

def deconvolve_traces(
    C: np.ndarray,
    frame_rate: float,
    decay_time: float = 0.4,
    method: str = 'oasis',
    penalty: float = 0,
    optimize_g: bool = True,
    noise_method: str = 'mean',
    s_min: float = 0.1,
    noise_gate_sigma: float = 3.5,
    robust_k_onset: float = 3.0,
    robust_k_peak: float = 5.0,
    robust_min_duration_s: float = 0.5,
    robust_safety_net: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Deconvolve calcium traces to infer spike trains.

    Parameters
    ----------
    C : array (N, T)
        Trace matrix (raw fluorescence or ΔF/F₀).  If traces appear to be
        raw fluorescence (median > 1), they are automatically converted to
        ΔF/F₀ before deconvolution.
    frame_rate : float
        Sampling rate in Hz.
    decay_time : float
        Indicator decay time constant in seconds (Fluo-4: ~0.4s).
    method : str
        'oasis' (AR deconvolution, recommended where it fits), 'threshold'
        (peak detection), or 'robust' (deterministic transient detector that
        prioritises catching obvious transients and rejecting noise, and never
        fails the way OASIS can).  On the 'oasis' path a robust safety net
        backfills any obvious transient OASIS missed unless disabled.
    penalty : float
        Sparsity penalty (L1). 0 = auto-tune (recommended for OASIS).
    optimize_g : bool
        Whether OASIS should optimise the AR coefficient from data.
    noise_method : str
        How OASIS estimates noise: 'mean', 'median', or 'logmexp'.
    s_min : float
        Minimum spike amplitude in ΔF/F₀.  Passed to OASIS's constrained_foopsi
        so the solver zeroes any inferred spike below this amplitude.  0 lets
        OASIS choose (no explicit floor).  Ignored by the threshold method.
    noise_gate_sigma : float
        After deconvolution, keep only spikes whose amplitude exceeds this
        multiple of the per-trace noise floor.  Applied on both the OASIS path
        (against OASIS's sn estimate) and the threshold path (against a MAD noise
        estimate), so the two methods stay consistent.  0 disables the gate.

    Returns
    -------
    dict with:
        'C_denoised'  : (N, T)  — denoised calcium traces (in ΔF/F₀)
        'S'           : (N, T)  — inferred spike trains (≥s_min and ≥noise gate)
        'bl'          : (N,)    — estimated baselines
        'noise'       : (N,)    — estimated noise levels
        'g'           : (N, p)  — AR coefficients per neuron
        'method'      : str     — method used
        'n_spikes'    : (N,)    — number of inferred spikes per neuron
        'C_dff'       : (N, T)  — ΔF/F₀ input used for deconvolution
    """
    N, T = C.shape

    # ── Ensure traces are ΔF/F₀ ──────────────────────────────────────────
    # OASIS expects small-valued ΔF/F₀ traces (values near 0, transients
    # as positive bumps of ~0.05–0.5).  If traces are raw fluorescence
    # (values in hundreds/thousands/millions), convert them first.
    C_dff = _ensure_dff(C, frame_rate)

    logger.info(f"Deconvolution: method={method}, {N} traces, "
                f"decay={decay_time}s, frame_rate={frame_rate} Hz")
    logger.info(f"  Input range: [{C_dff.min():.4f}, {C_dff.max():.4f}], "
                f"median={np.median(C_dff):.4f}")

    if method == 'oasis':
        try:
            result = _deconvolve_oasis(
                C_dff, frame_rate, decay_time,
                penalty=penalty, optimize_g=optimize_g,
                noise_method=noise_method,
                s_min=s_min, noise_gate_sigma=noise_gate_sigma,
            )
        except Exception as e:
            logger.warning(f"OASIS failed: {e}")
            logger.info("Falling back to threshold deconvolution")
            result = _deconvolve_threshold(C_dff, frame_rate, decay_time,
                                           noise_gate_sigma=noise_gate_sigma)
        # Safety net: OASIS can silently miss an obvious transient (a per-trace
        # solver failure zeroes the trace, or a conservative fit assigns a clear
        # event to baseline). Independently detect strong transients and backfill
        # spikes for any ROI where OASIS produced none but a clear event exists.
        # This guarantees obvious transients survive even when OASIS does not.
        if robust_safety_net:
            result = _apply_safety_net(
                result, C_dff, frame_rate, decay_time,
                robust_k_onset, robust_k_peak, robust_min_duration_s)
    elif method == 'threshold':
        result = _deconvolve_threshold(C_dff, frame_rate, decay_time,
                                       noise_gate_sigma=noise_gate_sigma)
    elif method == 'robust':
        result = _deconvolve_robust(
            C_dff, frame_rate, decay_time,
            k_onset=robust_k_onset, k_peak=robust_k_peak,
            min_duration_s=robust_min_duration_s)
    else:
        raise ValueError(f"Unknown deconvolution method: {method}")

    result['C_dff'] = C_dff
    return result


def _apply_safety_net(result, C_dff, frame_rate, decay_time,
                      k_onset, k_peak, min_duration_s):
    """Backfill robust-detector spikes for ROIs OASIS left empty but that carry a
    clear transient. Only fills empty ROIs, so it never overrides a real OASIS
    fit; it just stops obvious events being lost to an OASIS miss."""
    S = result['S']
    N = S.shape[0]
    empty = S.sum(axis=1) == 0
    if not empty.any():
        return result
    n_rescued = 0
    for i in np.where(empty)[0]:
        tr = C_dff[i].astype(np.float64)
        sn = _mad_noise(tr)
        s_i, n_ev = _robust_events(tr, sn, frame_rate,
                                   k_onset, k_peak, min_duration_s)
        if n_ev > 0:
            S[i] = s_i
            result['n_spikes'][i] = n_ev
            n_rescued += 1
    if n_rescued:
        logger.info(f"  Safety net: recovered {n_rescued} ROI(s) with clear "
                    f"transients that OASIS missed")
    result['n_spikes_rescued'] = n_rescued
    return result


def _ensure_dff(C: np.ndarray, frame_rate: float) -> np.ndarray:
    """
    Ensure traces are ΔF/F₀ for OASIS deconvolution.

    If data is raw fluorescence (median > 1.0): apply the standard
    rolling-percentile baseline correction (delegates to
    :func:`preprocessing.compute_dff_traces` so there is one canonical
    implementation).

    If data is already ΔF/F₀ (median ≤ 1.0): pass through without
    modification.  OASIS internally estimates its own baseline (the `bl`
    parameter in constrained_foopsi), so additional baseline subtraction
    here can interfere with its estimation and degrade spike detection.
    """
    median_val = np.median(C)

    if median_val <= 1.0:
        logger.info(f"  Traces appear to be ΔF/F₀ already (median={median_val:.4f})")
        logger.info(f"  Passing through without baseline adjustment — "
                    f"OASIS will estimate its own baseline internally")
        return C.copy().astype(np.float32)

    logger.info(f"  Traces appear to be raw fluorescence (median={median_val:.1f}), "
                f"converting to ΔF/F₀...")

    try:
        from preprocessing import compute_dff_traces
    except ImportError:
        from .preprocessing import compute_dff_traces

    C_dff, _, _ = compute_dff_traces(
        C, frame_rate=frame_rate,
        percentile=8.0,
        window_fraction=0.25,
        min_window=50, max_window=500,
        edge_trim=False,
    )

    logger.info(f"  Converted: range [{C_dff.min():.4f}, {C_dff.max():.4f}], "
                f"median={np.median(C_dff):.4f}")

    return C_dff.astype(np.float32)


def _deconvolve_oasis(
    C, frame_rate, decay_time, penalty, optimize_g, noise_method,
    s_min=0.1, noise_gate_sigma=3.5,
) -> Dict[str, np.ndarray]:
    """Run CaImAn's OASIS deconvolution.

    ``s_min`` is passed to constrained_foopsi, instructing OASIS to suppress
    any inferred spike whose amplitude is below that ΔF/F₀ level relative to
    baseline.  This is the principled way to gate on transient amplitude —
    letting the solver itself discard sub-threshold events rather than
    re-filtering its output post-hoc.  A second, noise-relative gate
    (``noise_gate_sigma`` × the trace's own noise floor) is then applied to the
    surviving spikes.  Both default to the calcium pipeline's values (0.1, 3.5)
    and are configurable via the deconvolution config section.
    """
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi

    N, T = C.shape
    dt = 1.0 / frame_rate

    # Initial AR(1) coefficient from decay time:  g = exp(-dt/τ)
    g_init = np.exp(-dt / decay_time)
    logger.info(f"  OASIS: g_init={g_init:.4f} (τ={decay_time}s, dt={dt:.4f}s), "
                f"s_min={s_min} ΔF/F₀, noise gate={noise_gate_sigma}σ")

    C_denoised = np.zeros((N, T), dtype=np.float32)
    S = np.zeros((N, T), dtype=np.float32)
    baselines = np.zeros(N, dtype=np.float32)
    noise_levels = np.zeros(N, dtype=np.float32)
    g_values = np.zeros((N, 1), dtype=np.float32)
    n_spikes = np.zeros(N, dtype=np.int32)

    n_success = 0
    n_failed = 0
    n_retry = 0

    def _foopsi(trace, smin):
        return constrained_foopsi(
            trace,
            g=[g_init] if not optimize_g else None,
            noise_method=noise_method,
            p=1,          # AR(1) model
            s_min=smin,
        )

    for i in range(N):
        trace = C[i].astype(np.float64)

        try:
            # constrained_foopsi returns: (c, bl, c1, g, sn, sp, lam)
            # s_min tells OASIS to zero any spike whose reconstructed amplitude is
            # below it (in ΔF/F₀).  On some traces (typically broad, high-amplitude
            # transients) the hard s_min constraint makes the solve infeasible and
            # constrained_foopsi raises.  Rather than dropping straight to crude
            # peak detection — which silently loses obvious transients (e.g. a 20σ
            # event) — first retry with s_min=0, letting OASIS place spikes without
            # the floor.  This recovers a proper deconvolution for most "failures".
            try:
                c, bl, c1, g, sn, sp, lam = _foopsi(trace, s_min)
            except Exception:
                if s_min <= 0:
                    raise                       # nothing to relax; go to fallback
                c, bl, c1, g, sn, sp, lam = _foopsi(trace, 0.0)
                n_retry += 1

            C_denoised[i] = c.astype(np.float32)
            S[i] = sp.astype(np.float32)
            baselines[i] = float(bl)
            noise_levels[i] = float(sn)
            g_values[i, 0] = float(g[0]) if len(g) > 0 else g_init
            n_success += 1

        except Exception as e:
            # Both OASIS attempts failed — use simple peak detection so a clear
            # transient is never silently lost. Denoised is zeroed so downstream
            # knows this trace was not properly deconvolved.
            C_denoised[i] = 0.0
            S[i] = _simple_spike_detect(trace, frame_rate, decay_time,
                                        noise_gate_sigma=noise_gate_sigma)
            noise_levels[i] = _mad_noise(trace)
            g_values[i, 0] = g_init
            n_failed += 1
            if n_failed <= 3:
                logger.warning(f"    OASIS failed on trace {i}: {e}")

        if (i + 1) % max(1, N // 5) == 0:
            logger.info(f"    Deconvolved {i + 1}/{N} traces")

    logger.info(f"  OASIS complete: {n_success} success "
                f"({n_retry} via s_min=0 retry), {n_failed} peak-detection fallback")
    if N and n_failed / N > 0.05:
        logger.warning(f"  OASIS fell back on {n_failed}/{N} traces "
                       f"({100*n_failed/N:.0f}%); check the trace units are ΔF/F₀ "
                       f"and consider lowering deconvolution.s_min")

    # ── Decay parameter diagnostics ──────────────────────────────────────
    dt = 1.0 / frame_rate
    g_fitted = g_values[:, 0]
    g_valid = g_fitted[g_fitted > 0]
    if len(g_valid) > 0:
        tau_fitted = -dt / np.log(np.clip(g_valid, 1e-10, 1 - 1e-10))
        logger.info(f"  ── DECAY PARAMETER DIAGNOSTICS ──")
        logger.info(f"  Initial g (from τ={decay_time}s): {g_init:.4f}")
        logger.info(f"  Fitted g:  median={np.median(g_valid):.4f}, "
                    f"mean={np.mean(g_valid):.4f}, "
                    f"range=[{np.min(g_valid):.4f}, {np.max(g_valid):.4f}]")
        logger.info(f"  Implied τ: median={np.median(tau_fitted):.3f}s, "
                    f"mean={np.mean(tau_fitted):.3f}s, "
                    f"range=[{np.min(tau_fitted):.3f}s, {np.max(tau_fitted):.3f}s]")
        logger.info(f"  Ratio τ_fitted/τ_initial: "
                    f"median={np.median(tau_fitted)/decay_time:.1f}×, "
                    f"range=[{np.min(tau_fitted)/decay_time:.1f}×, "
                    f"{np.max(tau_fitted)/decay_time:.1f}×]")
        pcts = np.percentile(tau_fitted, [5, 25, 50, 75, 95])
        logger.info(f"  τ percentiles: p5={pcts[0]:.3f}s, p25={pcts[1]:.3f}s, "
                    f"p50={pcts[2]:.3f}s, p75={pcts[3]:.3f}s, p95={pcts[4]:.3f}s")

    # ── Noise-relative spike gate ─────────────────────────────────────────
    # Each spike that survived s_min must also exceed noise_gate_sigma × the
    # trace noise floor (OASIS's own sn estimate). This ensures only clearly
    # visible transients are counted — spikes between s_min and the gate are
    # large enough in absolute terms but sit within the noise band of noisier
    # traces and produce ambiguous detections. noise_gate_sigma=0 disables it.
    n_noise_rejected = 0
    if noise_gate_sigma > 0:
        for i in range(N):
            sn = float(noise_levels[i])
            if sn <= 0:
                continue
            threshold = noise_gate_sigma * sn
            spike_frames = np.where(S[i] > 0)[0]
            for pk in spike_frames:
                if S[i, pk] < threshold:
                    S[i, pk] = 0.0
                    n_noise_rejected += 1

    if n_noise_rejected > 0:
        logger.info(f"  Noise gate (>{noise_gate_sigma}σ): "
                    f"removed {n_noise_rejected} sub-threshold spikes")

    # ── Transient duration gate ────────────────────────────────────────────
    # Reject traces where any single elevated episode on the *denoised*
    # trace lasts longer than max_transient_seconds.  Genuine calcium
    # transients decay within a few seconds (fitted τ ≈ 2s median); a
    # single sustained elevation lasting 80+ seconds indicates loading
    # artefacts, slow baseline shifts, or non-neuronal signals that OASIS
    # has fitted as activity rather than baseline.
    #
    # We measure the duration of "elevated" episodes on C_denoised,
    # defined as samples where the denoised trace exceeds the OASIS
    # baseline by more than a small per-trace tolerance.  The tolerance
    # is the larger of (a) 5% of the trace's peak-above-baseline (so
    # bright cells aren't false-positived by tiny baseline noise) and
    # (b) an absolute floor of 0.01 ΔF/F₀ (so dim cells aren't
    # over-sensitive to numerical noise from the OASIS solver).
    #
    # Closely-spaced genuine transients each resolve back to baseline
    # between events at this tolerance — bursts and rhythmic firing
    # are correctly seen as a series of short episodes, not one long one.
    max_transient_seconds = 80.0
    max_transient_frames = int(max_transient_seconds * frame_rate)
    n_duration_rejected = 0
    for i in range(N):
        denoised = C_denoised[i].astype(np.float32)
        bl = float(baselines[i])
        peak_above_bl = float(np.max(denoised) - bl)
        if peak_above_bl <= 0:
            continue   # flat or sub-baseline trace, nothing to gate

        tolerance = max(0.05 * peak_above_bl, 0.01)
        elevated = denoised > (bl + tolerance)
        if not elevated.any():
            continue

        # Longest continuous run of elevated frames.
        diffs = np.diff(np.concatenate([[0], elevated.astype(np.int8), [0]]))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        if len(starts) == 0:
            continue
        longest_run = int(np.max(ends - starts))

        if longest_run > max_transient_frames:
            C_denoised[i, :] = 0.0
            S[i, :] = 0.0
            n_duration_rejected += 1

    if n_duration_rejected > 0:
        logger.info(f"  Duration gate (>{max_transient_seconds:.0f}s): "
                    f"zeroed {n_duration_rejected} traces with sustained events")

    # ── Edge guard ────────────────────────────────────────────────────────
    # Suppress spikes at recording boundaries where baseline estimation is
    # unreliable regardless of the amplitude gate.
    edge_frames = max(2, int(frame_rate * 0.5))
    for i in range(N):
        S[i, :edge_frames] = 0.0
        S[i, -edge_frames:] = 0.0

    n_spikes = np.array([int(np.sum(S[i] > 0)) for i in range(N)], dtype=np.int32)

    total_spikes  = int(n_spikes.sum())
    median_spikes = float(np.median(n_spikes))
    logger.info(f"  Final: {total_spikes} spikes, median {median_spikes:.0f}/neuron")

    return {
        'C_denoised': C_denoised,
        'S':          S,
        'bl':         baselines,
        'noise':      noise_levels,
        'g':          g_values,
        'method':     'oasis',
        'n_spikes':   n_spikes,
    }


def _deconvolve_threshold(C, frame_rate, decay_time,
                          noise_gate_sigma=3.5) -> Dict[str, np.ndarray]:
    """Simple threshold-based spike detection fallback.

    Uses a ``noise_gate_sigma`` × MAD noise threshold to match the noise gate
    applied to the OASIS path, so the two methods are consistent when OASIS is
    unavailable.
    """
    from scipy.signal import find_peaks

    N, T = C.shape
    logger.info(f"  Threshold deconvolution: {N} traces, gate={noise_gate_sigma}σ")

    S = np.zeros((N, T), dtype=np.float32)
    baselines = np.zeros(N, dtype=np.float32)
    noise_levels = np.zeros(N, dtype=np.float32)
    n_spikes = np.zeros(N, dtype=np.int32)

    min_distance = max(1, int(decay_time * frame_rate * 0.5))

    for i in range(N):
        trace = C[i]
        noise = _mad_noise(trace)
        baseline = np.percentile(trace, 20)

        noise_levels[i] = noise
        baselines[i] = baseline

        if noise > 0:
            # noise_gate_sigma × noise matches the gate on the OASIS path
            height = baseline + noise_gate_sigma * noise
            peaks, _ = find_peaks(
                trace, height=height, distance=min_distance,
            )
            for pk in peaks:
                S[i, pk] = trace[pk] - baseline
            n_spikes[i] = len(peaks)

    logger.info(f"  Threshold deconvolution: median {np.median(n_spikes):.0f} "
                f"spikes/neuron")

    return {
        'C_denoised': C.copy(),
        'S':          S,
        'bl':         baselines,
        'noise':      noise_levels,
        'g':          np.full((N, 1), np.exp(-1.0 / (frame_rate * decay_time))),
        'method':     'threshold',
        'n_spikes':   n_spikes,
    }


def _simple_spike_detect(trace, frame_rate, decay_time, noise_gate_sigma=3.5):
    """Quick spike detection for a single trace (OASIS per-trace fallback).

    Uses a ``noise_gate_sigma`` × noise threshold to match the noise gate on the
    main OASIS path.
    """
    from scipy.signal import find_peaks
    S = np.zeros_like(trace)
    noise = _mad_noise(trace)
    baseline = np.percentile(trace, 20)
    if noise > 0:
        height = baseline + noise_gate_sigma * noise
        peaks, _ = find_peaks(trace, height=height,
                              distance=max(1, int(decay_time * frame_rate * 0.5)))
        for pk in peaks:
            S[pk] = trace[pk] - baseline
    return S


def _mad_noise(trace):
    """MAD-based noise estimate."""
    diff = np.diff(trace)
    return 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2)


def _robust_events(trace, noise, frame_rate, k_onset, k_peak, min_duration_s):
    """Deterministic transient detection for a single ΔF/F₀ trace.

    Independent of OASIS: it finds contiguous excursions above a
    ``k_onset`` × noise onset threshold that (a) last at least
    ``min_duration_s`` and (b) reach a ``k_peak`` × noise peak.  The dual
    threshold plus the minimum duration is what separates real transients from
    noise: a single-sample noise spike clears neither the duration nor,
    usually, the peak bar.  Noise is estimated robustly from successive
    differences (``_mad_noise``), so a broad, high-amplitude transient does not
    inflate it.  Returns a spike train with the peak ΔF/F₀ placed at each event
    maximum, and the event count.
    """
    T = len(trace)
    S = np.zeros(T, dtype=np.float32)
    if noise <= 0:
        return S, 0
    base = float(np.median(trace))
    onset = base + k_onset * noise
    peak_thr = base + k_peak * noise
    min_dur = max(2, int(round(min_duration_s * frame_rate)))

    above = trace > onset
    n_events = 0
    i = 0
    while i < T:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < T and above[j]:
            j += 1
        seg = trace[i:j]
        if (j - i) >= min_dur and seg.max() >= peak_thr:
            pk = i + int(np.argmax(seg))
            S[pk] = float(trace[pk] - base)
            n_events += 1
        i = j
    return S, n_events


def _deconvolve_robust(C, frame_rate, decay_time,
                       k_onset=3.0, k_peak=5.0, min_duration_s=0.5):
    """Deterministic transient detector as a standalone deconvolution method.

    Does no AR deconvolution, so it never fails the way OASIS can: its first
    priority is catching obvious transients while rejecting noise.  ``C_denoised``
    is a pass-through of the input (there is no model reconstruction); ``S`` holds
    the detected events.
    """
    N, T = C.shape
    S = np.zeros((N, T), dtype=np.float32)
    noise_levels = np.zeros(N, dtype=np.float32)
    n_spikes = np.zeros(N, dtype=np.int32)
    for i in range(N):
        tr = C[i].astype(np.float64)
        sn = _mad_noise(tr)
        noise_levels[i] = sn
        S[i], n_spikes[i] = _robust_events(tr, sn, frame_rate,
                                           k_onset, k_peak, min_duration_s)
    n_active = int((n_spikes > 0).sum())
    med_active = float(np.median(n_spikes[n_spikes > 0])) if n_active else 0.0
    logger.info(f"  Robust transient detection: {n_active}/{N} ROIs active, "
                f"median {med_active:.0f} events/active ROI "
                f"(k_onset={k_onset}, k_peak={k_peak}, min_dur={min_duration_s}s)")
    return {
        'C_denoised': C.copy(),
        'S':          S,
        'bl':         np.median(C, axis=1).astype(np.float32),
        'noise':      noise_levels,
        'g':          np.full((N, 1), np.exp(-1.0 / (frame_rate * decay_time))),
        'method':     'robust',
        'n_spikes':   n_spikes,
    }
