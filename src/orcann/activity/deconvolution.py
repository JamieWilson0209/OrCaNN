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
    baseline_percentile: float = 8.0,
    baseline_window_fraction: float = 0.25,
    baseline_min_window: int = 50,
    baseline_max_window: int = 500,
) -> Dict[str, np.ndarray]:
    """
    Deconvolve calcium traces to infer spike trains.

    Parameters
    ----------
    C : array (N, T)
        Trace matrix, **ΔF/F₀** (N, T).  Not raw fluorescence: convert with
        :func:`orcann.activity.baseline.compute_dff_traces` first, as
        ``run_activity._compute_dff`` does.  ``s_min`` and ``noise_gate_sigma``
        are both in ΔF/F₀ units, so raw input silently changes what they mean.
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

        'oasis' raises RuntimeError if CaImAn is not importable: a solver
        failure on a trace falls back to the threshold method, but a missing
        dependency does not, because that would silently change the detection
        method for the whole recording.
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
    baseline_percentile, baseline_window_fraction, baseline_min_window, baseline_max_window
        The rolling-percentile baseline the robust detector and the safety net
        measure against — the same settings the activity stage used for dF/F0,
        so the detector and the baseline stage agree on "resting".
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
        'n_spikes_censored' : (N,) — of those, how many touch a trace boundary
        'n_spikes_rescued'  : int  — ROIs backfilled by the safety net (oasis only)
        'C_dff'       : (N, T)  — ΔF/F₀ input used for deconvolution
    """
    N, T = C.shape

    # ── Ensure traces are ΔF/F₀ ──────────────────────────────────────────
    # OASIS expects small-valued ΔF/F₀ traces (values near 0, transients
    # as positive bumps of ~0.05–0.5).  If traces are raw fluorescence
    # (values in hundreds/thousands/millions), convert them first.
    if N == 0 or C.size == 0:
        # Guard before _ensure_dff: an empty array makes np.median NaN, which
        # sends it down the raw-fluorescence branch and crashes inside a log
        # line in baseline.py. Return the empty result the callers expect.
        logger.warning("  No ROIs to deconvolve — returning empty result")
        empty = C.astype(np.float32)
        return {'C_denoised': empty.copy(), 'S': np.zeros_like(empty),
                'bl': np.zeros(N, np.float32), 'noise': np.zeros(N, np.float32),
                'g': np.zeros((N, 1), np.float32), 'method': method,
                'n_spikes': np.zeros(N, np.int32),
                'n_spikes_censored': np.zeros(N, np.int32),
                'n_spikes_rescued': 0, 'C_dff': empty}

    # C is ΔF/F₀ by contract — run_activity._compute_dff converts before calling,
    # and it is the only caller. This used to run _ensure_dff, which inferred the
    # units from np.median(C) > 1.0 and converted if it decided they were raw.
    # That inference had no correct case here (the input is always already
    # converted) and three wrong ones: any NaN made the median NaN, and
    # `nan <= 1.0` is False, so one bad sample sent the whole matrix through a
    # second baseline correction; raw data normalised to [0, 1] passed as ΔF/F₀
    # with its photobleaching intact; and a genuine ΔF/F₀ trace with median > 1
    # was detrended twice, costing ~30% of event amplitude. It also hardcoded
    # baseline settings that ignored the config.
    #
    # Checked rather than inferred: a contract violation is loud, and nothing is
    # silently rescaled either way.
    C_dff = np.asarray(C, dtype=np.float32)
    _median = float(np.nanmedian(C_dff)) if np.isfinite(C_dff).any() else float("nan")
    if not np.isfinite(_median) or abs(_median) > _DFF_MEDIAN_SANITY:
        logger.warning(
            f"  Trace median is {_median:.4g}, which does not look like ΔF/F₀ "
            f"(expected |median| <= {_DFF_MEDIAN_SANITY}). deconvolve_traces "
            f"requires ΔF/F₀; pass traces through "
            f"orcann.activity.baseline.compute_dff_traces first. Continuing, but "
            f"s_min and the noise gate are in ΔF/F₀ units and will not mean what "
            f"they say.")

    logger.info(f"Deconvolution: method={method}, {N} traces, "
                f"decay={decay_time}s, frame_rate={frame_rate} Hz")
    logger.info(f"  Input range: [{C_dff.min():.4f}, {C_dff.max():.4f}], "
                f"median={np.median(C_dff):.4f}")

    if method == 'oasis':
        # OASIS *is* CaImAn's constrained_foopsi, so this method cannot run
        # without it. Checked up front, and never swallowed by the fallback
        # below: an absent dependency is a configuration error, not a numerical
        # one, and silently substituting a different detection method would
        # change the results while the run still reported success. This is
        # exactly what used to happen when the activity stage was run in the
        # torch env instead of the caiman env.
        try:
            from caiman.source_extraction.cnmf.deconvolution import (  # noqa: F401
                constrained_foopsi)
        except ImportError as e:
            raise RuntimeError(
                "deconvolution.method='oasis' requires CaImAn, which is not "
                f"importable in this environment ({e}). OASIS is CaImAn's "
                "constrained_foopsi, so there is no OASIS without it.\n"
                "  Run the activity stage in the caiman env (hpc/jobs/activity.sh "
                "uses CAIMAN_ENV; build it with `bash hpc/setup.sh caiman`), or "
                "set deconvolution.method='robust' to use the deterministic "
                "detector, which needs no CaImAn.") from e

        try:
            result = _deconvolve_oasis(
                C_dff, frame_rate, decay_time,
                penalty=penalty, optimize_g=optimize_g,
                noise_method=noise_method,
                s_min=s_min, noise_gate_sigma=noise_gate_sigma,
            )
        except ImportError:
            # A CaImAn submodule failing to load later is the same class of
            # problem as it being absent — do not degrade to another method.
            raise
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
                robust_k_onset, robust_k_peak, robust_min_duration_s,
                baseline_percentile=baseline_percentile,
                baseline_window_fraction=baseline_window_fraction,
                baseline_min_window=baseline_min_window,
                baseline_max_window=baseline_max_window)
    elif method == 'threshold':
        result = _deconvolve_threshold(C_dff, frame_rate, decay_time,
                                       noise_gate_sigma=noise_gate_sigma)
    elif method == 'robust':
        result = _deconvolve_robust(
            C_dff, frame_rate, decay_time,
            k_onset=robust_k_onset, k_peak=robust_k_peak,
            min_duration_s=robust_min_duration_s,
            baseline_percentile=baseline_percentile,
                baseline_window_fraction=baseline_window_fraction,
                baseline_min_window=baseline_min_window,
                baseline_max_window=baseline_max_window)
    else:
        raise ValueError(f"Unknown deconvolution method: {method}")

    # Every path must expose the same keys: _deconvolve_threshold never set
    # this, and the safety net's `if not empty.any(): return` skipped creating
    # it, so consumers hit KeyError on data-dependent runs. 0 is meaningful
    # here, not a placeholder — OASIS zeroes S[:, :edge_frames] and
    # S[:, -edge_frames:], so a fitted ROI has no boundary event by construction.
    if 'n_spikes_censored' not in result:
        result['n_spikes_censored'] = np.zeros(N, dtype=np.int32)
    result.setdefault('n_spikes_rescued', 0)
    result['C_dff'] = C_dff
    return result


# ── Shared detector policy ───────────────────────────────────────────────────
# The OASIS path applies two guards after fitting: it zeroes spikes within
# EDGE_SECONDS of either end (baseline estimation is unreliable there), and it
# rejects any trace whose elevation lasts longer than MAX_TRANSIENT_SECONDS
# (a loading artefact or a non-neuronal signal, not a calcium event). Both live
# here so the robust detector and the safety net apply the same policy rather
# than a different one — previously the safety net selected exactly the ROIs
# those guards had emptied and refilled them, defeating both.

# A ΔF/F₀ median far from 0 means the caller passed something else. Generous:
# real recordings measured here sit near 0.0004-0.08.
_DFF_MEDIAN_SANITY = 5.0

EDGE_SECONDS = 0.5
MAX_TRANSIENT_SECONDS = 80.0


def edge_frames_for(frame_rate: float) -> int:
    """Frames at each end where spikes are suppressed."""
    return max(2, int(frame_rate * EDGE_SECONDS))


def _apply_edge_guard(s_row: np.ndarray, frame_rate: float) -> np.ndarray:
    """Zero spikes at both trace boundaries, in place."""
    e = edge_frames_for(frame_rate)
    if 2 * e >= len(s_row):
        # The two slices would overlap and erase everything; a short recording
        # is not evidence of no activity.
        return s_row
    s_row[:e] = 0.0
    s_row[-e:] = 0.0
    return s_row


def _exceeds_duration_gate(trace: np.ndarray, base, frame_rate: float) -> bool:
    """True if the longest continuous elevation exceeds MAX_TRANSIENT_SECONDS.

    ``base`` must be a SCALAR resting level, not the rolling baseline used for
    detection. A rolling baseline rises to meet a sustained plateau, so the
    plateau stops looking elevated and this gate could never fire — which is the
    opposite of its purpose. OASIS references a single fitted ``bl`` here for the
    same reason.
    """
    max_frames = int(MAX_TRANSIENT_SECONDS * frame_rate)
    base = float(np.asarray(base, dtype=np.float64).min()
                 if np.ndim(base) else base)
    peak_above = float(np.max(trace - base))
    if peak_above <= 0:
        return False
    tolerance = max(0.05 * peak_above, 0.01)
    elevated = trace > (base + tolerance)
    if not elevated.any():
        return False
    d = np.diff(np.concatenate([[0], elevated.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    if len(starts) == 0:
        return False
    return int(np.max(ends - starts)) > max_frames


def detector_baseline(trace: np.ndarray, frame_rate: float, percentile: float,
                      window_fraction: float, min_window: int,
                      max_window: int) -> np.ndarray:
    """Per-frame baseline for the robust detector, using the configured method.

    The detector used to reference a single global ``np.median(trace)``. That
    fails in both directions: a drifting trace spends its later half above the
    median and is chopped into phantom events, while an ROI elevated for more
    than half the recording pulls the median up with it and goes silent. The
    same rolling percentile the activity stage uses for dF/F0 tracks drift and
    is unmoved by sustained activity, so the detector and the baseline stage
    now agree on what "resting" means.
    """
    from scipy import ndimage
    from orcann.activity.baseline import _adaptive_window
    T = len(trace)
    if T < 3:
        return np.full(T, float(np.median(trace)) if T else 0.0)
    window = _adaptive_window(T, window_fraction, min_window, max_window)
    window = min(window, T if T % 2 else T - 1)      # never wider than the trace
    window = max(3, window if window % 2 else window + 1)
    # percentile_filter directly, NOT baseline._rolling_baseline: that floors F0
    # at max(median*0.01, 1.0) to keep the dF/F0 division well-conditioned. This
    # detector only subtracts, so the floor is not needed here — and on a dF/F0
    # trace, whose values sit near 0.02, it would pin the baseline to a constant
    # 1.0 and suppress every event.
    return ndimage.percentile_filter(
        np.asarray(trace, np.float64), percentile, size=window, mode='reflect')


def _apply_safety_net(result, C_dff, frame_rate, decay_time,
                      k_onset, k_peak, min_duration_s,
                      baseline_percentile=8.0, baseline_window_fraction=0.25,
                       baseline_min_window=50, baseline_max_window=500):
    """Backfill robust-detector spikes for ROIs OASIS left empty but that carry a
    clear transient. Only fills empty ROIs, so it never overrides a real OASIS
    fit; it just stops obvious events being lost to an OASIS miss.

    The rescued spikes go through the same edge guard and duration gate the
    OASIS path applies. Without that, the net selected precisely the ROIs those
    guards had just emptied and refilled them, so one run applied two opposite
    policies and rescued ROIs carried systematically higher rates. A trace the
    duration gate rejects stays rejected; the net now only catches genuine
    solver misses, which is what it was for."""
    S = result['S']
    N = S.shape[0]
    empty = S.sum(axis=1) == 0
    if not empty.any():
        return result
    n_censored = result.get('n_spikes_censored')
    if n_censored is None:
        n_censored = np.zeros(N, dtype=np.int32)
        result['n_spikes_censored'] = n_censored
    n_rescued = 0
    n_gated = 0
    for i in np.where(empty)[0]:
        tr = C_dff[i].astype(np.float64)
        bl = detector_baseline(tr, frame_rate, baseline_percentile,
                               baseline_window_fraction, baseline_min_window,
                               baseline_max_window)
        rest = float(np.percentile(tr, baseline_percentile))
        if _exceeds_duration_gate(tr, rest, frame_rate):
            n_gated += 1
            continue                     # the duration gate rejected it; keep it rejected
        sn = _mad_noise(tr)
        s_i, n_ev, n_cens = _robust_events(tr, sn, frame_rate,
                                           k_onset, k_peak, min_duration_s,
                                           baseline=bl)
        _apply_edge_guard(s_i, frame_rate)
        n_ev = int((s_i > 0).sum())      # recount after the guard
        if n_ev > 0:
            S[i] = s_i
            result['n_spikes'][i] = n_ev
            n_censored[i] = n_cens
            n_rescued += 1
    if n_rescued:
        logger.info(f"  Safety net: recovered {n_rescued} ROI(s) with clear "
                    f"transients that OASIS missed")
    if n_gated:
        logger.info(f"  Safety net: {n_gated} empty ROI(s) left empty — the "
                    f"duration gate rejects them")
    result['n_spikes_rescued'] = n_rescued
    return result


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


def _robust_events(trace, noise, frame_rate, k_onset, k_peak, min_duration_s,
                   baseline=None):
    """Deterministic transient detection for a single ΔF/F₀ trace.

    Independent of OASIS: it finds contiguous excursions above a
    ``k_onset`` × noise onset threshold that (a) last at least
    ``min_duration_s`` and (b) reach a ``k_peak`` × noise peak.  The dual
    threshold plus the minimum duration is what separates real transients from
    noise: a single-sample noise spike clears neither the duration nor,
    usually, the peak bar.  Noise is estimated robustly from successive
    differences (``_mad_noise``), so a broad, high-amplitude transient does not
    inflate it.

    Returns ``(S, n_events, n_censored)``: a spike train with the peak ΔF/F₀
    placed at each event maximum, the event count, and how many of those events
    touch frame 0 or frame T-1 (see the note at the boundary test below).
    """
    T = len(trace)
    S = np.zeros(T, dtype=np.float32)
    if noise <= 0 or not np.isfinite(noise):
        return S, 0, 0
    # Per-frame baseline from the configured method (see detector_baseline).
    # None keeps the old global-median behaviour for direct callers.
    base = (np.full(T, float(np.median(trace))) if baseline is None
            else np.asarray(baseline, dtype=np.float64))
    onset = base + k_onset * noise
    peak_thr = base + k_peak * noise
    min_dur = max(2, int(round(min_duration_s * frame_rate)))

    above = trace > onset
    n_events = 0
    n_censored = 0
    i = 0
    while i < T:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < T and above[j]:
            j += 1
        seg = trace[i:j] - base[i:j]          # relative to the local baseline
        if (j - i) >= min_dur and np.any(trace[i:j] >= peak_thr[i:j]):
            pk = i + int(np.argmax(seg))
            S[pk] = float(trace[pk] - base[pk])
            n_events += 1
            # An excursion touching either end of the trace is censored. Right
            # (j == T): the event began inside the window but had not finished,
            # so its duration is unknown and its peak may lie beyond the trace.
            # Left (i == 0): it was already under way at frame 0, so its onset
            # is outside the recording entirely and counting it inflates the
            # rate. The two are not equivalent and only the right-censored case
            # clearly belongs in a rate. Both are kept and counted rather than
            # dropped, because excluding real events is a scientific decision,
            # not an engineering one — but the count is what lets the caller
            # make it. In both cases the amplitude in S is a lower bound.
            if i == 0 or j == T:
                n_censored += 1
        i = j
    return S, n_events, n_censored


def _deconvolve_robust(C, frame_rate, decay_time,
                       k_onset=3.0, k_peak=5.0, min_duration_s=0.5,
                       baseline_percentile=8.0, baseline_window_fraction=0.25,
                       baseline_min_window=50, baseline_max_window=500):
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
    n_censored = np.zeros(N, dtype=np.int32)
    n_duration_rejected = 0
    for i in range(N):
        tr = C[i].astype(np.float64)
        bl = detector_baseline(tr, frame_rate, baseline_percentile,
                               baseline_window_fraction, baseline_min_window,
                               baseline_max_window)
        sn = _mad_noise(tr)
        noise_levels[i] = sn
        rest = float(np.percentile(tr, baseline_percentile))
        if _exceeds_duration_gate(tr, rest, frame_rate):
            n_duration_rejected += 1     # same gate the OASIS path applies
            continue
        S[i], n_spikes[i], n_censored[i] = _robust_events(
            tr, sn, frame_rate, k_onset, k_peak, min_duration_s, baseline=bl)
        _apply_edge_guard(S[i], frame_rate)
        n_spikes[i] = int((S[i] > 0).sum())
    if n_duration_rejected:
        logger.info(f"  Duration gate (>{MAX_TRANSIENT_SECONDS:.0f}s): "
                    f"rejected {n_duration_rejected} trace(s)")
    n_active = int((n_spikes > 0).sum())
    med_active = float(np.median(n_spikes[n_spikes > 0])) if n_active else 0.0
    logger.info(f"  Robust transient detection: {n_active}/{N} ROIs active, "
                f"median {med_active:.0f} events/active ROI "
                f"(k_onset={k_onset}, k_peak={k_peak}, min_dur={min_duration_s}s)")
    n_cens_total = int(n_censored.sum())
    if n_cens_total:
        logger.info(f"  {n_cens_total} event(s) touch a trace boundary — real "
                    f"events, but duration unknown and amplitude a lower bound")
    return {
        'C_denoised': C.copy(),
        'S':          S,
        'bl':         np.median(C, axis=1).astype(np.float32),
        'noise':      noise_levels,
        'g':          np.full((N, 1), np.exp(-1.0 / (frame_rate * decay_time))),
        'method':     'robust',
        'n_spikes':   n_spikes,
        'n_spikes_censored': n_censored,
    }
