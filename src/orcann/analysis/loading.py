"""
Dataset loading and per-dataset feature extraction.

Loads pipeline output directories, applies edge ROI exclusion and distance
deduplication, scores neuron quality, and constructs a DatasetMetrics object
holding the per-recording summary plus per-neuron arrays for downstream
metric computation.

Also contains the name-parsing helpers used to recover genotype, organoid,
and cell-line identifiers from dataset folder names.
"""

import os
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def _abbrev(name: str, max_len: int = 20) -> str:
    """Create a short, readable dataset label for figure legends.

    Naming convention: {organoid}_{passage}_{date}_{region} - Denoised

    Examples
    --------
    'D109_3-63_040226_R7 - Denoised' -> 'D109_040226_R7'
    'D115_0-3_050326_R3 - Denoised'  -> 'D115_050326_R3'
    """
    # Strip common suffixes
    s = name.replace(' - Denoised', '').replace(' - denoised', '').strip()

    parts = s.split('_')
    line_id = parts[0] if parts else s[:4]  # 'D109'

    # Find the date part (6 digits, typically ddmmyy)
    date_part = ''
    for p in parts:
        if len(p) == 6 and p.isdigit():
            date_part = p
            break

    # Find the R part (region/repeat)
    r_part = ''
    for p in parts:
        if p.startswith('R') and len(p) <= 3 and p[1:].isdigit():
            r_part = p
            break

    # Build label: always include date for disambiguation
    if date_part and r_part:
        label = f"{line_id}_{date_part}_{r_part}"
    elif r_part:
        label = f"{line_id}_{r_part}"
    elif date_part:
        label = f"{line_id}_{date_part}"
    else:
        label = line_id

    return label[:max_len]


def _trace_snr(trace: np.ndarray) -> float:
    """
    Compute trace SNR following standard calcium imaging convention.

    SNR = peak ΔF/F amplitude / σ_baseline

    where σ_baseline is the MAD-based noise estimate from frame-to-frame
    differences (robust to transients).

    Returns values typically in the range 2–20 for good neurons.
    """
    diff = np.diff(trace)
    mad = np.median(np.abs(diff - np.median(diff)))
    noise = 1.4826 * mad / np.sqrt(2)  # MAD → σ
    if noise <= 0:
        return 0.0
    signal = np.percentile(trace, 95) - np.percentile(trace, 5)
    return float(signal / noise)


def _load_valid_mask(result_path) -> np.ndarray:
    """
    Return a boolean mask of valid ROIs for a dataset.

    Boundary-touching ROIs are excluded at detection time (v2.1+),
    so all saved ROIs are valid.  For backwards compatibility with
    older results that include boundary ROIs, the boundary_touching
    file is checked if present.
    """
    from pathlib import Path
    result_path = Path(result_path)

    # Try to infer N from spike trains
    spikes_path = result_path / 'data' / 'spike_trains.npy'
    traces_path = result_path / 'data' / 'temporal_traces.npy'
    if spikes_path.exists():
        N = np.load(spikes_path, mmap_mode='r').shape[0]
    elif traces_path.exists():
        N = np.load(traces_path, mmap_mode='r').shape[0]
    else:
        return None

    mask = np.ones(N, dtype=bool)

    # Backwards compat: exclude boundary ROIs from older pipeline runs
    boundary_path = result_path / 'boundary_touching.npy'
    if boundary_path.exists():
        boundary = np.load(boundary_path).astype(bool)
        if len(boundary) == N:
            mask &= ~boundary

    return mask


def _has_corrupted_values(trace: np.ndarray, threshold: float = 1e6) -> bool:
    """Check if a trace has corrupted/overflow values (e.g. 1e9+)."""
    return bool(np.any(np.abs(trace) > threshold))


@dataclass
class DatasetMetrics:
    """Per-dataset summary metrics extracted from pipeline outputs."""
    name: str
    filepath: str

    # Neuron counts
    n_neurons: int = 0
    n_confident: int = 0
    n_selected: int = 0
    n_hard_rejected: int = 0       # neurons rejected by hard gates (deconv failures)
    n_overlap_removed: int = 0     # duplicate ROIs removed by ≥50% spatial overlap
    n_distance_removed: int = 0    # ROIs removed by minimum centroid distance filter

    # Selected neuron info (for transparency)
    selected_indices: Optional[np.ndarray] = None    # indices into confident set
    selected_roi_indices: Optional[np.ndarray] = None  # original ROI indices (into full array)
    selected_quality: Optional[np.ndarray] = None    # quality scores
    selected_traces: Optional[np.ndarray] = None     # (n_selected, T) denoised
    selected_raw_traces: Optional[np.ndarray] = None # (n_selected, T) raw fluorescence
    selected_spikes: Optional[np.ndarray] = None     # (n_selected, T) spike trains
    selected_roi_crops: Optional[List] = None        # list of (crop_img, footprint_img) tuples per neuron

    # Per-neuron arrays (all confident neurons)
    all_quality_scores: Optional[np.ndarray] = None

    # Per-neuron arrays (selected neurons) — precomputed to avoid storing full traces
    neuron_spike_rates: Optional[np.ndarray] = None      # (n_selected,) events/10s
    neuron_spike_amplitudes: Optional[np.ndarray] = None  # (n_selected,) mean dF/F per neuron
    neuron_is_active: Optional[np.ndarray] = None         # (n_selected,) bool: ≥1 validated transient

    # Dataset-level metrics (computed from SELECTED neurons only)
    mean_spike_rate: float = 0.0
    median_spike_rate: float = 0.0
    mean_spike_amplitude: float = 0.0
    n_active: int = 0                    # active selected neurons (rate > 0)
    active_fraction: float = 0.0         # n_active / n_neurons (total detections)
    pairwise_correlation_mean: float = 0.0
    synchrony_index: float = 0.0
    mean_iei: float = 0.0
    cv_iei: float = 0.0
    n_network_bursts: int = 0
    burst_rate: float = 0.0
    mean_burst_participation: float = 0.0
    mean_quality_score: float = 0.0

    # Temporal
    frame_rate: float = 2.0
    n_frames: int = 0
    duration_seconds: float = 0.0

    # Motion quality
    motion_max_shift: float = 0.0
    motion_mean_shift: float = 0.0
    motion_residual_std: float = 0.0
    motion_excluded: bool = False

    # Baseline drift quality
    baseline_drift: float = 0.0       # population-mean drift ratio (Q4-Q1)/std
    baseline_drift_excluded: bool = False

    # Amplitude tracking (from run_info.json, populated if available)
    amplitude_tracking: Optional[List] = None  # per-stage amplitude diagnostics

    # Genotype (v2.0)
    genotype: str = ''                 # 'Control', 'Mutant', or 'Unknown'
    
    # Manual override
    manually_inactive: bool = False    # True if visually confirmed no activity
    line_id: str = ''                  # e.g. '3-63', '1-12'


# Feature names used for clustering (must match DatasetMetrics attributes)
# Focused on biological metrics — removed imaging-quality confounds
FEATURE_NAMES = [
    ('mean_spike_rate',          'Event Rate (events/10s)'),
    ('median_spike_rate',        'Median Event Rate (events/10s)'),
    ('mean_spike_amplitude',     'Mean Transient Amplitude (ΔF/F₀)'),
    ('pairwise_correlation_mean','Mean Pairwise Corr. (r)'),
    ('synchrony_index',          'Synchrony Index'),
    ('mean_iei',                 'Mean IEI (s)'),
    ('cv_iei',                   'IEI CV'),
    ('n_network_bursts',         'Network Bursts'),
    ('burst_rate',               'Burst Rate (bursts/10s)'),
    ('mean_burst_participation', 'Burst Participation'),
]



# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset_metrics(
    result_dir: str,
    name: str,

    frame_rate_override: Optional[float] = None,
    min_roi_distance: float = 15.0,
) -> Optional[DatasetMetrics]:
    """Load pipeline outputs, score neuron quality, select by threshold.

    Parameters
    ----------
        If > 0, select this many top neurons (legacy mode).
        Minimum quality score (0–1) for neuron inclusion. Default 0.8.
    min_roi_distance : float
        Minimum distance in pixels between selected neuron centroids.
        Pairs closer than this are deduplicated, keeping the higher-quality
        one.  Default 15.0 pixels.  Set to 0 to disable.
    """
    # Deferred import: metrics imports DatasetMetrics/FEATURE_NAMES from this
    # module, so a top-level import here would create a cycle.
    from .metrics import (
        _measure_transient_amplitudes,
        _pairwise_correlations,
        _synchrony_index,
        _inter_event_intervals_from_spikes,
        _network_bursts_from_spikes,
    )

    result_path = Path(result_dir)

    denoised_path = result_path / 'data' / 'traces_denoised.npy'
    spikes_path = result_path / 'data' / 'spike_trains.npy'
    traces_path = result_path / 'data' / 'temporal_traces.npy'

    if not traces_path.exists():
        logger.warning(f"  {name}: temporal_traces.npy not found, skipping")
        return None

    C_raw = np.load(traces_path)

    # Load raw fluorescence traces (for local ΔF/F amplitude measurement)
    raw_fluor_path = result_path / 'data' / 'temporal_traces_raw.npy'
    C_raw_fluorescence = np.load(raw_fluor_path) if raw_fluor_path.exists() else None

    # Check which amplitude method was used for this dataset
    amplitude_method = 'global_dff'  # default
    pipeline_json_path = result_path / 'pipeline_results.json'
    if pipeline_json_path.exists():
        try:
            import json as _json
            with open(pipeline_json_path) as _f:
                _pres = _json.load(_f)
            amplitude_method = _pres.get('amplitude_method', 'global_dff')
        except Exception:
            pass

    # Load deconvolved data
    has_deconv = denoised_path.exists() and spikes_path.exists()
    if has_deconv:
        C_denoised = np.load(denoised_path)
        S = np.load(spikes_path)
    else:
        logger.warning(f"  {name}: no deconvolution data — quality selection not possible, skipping")
        return None

    # Load and apply edge ROI exclusion
    boundary_path = result_path / 'boundary_touching.npy'
    if boundary_path.exists():
        boundary = np.load(boundary_path).astype(bool)
        n_edge = int(boundary.sum())
        if n_edge > 0:
            logger.info(f"  {name}: excluding {n_edge} edge ROIs")
    else:
        boundary = np.zeros(C_denoised.shape[0], dtype=bool)

    info_path = result_path / 'run_info.json'
    frame_rate = frame_rate_override or 2.0
    amp_tracking_data = None
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
            config = info.get('config', {})
            frame_rate = frame_rate_override or config.get('frame_rate', 2.0)
            amp_tracking_data = info.get('amplitude_tracking', None)
            if amp_tracking_data:
                logger.info(f"  {name}: loaded amplitude tracking ({len(amp_tracking_data)} stages)")

    N, T = C_denoised.shape
    duration = T / frame_rate
    dur_min = duration / 60.0

    # ── Selection: include all ROIs that survived deconvolution ──────────
    # Boundary-touching seeds are excluded at detection time.
    # The deconvolution stage gates on s_min, 3.5σ noise, and transient
    # duration — traces that fail have C_denoised and S zeroed.
    # Selection here simply checks for deconvolution output.
    has_events = np.array([np.sum(S[i] > 0) > 0 for i in range(N)])
    has_signal = np.array([np.max(C_denoised[i]) - np.min(C_denoised[i]) > 1e-10
                           for i in range(N)])
    sel_mask = has_events & has_signal

    n_deconv_pass = int(sel_mask.sum())
    n_deconv_fail = N - n_deconv_pass
    logger.info(f"  {name}: {N} total ROIs, {n_deconv_pass} with deconvolved events, "
                f"{n_deconv_fail} deconvolution failures")

    if n_deconv_pass < 3:
        logger.warning(f"  {name}: only {n_deconv_pass} ROIs passed deconvolution, skipping")
        return None

    sel_idx = np.where(sel_mask)[0]
    original_roi_idx = sel_idx.copy()
    actual_n = len(sel_idx)

    C_sel = C_denoised[sel_idx]
    S_sel = S[sel_idx]
    R_sel = C_raw[sel_idx]

    # Raw fluorescence traces for local ΔF/F amplitude measurement
    R_fluor_sel = None
    if C_raw_fluorescence is not None:
        R_fluor_sel = C_raw_fluorescence[sel_idx]

    # ── Load spatial footprints and compute ROI crops ────────────────────
    # Each crop is a dict with 'max_proj', 'baseline', 'contour' arrays
    roi_crops = None
    A_sparse = None
    footprint_path = result_path / 'data' / 'spatial_footprints.npz'
    max_proj_path = result_path / 'data' / 'max_projection.npy'
    mean_proj_path = result_path / 'data' / 'mean_projection.npy'
    info_path_spatial = result_path / 'run_info.json'
    try:
        if footprint_path.exists():
            from scipy.sparse import load_npz
            A_sparse = load_npz(footprint_path)

            # Get image dimensions
            dims = None
            if info_path_spatial.exists():
                with open(info_path_spatial) as _f:
                    _info = json.load(_f)
                    if 'dims' in _info:
                        dims = tuple(_info['dims'])
                    elif 'd1' in _info and 'd2' in _info:
                        dims = (int(_info['d1']), int(_info['d2']))

            # Load projection images if available
            max_proj = np.load(max_proj_path) if max_proj_path.exists() else None
            mean_proj = np.load(mean_proj_path) if mean_proj_path.exists() else None

            if dims is not None:
                d1, d2 = dims
                roi_crops = []
                crop_radius = 35  # pixels around centroid

                for sel_i in range(actual_n):
                    orig_roi = int(original_roi_idx[sel_i])
                    if orig_roi >= A_sparse.shape[1]:
                        roi_crops.append(None)
                        continue

                    # Get footprint as 2D
                    fp = A_sparse[:, orig_roi].toarray().ravel()
                    if len(fp) != d1 * d2:
                        roi_crops.append(None)
                        continue
                    fp_2d = fp.reshape(d1, d2)

                    # Find centroid
                    ys, xs = np.where(fp_2d > 0)
                    if len(ys) == 0:
                        roi_crops.append(None)
                        continue
                    cy, cx = int(np.mean(ys)), int(np.mean(xs))

                    # Crop region
                    y0 = max(0, cy - crop_radius)
                    y1 = min(d1, cy + crop_radius)
                    x0 = max(0, cx - crop_radius)
                    x1 = min(d2, cx + crop_radius)

                    # Binary contour of the footprint for overlay
                    fp_mask = (fp_2d > 0).astype(np.uint8)
                    contour_crop = fp_mask[y0:y1, x0:x1]

                    crop_data = {'contour': contour_crop}

                    # Crop from max projection (peak fluorescence)
                    if max_proj is not None and max_proj.shape == (d1, d2):
                        crop_data['max_proj'] = max_proj[y0:y1, x0:x1]

                    # Crop from mean projection (baseline)
                    if mean_proj is not None and mean_proj.shape == (d1, d2):
                        crop_data['baseline'] = mean_proj[y0:y1, x0:x1]

                    roi_crops.append(crop_data)

                n_with_crops = sum(1 for c in roi_crops if c is not None)
                has_projs = max_proj is not None or mean_proj is not None
                logger.info(f"  {name}: spatial crops for {n_with_crops}/{actual_n} neurons"
                            f" (projections: {'yes' if has_projs else 'no — run batch again to save projections'})")
            else:
                logger.info(f"  {name}: no image dims available, skipping spatial crops")
    except Exception as e:
        logger.warning(f"  {name}: spatial crop extraction failed: {e}")
        roi_crops = None

    # ── Distance deduplication: remove ROIs closer than min_roi_distance ──
    # Two detections with centres < min_roi_distance pixels apart are
    # effectively sampling the same structure.  Keeping both inflates n
    # and creates correlated duplicates.  The higher-SNR one is kept.
    roi_snr = np.array([_trace_snr(C_sel[j]) for j in range(actual_n)])
    n_distance_removed = 0
    if A_sparse is not None and min_roi_distance > 0 and actual_n >= 2:
        try:
            _dims = None
            _info_path = result_path / 'run_info.json'
            if _info_path.exists():
                with open(_info_path) as _f:
                    _info_data = json.load(_f)
                    if 'dims' in _info_data:
                        _dims = tuple(_info_data['dims'])
                    elif 'd1' in _info_data and 'd2' in _info_data:
                        _dims = (int(_info_data['d1']), int(_info_data['d2']))

            if _dims is not None:
                d1, d2 = _dims
                centroids = np.zeros((actual_n, 2))
                centroid_valid = np.ones(actual_n, dtype=bool)

                for ci in range(actual_n):
                    roi_col = int(original_roi_idx[ci])
                    if roi_col >= A_sparse.shape[1]:
                        centroid_valid[ci] = False
                        continue
                    fp = A_sparse[:, roi_col].toarray().ravel()
                    if len(fp) != d1 * d2:
                        centroid_valid[ci] = False
                        continue
                    fp_2d = fp.reshape(d1, d2)
                    ys, xs = np.where(fp_2d > 0)
                    if len(ys) == 0:
                        centroid_valid[ci] = False
                        continue
                    centroids[ci] = [np.mean(ys), np.mean(xs)]

                dist_keep = np.ones(actual_n, dtype=bool)
                for ci in range(actual_n):
                    if not dist_keep[ci] or not centroid_valid[ci]:
                        continue
                    for cj in range(ci + 1, actual_n):
                        if not dist_keep[cj] or not centroid_valid[cj]:
                            continue
                        dist = np.sqrt((centroids[ci, 0] - centroids[cj, 0])**2 +
                                       (centroids[ci, 1] - centroids[cj, 1])**2)
                        if dist < min_roi_distance:
                            if roi_snr[ci] >= roi_snr[cj]:
                                dist_keep[cj] = False
                            else:
                                dist_keep[ci] = False
                                break
                            n_distance_removed += 1

                if n_distance_removed > 0:
                    logger.info(f"  {name}: removing {n_distance_removed}/{actual_n} "
                                f"ROIs closer than {min_roi_distance:.0f}px")
                    sel_idx = sel_idx[dist_keep]
                    original_roi_idx = original_roi_idx[dist_keep]
                    C_sel = C_sel[dist_keep]
                    S_sel = S_sel[dist_keep]
                    R_sel = R_sel[dist_keep]
                    roi_snr = roi_snr[dist_keep]
                    if R_fluor_sel is not None:
                        R_fluor_sel = R_fluor_sel[dist_keep]
                    if roi_crops is not None:
                        roi_crops = [c for c, k in zip(roi_crops, dist_keep) if k]
                    actual_n = len(sel_idx)
                    if actual_n < 3:
                        logger.warning(f"  {name}: only {actual_n} neurons remain after "
                                       f"distance filter, skipping dataset")
                        return None
        except Exception as e:
            logger.warning(f"  {name}: centroid distance filter failed: {e}")

    logger.info(f"  {name}: selected {actual_n}/{N} ROIs "
                f"({n_deconv_fail} deconv failures, {n_distance_removed} distance-deduped)")

    # ── Compute metrics from SELECTED neurons ────────────────────────────
    # Spike rates (events per 10 seconds)
    spike_counts = np.array([np.sum(S_sel[j] > 0) for j in range(actual_n)])
    spike_rates = spike_counts / duration * 10.0 if duration > 0 else spike_counts
    mean_spike_rate = float(np.mean(spike_rates))
    median_spike_rate = float(np.median(spike_rates))

    # Spike amplitudes — method depends on pipeline configuration:
    #   direct / local_dff: measure each event as local ΔF/F from raw fluorescence
    #   global_dff / local_background: measure from corrected traces
    _use_local = amplitude_method in ('direct', 'local_dff')
    _amp_raw = R_fluor_sel if _use_local else None
    all_amps = _measure_transient_amplitudes(
        C_sel, S_sel, frame_rate, C_raw_fluorescence=_amp_raw)
    mean_spike_amp = float(np.mean(all_amps)) if all_amps else 0.0

    # Detailed per-neuron log for verification
    logger.info(f"  {name}: duration={duration:.1f}s ({dur_min:.2f} min), "
                f"spike counts per neuron: {spike_counts.tolist()}")
    logger.info(f"  {name}: rates/10s: {[f'{r:.1f}' for r in spike_rates]}")
    if all_amps:
        logger.info(f"  {name}: transient amplitudes (ΔF/F₀): {[f'{a:.3f}' for a in all_amps]}")

    # ── Correlation and synchrony ──────────────────────────────────────────
    # Uses DENOISED TRACES (C_sel) for correlation/synchrony - see docstrings
    # for rationale (trace correlations more reliable than spike correlations
    # at 2 Hz due to temporal discretization issues).
    # Minimum n=5: below this, pairwise correlation from too few pairs
    # (<10) is unreliable and excluded from statistical comparisons.
    MIN_N_CORR = 5
    if actual_n >= MIN_N_CORR:
        corr_mean, _ = _pairwise_correlations(C_sel, S=S_sel)
        sync_idx = _synchrony_index(C_sel, S=S_sel, frame_rate=frame_rate)
    else:
        corr_mean = float('nan')
        sync_idx  = float('nan')
        logger.info(f"  {name}: n_selected={actual_n} < {MIN_N_CORR} — "
                    f"correlation/synchrony set to NaN (insufficient pairs)")

    # ── IEI from SPIKE TRAINS ─────────────────────────────────────────────
    # Uses deconvolved spike events (S_sel > 0) for inter-event intervals
    mean_iei, cv_iei = _inter_event_intervals_from_spikes(S_sel, frame_rate)

    # ── Network bursts from SPIKE TRAINS ──────────────────────────────────
    # Uses population spike synchrony (fraction of neurons with S > 0)
    n_bursts, burst_rate_val, burst_part = _network_bursts_from_spikes(S_sel, frame_rate)

    ds = DatasetMetrics(
        name=name, filepath=str(result_path),
        n_neurons=N, n_confident=n_deconv_pass, n_selected=actual_n,
        n_hard_rejected=n_deconv_fail, n_overlap_removed=0,
        n_distance_removed=n_distance_removed,
        selected_indices=sel_idx,
        selected_roi_indices=original_roi_idx,
        selected_quality=roi_snr,
        selected_traces=C_sel,
        selected_raw_traces=R_sel,
        selected_spikes=S_sel,
        selected_roi_crops=roi_crops,
        all_quality_scores=None,
        mean_spike_rate=mean_spike_rate,
        median_spike_rate=median_spike_rate,
        mean_spike_amplitude=mean_spike_amp,
        pairwise_correlation_mean=corr_mean,
        synchrony_index=sync_idx,
        mean_iei=mean_iei, cv_iei=cv_iei,
        n_network_bursts=n_bursts,
        burst_rate=float(burst_rate_val),
        mean_burst_participation=burst_part,
        mean_quality_score=float(np.mean(roi_snr)),
        frame_rate=frame_rate,
        n_frames=T, duration_seconds=duration,
        genotype=_extract_genotype(name),
        line_id=_extract_line_id(name),
    )

    logger.info(f"    event_rate={mean_spike_rate:.1f}/10s, sync={sync_idx:.3f}, "
                f"mean_snr={np.mean(roi_snr):.3f}, "
                f"genotype={ds.genotype}, line={ds.line_id}")


    # ── Baseline drift (population-level) ────────────────────────────────
    # Measure drift on raw traces of selected neurons.
    # Compare mean fluorescence in first vs last quarter of recording.
    # High population-median drift ratio indicates bleach correction failure.
    drift_ratio = 0.0
    if R_sel is not None and R_sel.shape[1] > 10:
        T_drift = R_sel.shape[1]
        q1_slice = slice(0, T_drift // 4)
        q4_slice = slice(3 * T_drift // 4, T_drift)
        neuron_drifts = []
        for j in range(R_sel.shape[0]):
            trace = R_sel[j]
            q1_mean = np.mean(trace[q1_slice])
            q4_mean = np.mean(trace[q4_slice])
            trace_std = np.std(trace)
            if trace_std > 1e-10:
                neuron_drifts.append(abs(q4_mean - q1_mean) / trace_std)
        if neuron_drifts:
            drift_ratio = float(np.median(neuron_drifts))
    logger.info(f"    baseline_drift={drift_ratio:.3f}")

    # ── Motion quality ───────────────────────────────────────────────────
    shifts_path = result_path / 'data' / 'motion_shifts.npy'
    mc_info = {}
    if info_path.exists():
        with open(info_path) as f:
            mc_info = json.load(f).get('motion_correction', {})

    motion_max = mc_info.get('max_shift_y', 0.0) + mc_info.get('max_shift_x', 0.0)
    motion_mean = mc_info.get('mean_shift_y', 0.0) + mc_info.get('mean_shift_x', 0.0)

    motion_residual = 0.0
    if shifts_path.exists():
        shifts = np.load(shifts_path)
        if shifts.ndim == 2 and shifts.shape[0] > 2:
            shift_diffs = np.diff(shifts, axis=0)
            motion_residual = float(np.std(np.sqrt(
                shift_diffs[:, 0]**2 + shift_diffs[:, 1]**2
            )))

    ds.motion_max_shift = float(motion_max)
    ds.motion_mean_shift = float(motion_mean)
    ds.motion_residual_std = motion_residual
    ds.baseline_drift = drift_ratio
    ds.amplitude_tracking = amp_tracking_data

    # ── Precompute per-neuron arrays (for genotype analysis) ─────────────
    # These allow us to free the full (N, T) trace matrices below.
    dur_s = ds.duration_seconds if ds.duration_seconds > 0 else 1.0
    ds.neuron_spike_rates = np.array([
        np.sum(S_sel[j] > 0) / dur_s * 10.0
        for j in range(actual_n)
    ])
    
    # Active fraction: neurons with ≥1 validated transient in the selected
    # set, as a proportion of ALL detections for this dataset.
    # This measures how many of the total detected ROIs survived quality
    # selection AND showed genuine activity.
    ds.neuron_is_active = ds.neuron_spike_rates > 0
    ds.n_active = int(ds.neuron_is_active.sum())
    ds.active_fraction = ds.n_active / max(N, 1)
    
    logger.info(f"  {name}: active fraction = {ds.n_active}/{N} "
                f"({ds.active_fraction:.1%}) "
                f"[{ds.n_active} active selected out of {N} total detections]")

    # Inactive-recording gate: fewer than 3 neurons with any spikes makes
    # within-recording statistics (correlation, synchrony, MAD-based outlier
    # detection) unreliable. Classify these as inactive and drop from analysis.
    if ds.n_active < 3:
        logger.warning(f"  {name}: only {ds.n_active} active neurons, "
                       f"classifying as inactive and skipping")
        return None
    all_amps_list = _measure_transient_amplitudes(
        C_sel, S_sel, frame_rate, C_raw_fluorescence=_amp_raw)
    # _measure_transient_amplitudes skips neurons with no spikes, so its length
    # may be < actual_n. Build a per-neuron array with 0.0 for silent neurons.
    if len(all_amps_list) == actual_n:
        ds.neuron_spike_amplitudes = np.array(all_amps_list)
    else:
        # Recompute per-neuron to get correct alignment
        per_neuron_amps = np.zeros(actual_n)
        amp_idx = 0
        for j in range(actual_n):
            spike_frames = np.where(S_sel[j] > 0)[0]
            if len(spike_frames) > 0 and amp_idx < len(all_amps_list):
                per_neuron_amps[j] = all_amps_list[amp_idx]
                amp_idx += 1
        ds.neuron_spike_amplitudes = per_neuron_amps

    # ── Free large trace arrays to reduce memory ────────────────────────
    # The full (N, T) matrices are only needed for per-dataset diagnostic
    # figures; we keep them only for a limited number of datasets.
    # Dataset-level metrics are already computed above.
    ds.selected_traces = None
    ds.selected_raw_traces = None
    ds.selected_spikes = None

    return ds



# =============================================================================
# NAME PARSING (genotype / organoid / cell line)
# =============================================================================

def _extract_organoid_id(name: str) -> str:
    """Extract organoid identifier from dataset name.

    'D109_3-63_040226_R7 - Denoised' → 'D109'
    'D115_0-3_040226_R3 - Denoised'  → 'D115'

    Falls back to the full abbreviated name if no clear organoid prefix found.
    """
    s = name.replace(' - Denoised', '').replace(' - denoised', '').strip()
    parts = s.split('_')
    if parts and (parts[0].startswith('D') or parts[0].startswith('d')):
        return parts[0].upper()
    return _abbrev(name)



def _extract_genotype(name: str, genotype_map: dict = None) -> str:
    """Extract genotype from dataset name using a configurable prefix map.

    Naming convention: ``{day}_{line}_{date}_{region} - Denoised``

    The line field (second underscore-delimited part) encodes genotype
    via a prefix before the hyphen.  The mapping from prefix to genotype
    label is defined by ``genotype_map``.

    Parameters
    ----------
    name : str
        Dataset name.
    genotype_map : dict, optional
        Maps line-prefix strings to genotype labels.  Any prefix not in the
        map is assigned to ``'default'`` if present, otherwise ``'Unknown'``.

        Example::

            {'3': 'Control', 'default': 'Mutant'}

        Default (if None): ``{'3': 'Control', 'default': 'Mutant'}``.

    Examples
    --------
    >>> _extract_genotype('D109_3-63_040226_R7 - Denoised')
    'Control'
    >>> _extract_genotype('D109_1-12_040226_R3 - Denoised')
    'Mutant'

    Returns 'Unknown' if the line field cannot be parsed.
    """
    if genotype_map is None:
        genotype_map = {'3': 'Control', 'default': 'Mutant'}

    s = name.replace(' - Denoised', '').replace(' - denoised', '').strip()
    parts = s.split('_')

    if len(parts) < 2:
        return 'Unknown'

    line_field = parts[1]  # e.g. '3-63', '1-12', '0-3'

    # The genotype prefix is the digit(s) before the hyphen
    if '-' in line_field:
        prefix = line_field.split('-')[0].strip()
    else:
        prefix = line_field.strip()

    if prefix in genotype_map:
        return genotype_map[prefix]
    elif 'default' in genotype_map:
        return genotype_map['default']
    else:
        return 'Unknown'


def _extract_line_id(name: str) -> str:
    """Extract the full line identifier from dataset name.

    'D109_3-63_040226_R7 - Denoised' -> '3-63'
    """
    s = name.replace(' - Denoised', '').replace(' - denoised', '').strip()
    parts = s.split('_')
    return parts[1] if len(parts) >= 2 else 'unknown'


