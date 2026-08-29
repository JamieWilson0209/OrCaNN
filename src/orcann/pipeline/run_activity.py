"""Activity stage (CPU): results/spatial + movie -> results/activity.

The bridge from spatial segmentation to functional readouts, and the seam where
OrCaNN's detection hands off to the calcium pipeline's baseline / deconvolution /
gallery / analysis. For each segmented recording it:

  1. loads the per-ROI fluorescence traces, label image and centroids that
     `segment` wrote, plus the motion-corrected movie;
  2. baseline-corrects the traces to dF/F0 (`activity.baseline`);
  3. infers spike trains with OASIS (`activity.deconvolution`);
  4. writes the calcium-format per-recording folder the group analysis reads:
     data/{temporal_traces, temporal_traces_raw, traces_denoised, spike_trains,
     deconv_noise, spatial_footprints.npz, max_projection, mean_projection}
     plus run_info.json (dims + frame rate);
  5. renders the interactive HTML gallery.

Recordings whose temporal_traces.npy already exists are skipped unless force=True.
The output folder is named by recording_id, so the genotype/day parsing in the
analysis stage keys off it exactly as before.
"""
import json
import logging
import os

import numpy as np
from scipy.sparse import csc_matrix, save_npz

from orcann.pipeline import inference as infer
from orcann.pipeline.cli import list_recordings, list_spatial_recordings
from orcann.run_info import write as write_run_info, read_record, RunInfoError

logger = logging.getLogger(__name__)


def _movie_for(rec_id, pre):
    """The motion-corrected movie whose recording_id matches rec_id, or None."""
    for f in list_recordings(pre):
        if infer.recording_id(f) == rec_id:
            return f
    return None


def _motion_meta(movie_path):
    """Motion-correction metadata written beside the corrected movie, if present.

    `motion_correction` writes `<stem>_mc.json` (shift summary) and, when the
    array is available, `<stem>_mc_shifts.npy` (per-frame [dy, dx]) next to
    `<stem>_mc.tif`. Either may be absent — a recording corrected outside this
    pipeline has neither — and either being absent is carried through rather than
    filled in: no motion block is written, the analysis stage reads the metrics
    as NaN, and its motion gates exclude the recording for unknown motion rather
    than passing it as though it were perfectly still. Returns
    `(summary_dict_or_None, shifts_array_or_None)`.
    """
    if not movie_path:
        return None, None
    base = os.path.splitext(movie_path)[0]            # .../<stem>_mc
    summary, shifts = None, None
    jf = base + ".json"
    if os.path.isfile(jf):
        try:
            summary = read_record(jf, expect_stage="motion_correction")
        except RunInfoError as e:
            logger.error(f"  motion summary unusable, so this recording will be "
                         f"excluded for unknown motion: {e}")
    sf = base + "_shifts.npy"
    if os.path.isfile(sf):
        try:
            shifts = np.load(sf)
        except Exception as e:
            logger.warning(f"  could not read motion shifts {sf}: {e}")
    return summary, shifts


def _load_spatial(spatial_dir, rec_id):
    """Read traces / labels / centroids / max_projection for one recording."""
    d = os.path.join(spatial_dir, rec_id, infer.DATA_DIRNAME)
    traces = np.load(os.path.join(d, infer.TRACES_NPY))
    labels = np.load(os.path.join(d, infer.LABELS_NPY))
    cen_p = os.path.join(d, infer.CENTROIDS_NPY)
    centroids = np.load(cen_p) if os.path.exists(cen_p) else None
    mp_p = os.path.join(d, infer.MAXPROJ_NPY)
    max_proj = np.load(mp_p) if os.path.exists(mp_p) else None
    return traces, labels, centroids, max_proj


BASELINE_METHODS = ("global_dff",)


def _compute_dff(cfg, traces):
    """Baseline-correct raw fluorescence traces to dF/F0 per the config."""
    b = cfg.baseline
    fr = cfg.imaging.frame_rate
    # Validated rather than dispatched with a default branch: an unrecognised
    # value must not quietly select a method and change how dF/F0 is computed.
    if b.method not in BASELINE_METHODS:
        raise ValueError(
            f"baseline.method={b.method!r} is not recognised. "
            f"Valid values: {', '.join(BASELINE_METHODS)}.")
    from orcann.activity.baseline import compute_dff_traces
    c_dff, c_raw, _ = compute_dff_traces(
        traces, frame_rate=fr, percentile=b.percentile,
        window_fraction=b.window_fraction, min_window=b.min_window,
        max_window=b.max_window, presmooth_sigma=b.presmooth_sigma)
    return c_dff, c_raw


def _deconvolve(cfg, c_dff):
    """OASIS spike inference on dF/F0 traces.

    Returns (denoised, spikes, noise, censored, rejected, method_used).

    ``censored`` is the per-ROI count of detected events touching a trace
    boundary, whose duration is unknown and whose amplitude is a lower bound.

    ``rejected`` is the per-ROI boolean flag for traces excluded because they
    contained NaN or Inf. Those ROIs were never deconvolved, so their zero spike
    count is not a measurement and must not be read as a silent ROI.

    ``method_used`` is what the detector reported having done. There are no
    fallback paths left, so it agrees with the configured method; it is recorded
    rather than assumed so that a cohort mixing oasis and robust recordings —
    which is legitimate if the config changed mid-study — is still detectable.
    """
    if not cfg.deconvolution.enabled:
        return None, None, None, None, None, None
    from orcann.activity.deconvolution import deconvolve_traces
    d = cfg.deconvolution
    res = deconvolve_traces(
        c_dff, frame_rate=cfg.imaging.frame_rate, decay_time=cfg.decay_initialisation(),
        method=d.method, optimize_g=d.optimize_g,
        noise_method=d.noise_method, s_min=d.s_min,
        noise_gate_sigma=d.noise_gate_sigma,
        robust_safety_net=d.robust_safety_net,
        robust_k_onset=d.robust_k_onset, robust_k_peak=d.robust_k_peak,
        robust_min_duration_s=d.robust_min_duration_s,
        # The detector baselines with the configured method rather than a global
        # median, so it needs the same knobs the activity stage used.
        baseline_percentile=cfg.baseline.percentile,
        baseline_window_fraction=cfg.baseline.window_fraction,
        baseline_min_window=cfg.baseline.min_window,
        baseline_max_window=cfg.baseline.max_window)
    n_rej = int(res.get("n_traces_rejected") or 0)
    if n_rej:
        # Not carried per-ROI into the analysis: a trace that cannot be measured
        # and one with nothing to find are both a detected cell with no findable
        # activity, and the analysis counts them the same way. The count is the
        # part worth keeping, because a large one means an upstream problem.
        print(f"  WARNING: {n_rej} ROI(s) rejected for non-finite trace values "
              f"- see n_traces_rejected in run_info.json")
    return (res.get("C_denoised"), res.get("S"), res.get("noise"),
            res.get("n_spikes_censored"), res.get("trace_rejected"),
            res.get("method"))


def _write_outputs(out_dir, rec_id, cfg, *, c_dff, c_raw, denoised, spikes,
                   noise, censored, rejected, deconv_used, labels, mean_proj, max_proj, source,
                   motion=None, motion_shifts=None, global_intensity=None):
    """Write the calcium-format per-recording folder + run_info.json."""
    from orcann.activity.roi_adapter import footprints_from_labels

    data = os.path.join(out_dir, "data")
    os.makedirs(data, exist_ok=True)
    H, W = labels.shape

    np.save(os.path.join(data, "temporal_traces.npy"), c_dff.astype(np.float32))
    np.save(os.path.join(data, "temporal_traces_raw.npy"), c_raw.astype(np.float32))
    if denoised is not None:
        np.save(os.path.join(data, "traces_denoised.npy"), denoised.astype(np.float32))
    if spikes is not None:
        np.save(os.path.join(data, "spike_trains.npy"), spikes.astype(np.float32))
    if noise is not None:
        np.save(os.path.join(data, "deconv_noise.npy"), np.asarray(noise, np.float32))
    if censored is not None:
        np.save(os.path.join(data, "deconv_censored.npy"), np.asarray(censored, np.int32))
    if max_proj is not None:
        np.save(os.path.join(data, "max_projection.npy"), max_proj.astype(np.float32))
    np.save(os.path.join(data, "mean_projection.npy"), mean_proj.astype(np.float32))
    save_npz(os.path.join(data, "spatial_footprints.npz"),
             csc_matrix(footprints_from_labels(labels)))
    # Motion metadata carried through from the motion_correction stage so the
    # group analysis can apply its motion QC gates. Per-frame shifts feed the
    # residual-motion metric; the summary dict feeds the max/mean-shift gates.
    if motion_shifts is not None:
        np.save(os.path.join(data, "motion_shifts.npy"),
                np.asarray(motion_shifts, np.float32))

    payload = {
        "recording_id": rec_id,
        "frame_rate": float(cfg.imaging.frame_rate),
        "indicator": cfg.imaging.indicator,
        # The AR seed, not a measured transient decay.
        "decay_initialisation_s": cfg.decay_initialisation(),
        "dims": [int(H), int(W)],
        "n_roi": int(c_dff.shape[0]),
        "n_frames": int(c_dff.shape[1]),
        "baseline": {"method": cfg.baseline.method, "percentile": cfg.baseline.percentile},
        # method_used is what the detector reported having done, rather than
        # what was asked for. Since 10df777 removed the fallback paths the two
        # always agree, so there is no method_fallback flag any more — but the
        # observed value is still what gets recorded and what the cohort check
        # reads, so a future divergence cannot go unnoticed.
        "deconvolution": {"enabled": cfg.deconvolution.enabled,
                          "method": cfg.deconvolution.method,
                          "method_used": deconv_used,
                          "n_traces_rejected": (0 if rejected is None
                                                else int(np.asarray(rejected).sum()))},
        "amplitude_method": cfg.baseline.method,
        "source": os.path.abspath(source) if source else None,
    }
    if motion:
        payload["motion_correction"] = motion
    if global_intensity is not None:
        payload["global_intensity"] = global_intensity
    write_run_info(out_dir, payload)


def _write_galleries(cfg, out_dir, rec_id, movie, labels, centroids, max_proj,
                     c_dff, c_raw, denoised, spikes):
    """Interactive HTML gallery for one recording."""
    g = cfg.gallery
    if not g.interactive:
        return
    from orcann.activity.roi_adapter import build_seed_view, build_projections
    try:
        # Inside the try: a labels/projection shape mismatch raises here, and
        # outside it would escape the handler below and abandon the rest of the
        # batch after this recording's data had already been written.
        seeds = build_seed_view(labels, max_projection=max_proj, centroids=centroids)
        from orcann.activity.gallery import generate_interactive_gallery
        projections = build_projections(movie, max_projection=max_proj)
        generate_interactive_gallery(
            seeds, projections, movie,
            output_path=os.path.join(out_dir, "gallery.html"),
            title=f"{rec_id} - ROI Gallery", max_rois=g.max_rois,
            movie_processed=movie,
            traces_denoised=denoised, spike_trains=spikes,
            pipeline_traces_dff=c_dff, pipeline_traces_raw=c_raw)
        logger.info("  gallery.html written")
    except Exception as e:                       # a gallery failure must not lose data
        logger.warning(f"  interactive gallery failed: {e}")


def run(cfg, task_id=None, force=False):
    sp, pre, out = cfg.paths.spatial, cfg.paths.pre_processed, cfg.paths.activity
    recs = list_spatial_recordings(sp, task_id)
    if not recs:
        print(f"activity: no segmented recordings in {sp} (run segment first)")
        return
    os.makedirs(out, exist_ok=True)
    print(f"activity: {len(recs)} recording(s)  {sp} + {pre} -> {out}")
    for rec_id in recs:
        out_dir = os.path.join(out, rec_id)
        if os.path.exists(os.path.join(out_dir, "data", "temporal_traces.npy")) and not force:
            print(f"{rec_id:28s} (exists, skipped)")
            continue

        traces, labels, centroids, max_proj = _load_spatial(sp, rec_id)
        mv = _movie_for(rec_id, pre)
        if mv is None:
            print(f"{rec_id:28s} ERROR: source movie not found in {pre}, skipping")
            continue
        from orcann.pipeline.extraction import _load_movie
        movie = _load_movie(mv)
        if max_proj is not None and movie.shape[1:] != tuple(max_proj.shape):
            # segment worked at the model's resolution; match the movie to it so
            # projections / footprints / labels share one pixel grid.
            movie = infer.resample_to_shape(movie, max_proj.shape)
        mean_proj = movie.mean(axis=0).astype(np.float32)
        if max_proj is None:
            max_proj = movie.max(axis=0).astype(np.float32)

        # global intensity QC on the movie the activity stage consumes. A
        # plotting failure here must not lose the recording, so it is isolated.
        gi = None
        try:
            from orcann.activity.global_intensity import global_intensity_diagnostic
            gi = global_intensity_diagnostic(movie, out_dir, rec_id)
        except Exception as e:
            logger.warning(f"  global intensity diagnostic failed: {e}")

        c_dff, c_raw = _compute_dff(cfg, traces)
        denoised, spikes, noise, censored, rejected, deconv_used = _deconvolve(cfg, c_dff)

        motion, motion_shifts = _motion_meta(mv)
        _write_outputs(out_dir, rec_id, cfg, c_dff=c_dff, c_raw=c_raw,
                       denoised=denoised, spikes=spikes, noise=noise,
                       censored=censored, rejected=rejected,
                       deconv_used=deconv_used,
                       labels=labels, mean_proj=mean_proj, max_proj=max_proj,
                       source=mv, motion=motion, motion_shifts=motion_shifts,
                       global_intensity=gi)
        _write_galleries(cfg, out_dir, rec_id, movie, labels, centroids, max_proj,
                         c_dff, c_raw, denoised, spikes)

        n_spk = int((spikes > 0).sum()) if spikes is not None else 0
        print(f"{rec_id:28s} {int(c_dff.shape[0]):6d} cells  {n_spk:8d} spikes")
    print(f"activity outputs -> {out}/<recording_id>/  (now run: analysis)")
