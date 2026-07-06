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
  5. renders the interactive HTML gallery (and, optionally, the movie gallery).

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
    `<stem>_mc.tif`. Both are optional: recordings that were already corrected
    outside this pipeline, or corrected before this metadata was added, simply
    return `(None, None)` and the analysis then treats motion as clean (its motion
    QC gates default to zero shift). Returns `(summary_dict_or_None,
    shifts_array_or_None)`.
    """
    if not movie_path:
        return None, None
    base = os.path.splitext(movie_path)[0]            # .../<stem>_mc
    summary, shifts = None, None
    jf = base + ".json"
    if os.path.isfile(jf):
        try:
            with open(jf) as fh:
                summary = json.load(fh)
        except Exception as e:
            logger.warning(f"  could not read motion summary {jf}: {e}")
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


def _compute_dff(cfg, traces):
    """Baseline-correct raw fluorescence traces to dF/F0 per the config."""
    b = cfg.baseline
    fr = cfg.imaging.frame_rate
    if b.method == "direct":
        # OASIS receives raw traces; still hand a dF/F0-ish array to the gallery.
        from orcann.activity.baseline import compute_dff_traces
        c_dff, c_raw, _ = compute_dff_traces(traces, frame_rate=fr,
                                             percentile=b.percentile)
        return c_dff, c_raw
    if b.method == "local_background":
        from orcann.activity.baseline import compute_dff_local_background
        c_dff, c_raw, _ = compute_dff_local_background(traces, frame_rate=fr)
        return c_dff, c_raw
    from orcann.activity.baseline import compute_dff_traces
    c_dff, c_raw, _ = compute_dff_traces(
        traces, frame_rate=fr, percentile=b.percentile,
        window_fraction=b.window_fraction, min_window=b.min_window,
        max_window=b.max_window, presmooth_sigma=b.presmooth_sigma)
    return c_dff, c_raw


def _deconvolve(cfg, c_dff):
    """OASIS spike inference on dF/F0 traces; returns (denoised, spikes, noise)."""
    if not cfg.deconvolution.enabled:
        return None, None, None
    from orcann.activity.deconvolution import deconvolve_traces
    d = cfg.deconvolution
    res = deconvolve_traces(
        c_dff, frame_rate=cfg.imaging.frame_rate, decay_time=cfg.decay_time(),
        method=d.method, penalty=d.penalty, optimize_g=d.optimize_g,
        noise_method=d.noise_method, s_min=d.s_min,
        noise_gate_sigma=d.noise_gate_sigma,
        robust_safety_net=d.robust_safety_net,
        robust_k_onset=d.robust_k_onset, robust_k_peak=d.robust_k_peak,
        robust_min_duration_s=d.robust_min_duration_s)
    return res.get("C_denoised"), res.get("S"), res.get("noise")


def _write_outputs(out_dir, rec_id, cfg, *, c_dff, c_raw, denoised, spikes,
                   noise, labels, mean_proj, max_proj, source,
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

    run_info = {
        "recording_id": rec_id,
        "stage": "activity (baseline + deconvolution)",
        "frame_rate": float(cfg.imaging.frame_rate),
        "indicator": cfg.imaging.indicator,
        "decay_time_s": cfg.decay_time(),
        "dims": [int(H), int(W)],
        "d1": int(H), "d2": int(W),
        "n_roi": int(c_dff.shape[0]),
        "n_frames": int(c_dff.shape[1]),
        "baseline": {"method": cfg.baseline.method, "percentile": cfg.baseline.percentile},
        "deconvolution": {"enabled": cfg.deconvolution.enabled,
                          "method": cfg.deconvolution.method},
        "amplitude_method": cfg.baseline.method,
        "source": os.path.abspath(source) if source else None,
    }
    if motion:
        run_info["motion_correction"] = motion
    if global_intensity is not None:
        run_info["global_intensity"] = global_intensity
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=2)


def _write_galleries(cfg, out_dir, rec_id, movie, labels, centroids, max_proj,
                     c_dff, c_raw, denoised, spikes, noise):
    """Interactive (and optional movie) HTML gallery for one recording."""
    g = cfg.gallery
    if not (g.interactive or g.movie):
        return
    from orcann.activity.roi_adapter import build_seed_view, build_projections
    seeds = build_seed_view(labels, max_projection=max_proj, centroids=centroids)
    if g.interactive:
        try:
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
    if g.movie:
        try:
            from orcann.activity.movie_gallery import generate_movie_gallery
            generate_movie_gallery(
                movie, seeds, out_dir, frame_rate=cfg.imaging.frame_rate,
                subsample=g.movie_subsample, max_rois=g.max_rois,
                title=f"{rec_id} - Movie Gallery",
                traces_denoised=denoised, spike_trains=spikes,
                movie_processed=movie, deconv_noise=noise)
            logger.info("  movie gallery written")
        except Exception as e:
            logger.warning(f"  movie gallery failed: {e}")


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
        denoised, spikes, noise = _deconvolve(cfg, c_dff)

        motion, motion_shifts = _motion_meta(mv)
        _write_outputs(out_dir, rec_id, cfg, c_dff=c_dff, c_raw=c_raw,
                       denoised=denoised, spikes=spikes, noise=noise,
                       labels=labels, mean_proj=mean_proj, max_proj=max_proj,
                       source=mv, motion=motion, motion_shifts=motion_shifts,
                       global_intensity=gi)
        _write_galleries(cfg, out_dir, rec_id, movie, labels, centroids, max_proj,
                         c_dff, c_raw, denoised, spikes, noise)

        n_spk = int((spikes > 0).sum()) if spikes is not None else 0
        print(f"{rec_id:28s} {int(c_dff.shape[0]):6d} cells  {n_spk:8d} spikes")
    print(f"activity outputs -> {out}/<recording_id>/  (now run: analysis)")
