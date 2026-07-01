"""Temporal detection stage: results/spatial -> results/transients.

Runs the trained temporal model over each recording's traces (from the spatial
stage) and writes the canonical <transients>/<recording_id>/ folder: data/{traces,
rates, events, meta} and figures/roi_<i>.png. Recordings whose events already
exist are skipped unless force=True.
"""
import os

from orcann.pipeline import inference as infer
from orcann.pipeline.cli import list_spatial_recordings
from orcann.pipeline.model_io import load_model


def run(cfg, task_id=None, force=False):
    if cfg.models.temporal is None:
        raise SystemExit("set models.temporal in the config")
    spatial_dir, out = cfg.paths.spatial, cfg.paths.transients
    tp, fg = cfg.temporal, cfg.figures
    recs = list_spatial_recordings(spatial_dir, task_id)
    if not recs:
        print(f"detect_transients: no spatial outputs in {spatial_dir} "
              f"(run segment first)"); return

    det = tp.detection()
    model = load_model(cfg.models.temporal, map_location="cpu")
    os.makedirs(out, exist_ok=True)
    models = {"temporal": os.path.abspath(cfg.models.temporal)}

    print(f"detect_transients: {len(recs)} recording(s)  {spatial_dir} -> {out}")
    print(f"{'recording':28s} {'ROIs':>6} {'events':>7}")
    for rec_id in recs:
        if os.path.exists(os.path.join(out, rec_id, "data", "events.npz")) and not force:
            print(f"{rec_id:28s} {'(exists, skipped)':>14}")
            continue
        traces_path = os.path.join(spatial_dir, rec_id, "data", "traces.npy")
        traces = infer.load_traces(traces_path)
        rates, events = infer.detect_all(model, traces, tp.frame_rate, **det)
        rec_dir = infer.write_recording(
            out, rec_id, traces=traces, rates=rates, events=events,
            frame_rate=tp.frame_rate, detection=det,
            stage="temporal (transient detection)", models=models, source=traces_path)
        if fg.enabled:
            infer.write_figures(rec_dir, model, traces, rates, tp.frame_rate, det,
                                max_roi_figures=fg.max_roi_figures)
        print(f"{rec_id:28s} {traces.shape[0]:6d} {len(events['roi']):7d}")
    print(f"transient outputs -> {out}/<recording_id>/")
