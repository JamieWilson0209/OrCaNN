"""Inference stage (GPU): data/pre_processed -> results/infer/<rec>/.

The parameter-independent half of spatial detection. Runs the segmenter once per
recording and caches the soma-probability map + max projection, so threshold /
min_radius tuning (the `segment` stage) never re-runs the GPU. When figures are
enabled it also writes prob_overlay.png, a QC image of the gamma-stretched
probability map over a translucent max projection. Recordings whose prob.npy
already exists are skipped unless force=True.
"""
import json
import os

import numpy as np

from orcann.pipeline import inference as infer
from orcann.pipeline.cli import list_recordings
from orcann.pipeline.model_io import load_model


def _pick_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(cfg, task_id=None, force=False):
    if cfg.models.spatial is None:
        raise SystemExit("set models.spatial in the config")
    pre, out = cfg.paths.pre_processed, cfg.paths.infer
    sp = cfg.spatial
    files = list_recordings(pre, task_id)
    if not files:
        print(f"infer: no recordings in {pre}"); return

    device = _pick_device()
    model = load_model(cfg.models.spatial).to(device)
    os.makedirs(out, exist_ok=True)

    print(f"infer: device {device}  |  {len(files)} recording(s)  {pre} -> {out}")
    for f in files:
        rec_id = infer.recording_id(f)
        d = os.path.join(out, rec_id)
        if os.path.exists(os.path.join(d, infer.PROB_NPY)) and not force:
            print(f"{rec_id:28s} (cached, skipped)")
            continue
        prob, maxproj, n_frames = infer.infer_prob(
            f, model, resize_to=sp.resize_to, train_um_override=sp.train_um_per_px)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, infer.PROB_NPY), prob)
        np.save(os.path.join(d, infer.MAXPROJ_NPY), maxproj)
        with open(os.path.join(d, infer.META_JSON), "w") as fh:
            json.dump({"recording_id": rec_id, "stage": "infer",
                       "working_hw": [int(prob.shape[0]), int(prob.shape[1])],
                       "n_frames": n_frames, "source": os.path.abspath(f),
                       "model": os.path.abspath(cfg.models.spatial)}, fh, indent=2)
        if cfg.figures.enabled:
            from orcann.pipeline.figures import prob_overlay_figure
            prob_overlay_figure(os.path.join(d, "prob_overlay.png"), prob, maxproj,
                                title=f"{rec_id}: soma probability over max projection")
        print(f"{rec_id:28s} prob {prob.shape[0]}x{prob.shape[1]}  T={n_frames}")
    print(f"cached probability maps -> {out}/<recording_id>/  (now run: segment)")
