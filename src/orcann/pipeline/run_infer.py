"""Inference stage (GPU): data/pre_processed -> results/infer/<rec>/.

The parameter-independent half of spatial detection. Runs the segmenter once per
recording and caches the soma-probability map + max projection, so threshold /
min_radius tuning (the `segment` stage) never re-runs the GPU. When figures are
enabled it also writes prob_overlay.png, a QC image of the gamma-stretched
probability map over a translucent max projection. Recordings whose prob.npy
already exists are skipped unless force=True.
"""
import os

import numpy as np

from orcann.pipeline import inference as infer
from orcann.pipeline.cli import list_recordings
from orcann.pipeline.model_io import (
    ModelStoreError, load_model, read_identity, resolve_model)
from orcann.run_info import RunInfoError, read_record, write_record


def _pick_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cached_model(d):
    """(identity, digest) recorded for an existing prob map, or (None, None).

    A record this stage cannot read is treated as no record: the cache then has
    no provenance, so it is rebuilt rather than trusted.
    """
    try:
        rec = read_record(os.path.join(d, infer.META_JSON), expect_stage="infer")
    except (RunInfoError, OSError, ValueError):
        return None, None
    return rec.get("model_identity"), rec.get("model_digest")


def _is_current(d, identity, dgst):
    """Whether a cached prob map was produced by the model about to be used.

    Identity first, because it is a string compare against a record already on
    disk. Digest second, so a model promoted twice under different names -- same
    content, different timestamp -- does not force the whole GPU stage to re-run
    for nothing.
    """
    was_id, was_dgst = _cached_model(d)
    if was_id and was_id == identity:
        return True
    return bool(dgst) and was_dgst == dgst


def run(cfg, task_id=None, force=False):
    pre, out = cfg.paths.pre_processed, cfg.paths.infer
    sp = cfg.spatial
    files = list_recordings(pre, task_id)
    if not files:
        print(f"infer: no recordings in {pre}"); return

    try:
        identity, model_path = resolve_model(cfg.models.dir, cfg.models.spatial)
    except ModelStoreError as e:
        raise SystemExit(str(e))
    ident = read_identity(model_path)
    dgst = ident.get("digest")

    # Which recordings actually need the GPU, decided from records on disk. The
    # model is loaded only if the answer is "some": a fully cached re-run used to
    # occupy the queue with a model on the device purely to print skip lines.
    todo = [f for f in files
            if force or not os.path.exists(
                os.path.join(out, infer.recording_id(f), infer.PROB_NPY))
            or not _is_current(os.path.join(out, infer.recording_id(f)),
                               identity, dgst)]
    os.makedirs(out, exist_ok=True)
    print(f"infer: model {identity}  |  {len(files)} recording(s)  {pre} -> {out}")
    for f in files:
        if f not in todo:
            print(f"{infer.recording_id(f):28s} (cached, skipped)")
    if not todo:
        print(f"every recording is current for {identity}; nothing to do")
        return True

    device = _pick_device()
    model = load_model(model_path).to(device)
    print(f"  device {device}  |  {len(todo)} to compute")
    for f in todo:
        rec_id = infer.recording_id(f)
        d = os.path.join(out, rec_id)
        prob, maxproj, prov = infer.infer_prob(
            f, model, resize_to=sp.resize_to, train_um_override=sp.train_um_per_px)
        n_frames = prov["n_frames"]
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, infer.PROB_NPY), prob)
        np.save(os.path.join(d, infer.MAXPROJ_NPY), maxproj)
        # What the recording said about itself and which resampling rule ran, both
        # from the call that did the work. Whether a movie was rescaled, and by how
        # much, cannot be read back off prob.npy; unrecorded it is unanswerable.
        write_record(os.path.join(d, infer.META_JSON), "infer",
                     dict(prov,
                          recording_id=rec_id,
                          working_hw=[int(prob.shape[0]), int(prob.shape[1])],
                          model_identity=identity,
                          model_digest=dgst,
                          model=os.path.abspath(model_path),
                          model_pixel_um=getattr(model, "config", {}).get("pixel_um"),
                          model_train_hw=getattr(model, "config", {}).get("train_hw")))
        if cfg.figures.enabled:
            from orcann.pipeline.figures import prob_overlay_figure
            prob_overlay_figure(os.path.join(d, "prob_overlay.png"), prob, maxproj,
                                title=f"{rec_id}: soma probability over max projection")
        print(f"{rec_id:28s} prob {prob.shape[0]}x{prob.shape[1]}  T={n_frames}")
    print(f"cached probability maps -> {out}/<recording_id>/  (now run: segment)")
    return True
