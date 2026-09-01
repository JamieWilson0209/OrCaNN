"""Inference stage (GPU): a motion-correction store -> results/infer/<store>/<rec>/.

The parameter-independent half of spatial detection. Runs the segmenter once per
recording and caches the soma-probability map + max projection, so threshold /
min_radius tuning (the `segment` stage) never re-runs the GPU. When figures are
enabled it also writes prob_overlay.png, a QC image of the gamma-stretched
probability map over a translucent max projection.

The model that ran is part of the output store's name, so promoting a different
model does not overwrite these maps and does not leave them looking current: they
are a different store, and this one is still there under its own name. A
recording whose record is already in the store for these settings is skipped
unless force=True.
"""
import os

import numpy as np

from orcann.pipeline import inference as infer
from orcann.pipeline import provenance as prov
from orcann.pipeline.cli import store_movies
from orcann.pipeline.model_io import load_model
from orcann.run_info import write_record


def _pick_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(cfg, task_id=None, force=False):
    sp = cfg.spatial
    try:
        chain = prov.chain(cfg)
    except prov.ProvenanceError as e:
        raise SystemExit(str(e))
    pairs = store_movies(chain.motion_dir, chain.motion, task_id)
    if not pairs:
        print(f"infer: no corrected recordings in {chain.motion_dir}")
        return False

    out = prov.store_dir(cfg.paths.infer, chain.infer)
    os.makedirs(out, exist_ok=True)
    print(f"infer: model {chain.model}  |  {len(pairs)} recording(s)  "
          f"{chain.motion_dir} -> {out}")

    # Which recordings actually need the GPU, decided from the records on disk.
    # The model is loaded only if the answer is "some": a fully current re-run
    # used to occupy the queue with a model on the device purely to print skip
    # lines.
    todo = [(f, i) for f, i in pairs
            if force or not prov.is_done(
                os.path.join(out, i, infer.META_JSON), prov.STAGE_INFER)]
    for f, ident in pairs:
        if (f, ident) not in todo:
            print(f"{ident:32s} (current, skipped)")
    if not todo:
        print(f"every recording is current in {chain.infer}; nothing to do")
        return True

    device = _pick_device()
    model = load_model(chain.model_path).to(device)
    print(f"  device {device}  |  {len(todo)} to compute")
    for f, rec_id in todo:
        d = os.path.join(out, rec_id)
        prob, maxproj, prov_meta = infer.infer_prob(
            f, model, resize_to=sp.resize_to, train_um_override=sp.train_um_per_px)
        n_frames = prov_meta["n_frames"]
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, infer.PROB_NPY), prob)
        np.save(os.path.join(d, infer.MAXPROJ_NPY), maxproj)
        if cfg.figures.enabled:
            from orcann.pipeline.figures import prob_overlay_figure
            prob_overlay_figure(os.path.join(d, "prob_overlay.png"), prob, maxproj,
                                title=f"{rec_id}: soma probability over max projection")
        # What the recording said about itself and which resampling rule ran, both
        # from the call that did the work. Whether a movie was rescaled, and by how
        # much, cannot be read back off prob.npy; unrecorded it is unanswerable.
        # Written last, so a run killed mid-recording leaves no record and this
        # recording is recomputed rather than half-trusted.
        write_record(os.path.join(d, infer.META_JSON), prov.STAGE_INFER,
                     dict(prov_meta,
                          recording_id=rec_id,
                          upstream_store=chain.motion,
                          stage_version=prov.INFER_VERSION,
                          working_hw=[int(prob.shape[0]), int(prob.shape[1])],
                          model_identity=chain.model,
                          model_digest=chain.model_digest,
                          model_pixel_um=getattr(model, "config", {}).get("pixel_um"),
                          model_train_hw=getattr(model, "config", {}).get("train_hw")))
        print(f"{rec_id:32s} prob {prob.shape[0]}x{prob.shape[1]}  T={n_frames}")
    print(f"cached probability maps -> {out}/<recording>/  (now run: segment)")
    return True
