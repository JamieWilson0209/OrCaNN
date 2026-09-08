"""Inference stage (GPU): a motion-correction store -> results/infer/<store>/<rec>/.

The parameter-independent half of spatial detection. Runs the segmenter once per
recording and caches the soma-probability map + max projection, so threshold /
min_radius tuning (the `segment` stage) never re-runs the GPU. When figures are
enabled it also writes prob_overlay.png, a QC image of the gamma-stretched
probability map over a translucent max projection.

The model that ran is part of the key recorded beside every map, so promoting a
different model leaves these maps reading as out of date and they are recomputed
rather than mistaken for the new model's. A recording whose record was written
under the settings in this config is skipped unless force=True.
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
        keys = prov.stage_keys(cfg, chain)
    except prov.ProvenanceError as e:
        raise SystemExit(str(e))
    found = store_movies(chain.motion_dir, chain.motion_external)
    triples, surplus = prov.for_task(found, task_id)
    if surplus:
        print(f"infer: task {task_id} is past the end of {len(found)} "
              f"recording(s); nothing to do")
        return True
    if not triples:
        print(f"infer: no corrected recordings in {chain.motion_dir}")
        return False

    out = prov.output_store(cfg.paths.infer, prov.STAGE_INFER, chain.run_key,
                            keys[prov.STAGE_INFER], infer.META_JSON)
    os.makedirs(out, exist_ok=True)
    print(f"infer: model {chain.model}  |  {len(triples)} recording(s)  "
          f"{chain.motion_dir} -> {out}")

    # Which recordings actually need the GPU, decided from the records on disk.
    # The model is loaded only if the answer is "some": a fully current re-run
    # used to occupy the queue with a model on the device purely to print skip
    # lines.
    todo = []
    for f, rec_id, tag in triples:
        key = dict(keys[prov.STAGE_INFER], recording={"content_tag": tag})
        ok, why = prov.is_current(os.path.join(out, rec_id, infer.META_JSON),
                                  prov.STAGE_INFER, key)
        if ok and not force:
            print(f"{rec_id:32s} (current, skipped)")
            continue
        if not ok and why != "no record":
            print(f"{rec_id:32s} ({why})")
        todo.append((f, rec_id, key))
    if not todo:
        print(f"every recording is current in {out}; nothing to do")
        return True

    device = _pick_device()
    model = load_model(chain.model_path).to(device)
    print(f"  device {device}  |  {len(todo)} to compute")
    for f, rec_id, key in todo:
        d = os.path.join(out, rec_id)
        prob, maxproj, prov_meta = infer.infer_prob(f, model,
                                                    resize_to=sp.resize_to)
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
        write_record(os.path.join(d, infer.META_JSON), prov.STAGE_INFER, key,
                     chain.run_key,
                     dict(prov_meta,
                          recording_id=rec_id,
                          upstream_store=chain.motion_dir,
                          working_hw=[int(prob.shape[0]), int(prob.shape[1])],
                          model_identity=chain.model,
                          model_digest=chain.model_digest,
                          model_train_hw=getattr(model, "config", {}).get("train_hw")))
        print(f"{rec_id:32s} prob {prob.shape[0]}x{prob.shape[1]}  T={n_frames}")
    print(f"cached probability maps -> {out}/<recording>/  (now run: segment)")
    return True
