"""Training stage: fit the spatial segmenter (drives orcann.spatial.training).

Reads the train_spatial section of the config (movies, masks, out, report, and
the training knobs). Masks are instance-label .npy (from rasterize_rois) or ImageJ
ROI sets, paired to movies by filename stem. Use synthetic=True for a self-test.
"""
import glob
import json
import os

import numpy as np

from orcann.spatial import (
    train_segmenter, load_seg_recording, synthetic_sources,
    predict_prob, segment_instances, best_iou, SegRecording)
from orcann.pipeline.model_io import save_model


def find_pairs(movies_dir, masks_dir):
    pairs = {}
    for m in glob.glob(os.path.join(movies_dir, "*.tif")) + \
             glob.glob(os.path.join(movies_dir, "*.tiff")):
        pairs.setdefault(os.path.splitext(os.path.basename(m))[0], [None, None])[0] = m
    for a in glob.glob(os.path.join(masks_dir, "*.npy")) + \
             glob.glob(os.path.join(masks_dir, "*.zip")):
        st = os.path.splitext(os.path.basename(a))[0]
        if st in pairs:
            pairs[st][1] = a
    return [(v[0], v[1]) for v in pairs.values() if v[0] and v[1]]


def run(cfg, synthetic=False):
    t = cfg.train_spatial
    channels = {"use_structural": False, "use_max": False,
                "use_variance": False, "use_correlation": False}
    channels.update({f"use_{c}": True for c in t.channels})
    radii = tuple(float(x) for x in t.radii)
    if not radii:
        raise SystemExit("train_spatial.radii is empty")
    print(f"LoG bank: {len(radii)} scale(s) -> radii_px={list(radii)}")

    rng = np.random.default_rng(0)
    if synthetic:
        sources, loader = synthetic_sources(), None
    else:
        for k in ("movies", "masks", "out"):
            if getattr(t, k) is None:
                raise SystemExit(f"set train_spatial.{k} in the config (or run --synthetic)")
        pairs = find_pairs(t.movies, t.masks)
        if not pairs:
            print("no movie/mask pairs found"); return
        loader = load_seg_recording
        if t.min_cell_area > 0:
            from functools import partial
            loader = partial(load_seg_recording, min_area=t.min_cell_area)
        sources = pairs
        print(f"{len(pairs)} recordings")

    idx = list(range(len(sources))); rng.shuffle(idx)
    if not t.holdout:
        train_i, val_i = idx, []
        print(f"no-holdout: training final model on all {len(idx)} recordings")
    else:
        n_val = max(1, int(len(idx) * t.val_frac))
        val_i, train_i = idx[:n_val], idx[n_val:]
    train = [sources[i] for i in train_i]
    val = [sources[i] for i in val_i]

    out_dir = t.out or "/tmp/seg_synth"
    os.makedirs(out_dir, exist_ok=True)
    model = train_segmenter(train, channels=channels, radii_px=radii,
                            patch=t.patch, epochs=t.epochs, loader=loader,
                            pixel_um=t.pixel_um,
                            checkpoint_path=os.path.join(out_dir, "segmenter.pt"))
    save_model(model, os.path.join(out_dir, "segmenter.pt"))

    metrics = {"channels": list(t.channels), "radii": list(radii),
               "n_train": len(train), "held_out": t.holdout}
    if val:
        ious = []
        for s in val:
            rec = s if isinstance(s, SegRecording) else loader(*s)
            prob = predict_prob(model, rec.movie)
            iou, thr = best_iou(prob, (rec.label > 0))
            ious.append(iou)
            inst = segment_instances(prob, rec.centroids, threshold=thr)
            print(f"  {rec.rid:28s} IoU {iou:.3f} @thr {thr:.2f}  instances pred/true "
                  f"{int(inst.max())}/{len(rec.centroids)}")
        metrics["val_iou_mean"] = float(np.mean(ious))
        print("held-out mean IoU (best threshold):", round(metrics["val_iou_mean"], 3))
    else:
        print("final model trained on all data; assess from downstream results.")
    if t.report:
        os.makedirs(os.path.dirname(t.report) or ".", exist_ok=True)
        with open(t.report, "w") as f:
            json.dump(metrics, f, indent=2)
