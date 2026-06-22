#!/usr/bin/env python
"""Train the spatial SEGMENTER (per-pixel soma probability).

    # synthetic self-test
    python scripts/run_train_seg.py --synthetic --epochs 30 --out models/seg

    # real data: movies/<stem>.tif paired with masks/<stem>.npy
    # (instance labels from scripts/rasterize_rois.py)
    python scripts/run_train_seg.py --movies movies --masks masks \
        --channels structural,max,variance --radii 5,7,9,13 \
        --out models/seg --report models/seg/report.json
"""
import argparse, glob, json, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orcann.spatial_seg import (                                    # noqa: E402
    train_segmenter, load_seg_recording, synthetic_sources,
    predict_prob, segment_instances, best_iou, SegRecording)
from orcann.io import save_model                                    # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--movies"); ap.add_argument("--masks")
    ap.add_argument("--channels", default="structural,max,variance")
    ap.add_argument("--radii", default="3:3.7:4.5:5.5:6.7:8.2:10")
    ap.add_argument("--min-cell-area", type=int, default=0,
                    help="strip ROIs smaller than this many px from the TRAINING "
                         "masks so the model never learns to label tiny spots")
    ap.add_argument("--pixel-um", type=float, default=None,
                    help="microns/px of the training movies; recorded in the model "
                         "so inference can auto-rescale new recordings to match")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-holdout", action="store_true",
                    help="train the FINAL model on ALL recordings (no held-out "
                         "split, no validation eval); assess from downstream results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    flags = {f"use_{c.strip()}": True for c in a.channels.split(",")}
    channels = {"use_structural": False, "use_max": False,
                "use_variance": False, "use_correlation": False, **flags}
    # accept comma, colon, or whitespace separated radii. qsub -v splits on
    # commas, so a multi-value bank must be submitted colon-separated; tolerating
    # both here means the job parses whichever form arrives, with no shell rewrite.
    radii = tuple(float(x) for x in re.split(r"[,:\s]+", a.radii.strip()) if x)
    if len(radii) < 1:
        raise SystemExit(f"--radii parsed to an empty bank from {a.radii!r}")
    print(f"LoG bank: {len(radii)} scale(s) -> radii_px={list(radii)}")

    rng = np.random.default_rng(0)
    if a.synthetic:
        recs = synthetic_sources()
        loader = None
        sources = recs
    else:
        pairs = find_pairs(a.movies, a.masks)
        if not pairs:
            print("no movie/mask pairs found"); return
        loader = load_seg_recording
        if a.min_cell_area > 0:
            from functools import partial
            loader = partial(load_seg_recording, min_area=a.min_cell_area)
        sources = pairs
        print(f"{len(pairs)} recordings")

    idx = list(range(len(sources))); rng.shuffle(idx)
    if a.no_holdout:
        train_i, val_i = idx, []                      # final model: use everything
        print(f"no-holdout: training final model on all {len(idx)} recordings")
    else:
        n_val = max(1, int(len(idx) * a.val_frac))
        val_i, train_i = idx[:n_val], idx[n_val:]
    train = [sources[i] for i in train_i]
    val = [sources[i] for i in val_i]

    model = train_segmenter(train, channels=channels, radii_px=radii,
                            patch=a.patch, epochs=a.epochs, loader=loader,
                            pixel_um=a.pixel_um,
                            checkpoint_path=os.path.join(a.out, "segmenter.pt"))
    save_model(model, os.path.join(a.out, "segmenter.pt"))

    metrics = {"channels": a.channels, "radii": list(radii),
               "n_train": len(train), "held_out": not a.no_holdout}
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
        print("final model trained on all data; assess performance from "
              "downstream inference results.")
    if a.report:
        with open(a.report, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
