#!/usr/bin/env python
"""Train the spatial scattering detector on annotated recordings.

Pairs movies and ROI annotations by filename stem, trains the chosen channel
configuration, evaluates on a held-out split, and saves the model + a metrics
report. Use --synthetic for an end-to-end self-test with no data on disk.

DATA-INTAKE: movie<->ROI pairing assumes matching filename stems
(movies/<stem>.npy  <->  rois/<stem>.npy|.tif|.csv). Adjust if your naming
differs, and confirm the ROI format is handled by orcann loader.
"""
import argparse, glob, json, os

import numpy as np
import torch

from orcann.train_spatial import (
    load_recording, synthetic_annotated_bank, train_detector, evaluate,
    radius_bank_from_recordings, _load_annotation, AnnotatedRecording,
)


def parse_channels(s: str) -> dict:
    keys = {"structural": "use_structural", "max": "use_max",
            "variance": "use_variance", "corr": "use_correlation",
            "coherence": "use_correlation"}
    chosen = {"use_structural": False, "use_max": False,
              "use_variance": False, "use_correlation": False}
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok:
            chosen[keys[tok]] = True
    return chosen


def find_pairs(movies_dir, rois_dir):
    """(movie_path, roi_path) pairs by matching filename stem — NOT loaded."""
    pairs = []
    for mv in sorted(glob.glob(os.path.join(movies_dir, "*"))):
        stem = os.path.splitext(os.path.basename(mv))[0]
        roi = next(iter(sorted(glob.glob(os.path.join(rois_dir, stem + ".*")))), None)
        if roi is None:
            print(f"  WARNING: no ROI for {stem}, skipping"); continue
        pairs.append((mv, roi))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies"); ap.add_argument("--rois")
    ap.add_argument("--channels", default="structural,max,variance")
    ap.add_argument("--radii", default="auto")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--n-patch", type=int, default=16)
    ap.add_argument("--n-energy-frames", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default="models/spatial")
    ap.add_argument("--report", default=None)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    if a.synthetic:
        sources = synthetic_annotated_bank(n_recordings=5)
        loader = None
        n_cells = sum(len(r.centroids) for r in sources)
    else:
        sources = find_pairs(a.movies, a.rois)        # streamed, not loaded
        loader = load_recording
        n_cells = sum(len(_load_annotation(roi, (1, 1), 6.0)[0]) for _, roi in sources)
    print(f"{len(sources)} recordings, {n_cells} annotated cells")

    n_val = max(1, int(len(sources) * a.val_frac))
    train, val = sources[:-n_val], sources[-n_val:]

    channels = parse_channels(a.channels)
    if a.radii.strip().lower() == "auto":
        if a.synthetic:
            radii = radius_bank_from_recordings(train)
        else:                                         # radii from ROI files only
            stub = [AnnotatedRecording(np.zeros((1, 1, 1), np.float32),
                                       *_load_annotation(roi, (1, 1), 6.0), recording_id=roi)
                    for _, roi in train]
            radii = radius_bank_from_recordings(stub)
        print(f"radius bank from annotations: {radii}")
    else:
        radii = tuple(float(x) for x in a.radii.split(","))

    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.out, "detector.pt")
    model = train_detector(train, channels=channels, radii_px=radii, epochs=a.epochs,
                           patch=a.patch, n_patch=a.n_patch,
                           n_energy_frames=a.n_energy_frames,
                           loader=loader, checkpoint_path=ckpt)
    metrics = evaluate(model, val, loader=loader)
    print("held-out:", metrics)

    torch.save(model, ckpt)
    if a.report:
        os.makedirs(os.path.dirname(a.report), exist_ok=True)
        with open(a.report, "w") as f:
            json.dump({"channels": channels, "radii": list(radii), **metrics}, f, indent=2)
    print(f"saved model to {ckpt}")


if __name__ == "__main__":
    main()
