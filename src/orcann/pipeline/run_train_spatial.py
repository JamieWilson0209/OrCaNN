"""Training stage: fit the spatial segmenter (drives orcann.spatial.training).

Reads the train_spatial section of the config (movies, masks, out, report, and
the training knobs). Masks are instance-label .npy (from rasterize_rois) or ImageJ
ROI sets, paired to movies by an exactly equal filename stem: blind_0001.tif
pairs with blind_0001.zip and with nothing else. Use synthetic=True for a self-test.
"""
import glob
import os

import numpy as np

from orcann.spatial import (
    train_segmenter, load_seg_recording, synthetic_sources,
    predict_prob, best_iou, SegRecording)
from orcann.pipeline.model_io import save_trained_model
from orcann.spatial.detection.laplacian import EDGE_CORRECTIONS


def find_pairs(movies_dir, masks_dir):
    """Movies and masks sharing a filename stem, as (movie, mask) paths.

    The stems must be equal character for character: the mask for blind_0001.tif
    is blind_0001.npy or blind_0001.zip. A decorated name -- blind_0001_allroi.zip,
    the form ImageJ writes a saved ROI set under -- is a different stem and pairs
    with nothing, so rename it or run rasterize_rois, which names its output after
    the movie it matched.
    """
    movies = _stems(movies_dir, ("*.tif", "*.tiff"))
    masks = _stems(masks_dir, ("*.npy", "*.zip"))
    # sorted, so the seeded shuffle below splits train/val the same way on any
    # machine; glob order follows the filesystem
    return [(movies[s], masks[s]) for s in sorted(movies.keys() & masks.keys())]


def _stems(dirpath, patterns):
    """Filename stem -> path, for every file in dirpath matching patterns."""
    found = {}
    for pat in patterns:
        for p in glob.glob(os.path.join(dirpath, pat)):
            found[os.path.splitext(os.path.basename(p))[0]] = p
    return found


def _unpaired_message(movies_dir, masks_dir):
    """What to say when the directories hold files but share no stem. The cause
    is usually a suffix on the mask names, which pairing does not strip, so the
    message shows a name from each side rather than only the counts."""
    movies = sorted(_stems(movies_dir, ("*.tif", "*.tiff")))
    masks = sorted(_stems(masks_dir, ("*.npy", "*.zip")))
    lines = [f"no movie/mask pairs: {len(movies)} movie(s) in {movies_dir}, "
             f"{len(masks)} mask(s) in {masks_dir}, no stem in common."]
    if not movies:
        lines.append(f"  {movies_dir} holds no .tif/.tiff")
    elif not masks:
        lines.append(f"  {masks_dir} holds no .npy/.zip")
    else:
        lines.append(f"  a movie stem: {movies[0]}")
        lines.append(f"  a mask stem:  {masks[0]}")
        lines.append("A mask pairs only with the movie whose stem it equals "
                     "exactly. Rename the masks to match, or run "
                     "orcann.spatial.training.rasterize_rois, which names its "
                     "output after the movie it matched.")
    return "\n".join(lines)


def run(cfg, synthetic=False):
    t = cfg.train_spatial
    channels = {"use_structural": False, "use_max": False,
                "use_variance": False, "use_correlation": False}
    channels.update({f"use_{c}": True for c in t.channels})
    radii = tuple(float(x) for x in t.radii)
    if not radii:
        raise SystemExit("train_spatial.radii is empty")
    if t.n_patch < 1:
        raise SystemExit(f"train_spatial.n_patch is {t.n_patch}; it must be at least 1")
    if not 0.0 <= t.fg_frac <= 1.0:
        raise SystemExit(f"train_spatial.fg_frac is {t.fg_frac}; it is a fraction, 0 to 1")
    if t.edge_correction not in EDGE_CORRECTIONS:
        raise SystemExit(
            f"train_spatial.edge_correction is {t.edge_correction!r}; it must be "
            f"one of {', '.join(EDGE_CORRECTIONS)}")
    if t.n_energy_frames is not None and t.n_energy_frames < 1:
        raise SystemExit(
            f"train_spatial.n_energy_frames is {t.n_energy_frames}; it must be at "
            "least 1, or null to pool every frame")
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
            raise SystemExit(_unpaired_message(t.movies, t.masks))
        loader = load_seg_recording
        if t.min_cell_area > 0:
            from functools import partial
            loader = partial(load_seg_recording, min_area=t.min_cell_area)
        sources = pairs
        print(f"{len(pairs)} recordings")

    idx = list(range(len(sources))); rng.shuffle(idx)
    if t.holdout:
        n_val = max(1, int(len(idx) * t.val_frac))
        train_i, val_i = idx[n_val:], idx[:n_val]
    else:
        train_i, val_i = idx, []
        print(f"no-holdout: training final model on all {len(idx)} recordings")

    out_dir = t.out or "/tmp/seg_synth"
    os.makedirs(out_dir, exist_ok=True)
    # The per-epoch checkpoint is scratch and goes somewhere of its own. It used
    # to be written over the model the pipeline was running, so a job killed at
    # any epoch left an under-trained model in service with the previous one
    # already gone.
    checkpoint_path = t.checkpoint or None

    def fit(train_i, val_i):
        # Progress frames are named for the run, not for that scratch checkpoint,
        # whose path defaults to one _in_progress.pt every run overwrites. The
        # run's own output directory is named for an identity that does not exist
        # until the fit finishes, which leaves train_spatial.name -- the same
        # component that identity is built from.
        progress_dir = None
        if t.progress_frames:
            progress_dir = os.path.join(t.out or ".", f"{t.name}_progress")
        return train_segmenter([sources[i] for i in train_i], channels=channels,
                               radii_px=radii, patch=t.patch, n_patch=t.n_patch,
                               fg_frac=t.fg_frac, n_energy_frames=t.n_energy_frames,
                               epochs=t.epochs, loader=loader,
                               edge_correction=t.edge_correction,
                               val_sources=[sources[i] for i in val_i],
                               progress_dir=progress_dir,
                               checkpoint_path=checkpoint_path)

    def score(model, val_i):
        rows = []
        for i in val_i:
            src = sources[i]
            rec = src if isinstance(src, SegRecording) else loader(*src)
            prob = predict_prob(model, rec.movie)
            iou, thr = best_iou(prob, (rec.label > 0))
            rows.append({"recording": os.path.basename(rec.rid),
                         "iou": float(iou), "threshold": float(thr),
                         "n_cells": int(len(rec.centroids))})
            print(f"  {os.path.basename(rec.rid):28s} IoU {iou:.3f} @thr {thr:.2f}  "
                  f"{rows[-1]['n_cells']} annotated cells")
        return rows

    model = fit(train_i, val_i)
    per_recording = score(model, val_i)

    metrics = {"channels": list(t.channels), "radii": list(radii),
               "n_recordings": len(sources),
               "n_train": len(train_i), "n_val": len(val_i),
               "held_out": t.holdout,
               "patch": t.patch, "epochs": t.epochs, "synthetic": bool(synthetic),
               "edge_correction": t.edge_correction, "name": t.name}
    if per_recording:
        ious = [r["iou"] for r in per_recording]
        metrics["val_iou_mean"] = float(np.mean(ious))
        metrics["val_iou_std"] = float(np.std(ious))
        metrics["val_iou_per_recording"] = per_recording
        print(f"mean IoU (best threshold) over {len(ious)} recordings: "
              f"{metrics['val_iou_mean']:.3f} +/- {metrics['val_iou_std']:.3f}")
    else:
        print("final model trained on all data; assess from downstream results.")
    # Written once, under an identity derived from what it is, beside its own
    # report. This is train_spatial.out, not models.dir: finishing a run does not
    # put a model into service -- promoting it does.
    identity = save_trained_model(model, out_dir, t.name, cfg.run.key,
                                  report=metrics)
    print(f"trained model -> {os.path.join(out_dir, identity)}")
    print(f"  promote it into {cfg.models.dir} to run it "
          f"(models.spatial: {identity}, or 'latest')")
    return True
