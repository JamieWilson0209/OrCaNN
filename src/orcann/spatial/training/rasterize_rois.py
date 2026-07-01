#!/usr/bin/env python
"""Rasterise ImageJ ROI sets (the '*_allroi.zip' freehand footprints) into
per-recording INSTANCE LABEL images for segmentation training.

Each output is an (H, W) int32 label image: 0 = background, k = the k-th ROI.
Foreground mask is simply (label > 0); instance id is preserved so touching
cells can be separated later (e.g. seeded watershed from the centroid list).

ImageJ stores x=col, y=row; that mapping is applied explicitly. ROIs were
traced on the downscaled projection (~512 frame). If your movies are at a
different resolution, pass --movies (or --target-dim) and the vector ROIs are
scaled to the movie grid BEFORE rasterising, so labels and movie align.

    python scripts/rasterize_rois.py --rois ROIs --out masks --src-dim 512 \
        --movies movies --check qc
"""
import argparse, glob, os
import numpy as np
import roifile
from skimage.draw import polygon as skpoly
from scipy.ndimage import label as cc_label


def movie_hw(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0].shape
    return (s[-2], s[-1])


def match_movie(stem, movies):
    base = stem.replace("_allroi", "")
    for ms, p in movies.items():
        if ms.lower() == base.lower() or base.lower() in ms.lower() or ms.lower() in base.lower():
            return ms, p
    return None, None


def rasterize(rois, H, W, scale):
    inst = np.zeros((H, W), np.int32)
    overlap = np.zeros((H, W), np.int16)
    n = 0
    for r in rois:
        xy = np.asarray(r.coordinates(), float) * scale
        if xy.ndim != 2 or len(xy) < 3:
            continue
        n += 1
        rr, cc = skpoly(xy[:, 1], xy[:, 0], shape=(H, W))   # rows=y, cols=x
        inst[rr, cc] = n
        overlap[rr, cc] += 1
    return inst, overlap, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rois", required=True, help="dir of *_allroi.zip ROI sets")
    ap.add_argument("--out", required=True, help="dir for <stem>.npy instance labels")
    ap.add_argument("--src-dim", type=float, default=512.0,
                    help="frame size the ROIs were drawn on (default 512)")
    ap.add_argument("--movies", default=None, help="movies dir; sets target res + naming")
    ap.add_argument("--target-dim", type=int, default=None,
                    help="force output frame size (square) if no movies given")
    ap.add_argument("--check", default=None, help="dir for QC overlays")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.check:
        os.makedirs(a.check, exist_ok=True)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    movies = {}
    if a.movies:
        for m in glob.glob(os.path.join(a.movies, "*.tif")) + glob.glob(os.path.join(a.movies, "*.tiff")):
            movies[os.path.splitext(os.path.basename(m))[0]] = m

    files = sorted(f for f in glob.glob(os.path.join(a.rois, "*.zip"))
                   if "__MACOSX" not in f)
    print(f"{'recording':16s} {'rois':>5} {'HxW':>11} {'scale':>6} {'fg%':>6} {'touch':>6} {'rmed':>5}")
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        rois = roifile.roiread(f)
        if not isinstance(rois, (list, tuple)):
            rois = [rois]
        out_stem = stem.replace("_allroi", "")
        scale, H, W = 1.0, None, None
        if movies:
            ms, mp = match_movie(stem, movies)
            if mp:
                H, W = movie_hw(mp); scale = ((H + W) / 2.0) / a.src_dim
                out_stem = ms
        if H is None:
            D = a.target_dim or int(round(a.src_dim))
            H = W = D
            scale = D / a.src_dim if a.target_dim else 1.0

        inst, overlap, n = rasterize(rois, H, W, scale)
        np.save(os.path.join(a.out, out_stem + ".npy"), inst)
        fg = inst > 0
        areas = np.bincount(inst.ravel())[1:]
        rmed = float(np.median(np.sqrt(areas / np.pi))) if len(areas) else 0.0
        _, nblob = cc_label(fg)
        touch = n - nblob
        print(f"{out_stem:16s} {n:5d} {f'{H}x{W}':>11} {scale:6.2f} "
              f"{100*fg.mean():6.2f} {touch:6d} {rmed:5.1f}")

        if a.check:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(fg, cmap="gray")
            ax.set_title(f"{out_stem}: {n} footprints @ {H}x{W}"); ax.axis("off")
            plt.tight_layout(); plt.savefig(os.path.join(a.check, out_stem + "_mask.png"), dpi=100)
            plt.close(fig)

    print(f"\ninstance labels -> {a.out}  (foreground = label>0; ids preserved for watershed)")


if __name__ == "__main__":
    main()
