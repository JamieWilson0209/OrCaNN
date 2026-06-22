#!/usr/bin/env python
"""Run the trained spatial SEGMENTER on new recordings (.nd2 / .tif / .npy).

No annotations are needed: the per-pixel soma probability is predicted,
thresholded, and labelled by connected components (touching cells merge into
one for now; pass --watershed to split them with peak-seeded watershed later).
A weighted-average trace is extracted per labelled region. Per recording:
    <stem>_labels.npy     (H,W) int32 instance labels (0 = background)
    <stem>_traces.npy     (N,T) float32 weighted-average traces
    <stem>_centroids.npy  (N,2) region centroids (row,col)
    <stem>_overlay.png    QC: region boundaries on the max projection

    python scripts/run_seg_infer.py --model models/seg_final/segmenter.pt \
        --movies new_nd2 --out results/seg_infer --threshold 0.55

SCALE is handled automatically for .nd2: the recording's microns/pixel is read
from its metadata and the movie is resampled to the pixel size the model was
trained at (stored in the model when trained with --pixel-um, or pass
--train-um-per-px). For inputs without pixel metadata (e.g. .tif), use
--resize-to N as a fallback, or accept native scale.
"""
import argparse, glob, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orcann.io import load_model                                   # noqa: E402
from orcann.extract import _load_movie                             # noqa: E402
from orcann.spatial_log import extract_instances                   # noqa: E402
from orcann.spatial_seg import predict_prob, segment_instances     # noqa: E402


def nd2_um_per_px(path):
    """Microns/pixel from ND2 metadata (xy), or None if unavailable."""
    try:
        import nd2
        with nd2.ND2File(path) as f:
            vs = f.voxel_size()            # namedtuple (x, y, z) in microns
        return float(vs.x)
    except Exception:
        return None


def traces_from_labels(movie, labels, prob):
    """Weighted-average trace per instance; weights = soma probability."""
    T = movie.shape[0]
    ids = np.unique(labels); ids = ids[ids != 0]
    traces = np.zeros((len(ids), T), np.float32)
    cents = np.zeros((len(ids), 2), np.float32)
    flat_mov = movie.reshape(T, -1)
    for i, k in enumerate(ids):
        m = labels == k
        w = prob[m].astype(np.float32)
        s = float(w.sum())
        if s <= 0:
            w = np.ones_like(w); s = float(w.sum())
        traces[i] = (flat_mov[:, m.ravel()] * w[None, :]).sum(1) / s
        ys, xs = np.where(m); cents[i] = (ys.mean(), xs.mean())
    return traces, cents


def resize_movie(movie, scale=None, target=None, target_hw=None):
    from scipy.ndimage import zoom
    T, H, W = movie.shape
    if scale is not None:
        f = (1, scale, scale)
    elif target_hw is not None:
        f = (1, target_hw[0] / H, target_hw[1] / W)
    else:
        f = (1, target / H, target / W)
    return zoom(movie, f, order=1).astype(np.float32)


def save_overlay(movie, labels, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    mp = movie.max(0).astype(float); mp /= max(np.percentile(mp, 99.5), 1e-6)
    b = find_boundaries(labels, mode="outer")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.clip(mp, 0, 1), cmap="gray")
    ax.imshow(np.ma.masked_where(~b, b), cmap="autumn", alpha=0.9)
    ax.set_title(f"{os.path.basename(path)[:-12]}: {int(labels.max())} cells"); ax.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--movies", help="dir of .nd2 / .tif / .npy recordings")
    ap.add_argument("--movie", help="single recording (array job, one per task)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="soma probability cut (fixed at inference; ~0.5-0.6)")
    ap.add_argument("--watershed", action="store_true",
                    help="split touching cells with a peak-seeded watershed; "
                         "default is connected components (touching cells merge)")
    ap.add_argument("--min-distance", type=int, default=4,
                    help="min peak separation in px for watershed seeding")
    ap.add_argument("--train-um-per-px", type=float, default=None,
                    help="override the model's recorded training pixel size for "
                         "auto-rescaling; usually unnecessary if the model carries it")
    ap.add_argument("--resize-to", type=int, default=0,
                    help="force each frame to NxN (fallback when pixel size is unknown)")
    ap.add_argument("--min-area", type=int, default=4,
                    help="drop detected regions smaller than this many px (cleans "
                         "hotspot/noise specks; 0 disables)")
    ap.add_argument("--min-radius", type=float, default=0.0,
                    help="drop detected regions below this equivalent radius in px "
                         "(convenience for --min-area; uses pi*r^2)")
    ap.add_argument("--save-prob", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--motion-correct", action="store_true",
                    help="run NoRMCorre before segmenting (needs caiman in THIS "
                         "env; otherwise pre-correct with scripts/run_motion_correct.py)")
    ap.add_argument("--mc-mode", default="auto",
                    choices=["rigid", "piecewise_rigid", "auto"])
    ap.add_argument("--mc-max-shift", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model = load_model(a.model)
    cfg = getattr(model, "config", {})
    target_um = a.train_um_per_px or cfg.get("pixel_um")
    target_hw = cfg.get("train_hw")
    files = ([a.movie] if a.movie else
             sorted(sum([glob.glob(os.path.join(a.movies, e))
                         for e in ("*.nd2", "*.tif", "*.tiff", "*.npy")], []))
             if a.movies else None)
    if files is None:
        ap.error("give --movies DIR (serial) or --movie FILE (array task)")
    if not files:
        print("no recordings found"); return
    if target_um is None and target_hw is None and not a.resize_to:
        print("note: this model recorded no training scale (trained before scale "
              "tracking). Running at native scale; retrain so the model stores "
              "train_hw, or pass --resize-to / --train-um-per-px.")

    min_area = a.min_area
    if a.min_radius > 0:
        min_area = max(min_area, int(round(np.pi * a.min_radius ** 2)))
    if min_area > 0:
        print(f"stripping detected regions below {min_area} px"
              + (f" (radius {a.min_radius:g})" if a.min_radius > 0 else ""))

    print(f"{'recording':24s} {'in HxW':>11} {'um/px':>7} {'-> HxW':>11} {'cells':>6} {'T':>6}")
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        movie = _load_movie(f)
        in_hw = movie.shape[1:]
        in_um = nd2_um_per_px(f) if f.endswith(".nd2") else None

        if a.motion_correct:                          # correct on native movie first
            from orcann.motion_correction import correct_motion
            movie = correct_motion(movie.astype(np.float32), mode=a.mc_mode,
                                   max_shift=a.mc_max_shift).corrected.astype(np.float32)
            in_hw = movie.shape[1:]                    # NoRMCorre may crop borders

        if target_um and in_um:                       # physical (um/px) scale match
            movie = resize_movie(movie, scale=in_um / target_um)
        elif a.resize_to:                             # explicit override
            movie = resize_movie(movie, target=a.resize_to)
        elif target_hw and tuple(in_hw) != tuple(target_hw):   # auto frame-size match
            movie = resize_movie(movie, target_hw=target_hw)

        prob = predict_prob(model, movie)
        if a.watershed:
            seeds, _ = extract_instances(prob, min_distance=a.min_distance,
                                         threshold=a.threshold)
            labels = segment_instances(prob, seeds, threshold=a.threshold)
        else:
            from scipy.ndimage import label as cc_label
            labels = cc_label(prob >= a.threshold)[0].astype(np.int32)

        if min_area > 0:                              # drop tiny noise specks
            sizes = np.bincount(labels.ravel())
            drop = np.where(sizes < min_area)[0]
            drop = drop[drop != 0]
            if drop.size:
                labels[np.isin(labels, drop)] = 0
            # relabel remaining regions to contiguous 1..N
            keep = np.unique(labels); keep = keep[keep != 0]
            remap = np.zeros(int(labels.max()) + 1, np.int32)
            remap[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
            labels = remap[labels]

        traces, cents = traces_from_labels(movie, labels, prob)

        np.save(os.path.join(a.out, stem + "_labels.npy"), labels.astype(np.int32))
        np.save(os.path.join(a.out, stem + "_traces.npy"), traces)
        np.save(os.path.join(a.out, stem + "_centroids.npy"), cents)
        if a.save_prob:
            np.save(os.path.join(a.out, stem + "_prob.npy"), prob.astype(np.float32))
        if not a.no_overlay:
            save_overlay(movie, labels, os.path.join(a.out, stem + "_overlay.png"))

        T, H, W = movie.shape
        um_s = f"{in_um:.3f}" if in_um else "-"
        print(f"{stem:24s} {f'{in_hw[0]}x{in_hw[1]}':>11} {um_s:>7} "
              f"{f'{H}x{W}':>11} {int(labels.max()):6d} {T:6d}")

    print(f"\noutputs -> {a.out}  (labels + traces + centroids per recording)")


if __name__ == "__main__":
    main()
