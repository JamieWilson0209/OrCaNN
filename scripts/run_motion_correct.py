#!/usr/bin/env python
"""Batch motion correction (CaImAn NoRMCorre) to run BEFORE segmentation.

NoRMCorre needs caiman, which is NOT in the segmenter inference env, so this is
a separate step run in the caiman env. It reads .nd2/.tif recordings, corrects
each, and writes a motion-corrected movie that run_seg_infer.py then consumes:
    <stem>_mc.tif    (T,H,W) float32 corrected movie  (or _mc.npy with --save-npy)
    <stem>_mc.json   shift summary (max/mean dy,dx, border crop, elapsed)

    python scripts/run_motion_correct.py --movies new_nd2 --out mc_movies \
        --mode auto --max-shift 20

Then segment the corrected movies:
    python scripts/run_seg_infer.py --model .../segmenter.pt --movies mc_movies --out seg

NOTE: writing to .tif drops the .nd2 pixel-size metadata, so inference falls
back to frame-size scale matching (correct for same-magnification data). If you
need the physical um/px match, segment the .nd2 directly with the integrated
--motion-correct flag instead (requires caiman in the inference env).
"""
import argparse, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orcann.extract import _load_movie                            # noqa: E402
from orcann.motion_correction import correct_motion              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", default=None, help="dir of .nd2 / .tif recordings")
    ap.add_argument("--movie", default=None,
                    help="single recording (used by the array job, one per task)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="auto",
                    choices=["rigid", "piecewise_rigid", "auto"])
    ap.add_argument("--max-shift", type=int, default=20)
    ap.add_argument("--n-processes", type=int, default=1)
    ap.add_argument("--resize-to", type=int, default=0,
                    help="downscale each frame to NxN BEFORE correcting; biggest "
                         "speedup, and lossless for you since you segment at 512")
    ap.add_argument("--niter-rig", type=int, default=2,
                    help="rigid template iterations; 1 is faster")
    ap.add_argument("--num-frames-split", type=int, default=100)
    ap.add_argument("--pw-stride", type=int, default=96,
                    help="piecewise patch stride; larger = fewer patches = faster")
    ap.add_argument("--pw-overlap", type=int, default=48)
    ap.add_argument("--save-npy", action="store_true",
                    help="write .npy instead of .tif")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import tifffile
    if a.movie:
        files = [a.movie]
    elif a.movies:
        files = sorted(sum([glob.glob(os.path.join(a.movies, e))
                            for e in ("*.nd2", "*.tif", "*.tiff", "*.npy")], []))
    else:
        ap.error("give --movies DIR (serial) or --movie FILE (array task)")
    if not files:
        print("no recordings found"); return

    print(f"{'recording':24s} {'T,H,W':>16} {'max dy,dx':>13} {'mean dy,dx':>13} {'sec':>6}")
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        movie = _load_movie(f).astype(np.float32)
        if a.resize_to and movie.shape[1] != a.resize_to:
            from scipy.ndimage import zoom
            T, H, W = movie.shape
            movie = zoom(movie, (1, a.resize_to / H, a.resize_to / W),
                         order=1).astype(np.float32)
        res = correct_motion(movie, mode=a.mode, max_shift=a.max_shift,
                             n_processes=a.n_processes, niter_rig=a.niter_rig,
                             num_frames_split=a.num_frames_split,
                             pw_strides=(a.pw_stride, a.pw_stride),
                             pw_overlaps=(a.pw_overlap, a.pw_overlap))
        ext = "_mc.npy" if a.save_npy else "_mc.tif"
        out_mov = os.path.join(a.out, stem + ext)
        if a.save_npy:
            np.save(out_mov, res.corrected.astype(np.float32))
        else:
            tifffile.imwrite(out_mov, res.corrected.astype(np.float32))
        s = res.summary()
        with open(os.path.join(a.out, stem + "_mc.json"), "w") as fh:
            json.dump(s, fh, indent=2)

        T, H, W = res.corrected.shape
        mx = f"{s['max_shift_y']:.1f},{s['max_shift_x']:.1f}"
        mn = f"{s['mean_shift_y']:.1f},{s['mean_shift_x']:.1f}"
        print(f"{stem:24s} {f'{T},{H},{W}':>16} {mx:>13} {mn:>13} {s['elapsed_seconds']:6.0f}")

    print(f"\ncorrected movies -> {a.out}  (pass this dir to run_seg_infer.py --movies)")


if __name__ == "__main__":
    main()
