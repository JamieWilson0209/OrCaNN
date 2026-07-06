"""Motion-correction stage: data/raw -> data/pre_processed (caiman env).

NoRMCorre needs caiman, which is NOT in the segmenter/inference env, so this stage
runs in the caiman env. Each recording is corrected and written as
<stem>_mc.tif (T,H,W float32) plus <stem>_mc.json (shift summary). Recordings whose
corrected movie already exists are skipped unless force=True.
"""
import json
import os

import numpy as np

from orcann.pipeline.cli import list_recordings


def _done(pre_dir, stem):
    return (os.path.exists(os.path.join(pre_dir, stem + "_mc.tif"))
            or os.path.exists(os.path.join(pre_dir, stem + "_mc.npy")))


def run(cfg, task_id=None, force=False):
    from orcann.pipeline.extraction import _load_movie
    from orcann.pipeline.motion_correction import correct_motion
    import tifffile

    raw, pre = cfg.paths.raw, cfg.paths.pre_processed
    mc = cfg.motion_correction
    os.makedirs(pre, exist_ok=True)
    files = list_recordings(raw, task_id)
    if not files:
        print(f"motion_correction: no recordings in {raw}"); return

    print(f"motion_correction: {len(files)} recording(s)  {raw} -> {pre}")
    print(f"{'recording':24s} {'T,H,W':>16} {'max dy,dx':>13} {'sec':>6}")
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if _done(pre, stem) and not force:
            print(f"{stem:24s} {'(exists, skipped)':>16}")
            continue
        movie = _load_movie(f).astype(np.float32)
        res = correct_motion(movie, mode=mc.mode, max_shift=mc.max_shift)
        out_mov = os.path.join(pre, stem + "_mc.tif")
        tifffile.imwrite(out_mov, res.corrected.astype(np.float32))
        s = res.summary()
        with open(os.path.join(pre, stem + "_mc.json"), "w") as fh:
            json.dump(s, fh, indent=2)
        # Per-frame [dy, dx] shifts, so the activity stage can carry them into
        # results/activity/ for the analysis stage's residual-motion QC metric.
        if getattr(res, "shifts", None) is not None:
            np.save(os.path.join(pre, stem + "_mc_shifts.npy"),
                    np.asarray(res.shifts, np.float32))
        T, H, W = res.corrected.shape
        mx = f"{s['max_shift_y']:.1f},{s['max_shift_x']:.1f}"
        print(f"{stem:24s} {f'{T},{H},{W}':>16} {mx:>13} {s['elapsed_seconds']:6.0f}")
    print(f"corrected movies -> {pre}")
