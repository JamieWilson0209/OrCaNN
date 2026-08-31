"""Motion-correction stage: data/raw -> data/pre_processed (caiman env).

NoRMCorre needs caiman, which is NOT in the segmenter/inference env, so this stage
runs in the caiman env. Each recording is corrected and written as
<stem>_mc.tif (T,H,W float32) plus <stem>_mc.json (shift summary). Recordings whose
corrected movie already exists are skipped unless force=True.

The corrected TIFF carries the source's own axes and pixel size, so the stages
below it read a self-describing file rather than assuming (T, H, W) at an unknown
scale.
"""
import os

import numpy as np

from orcann.pipeline.cli import list_recordings
from orcann.run_info import write_record


def _write_corrected(path, movie, meta):
    """Write the corrected movie as an ImageJ TIFF carrying the source's own scale.

    The pixel size is known only while the raw file is open. A plain TIFF drops
    it, and intake then reads the corrected movie as a 'QYX' stack of unknown
    scale -- which is why the physical-scale branch of
    ``inference.resample_for_model`` has never had a µm/px to match against.
    ImageJ is the only container ``extraction._tiff_scale`` reads a unit from.

    The ImageJ header is written by hand rather than through ``imagej=True``.
    This stage runs in the caiman env, whose tifffile is six years older than
    the one every stage below it uses, and the two do not agree on what
    ``imagej=True`` means for a 3-D array: the old writer files the stack as
    ``channels`` and the new one as ``frames``. A corrected movie written by the
    old one reads back as a channel stack, which intake refuses -- correctly,
    since a channel stack is not a time series. Spelling the header out produces
    a byte-identical file from both versions. See dev/two-env-split.md.

    Only what the source declared is written. A recording that named no pixel
    size leaves the unit and resolution off rather than carrying a default; the
    axes are still declared, so the movie stops being read as 'QYX' by
    convention either way.

    The µm/px does not survive bit-exactly: TIFF stores resolution as a
    RATIONAL, so 0.6864907163890155 reads back as 0.6864907163690688. That is
    2e-11 µm at this scale -- a limit of the container, not a substituted value.
    """
    import tifffile

    # ImageJ keys, one per line, the format's own header. The version string is
    # what marks the description as ImageJ metadata at all; nothing reads it.
    lines = ["ImageJ=1.52p", f"images={movie.shape[0]}",
             f"frames={movie.shape[0]}", "slices=1", "channels=1"]
    kw = {}
    ux = meta.um_per_px
    if ux:
        lines.append("unit=um")
        kw["resolution"] = (1.0 / ux, 1.0 / (meta.um_per_px_y or ux))
    if meta.frame_interval_s:
        lines.append(f"finterval={float(meta.frame_interval_s)}")
    # photometric is not cosmetic here: without it tifffile reads a leading
    # dimension of 3 or 4 as colour samples, so a four-frame recording comes back
    # as one 'SYX' plane and intake refuses it.
    tifffile.imwrite(path, movie, description="\n".join(lines) + "\n",
                     metadata=None, photometric="minisblack", **kw)


def _done(pre_dir, stem):
    return (os.path.exists(os.path.join(pre_dir, stem + "_mc.tif"))
            or os.path.exists(os.path.join(pre_dir, stem + "_mc.npy")))


def run(cfg, task_id=None, force=False):
    from orcann.pipeline.extraction import open_recording
    from orcann.pipeline.motion_correction import correct_motion

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
        # No require_finite here: this stage is entitled to non-finite borders and
        # sanitises them itself. The metadata is read alongside the pixels so the
        # corrected copy can carry the scale forward.
        with open_recording(f) as rec:
            movie = rec.array().astype(np.float32)
            meta = rec.meta
        res = correct_motion(movie, mode=mc.mode, max_shift=mc.max_shift)
        out_mov = os.path.join(pre, stem + "_mc.tif")
        _write_corrected(out_mov, res.corrected.astype(np.float32), meta)
        s = res.summary()
        write_record(os.path.join(pre, stem + "_mc.json"),
                     "motion_correction", dict(s, recording_id=stem))
        # Per-frame [dy, dx] shifts, so the activity stage can carry them into
        # results/activity/ for the analysis stage's residual-motion QC metric.
        if getattr(res, "shifts", None) is not None:
            np.save(os.path.join(pre, stem + "_mc_shifts.npy"),
                    np.asarray(res.shifts, np.float32))
        T, H, W = res.corrected.shape
        mx = f"{s['max_shift_y']:.1f},{s['max_shift_x']:.1f}"
        print(f"{stem:24s} {f'{T},{H},{W}':>16} {mx:>13} {s['elapsed_seconds']:6.0f}")
    print(f"corrected movies -> {pre}")
