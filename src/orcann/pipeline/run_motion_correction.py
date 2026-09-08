"""Motion-correction stage: data/raw -> data/pre_processed (caiman env).

NoRMCorre needs caiman, which is NOT in the segmenter/inference env, so this stage
runs in the caiman env. Each recording is corrected and written into a store named
for the stage and the run key, as <recording>.tif (T,H,W float32) beside
<recording>.json (shift summary). A recording whose record was written under the
settings in this config is skipped unless force=True.

The corrected TIFF declares its own axes, so the stages below it read a file that
says it is a time series rather than one assumed to be (T, H, W).
"""
import os

import numpy as np

from orcann.pipeline import provenance as prov
from orcann.pipeline.cli import identified_recordings
from orcann.run_info import write_record


def _write_corrected(path, movie, meta):
    """Write the corrected movie as an ImageJ TIFF that declares its axes.

    A plain TIFF names no time axis, so intake reads the corrected movie back as
    a 'QYX' stack and says so on every recording. The ImageJ header is what
    declares the movie a time series, and it is also the only container the
    frame interval survives in.

    The header is written by hand rather than through ``imagej=True``.
    This stage runs in the caiman env, whose tifffile is six years older than
    the one every stage below it uses, and the two do not agree on what
    ``imagej=True`` means for a 3-D array: the old writer files the stack as
    ``channels`` and the new one as ``frames``. A corrected movie written by the
    old one reads back as a channel stack, which intake refuses -- correctly,
    since a channel stack is not a time series. Spelling the header out produces
    a byte-identical file from both versions. See dev/two-env-split.md.

    Only what the source declared is written. A recording that named no frame
    interval leaves the key off rather than carrying a default.
    """
    import tifffile

    # ImageJ keys, one per line, the format's own header. The version string is
    # what marks the description as ImageJ metadata at all; nothing reads it.
    lines = ["ImageJ=1.52p", f"images={movie.shape[0]}",
             f"frames={movie.shape[0]}", "slices=1", "channels=1"]
    if meta.frame_interval_s:
        lines.append(f"finterval={float(meta.frame_interval_s)}")
    # photometric is not cosmetic here: without it tifffile reads a leading
    # dimension of 3 or 4 as colour samples, so a four-frame recording comes back
    # as one 'SYX' plane and intake refuses it.
    tifffile.imwrite(path, movie, description="\n".join(lines) + "\n",
                     metadata=None, photometric="minisblack")


def run(cfg, task_id=None, force=False):
    from orcann.pipeline.extraction import open_recording
    from orcann.pipeline.motion_correction import correct_motion

    raw, pre = cfg.paths.raw, cfg.paths.pre_processed
    mc = cfg.motion_correction
    try:
        run_key = prov.check_run_key(cfg.run.key)
    except prov.ProvenanceError as e:
        raise SystemExit(str(e))
    store = prov.output_store(pre, prov.STAGE_MOTION, run_key,
                              prov.stage_key(cfg, prov.STAGE_MOTION), None)
    os.makedirs(store, exist_ok=True)
    found = identified_recordings(raw)
    triples, surplus = prov.for_task(found, task_id)
    if surplus:
        print(f"motion_correction: task {task_id} is past the end of "
              f"{len(found)} recording(s); nothing to do")
        return True
    if not triples:
        print(f"motion_correction: no recordings in {raw}")
        return False

    print(f"motion_correction: {len(triples)} recording(s)  {raw} -> {store}")
    print(f"{'recording':32s} {'T,H,W':>16} {'max dy,dx':>13} {'sec':>6}")
    for f, rec_id, tag in triples:
        key = prov.stage_key(cfg, prov.STAGE_MOTION, content_tag=tag)
        record = os.path.join(store, rec_id + ".json")
        ok, why = prov.is_current(record, prov.STAGE_MOTION, key)
        if ok and not force:
            print(f"{rec_id:32s} {'(current, skipped)':>16}")
            continue
        if not ok and why != "no record":
            print(f"{rec_id:32s} ({why})")
        # No require_finite here: this stage is entitled to non-finite borders and
        # sanitises them itself. The metadata is read alongside the pixels so the
        # corrected copy can declare what the source declared.
        with open_recording(f) as rec:
            movie = rec.array().astype(np.float32)
            meta = rec.meta
        res = correct_motion(movie, mode=mc.mode, max_shift=mc.max_shift)
        _write_corrected(os.path.join(store, rec_id + ".tif"),
                         res.corrected.astype(np.float32), meta)
        # Per-frame [dy, dx] shifts, so the activity stage can carry them into
        # the activity store for the analysis stage's residual-motion QC metric.
        if getattr(res, "shifts", None) is not None:
            np.save(os.path.join(store, rec_id + "_shifts.npy"),
                    np.asarray(res.shifts, np.float32))
        s = res.summary()
        # The record last, and only now: it is what marks this recording done, so
        # a job killed between the two writes leaves a movie the next run cannot
        # see rather than one it would trust.
        write_record(record, prov.STAGE_MOTION, key, run_key,
                     dict(s, recording_id=rec_id, source=os.path.abspath(f)))
        T, H, W = res.corrected.shape
        mx = f"{s['max_shift_y']:.1f},{s['max_shift_x']:.1f}"
        print(f"{rec_id:32s} {f'{T},{H},{W}':>16} {mx:>13} {s['elapsed_seconds']:6.0f}")
    print(f"corrected movies -> {store}  (now run: infer)")
    return True
