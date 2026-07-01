"""Full pipeline: motion correction (if needed) -> infer -> segment -> detect_transients.

Reads everything from the config. Motion correction is skipped when every raw
recording already has a corrected movie in data/pre_processed. `infer` runs the
GPU model and caches the probability maps; `segment` thresholds + extracts (CPU,
no model). Each stage skips recordings whose output exists; pass force=True to redo.
"""
import os

from orcann.pipeline import run_infer, run_segment, detect_transients, run_motion_correction
from orcann.pipeline.cli import list_recordings


def _mc_complete(cfg):
    raw = list_recordings(cfg.paths.raw)
    if not raw:
        return True
    pre = cfg.paths.pre_processed
    for f in raw:
        stem = os.path.splitext(os.path.basename(f))[0]
        if not (os.path.exists(os.path.join(pre, stem + "_mc.tif"))
                or os.path.exists(os.path.join(pre, stem + "_mc.npy"))):
            return False
    return True


def run(cfg, task_id=None, force=False):
    print("=== run_pipeline ===")
    if force or not _mc_complete(cfg):
        run_motion_correction.run(cfg, task_id=task_id, force=force)
    else:
        print(f"motion_correction: {cfg.paths.pre_processed} already complete, skipping")
    run_infer.run(cfg, task_id=task_id, force=force)
    run_segment.run(cfg, task_id=task_id, force=force)
    detect_transients.run(cfg, task_id=task_id, force=force)
    print("=== pipeline complete ===")
