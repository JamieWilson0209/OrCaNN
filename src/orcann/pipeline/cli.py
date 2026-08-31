"""The single `orcann` entry point: subcommands that each read the YAML config.

    orcann motion_correction --config config.yaml [--force] [--task-id N]
    orcann infer             --config config.yaml [--force] [--task-id N]
    orcann segment           --config config.yaml [--force] [--task-id N] [--sweep k=v1,v2]
    orcann activity          --config config.yaml [--force] [--task-id N]
    orcann analysis          --config config.yaml [--force]
    orcann train_spatial     --config config.yaml [--synthetic]

Spatial detection is split: `infer` runs the GPU model once and caches the
probability map; `segment` (CPU, no model) thresholds + extracts and is the cheap
stage you re-run while tuning. `activity` then baseline-corrects, deconvolves
(OASIS) and renders the per-recording HTML gallery; `analysis` aggregates across
recordings. Every subcommand accepts --set section.key=value (repeatable) and
--dump-config PATH. Paths and the model live in the config.
"""
import argparse
import glob
import os

from orcann.configLoader import Config
from orcann.logging_setup import LEVELS, DEFAULT_LEVEL, configure as configure_logging

_EXTS = ("*.nd2", "*.tif", "*.tiff", "*.npy")
# Sidecars the motion_correction stage writes next to each corrected movie in
# data/pre_processed. They must not be mistaken for recordings: <stem>_mc.json is
# already excluded (not a movie extension), but <stem>_mc_shifts.npy matches the
# *.npy glob and would otherwise be counted as an extra recording, doubling the
# infer/segment array size and mapping tasks onto shift arrays.
_SIDECAR_SUFFIXES = ("_mc_shifts.npy", "_shifts.npy")


def list_recordings(dirpath, task_id=None):
    """Sorted recordings in a directory; with task_id, just the 1-based Nth (SGE array)."""
    if not dirpath or not os.path.isdir(dirpath):
        return []
    files = sorted(sum([glob.glob(os.path.join(dirpath, e)) for e in _EXTS], []))
    files = [f for f in files
             if not os.path.basename(f).endswith(_SIDECAR_SUFFIXES)]
    if task_id is not None:
        return [files[task_id - 1]] if 1 <= task_id <= len(files) else []
    return files


def list_infer_recordings(infer_dir, task_id=None):
    """recording_ids under a results/infer dir that have a cached prob.npy."""
    if not os.path.isdir(infer_dir):
        return []
    recs = sorted(d for d in os.listdir(infer_dir)
                  if os.path.isfile(os.path.join(infer_dir, d, "prob.npy")))
    if task_id is not None:
        return [recs[task_id - 1]] if 1 <= task_id <= len(recs) else []
    return recs


def list_spatial_recordings(spatial_dir, task_id=None):
    """recording_ids under a results/spatial dir that have data/traces.npy."""
    if not os.path.isdir(spatial_dir):
        return []
    recs = sorted(d for d in os.listdir(spatial_dir)
                  if os.path.isfile(os.path.join(spatial_dir, d, "data", "traces.npy")))
    if task_id is not None:
        return [recs[task_id - 1]] if 1 <= task_id <= len(recs) else []
    return recs


def _models(cfg, promote):
    """List what has been trained and what is in service, or promote one model.

    Promotion is its own command because it is its own decision: training
    produces a model, and a person choosing to run it is a separate event that
    should appear in the shell history of whoever made it.
    """
    from orcann.pipeline import model_io as mio
    if promote:
        try:
            dst = mio.promote(cfg.train_spatial.out, cfg.models.dir, promote)
        except mio.ModelStoreError as e:
            raise SystemExit(str(e))
        print(f"promoted -> {dst}")
        print(f"  run it with models.spatial: {promote}  (or 'latest')")
        return True
    trained = mio.list_models(cfg.train_spatial.out)
    in_use = mio.list_models(cfg.models.dir)
    print(f"trained ({cfg.train_spatial.out}) -- not in service until promoted:")
    for i in trained:
        print(f"  {i}{'   [promoted]' if i in in_use else ''}")
    print(f"in use ({cfg.models.dir}):")
    for i in in_use:
        print(f"  {i}")
    if in_use:
        print(f"latest -> {in_use[-1]}")
    if not trained and not in_use:
        print("  nothing yet; run train_spatial")
    return True


def _common(sp):
    sp.add_argument("--config", default="config.yaml",
                    help="YAML config file (default: ./config.yaml)")
    sp.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="section.key=value",
                    help="one-off override of a config value (repeatable)")
    sp.add_argument("--log-level", default=DEFAULT_LEVEL, choices=LEVELS,
                    help="how much the run reports (default: %(default)s)")
    sp.add_argument("--dump-config", metavar="PATH",
                    help="write a commented config to PATH and exit")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="orcann", description="OrCaNN calcium-imaging pipeline (config-driven).")
    sub = ap.add_subparsers(dest="stage", required=True, metavar="<stage>")

    for name, helptext in [
            ("motion_correction", "data/raw -> data/pre_processed (caiman env)"),
            ("infer", "pre_processed -> results/infer (GPU; cache prob maps)"),
            ("activity", "results/spatial -> results/activity (dF/F0, OASIS, gallery)")]:
        p = sub.add_parser(name, help=helptext)
        _common(p)
        p.add_argument("--force", action="store_true", help="redo recordings whose output exists")
        p.add_argument("--task-id", type=int, default=None,
                       help="process only the Nth recording (SGE array)")

    p = sub.add_parser("segment", help="results/infer + movie -> results/spatial (CPU; threshold+extract)")
    _common(p)
    p.add_argument("--force", action="store_true", help="redo recordings whose output exists")
    p.add_argument("--task-id", type=int, default=None,
                   help="process only the Nth recording (SGE array)")
    p.add_argument("--sweep", action="append", default=[], metavar="key=v1,v2,...",
                   help="preview a parameter grid (montage + table), no extraction; repeatable")

    p = sub.add_parser("analysis", help="results/activity -> results/analysis (group figures + tables)")
    _common(p)
    p.add_argument("--force", action="store_true", help="rebuild even if outputs exist")

    p = sub.add_parser("train_spatial", help="train the segmenter")
    _common(p)
    p.add_argument("--synthetic", action="store_true", help="self-test on synthetic data")

    p = sub.add_parser("models", help="list trained models, or put one into service")
    _common(p)
    p.add_argument("--promote", metavar="IDENTITY",
                   help="copy this model from train_spatial.out into models.dir")

    return ap


def _resolve_config(a):
    """Handle --dump-config (returns None), else load config + apply --set."""
    if a.dump_config:
        Config().apply_overrides(a.overrides).dump(a.dump_config)
        print(f"wrote config -> {a.dump_config}")
        return None
    synthetic = getattr(a, "synthetic", False)
    if not synthetic and not os.path.exists(a.config):
        raise SystemExit(f"config not found: {a.config} "
                         f"(create one with: orcann {a.stage} --dump-config {a.config})")
    path = a.config if os.path.exists(a.config) else None
    return Config.load(path).apply_overrides(a.overrides).resolve_paths()


def main(argv=None) -> int:
    """Run one stage. Returns a process exit code.

    A stage runner signals failure by returning False; anything else (including
    None, which the runners that have not adopted the convention still return)
    counts as success. Raising is also a failure — argparse and the runners let
    real exceptions through, and the shell sees a traceback and a non-zero code.
    """
    a = build_parser().parse_args(argv)
    # Before anything else: a stage that logs its way through a config problem
    # should be heard doing it.
    configure_logging(a.log_level)
    cfg = _resolve_config(a)
    if cfg is None:
        return 2

    if a.stage == "motion_correction":
        from orcann.pipeline import run_motion_correction
        ok = run_motion_correction.run(cfg, task_id=a.task_id, force=a.force)
    elif a.stage == "infer":
        from orcann.pipeline import run_infer
        ok = run_infer.run(cfg, task_id=a.task_id, force=a.force)
    elif a.stage == "segment":
        from orcann.pipeline import run_segment
        ok = run_segment.run(cfg, task_id=a.task_id, force=a.force, sweeps=a.sweep)
    elif a.stage == "activity":
        from orcann.pipeline import run_activity
        ok = run_activity.run(cfg, task_id=a.task_id, force=a.force)
    elif a.stage == "train_spatial":
        from orcann.pipeline import run_train_spatial
        ok = run_train_spatial.run(cfg, synthetic=a.synthetic)
    elif a.stage == "models":
        ok = _models(cfg, a.promote)
    elif a.stage == "analysis":
        from orcann.pipeline import run_analysis
        ok = run_analysis.run(cfg, force=a.force)
    else:                                    # unreachable: subparser is required
        return 2

    return 1 if ok is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
