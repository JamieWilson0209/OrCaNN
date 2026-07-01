"""The single `orcann` entry point: subcommands that each read the YAML config.

    orcann run_pipeline      --config config.yaml
    orcann motion_correction --config config.yaml [--force] [--task-id N]
    orcann infer             --config config.yaml [--force] [--task-id N]
    orcann segment           --config config.yaml [--force] [--task-id N] [--sweep k=v1,v2]
    orcann detect_transients --config config.yaml [--force] [--task-id N]
    orcann analysis          --config config.yaml [--force]
    orcann train_spatial     --config config.yaml [--synthetic]
    orcann train_temporal    --config config.yaml [--synthetic]

Spatial detection is split: `infer` runs the GPU model once and caches the
probability map; `segment` (CPU, no model) thresholds + extracts and is the cheap
stage you re-run while tuning. Every subcommand accepts --set section.key=value
(repeatable) and --dump-config PATH. Paths and models live in the config.
"""
import argparse
import glob
import os

from orcann.configLoader import Config

_EXTS = ("*.nd2", "*.tif", "*.tiff", "*.npy")


def list_recordings(dirpath, task_id=None):
    """Sorted recordings in a directory; with task_id, just the 1-based Nth (SGE array)."""
    if not dirpath or not os.path.isdir(dirpath):
        return []
    files = sorted(sum([glob.glob(os.path.join(dirpath, e)) for e in _EXTS], []))
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


def _common(sp):
    sp.add_argument("--config", default="config.yaml",
                    help="YAML config file (default: ./config.yaml)")
    sp.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="section.key=value",
                    help="one-off override of a config value (repeatable)")
    sp.add_argument("--dump-config", metavar="PATH",
                    help="write a commented config to PATH and exit")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="orcann", description="OrCaNN calcium-imaging pipeline (config-driven).")
    sub = ap.add_subparsers(dest="stage", required=True, metavar="<stage>")

    p = sub.add_parser("run_pipeline",
                       help="motion correction -> infer -> segment -> detect_transients")
    _common(p)
    p.add_argument("--force", action="store_true", help="redo stages even if outputs exist")
    p.add_argument("--task-id", type=int, default=None,
                   help="process only the Nth recording (SGE array)")

    for name, helptext in [
            ("motion_correction", "data/raw -> data/pre_processed (caiman env)"),
            ("infer", "pre_processed -> results/infer (GPU; cache prob maps)"),
            ("detect_transients", "results/spatial -> results/transients")]:
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

    p = sub.add_parser("analysis", help="results/transients -> results/analysis (group figures + tables)")
    _common(p)
    p.add_argument("--force", action="store_true", help="rebuild even if outputs exist")

    for name, helptext in [("train_spatial", "train the segmenter"),
                           ("train_temporal", "train/evaluate the temporal head")]:
        p = sub.add_parser(name, help=helptext)
        _common(p)
        p.add_argument("--synthetic", action="store_true", help="self-test on synthetic data")

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


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = _resolve_config(a)
    if cfg is None:
        return

    if a.stage == "run_pipeline":
        from orcann.pipeline import run_pipeline
        run_pipeline.run(cfg, task_id=a.task_id, force=a.force)
    elif a.stage == "motion_correction":
        from orcann.pipeline import run_motion_correction
        run_motion_correction.run(cfg, task_id=a.task_id, force=a.force)
    elif a.stage == "infer":
        from orcann.pipeline import run_infer
        run_infer.run(cfg, task_id=a.task_id, force=a.force)
    elif a.stage == "segment":
        from orcann.pipeline import run_segment
        run_segment.run(cfg, task_id=a.task_id, force=a.force, sweeps=a.sweep)
    elif a.stage == "detect_transients":
        from orcann.pipeline import detect_transients
        detect_transients.run(cfg, task_id=a.task_id, force=a.force)
    elif a.stage == "train_spatial":
        from orcann.pipeline import run_train_spatial
        run_train_spatial.run(cfg, synthetic=a.synthetic)
    elif a.stage == "train_temporal":
        from orcann.pipeline import run_train_temporal
        run_train_temporal.run(cfg, synthetic=a.synthetic)
    elif a.stage == "analysis":
        from orcann.pipeline import run_analysis
        run_analysis.run(cfg, force=a.force)


if __name__ == "__main__":
    main()
