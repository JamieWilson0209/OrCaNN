"""The single `orcann` entry point: subcommands that each read the YAML config.

    orcann motion_correction --config config.yaml [--force] [--task-id N]
    orcann infer             --config config.yaml [--force] [--task-id N]
    orcann segment           --config config.yaml [--force] [--task-id N] [--sweep k=v1,v2]
    orcann activity          --config config.yaml [--force] [--task-id N]
    orcann analysis          --config config.yaml [--force]
    orcann train_spatial     --config config.yaml [--synthetic]
    orcann status            --config config.yaml

Spatial detection is split: `infer` runs the GPU model once and caches the
probability map; `segment` (CPU, no model) thresholds + extracts and is the cheap
stage you re-run while tuning. `activity` then baseline-corrects, deconvolves
(OASIS) and renders the per-recording HTML gallery; `analysis` aggregates across
recordings. `status` runs nothing and prints which stages are already current.

Every stage writes into a folder named for the settings it ran under, so a stage
re-run with those settings unchanged finds its own output and skips, and a
changed setting writes somewhere new rather than over it. Every subcommand
accepts --set section.key=value (repeatable) and --dump-config PATH. Paths and
the model live in the config.
"""
import argparse
import glob
import os

from orcann.configLoader import Config
from orcann.logging_setup import LEVELS, DEFAULT_LEVEL, configure as configure_logging

_EXTS = ("*.nd2", "*.tif", "*.tiff", "*.npy")
# Sidecars written beside each corrected movie in a motion-correction store. They
# must not be mistaken for recordings: <identity>.json is already excluded (not a
# movie extension), but <identity>_shifts.npy matches the *.npy glob and would
# otherwise be counted as an extra recording, doubling the infer/segment array
# size and mapping tasks onto shift arrays.
_SIDECAR_SUFFIXES = ("_shifts.npy",)


def list_recordings(dirpath, task_id=None):
    """Sorted movie files in a directory; with task_id, just the 1-based Nth (SGE array).

    Files only, so the store folders under a results root are passed over.
    """
    if not dirpath or not os.path.isdir(dirpath):
        return []
    files = sorted(sum([glob.glob(os.path.join(dirpath, e)) for e in _EXTS], []))
    files = [f for f in files
             if not os.path.basename(f).endswith(_SIDECAR_SUFFIXES)]
    if task_id is not None:
        return [files[task_id - 1]] if 1 <= task_id <= len(files) else []
    return files


def identified_recordings(dirpath, task_id=None):
    """``[(path, identity)]`` for the movies in a directory, in path order.

    Identity is minted here, at the edge of the pipeline, by reading two frames
    of each file; every stage below copies the string rather than deriving it
    again. Two files that mint the same identity are the same recording twice —
    the same pixels in two containers, say — and the run stops naming both,
    because either choice would silently drop one of them and the results would
    not say which was used.
    """
    from orcann.pipeline.extraction import IntakeError
    from orcann.pipeline.provenance import recording_identity

    files = list_recordings(dirpath)
    seen = {}
    for f in files:
        # A file that cannot be read stops the run rather than being passed over:
        # a directory the pipeline cannot fully account for is not a cohort. It is
        # named here, at enumeration, because that is where it is noticed -- and
        # every task of an array will say the same thing, which is the point.
        try:
            seen.setdefault(recording_identity(f), []).append(f)
        except (IntakeError, OSError, ValueError) as e:
            raise SystemExit(
                f"{f}: cannot be read as a recording, so the recordings in "
                f"{dirpath} cannot be enumerated ({e}). Move it out of the "
                f"directory or correct it.")
    clashes = {i: fs for i, fs in seen.items() if len(fs) > 1}
    if clashes:
        lines = "\n".join(f"  {i}\n    " + "\n    ".join(fs)
                          for i, fs in sorted(clashes.items()))
        raise SystemExit(
            f"{dirpath}: these files are the same recording under more than one "
            f"name -- same frame count, frame size and end frames:\n{lines}\n"
            f"Remove or rename all but one of each.")
    pairs = [(f, i) for i, fs in seen.items() for f in fs]
    pairs.sort()
    if task_id is not None:
        return [pairs[task_id - 1]] if 1 <= task_id <= len(pairs) else []
    return pairs


def store_movies(store, store_identity, task_id=None):
    """``[(path, identity)]`` for the movies a motion-correction store offers.

    A store this pipeline wrote names its movies by identity and vouches for
    them with a record each, so nothing needs re-reading. Movies corrected
    elsewhere and dropped into ``paths.pre_processed`` have neither, so they are
    identified the same way raw recordings are.
    """
    from orcann.pipeline import provenance as prov

    if store_identity == prov.EXTERNAL:
        return identified_recordings(store, task_id)
    out = []
    for ident in prov.list_flat_records(store, prov.STAGE_MOTION, task_id):
        mv = prov.movie_in_store(store, ident)
        if mv is None:
            raise SystemExit(
                f"{store}: {ident} has a motion-correction record but no movie "
                f"beside it. Delete the record and re-run motion_correction.")
        out.append((mv, ident))
    return out


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


def _status(cfg):
    """Which stages are current for which recordings, and where their output is.

    The one place that answers "will this run do anything?" without running it.
    Every state it prints is read from a record, so it reports what a stage will
    actually decide rather than what the config intends.
    """
    from orcann.pipeline import inference as infer
    from orcann.pipeline import provenance as prov

    mc_id = prov.motion_identity(cfg)
    try:
        where = prov.motion_store(cfg)
    except prov.ProvenanceError:
        where = (prov.store_dir(cfg.paths.pre_processed, mc_id), mc_id)
    try:
        chain = prov.chain(cfg, motion=where)
    except prov.ProvenanceError as e:
        print(f"status: {e}")
        return False

    stores = [("motion_correction", chain.motion_dir),
              ("infer", prov.store_dir(cfg.paths.infer, chain.infer)),
              ("segment", prov.store_dir(cfg.paths.spatial, chain.segment)),
              ("activity", prov.store_dir(cfg.paths.activity, chain.activity)),
              ("analysis", prov.store_dir(cfg.paths.analysis, chain.analysis))]
    print(f"model in service: {chain.model}")
    for name, d in stores:
        print(f"  {name:18s} {d}{'' if os.path.isdir(d) else '   (not yet)'}")

    # Raw recordings when they are readable, and whatever the corrected store
    # holds otherwise: a cohort whose raw files live on the cluster still has a
    # status worth printing on the machine holding the results.
    try:
        pairs = identified_recordings(cfg.paths.raw)
    except SystemExit as e:
        print(f"\n{e}")
        return False
    idents = [i for _, i in pairs]
    if not idents:
        idents = prov.list_flat_records(chain.motion_dir, prov.STAGE_MOTION) \
            if chain.motion != prov.EXTERNAL else [i for _, i in store_movies(*where)]

    done = {
        "mc": set(prov.list_flat_records(chain.motion_dir, prov.STAGE_MOTION)),
        "infer": set(prov.list_dir_records(
            prov.store_dir(cfg.paths.infer, chain.infer),
            infer.META_JSON, prov.STAGE_INFER)),
        "segment": set(prov.list_dir_records(
            prov.store_dir(cfg.paths.spatial, chain.segment),
            os.path.join(infer.DATA_DIRNAME, infer.META_JSON), prov.STAGE_SEGMENT)),
        "activity": set(prov.list_dir_records(
            prov.store_dir(cfg.paths.activity, chain.activity),
            prov.RUN_INFO, prov.STAGE_ACTIVITY)),
    }
    if chain.motion == prov.EXTERNAL:
        done["mc"] = set(idents)          # corrected elsewhere; taken as given

    # Recordings in a store with no raw file behind them: a recording that was
    # replaced or removed since it was processed. Its results are not wrong, but
    # nothing upstream will refresh them, and the analysis stage refuses a cohort
    # holding one acquisition twice -- so they are shown rather than left out.
    stored = sorted(set().union(*done.values()) - set(idents))
    cols = ["mc", "infer", "segment", "activity"]
    print(f"\n{'recording':40s} " + " ".join(f"{c:>9s}" for c in cols))
    for i in sorted(idents) + stored:
        marks = " ".join(f"{('current' if i in done[c] else '-'):>9s}" for c in cols)
        print(f"{i:40s} {marks}{'   (no raw file)' if i in stored else ''}")
    if not idents and not stored:
        print("(no recordings)")
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
            ("motion_correction", "data/raw -> a pre_processed store (caiman env)"),
            ("infer", "corrected movies -> an infer store (GPU; cache prob maps)"),
            ("activity", "a segment store -> an activity store (dF/F0, OASIS, gallery)")]:
        p = sub.add_parser(name, help=helptext)
        _common(p)
        p.add_argument("--force", action="store_true",
                       help="redo recordings that are already current")
        p.add_argument("--task-id", type=int, default=None,
                       help="process only the Nth recording (SGE array)")

    p = sub.add_parser("segment", help="an infer store + movies -> a segment store (CPU; threshold+extract)")
    _common(p)
    p.add_argument("--force", action="store_true",
                   help="redo recordings that are already current")
    p.add_argument("--task-id", type=int, default=None,
                   help="process only the Nth recording (SGE array)")
    p.add_argument("--sweep", action="append", default=[], metavar="key=v1,v2,...",
                   help="preview a parameter grid (montage + table), no extraction; repeatable")

    p = sub.add_parser("analysis", help="an activity store -> an analysis store (group figures + tables)")
    _common(p)
    p.add_argument("--force", action="store_true",
                   help="rebuild even if the outputs are current")

    p = sub.add_parser("train_spatial", help="train the segmenter")
    _common(p)
    p.add_argument("--synthetic", action="store_true", help="self-test on synthetic data")

    p = sub.add_parser("status", help="which stages are current, for which recordings")
    _common(p)

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
    elif a.stage == "status":
        ok = _status(cfg)
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
