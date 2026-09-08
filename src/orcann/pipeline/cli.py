"""The single `orcann` entry point: subcommands that each read the YAML config.

    orcann motion_correction --config config.yaml [--force] [--task-id N]
    orcann infer             --config config.yaml [--force] [--task-id N]
    orcann segment           --config config.yaml [--force] [--task-id N] [--sweep k=v1,v2]
    orcann activity          --config config.yaml [--force] [--task-id N]
    orcann analysis          --config config.yaml [--force]
    orcann train_spatial     --config config.yaml [--synthetic]
    orcann status            --config config.yaml
    orcann runs              --config config.yaml
    orcann snapshot          --config config.yaml

Spatial detection is split: `infer` runs the GPU model once and caches the
probability map; `segment` (CPU, no model) thresholds + extracts and is the cheap
stage you re-run while tuning. `activity` then baseline-corrects, deconvolves
(OASIS) and renders the per-recording HTML gallery; `analysis` aggregates across
recordings. `status` runs nothing and prints which stages are already current.

Every stage writes into a folder named for the stage and `run.key`, and records
beside each output the settings it ran under; a stage re-run with those settings
unchanged skips, and a changed setting makes the affected recordings stale and
recomputes them. Every subcommand accepts --set section.key=value (repeatable)
and --dump-config PATH. Paths and the model live in the config.
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


def identified_recordings(dirpath):
    """``[(path, recording, content_tag)]`` for the movies in a directory.

    The tag is minted here, at the edge of the pipeline, by reading two frames of
    each file; every stage below copies it out of the record above rather than
    re-reading the movie. Two kinds of collision stop the run rather than being
    resolved by guessing, because either choice would silently drop a recording
    and the results would not say which was used: two files that mint the same
    tag are the same recording twice, and two files that reduce to the same
    acquisition name would write into one another's output folder.
    """
    from orcann.pipeline.extraction import IntakeError
    from orcann.pipeline import provenance as prov

    files = list_recordings(dirpath)
    by_tag, by_name, out = {}, {}, []
    for f in files:
        # A file that cannot be read stops the run rather than being passed over:
        # a directory the pipeline cannot fully account for is not a cohort. It is
        # named here, at enumeration, because that is where it is noticed -- and
        # every task of an array will say the same thing, which is the point.
        try:
            tag = prov.content_tag(f)
        except (IntakeError, OSError, ValueError) as e:
            raise SystemExit(
                f"{f}: cannot be read as a recording, so the recordings in "
                f"{dirpath} cannot be enumerated ({e}). Move it out of the "
                f"directory or correct it.")
        rec_id = prov.recording_id(f)
        by_tag.setdefault(tag, []).append(f)
        by_name.setdefault(rec_id, []).append(f)
        out.append((f, rec_id, tag))

    same = {t: fs for t, fs in by_tag.items() if len(fs) > 1}
    if same:
        lines = "\n".join("  " + "\n  ".join(fs) for fs in same.values())
        raise SystemExit(
            f"{dirpath}: these files are the same recording under more than one "
            f"name -- same frame count, frame size and end frames:\n{lines}\n"
            f"Remove or rename all but one of each.")
    clash = {n: fs for n, fs in by_name.items() if len(fs) > 1}
    if clash:
        lines = "\n".join(f"  {n}\n    " + "\n    ".join(fs)
                          for n, fs in sorted(clash.items()))
        raise SystemExit(
            f"{dirpath}: these files reduce to the same recording name, so they "
            f"would write into one folder:\n{lines}\n"
            f"Rename all but one of each.")

    out.sort()
    return out


def store_movies(store, external):
    """``[(path, recording, content_tag)]`` for the movies a store offers.

    A store this pipeline wrote names its movies by recording and vouches for
    each with a record, which carries the tag of the raw file the correction was
    made from -- so the tag follows the chain down and a raw file replaced after
    correction reaches every stage below. Movies corrected elsewhere have no
    record, so they are identified the way raw recordings are.
    """
    from orcann.pipeline import provenance as prov

    if external:
        return identified_recordings(store)
    out = []
    for rec_id in prov.list_flat_records(store, prov.STAGE_MOTION):
        mv = prov.movie_in_store(store, rec_id)
        if mv is None:
            raise SystemExit(
                f"{store}: {rec_id} has a motion-correction record but no movie "
                f"beside it. Delete the record and re-run motion_correction.")
        rec = prov.record_of(store, rec_id, None, prov.STAGE_MOTION)
        out.append((mv, rec_id, rec.get("key", {}).get("recording", {}).get("content_tag")))
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


def _status(cfg, as_json=False):
    """Which stages are current for which recordings, where, and why not.

    The one place that answers "will this run do anything?" without running it.
    Every state it prints is read from a record and compared against the key
    this config implies, so it reports what a stage will actually decide. With
    --json it prints the same computation for the submission scripts, so what a
    person reads and what a launcher acts on cannot drift apart.
    """
    import json as _json

    from orcann.pipeline import provenance as prov

    try:
        ch = prov.tolerant_chain(cfg)
        plans = prov.plan(cfg, ch)
        ana = prov.analysis_plan(cfg, ch, plans[-1])
    except prov.ProvenanceError as e:
        if as_json:
            print(_json.dumps({"error": str(e)}))
            return False
        print(f"status: {e}")
        return False

    if as_json:
        print(_json.dumps({"run_key": ch.run_key, "model": ch.model,
                           "stages": [p.to_dict() for p in plans + [ana]]},
                          indent=2))
        return True

    print(f"run key         : {ch.run_key}")
    print(f"model in service: {ch.model}")
    for p in plans:
        where = p.store or p.write_dir
        note = ("   (not yet)" if p.store is None
                else f"   (reusing {os.path.basename(p.store)})" if p.reused
                else "")
        print(f"  {p.stage:18s} {where}{note}")
    if ch.motion_external:
        print("  motion_correction is external: movies taken as given")

    idents = sorted({r for p in plans for r in p.recordings})
    cols = [(p.stage, max(9, len(p.stage))) for p in plans]
    width = max([len(r) for r in idents] + [len("recording")])
    print(f"\n{'recording':{width}s} " + " ".join(f"{c:>{w}s}" for c, w in cols))
    for rec in idents:
        row = []
        for p in plans:
            row.append("current" if rec in p.current
                       else "stale" if rec in p.stale else "-")
        print(f"{rec:{width}s} " + " ".join(f"{m:>{w}s}"
                                            for m, (_, w) in zip(row, cols)))
    if not idents:
        print("(no recordings)")

    reasons = {}
    for p in plans:
        for rec, why in p.stale.items():
            reasons.setdefault((p.stage, why), []).append(rec)
    if reasons:
        print("\nstale because:")
        for (stage, why), recs in sorted(reasons.items()):
            print(f"  {stage:18s} {why}   ({len(recs)} recording(s))")

    state = ("current" if ana.current else ana.blocked or
             "  ".join(ana.stale.values()) or "no record")
    print(f"\nanalysis: {state} ({len(ana.recordings)} recording(s) -> "
          f"{ana.write_dir})")
    return True


def _snapshot(cfg):
    """Write the config a submitted run will read, and print where it went.

    Called by the submission scripts before any job is queued, so every task of
    every array reads one file that cannot change under them. It is the resolved
    config, overrides included, with every path already absolute — the jobs run
    from the repo root and must not re-anchor a relative path against it.
    """
    from orcann.pipeline import provenance as prov

    try:
        d = prov.run_dir(cfg)
    except prov.ProvenanceError as e:
        raise SystemExit(str(e))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "config.yaml")
    cfg.dump(path)
    print(path)
    return True


def _runs(cfg):
    """Every run key present under the results roots, and how far each got.

    The answer to "which run was that", which is the question a name without a
    settings digest in it cannot answer on its own.
    """
    from orcann.pipeline import inference as infer
    from orcann.pipeline import provenance as prov

    seg_rel = os.path.join(infer.DATA_DIRNAME, infer.META_JSON)
    layout = [(prov.STAGE_MOTION, cfg.paths.pre_processed, None),
              (prov.STAGE_INFER, cfg.paths.infer, infer.META_JSON),
              (prov.STAGE_SEGMENT, cfg.paths.spatial, seg_rel),
              (prov.STAGE_ACTIVITY, cfg.paths.activity, prov.RUN_INFO),
              (prov.STAGE_ANALYSIS, cfg.paths.analysis, None)]

    counts = {}
    for stage, root, rel in layout:
        for run_key, d in prov.list_stores(root, stage):
            n = (len(prov.list_flat_records(d, stage)) if rel is None
                 else len(prov.list_dir_records(d, rel, stage)))
            counts.setdefault(run_key, {})[stage] = n
    if not counts:
        print("no runs yet")
        return True
    width = max(len(k) for k in counts)
    for run_key in sorted(counts):
        done = "  ".join(f"{s} {counts[run_key][s]}"
                         for s, _, _ in layout if s in counts[run_key])
        here = "  <- run.key" if run_key == cfg.run.key else ""
        print(f"{run_key:{width}s}  {done}{here}")
    print("\norcann status shows what each stage would do under the current config.")
    return True


def _common(sp):
    sp.add_argument("--config", default="config.yaml",
                    help="YAML config file (default: ./config.yaml)")
    sp.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="section.key=value",
                    help="one-off override of a config value (repeatable)")
    # No default here: `status --json` writes a document to stdout and has to be
    # the only thing on it, so it drops to ERROR unless a level was asked for.
    sp.add_argument("--log-level", default=None, choices=LEVELS,
                    help=f"how much the run reports (default: {DEFAULT_LEVEL})")
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
    p.add_argument("--json", action="store_true",
                   help="print the same plan as JSON, for the submission scripts")

    p = sub.add_parser("runs", help="which run keys exist, and how far each got")
    _common(p)

    p = sub.add_parser("snapshot",
                       help="write <paths.runs>/<run.key>/config.yaml for a submitted run")
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
    quiet = getattr(a, "json", False)
    configure_logging(a.log_level or ("ERROR" if quiet else DEFAULT_LEVEL))
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
        ok = _status(cfg, as_json=a.json)
    elif a.stage == "runs":
        ok = _runs(cfg)
    elif a.stage == "snapshot":
        ok = _snapshot(cfg)
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
