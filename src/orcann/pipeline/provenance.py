"""Two questions, answered in two places: which recording is this, and is the
result on disk still the one this config asks for.

**A recording is named by its acquisition.** ``D109_3-63_040226_R7``, the name
the microscope gave it, which the analysis stage also parses genotype and day
out of. What the file *holds* is summarised as a content tag — frame count,
frame size, dtype, and the first and last frame — and that tag goes in the
record, not in the name. Replace a file with different data under the same name
and the tag moves, so every result made from it reads as out of date and is
recomputed. The tag is a sample, not a proof: a file whose ends and shape match
is treated as the same recording, because reading the middle costs the whole
movie and the person who filled ``data/raw`` is the one who knows what is in it.

**A store is named by the stage that wrote it and the run key.**
``segment_outputs_main``. The run key comes from ``config.yaml`` and is chosen
by a person; nothing about the settings is encoded in the name, because a name
that encodes half the settings is read as though it encoded all of them.

**What decides a re-run is the key on the record.** Every record carries the
settings its stage ran under, the digest of the record above it, and the
recording's content tag, as named fields. Currency is a field-by-field
comparison against the key the config implies now, and the first field that
differs is the reason, which is what ``orcann status`` prints. A record that
cannot be read at the current schema counts as absent, so a run killed halfway
leaves work invisible rather than half-trusted, and every stage writes its
record last.

**A store is not tied to the run key that made it.** The key deliberately
excludes the run key, so a stage whose settings are unchanged finds an earlier
run's store by scanning the sibling stores under its root, reads it, and
computes nothing. That is why a run may have no store of its own for a stage it
did not need to run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from orcann.run_info import RunInfoError, json_safe, read_record

# The ``stage`` a record opens with, and the name a reader checks it against.
# One spelling per stage, here, so a reader cannot ask for a stage by a name the
# writer never used and be told the record is missing.
STAGE_MOTION = "motion_correction"
STAGE_INFER = "infer"
STAGE_SEGMENT = "segment"
STAGE_ACTIVITY = "activity"
STAGE_ANALYSIS = "analysis"

STAGES = (STAGE_MOTION, STAGE_INFER, STAGE_SEGMENT, STAGE_ACTIVITY, STAGE_ANALYSIS)

# What separates the stage name from the run key in a store's name. Spelled out
# rather than abbreviated: the folder is read far more often than it is typed.
STORE_INFIX = "_outputs_"

# Where each stage keeps the record that says a recording is done, relative to
# that recording's folder in the store.
RUN_INFO = "run_info.json"

# A run key names directories, so it is held to what is safe in a path. It may
# contain underscores; store names are parsed by stripping a known stage prefix,
# so the remainder is the key whatever it contains.
_RUN_KEY_OK = re.compile(r"^[A-Za-z0-9._-]+$")

_HEX = re.compile(r"^[0-9a-f]+$")

_MOVIE_EXTS = (".nd2", ".tif", ".tiff", ".npy")


class ProvenanceError(Exception):
    """A key could not be built, or a store could not be found."""


# KEYS -- what a stage ran under, and what makes an earlier result out of date.

def key_digest(key: Dict) -> str:
    """sha256 over a key, stable across processes and insertion order.

    JSON with sorted keys rather than ``repr`` of a dict, whose order is the
    order it was built in. Types survive the encoding, so 0.3 and "0.3" stay
    different settings — one reaches a threshold as a float and the other would
    not reach it at all.
    """
    text = json.dumps(json_safe(key), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _fields(section) -> Dict[str, object]:
    """Every value in a config section, by name.

    A whole section rather than a chosen few: a setting added to the section
    later is then in the key from the moment it exists, where a hand-written list
    would silently leave it out and let a changed setting reuse an old result.
    """
    return {k: v for k, v in sorted(vars(section).items())
            if not k.startswith("_")}


# Bump one when that stage computes something different from the same settings --
# a changed algorithm, a changed default the config does not carry, a fixed
# defect. Nothing detects that you should have: a stage whose code changed
# without its version will reuse output it would no longer produce. The package
# version and commit are recorded beside every record, but deliberately not in
# the key, because invalidating a GPU cohort on a comment change is worse than
# the gap this leaves.
# 2: the motion QC statistics were being computed from the elastic worst-patch
#    summary rather than from a whole-frame trajectory. max_shift_* and the
#    saved _shifts.npy now come from shifts_rig; the border crop still comes
#    from the worst patch, which is what it wants. The corrected movies are
#    unchanged -- same NoRMCorre call, same crop -- but the recorded motion
#    metadata is not, and the QC gate downstream reads it.
MOTION_VERSION = 2
INFER_VERSION = 1
SEGMENT_VERSION = 1
ACTIVITY_VERSION = 1
ANALYSIS_VERSION = 1

_VERSIONS = {STAGE_MOTION: MOTION_VERSION, STAGE_INFER: INFER_VERSION,
             STAGE_SEGMENT: SEGMENT_VERSION, STAGE_ACTIVITY: ACTIVITY_VERSION,
             STAGE_ANALYSIS: ANALYSIS_VERSION}

# The upstream a stage records when its input was corrected outside this
# pipeline -- the documented route for movies dropped into paths.pre_processed.
# It stands in for settings that are unknown by definition; the recording is
# still identified by content, so a swapped file is still noticed.
EXTERNAL = {"stage": STAGE_MOTION, "external": True, "digest": None}


def _params(cfg, stage: str, model_digest: Optional[str] = None) -> Dict:
    """The settings that change what one stage computes, and nothing else.

    Not the run key, and not which recordings are in the cohort: a store whose
    settings are unchanged must read as unchanged, or an earlier run's output
    could never be reused and a cohort that gains a recording would invalidate
    the twenty-two already done.
    """
    if stage == STAGE_MOTION:
        return _fields(cfg.motion_correction)
    if stage == STAGE_INFER:
        # Keyed on the model's digest, not the selector and not its identity.
        # The selector is often "latest", which resolves to a different model
        # over time and could never invalidate anything; the identity carries a
        # timestamp, so the same weights promoted twice would send the whole GPU
        # stage over maps it had already computed.
        return {"model": model_digest, "resize_to": cfg.spatial.resize_to}
    if stage == STAGE_SEGMENT:
        return {"threshold": cfg.spatial.threshold,
                "min_area": cfg.spatial.min_area,
                "min_radius": cfg.spatial.min_radius}
    if stage == STAGE_ACTIVITY:
        return {"baseline": _fields(cfg.baseline),
                "deconvolution": _fields(cfg.deconvolution),
                "frame_rate": cfg.imaging.frame_rate,
                "indicator": cfg.imaging.indicator}
    if stage == STAGE_ANALYSIS:
        return {"analysis": _fields(cfg.analysis),
                "frame_rate": cfg.imaging.frame_rate}
    raise ProvenanceError(f"no such stage: {stage!r}")


def stage_key(cfg, stage: str, *, upstream: Optional[Dict] = None,
              content_tag: Optional[str] = None,
              cohort: Optional[List] = None,
              model_digest: Optional[str] = None) -> Dict:
    """The key one stage runs one recording under.

    ``upstream`` is the link to the record above — ``{"stage", "digest"}``, or
    ``EXTERNAL``. ``content_tag`` identifies the recording for the stages that
    read a movie. ``cohort`` is for ``analysis``, which has no recording of its
    own and must instead notice when the set of recordings it aggregated changes.
    """
    key: Dict[str, object] = {"stage": stage, "stage_version": _VERSIONS[stage],
                              "params": _params(cfg, stage, model_digest)}
    if upstream is not None:
        key["upstream"] = upstream
    if content_tag is not None:
        key["recording"] = {"content_tag": content_tag}
    if cohort is not None:
        key["cohort"] = cohort
    return key


def store_digest(key: Dict) -> str:
    """The digest of a key with its per-recording parts removed.

    What a downstream stage links to. It has to be the same value for every
    recording in a store, or two recordings processed under identical settings
    would key their outputs differently and no store could be matched as a
    whole.
    """
    return key_digest({k: v for k, v in key.items()
                       if k not in ("recording", "cohort")})


def stage_keys(cfg, ch: "Chain") -> Dict[str, Dict]:
    """Every stage's key, top to bottom, with each linked to the one above.

    Derived from the config alone: an upstream link is the digest of the
    upstream *key*, not of any record, so the whole chain follows from
    ``config.yaml`` and a change at any level reaches every level below it.
    """
    keys = {STAGE_MOTION: stage_key(cfg, STAGE_MOTION)}
    up = EXTERNAL if ch.motion_external else {
        "stage": STAGE_MOTION, "digest": store_digest(keys[STAGE_MOTION])}
    keys[STAGE_INFER] = stage_key(cfg, STAGE_INFER, upstream=up,
                                  model_digest=ch.model_digest)
    for stage, above in ((STAGE_SEGMENT, STAGE_INFER),
                         (STAGE_ACTIVITY, STAGE_SEGMENT),
                         (STAGE_ANALYSIS, STAGE_ACTIVITY)):
        keys[stage] = stage_key(
            cfg, stage,
            upstream={"stage": above, "digest": store_digest(keys[above])})
    return keys


def _difference(have, want, path: str = "") -> Optional[str]:
    """The first field in which two keys differ, phrased for a person.

    Walked in the key's own order so the reason names the most specific thing
    that moved, and reported as one line because that line is what ``status``
    prints next to a recording.
    """
    if isinstance(want, dict) and isinstance(have, dict):
        for k in want:
            here = f"{path}.{k}" if path else k
            if k not in have:
                return f"{here} not recorded"
            d = _difference(have[k], want[k], here)
            if d:
                return d
        extra = [k for k in have if k not in want]
        return f"{path}.{extra[0]} no longer a setting" if extra else None
    if isinstance(want, list) and isinstance(have, list):
        if have == want:
            return None
        if len(have) != len(want):
            return f"{path} changed ({len(have)} -> {len(want)} entries)"
        # Same length, so it is a member that moved rather than the membership.
        # Naming the first one is what makes a cohort reason actionable: it says
        # which recording was reprocessed, not merely that one was.
        for a, b in zip(have, want):
            if a != b:
                name = a[0] if isinstance(a, list) and a else a
                return f"{path} entry changed ({name})"
        return f"{path} changed"
    if have != want:
        return f"{path}  {_short(have)} -> {_short(want)}"
    return None


def _short(value) -> str:
    """A value as it reads in a one-line reason.

    A digest is abbreviated because the reason is for a person deciding what to
    do, and no two digests that differ at all differ only past the eighth
    character in a way anyone will act on.
    """
    if isinstance(value, str) and len(value) == 64 and _HEX.match(value):
        return f"{value[:8]}..."
    return repr(value)


def is_current(record_path: str, stage: str, key: Dict) -> Tuple[bool, str]:
    """Whether a record is present, readable, and made under ``key``.

    Returns the reason it is not, which is the whole point of comparing named
    fields rather than one digest: "params.threshold 0.3 -> 0.6" tells a person
    what to do, and "the digest differs" tells them to go and diff two logs.
    """
    try:
        doc = read_record(record_path, expect_stage=stage)
    except (RunInfoError, OSError, ValueError):
        return False, "no record"
    have = doc.get("key")
    if not isinstance(have, dict):
        return False, "record carries no key"
    diff = _difference(have, key)
    return (False, diff) if diff else (True, "")


# STORES -- one folder per stage per run key, under the root the config names.

def check_run_key(run_key: str) -> str:
    """The run key, or a refusal. It names directories, so it is checked once."""
    if not run_key or not _RUN_KEY_OK.match(run_key):
        raise ProvenanceError(
            f"run.key {run_key!r} is not usable in a directory name. Use "
            f"letters, digits, dot, dash or underscore.")
    return run_key


def run_dir(cfg) -> str:
    """Where a run keeps the config it was launched from.

    One directory per run key, holding the snapshot the cluster jobs read. It is
    the plan the run was submitted under, so editing ``config.yaml`` while an
    array sits in the queue changes nothing about what those jobs compute.
    """
    return os.path.join(cfg.paths.runs, check_run_key(cfg.run.key))


def store_name(stage: str, run_key: str) -> str:
    if stage not in STAGES:
        raise ProvenanceError(f"no such stage: {stage!r}")
    return f"{stage}{STORE_INFIX}{check_run_key(run_key)}"


def store_dir(root: str, stage: str, run_key: str) -> str:
    return os.path.join(root, store_name(stage, run_key))


def parse_store(name: str) -> Optional[Tuple[str, str]]:
    """``(stage, run_key)`` for a store folder, or None for anything else.

    The stage prefix is matched first and the whole remainder is the key, so a
    key containing underscores parses. No stage name is a prefix of another
    followed by the infix, so at most one prefix can match.
    """
    for stage in STAGES:
        head = stage + STORE_INFIX
        if name.startswith(head) and len(name) > len(head):
            return stage, name[len(head):]
    return None


def list_stores(root: str, stage: Optional[str] = None) -> List[Tuple[str, str]]:
    """``[(run_key, directory)]`` for the stores under ``root``, sorted by key."""
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        parsed = parse_store(name) if os.path.isdir(d) else None
        if parsed and (stage is None or parsed[0] == stage):
            out.append((parsed[1], d))
    return out


def _store_key(store: str, stage: str, record_rel: str) -> Optional[Dict]:
    """The recording-independent key a store was written under, or None.

    Read from the first record that parses, because the records are the source
    of truth and a separate store-level file would be a second copy to keep
    consistent. Every record in a store shares this part of its key; only the
    recording block differs.
    """
    if not os.path.isdir(store):
        return None
    for entry in sorted(os.listdir(store)):
        path = (os.path.join(store, entry) if record_rel is None
                else os.path.join(store, entry, record_rel))
        try:
            doc = read_record(path, expect_stage=stage)
        except (RunInfoError, OSError, ValueError):
            continue
        key = doc.get("key")
        if isinstance(key, dict):
            return {k: v for k, v in key.items()
                    if k not in ("recording", "cohort")}
    return None


def find_store(root: str, stage: str, run_key: str, key: Dict,
               record_rel: Optional[str]) -> Optional[str]:
    """The store holding results for ``key``: this run's, or an earlier run's.

    This run's store wins when it matches, so a run reads its own output in
    preference to anything else. Otherwise the sibling stores are scanned and
    the first whose key matches is adopted, which is how a run that changed only
    a downstream setting reuses the inference it did not need to redo. The
    adopted directory is recorded by the stage that read it, since after
    adoption there is no store under this run key to point at.
    """
    want = {k: v for k, v in key.items() if k not in ("recording", "cohort")}
    own = store_dir(root, stage, run_key)
    if _store_key(own, stage, record_rel) == want:
        return own
    for other_key, d in list_stores(root, stage):
        if other_key != run_key and _store_key(d, stage, record_rel) == want:
            return d
    return None


def output_store(root: str, stage: str, run_key: str, key: Dict,
                 record_rel: Optional[str]) -> str:
    """Where a stage writes: the store already holding results for these
    settings, or a new one named for this run.

    A run key names a store when its settings are new. When an earlier run
    computed exactly these settings, its store is the one that gets added to,
    because two directories of identical results would be a copy to keep in step
    and a stage below choosing between them would have to guess which was
    complete. Which run produced each recording is in the record, not the path.
    """
    return (find_store(root, stage, run_key, key, record_rel)
            or store_dir(root, stage, run_key))


def require_store(root: str, stage: str, run_key: str, key: Dict,
                  record_rel: Optional[str]) -> str:
    """``find_store``, or a refusal naming the run keys that are there.

    Reached when a stage's input does not exist for the settings in the config —
    either the upstream stage has not run under them, or the config changed
    after it did. Both are the same instruction to the person running it, and
    adopting a nearby store would attribute one run's results to another's
    settings.
    """
    found = find_store(root, stage, run_key, key, record_rel)
    if found:
        return found
    have = [k for k, _ in list_stores(root, stage)]
    listing = ", ".join(have) if have else "(nothing)"
    raise ProvenanceError(
        f"no {stage} output for the settings in this config, under run key "
        f"{run_key!r} or any other. Run {stage}, or set run.key back to the run "
        f"whose settings you want.\n  {stage} output present for run keys: "
        f"{listing}\n  looked under: {root}")


def for_task(found: List, task_id: Optional[int]) -> Tuple[List, bool]:
    """What one SGE array task handles, and whether its index is past the end.

    Every stage in a chained run is sized from the same recording count, taken
    before any of them has produced anything. A task index beyond the end
    therefore means a stage above lost a recording, not that this stage has no
    input: it has nothing to do and exits saying so, rather than reporting an
    empty store and failing N tasks for one upstream failure.
    """
    if task_id is None:
        return found, False
    if 1 <= task_id <= len(found):
        return [found[task_id - 1]], False
    return [], bool(found)


def list_flat_records(store: str, stage: str) -> List[str]:
    """Recordings in a store that keeps one record file per recording.

    The motion-correction store: ``<recording>.tif`` beside ``<recording>.json``.
    Enumerating records rather than movies is what keeps a killed run invisible.
    """
    if not os.path.isdir(store):
        return []
    found = []
    for entry in sorted(os.listdir(store)):
        if not entry.endswith(".json"):
            continue
        try:
            read_record(os.path.join(store, entry), expect_stage=stage)
        except (RunInfoError, OSError, ValueError):
            continue
        found.append(entry[:-len(".json")])
    return found


def list_dir_records(store: str, record_rel: str, stage: str) -> List[str]:
    """Recordings in a store that keeps a directory per recording.

    The probability map may be there and truncated, but without a readable
    record beside it the recording is simply not done.
    """
    if not os.path.isdir(store):
        return []
    found = []
    for entry in sorted(os.listdir(store)):
        if not os.path.isdir(os.path.join(store, entry)):
            continue
        try:
            read_record(os.path.join(store, entry, record_rel), expect_stage=stage)
        except (RunInfoError, OSError, ValueError):
            continue
        found.append(entry)
    return found


def record_key(record_path: str, stage: str) -> Optional[Dict]:
    """The key a record was written under, or None when it cannot be read."""
    try:
        return read_record(record_path, expect_stage=stage).get("key")
    except (RunInfoError, OSError, ValueError):
        return None


def record_of(store: str, rec_id: str, record_rel: Optional[str],
              stage: str) -> Dict:
    """One recording's record from a store, or a refusal naming the file."""
    path = (os.path.join(store, rec_id + ".json") if record_rel is None
            else os.path.join(store, rec_id, record_rel))
    try:
        return read_record(path, expect_stage=stage)
    except (RunInfoError, OSError, ValueError) as e:
        raise ProvenanceError(str(e))


# RECORDINGS -- what a movie file is, and what it holds.

def content_tag(path: str) -> str:
    """Full digest of what a movie file holds, from its shape and its two ends.

    Reads two frames. The frame count is in the digest as well as the pixels, so
    a truncated or extended recording is a different recording even when the
    ends it still has are unchanged.
    """
    from orcann.pipeline.extraction import open_recording

    with open_recording(path) as rec:
        m = rec.meta
        idx = [0] if m.n_frames == 1 else [0, m.n_frames - 1]
        ends = rec.frames(idx)
    h = hashlib.sha256()
    h.update(f"{m.n_frames}x{m.height}x{m.width}:{m.dtype_in}".encode())
    h.update(np.ascontiguousarray(ends, np.float32).tobytes())
    return h.hexdigest()


def recording_id(path: str) -> str:
    """The acquisition name a movie file is filed under."""
    from orcann.pipeline.inference import recording_id as _rid

    return _rid(path)


def holds_movies(d: str) -> bool:
    """Whether a directory holds movie files of its own, store folders aside."""
    return os.path.isdir(d) and any(
        f.lower().endswith(_MOVIE_EXTS) for f in os.listdir(d))


def movie_in_store(store: str, rec_id: str) -> Optional[str]:
    """The corrected movie a store holds for one recording, or None."""
    for ext in (".tif", ".npy"):
        p = os.path.join(store, rec_id + ext)
        if os.path.isfile(p):
            return p
    return None


# THE CHAIN -- what one config resolves to before any stage reads anything.

@dataclass(frozen=True)
class Chain:
    """The run key, where the corrected movies are, and the model in service.

    Store names follow from the run key alone, so there is nothing else to
    resolve up front; which store a stage *reads* depends on keys that are only
    known per recording, and is resolved there.
    """
    run_key: str
    motion_dir: str
    motion_external: bool
    model: str
    model_digest: Optional[str]
    model_path: str


def motion_source(cfg) -> Tuple[str, bool]:
    """``(directory, external)`` for the corrected movies the stages below read.

    The store this run key names, if it is there; failing that, an earlier run's
    motion store written under the same settings; failing that,
    ``paths.pre_processed`` itself when it holds movies, which is the documented
    route for recordings corrected outside this pipeline. A store wins over
    loose movies beside it, so a stray file in the root cannot divert a run that
    has a corrected cohort of its own.
    """
    root = cfg.paths.pre_processed
    found = find_store(root, STAGE_MOTION, cfg.run.key,
                       stage_key(cfg, STAGE_MOTION), None)
    if found:
        return found, False
    if holds_movies(root):
        return root, True
    raise ProvenanceError(
        f"no motion-correction output for the settings in this config, and "
        f"{root} holds no movies: nothing to read. Run motion_correction, or "
        f"put movies corrected elsewhere in {root}.")


def chain(cfg, motion: Optional[Tuple[str, bool]] = None) -> Chain:
    """Resolve the run key, the corrected movies and the model, in one place.

    ``motion`` overrides where the corrected movies are taken to be. Only
    ``status`` passes it, so it can describe a pipeline whose first stage has not
    run yet; a stage that is about to read those movies wants the refusal.
    """
    from orcann.pipeline.model_io import ModelStoreError, read_identity, resolve_model

    check_run_key(cfg.run.key)
    motion_dir, external = motion if motion else motion_source(cfg)
    try:
        model, model_path = resolve_model(cfg.models.dir, cfg.models.spatial)
    except ModelStoreError as e:
        raise ProvenanceError(str(e))
    digest = read_identity(model_path).get("digest")
    return Chain(run_key=cfg.run.key, motion_dir=motion_dir,
                 motion_external=external, model=model, model_digest=digest,
                 model_path=model_path)



# THE PLAN -- what every stage would do, computed without running anything.

@dataclass(frozen=True)
class StagePlan:
    """One stage's situation under the config as it stands.

    ``store`` is where this stage's results are or would be read from, which may
    belong to another run key; ``write_dir`` is where this run would put new
    ones. ``blocked`` says why nothing can be counted yet — the stage above has
    not run, which is normal at the head of a chained submission and is not a
    reason to refuse.
    """
    stage: str
    store: Optional[str]
    write_dir: str
    reused: bool
    recordings: List[str]
    current: List[str]
    stale: Dict[str, str]
    blocked: Optional[str]

    @property
    def todo(self) -> int:
        return len(self.recordings) - len(self.current)

    def to_dict(self) -> Dict:
        return {"stage": self.stage, "store": self.store,
                "write_dir": self.write_dir, "reused": self.reused,
                "recordings": self.recordings, "current": self.current,
                "stale": self.stale, "blocked": self.blocked,
                "n_recordings": len(self.recordings), "n_todo": self.todo}


# Where each stage keeps a recording's record, relative to that recording's
# folder in the store. None is the motion store's flat <recording>.json beside
# the corrected movie.
def record_rel(stage: str) -> Optional[str]:
    from orcann.pipeline import inference as infer

    return {STAGE_MOTION: None,
            STAGE_INFER: infer.META_JSON,
            STAGE_SEGMENT: os.path.join(infer.DATA_DIRNAME, infer.META_JSON),
            STAGE_ACTIVITY: RUN_INFO,
            STAGE_ANALYSIS: RUN_INFO}[stage]


def stage_root(cfg, stage: str) -> str:
    return {STAGE_MOTION: cfg.paths.pre_processed,
            STAGE_INFER: cfg.paths.infer,
            STAGE_SEGMENT: cfg.paths.spatial,
            STAGE_ACTIVITY: cfg.paths.activity,
            STAGE_ANALYSIS: cfg.paths.analysis}[stage]


def tolerant_chain(cfg) -> Chain:
    """``chain``, but describing a workspace whose first stage has not run.

    A planner has to be able to say "motion_correction has not run yet"; only a
    stage about to read those movies wants the refusal.
    """
    try:
        motion = motion_source(cfg)
    except ProvenanceError:
        motion = (store_dir(cfg.paths.pre_processed, STAGE_MOTION, cfg.run.key),
                  False)
    return chain(cfg, motion=motion)


def content_tags(cfg, ch: Chain) -> Dict[str, str]:
    """The tag each recording is known by, from as far up the chain as is readable.

    The tag is minted from the raw file and copied downward, so the raw directory
    is the right place to read it. On a machine holding only results it is taken
    from the motion records instead, which means a raw file replaced elsewhere
    cannot be noticed there — a plan reports what it can see rather than
    implying it checked.
    """
    from orcann.pipeline.cli import identified_recordings, store_movies

    try:
        tags = {r: t for _, r, t in identified_recordings(cfg.paths.raw)}
    except SystemExit:
        tags = {}
    if not tags and not ch.motion_external:
        tags = {r: t for _, r, t in store_movies(ch.motion_dir, False)}
    return tags


def plan(cfg, ch: Optional[Chain] = None) -> List[StagePlan]:
    """Every stage's situation, in order, for one config.

    One computation behind ``orcann status``, ``orcann status --json`` and the
    submission scripts, so what a person is shown and what the launcher acts on
    cannot drift apart.
    """
    ch = ch or tolerant_chain(cfg)
    keys = stage_keys(cfg, ch)
    tags = content_tags(cfg, ch)
    plans: List[StagePlan] = []
    upstream_recs: Optional[List[str]] = sorted(tags)
    blocked_above = None if tags else "no recordings in the raw directory"

    for stage in (STAGE_MOTION, STAGE_INFER, STAGE_SEGMENT, STAGE_ACTIVITY):
        root, rel = stage_root(cfg, stage), record_rel(stage)
        if stage == STAGE_MOTION and ch.motion_external:
            # Movies corrected elsewhere are taken as given: there is no store to
            # find and nothing for this stage to do.
            plans.append(StagePlan(stage, ch.motion_dir, ch.motion_dir, False,
                                   list(upstream_recs or []),
                                   list(upstream_recs or []), {}, None))
            upstream_recs = list(upstream_recs or [])
            continue
        store = find_store(root, stage, ch.run_key, keys[stage], rel)
        write_dir = store or store_dir(root, stage, ch.run_key)
        recs = list(upstream_recs or [])
        current, stale = [], {}
        if store is not None:
            for rec in recs:
                path = (os.path.join(store, rec + ".json") if rel is None
                        else os.path.join(store, rec, rel))
                key = dict(keys[stage])
                tag = tags.get(rec)
                if tag is None:
                    tag = ((record_key(path, stage) or {})
                           .get("recording", {}).get("content_tag"))
                key["recording"] = {"content_tag": tag}
                ok, why = is_current(path, stage, key)
                (current.append(rec) if ok
                 else stale.__setitem__(rec, why))
        plans.append(StagePlan(stage, store, write_dir,
                               bool(store) and os.path.abspath(store) != os.path.abspath(write_dir),
                               recs, current, stale, blocked_above))
        # What this stage will hand down: the recordings it has finished, which
        # is what the stage below will actually find on disk.
        if store is None:
            upstream_recs, blocked_above = [], f"{stage} has not run"
        else:
            done = (list_flat_records(store, stage) if rel is None
                    else list_dir_records(store, rel, stage))
            upstream_recs = sorted(set(done) | set(current))
            blocked_above = None if upstream_recs else f"{stage} has produced nothing"
    return plans


def analysis_plan(cfg, ch: Optional[Chain] = None,
                  activity: Optional[StagePlan] = None) -> StagePlan:
    """The aggregate stage, whose cohort is part of what makes it out of date."""
    ch = ch or tolerant_chain(cfg)
    keys = stage_keys(cfg, ch)
    write_dir = output_store(cfg.paths.analysis, STAGE_ANALYSIS, ch.run_key,
                            keys[STAGE_ANALYSIS], None)
    src = (activity.store if activity is not None else
           find_store(cfg.paths.activity, STAGE_ACTIVITY, ch.run_key,
                      keys[STAGE_ACTIVITY], RUN_INFO))
    if src is None:
        return StagePlan(STAGE_ANALYSIS, None, write_dir, False, [], [], {},
                         "activity has not run")
    have = list_dir_records(src, RUN_INFO, STAGE_ACTIVITY)
    cohort = sorted([r, record_of(src, r, RUN_INFO, STAGE_ACTIVITY).get("key_digest")]
                    for r in have)
    ok, why = is_current(os.path.join(write_dir, RUN_INFO), STAGE_ANALYSIS,
                         dict(keys[STAGE_ANALYSIS], cohort=cohort))
    return StagePlan(STAGE_ANALYSIS, write_dir if ok else None, write_dir, False,
                     have, have if ok else [], {} if ok else {"cohort": why}, None)


def stage_inputs(cfg, stage: str) -> List[str]:
    """The recordings a stage would work through, current or not.

    What sizes an SGE array, so it is the enumeration the stage itself does and
    a task index means the same recording in the submitter and in the worker. It
    deliberately does not filter on currency: the set of stale recordings moves
    while the array sits in the queue, and a task that finds its own recording
    current exits in a second.
    """
    from orcann.pipeline import inference as infer
    from orcann.pipeline.cli import identified_recordings, store_movies

    if stage in ("motion_correct", STAGE_MOTION):
        return [r for _, r, _ in identified_recordings(cfg.paths.raw)]
    ch = chain(cfg)
    keys = stage_keys(cfg, ch)
    if stage == STAGE_INFER:
        return [r for _, r, _ in store_movies(ch.motion_dir, ch.motion_external)]
    if stage == STAGE_SEGMENT:
        store = require_store(cfg.paths.infer, STAGE_INFER, ch.run_key,
                              keys[STAGE_INFER], infer.META_JSON)
        return list_dir_records(store, infer.META_JSON, STAGE_INFER)
    if stage == STAGE_ACTIVITY:
        rel = os.path.join(infer.DATA_DIRNAME, infer.META_JSON)
        store = require_store(cfg.paths.spatial, STAGE_SEGMENT, ch.run_key,
                              keys[STAGE_SEGMENT], rel)
        return list_dir_records(store, rel, STAGE_SEGMENT)
    if stage == STAGE_ANALYSIS:
        store = require_store(cfg.paths.activity, STAGE_ACTIVITY, ch.run_key,
                              keys[STAGE_ACTIVITY], RUN_INFO)
        return list_dir_records(store, RUN_INFO, STAGE_ACTIVITY)
    raise ProvenanceError(f"no such stage: {stage!r}")
