"""Two identities: which recording this is, and which settings produced a folder.

They answer different questions and are computed in different places.

**A recording's identity** is its filename plus a tag over what the file holds —
frame count, frame size, dtype, and the first and last frame. It is minted once,
where raw recordings enter the pipeline, and every stage below copies the string:
``<store>/REC_5c1d.tif``, then ``<store>/REC_5c1d/prob.npy``, then
``<store>/REC_5c1d/data/traces.npy``. Two files whose names normalise alike but
whose contents differ get different tags, so a raw recording and a denoised
export of it are two recordings rather than one collision; replace a file with
different data under the same name and the identity moves, so the work re-runs
and the old result stays where it was instead of being overwritten.

The tag is a sample, not a proof. A file whose ends and shape match is treated as
the same recording — the middle is not read, because reading it costs the whole
movie and the person running the pipeline is the one who knows what they put in
``data/raw``.

**A store's identity** is the settings a stage ran under, plus the identity of
the store it read: ``mc_auto_ms20_3f9ac41e``. Same settings, same name, so a
stage finds its own earlier output and skips; different settings, different name,
so nothing is overwritten and two parameter choices sit side by side. The
recordings that went in are deliberately *not* part of it — a cohort that gains
a recording must not rename the folder holding the twenty-two already corrected.

A stage never selects a store. It computes its own name from the config and its
input's name the same way, so the whole chain follows from ``config.yaml``.

**A record is the completion marker.** A stage's output is present when its
record reads back at the current schema, never when a file merely exists: a run
killed halfway leaves arrays with no record, and those are invisible to the next
run rather than mistaken for finished work. Every stage writes its record last.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from orcann.run_info import RunInfoError, read_record

# Four hex of the content tag names the recording, eight of the settings digest
# names the store. Both are labels for a human reading `ls`; the value they are
# drawn from is what actually distinguishes, and it is recorded in full nowhere
# because nothing reads it -- see the module docstring on samples versus proofs.
RECORDING_TAG_CHARS = 4
STORE_TAG_CHARS = 8

# What may appear in the readable half of a store name. Anything else in a
# parameter value becomes a hyphen rather than being escaped, since the digest
# carries the actual value and the label is only there to be read. Underscore is
# excluded so it stays the separator between a name's parts.
_LABEL_OK = re.compile(r"[^A-Za-z0-9.-]+")


# The ``stage`` a record opens with, and the name a reader checks it against.
# One spelling per stage, here, so a reader cannot ask for a stage by a name the
# writer never used and be told the record is missing.
STAGE_MOTION = "motion_correction"
STAGE_INFER = "infer"
STAGE_SEGMENT = "segment"
STAGE_ACTIVITY = "activity"
STAGE_ANALYSIS = "analysis"

# Where each stage keeps the record that says a recording is done, relative to
# that recording's folder in the store.
RUN_INFO = "run_info.json"


class ProvenanceError(Exception):
    """An identity could not be minted, or a store could not be found."""


def _digest(parts: Dict[str, object]) -> str:
    """sha256 over ``key=repr(value)`` pairs in key order.

    ``repr`` rather than ``str`` so 0.3 and "0.3" are different settings, which
    they are: one reaches a threshold as a float and the other would not reach
    it at all.
    """
    h = hashlib.sha256()
    for k in sorted(parts):
        h.update(f"{k}={parts[k]!r}".encode())
    return h.hexdigest()


def content_tag(path: str) -> str:
    """Full digest of what a movie file holds, from its shape and its two ends.

    Reads two frames. The frame count is in the digest as well as the pixels, so
    a truncated or extended recording is a different recording even when the ends
    it still has are unchanged.
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


def recording_identity(path: str) -> str:
    """``<normalised stem>_<4 hex>`` for a movie file.

    The stem keeps the acquisition name the analysis stage parses genotype and
    day out of; the tag is appended, so ``D109_3-63_040226_R7`` becomes
    ``D109_3-63_040226_R7_5c1d`` and the parsers, which read the leading fields,
    are unaffected.
    """
    from orcann.pipeline.inference import recording_id

    return f"{recording_id(path)}_{content_tag(path)[:RECORDING_TAG_CHARS]}"


def stem_of(identity: str) -> str:
    """The acquisition name an identity was minted from, tag removed.

    Two identities sharing a stem are the same recording under two contents --
    the file was replaced or re-exported between runs. Both are kept, because
    neither result is wrong for the data it was made from, but anything that
    aggregates across recordings has to notice.
    """
    head, _, tag = identity.rpartition("_")
    return head if head and len(tag) == RECORDING_TAG_CHARS else identity


def store_identity(prefix: str, label: str, key: Dict[str, object]) -> str:
    """``<prefix>_<label>_<8 hex>``: what a stage's output folder is called.

    ``key`` must hold every setting that changes what the stage computes,
    including the identity of the store it reads and the stage's own version
    number. It must not hold which recordings were processed.

    A stage below keys on this whole string, readable half included, so changing
    how labels are built moves every store beneath it too. That is the intended
    behaviour of a name that is also a key, and it costs one re-run.
    """
    lab = _LABEL_OK.sub("-", str(label)).strip("-") if label else ""
    tag = _digest(key)[:STORE_TAG_CHARS]
    return f"{prefix}_{lab}_{tag}" if lab else f"{prefix}_{tag}"


def is_done(record_path: str, stage: str) -> bool:
    """Whether a stage's record for one recording is present and readable.

    A record this code cannot read counts as absent, so the work is redone rather
    than trusted: an unreadable record is either a half-written run or an older
    schema, and neither says what the arriving arrays mean.
    """
    try:
        read_record(record_path, expect_stage=stage)
        return True
    except (RunInfoError, OSError, ValueError):
        return False


# STORES -- one folder per settings choice, under the root the config names.

def store_dir(root: str, identity: str) -> str:
    return os.path.join(root, identity)


def _looks_like_store(name: str) -> bool:
    return bool(re.match(r"^[a-z]+_.*[0-9a-f]{8}$", name))


def list_stores(root: str) -> List[str]:
    """Every store folder under ``root``, sorted. Anything else is ignored."""
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and _looks_like_store(d))


def require_store(root: str, identity: str, stage_that_writes: str) -> str:
    """The store ``identity`` under ``root``, or a refusal naming what is there.

    Reached when a stage's input does not exist for the settings in the config —
    either the upstream stage has not run under them, or the config changed after
    it did. Both are the same instruction to the person running it, and guessing
    at a nearby store would attribute one run's results to another's settings.
    """
    d = store_dir(root, identity)
    if os.path.isdir(d):
        return d
    have = list_stores(root)
    listing = "\n  ".join(have) if have else "(nothing)"
    raise ProvenanceError(
        f"{d} does not exist: no {stage_that_writes} output for the settings in "
        f"this config. Run {stage_that_writes}, or set the config back to the "
        f"settings that produced one of:\n  {listing}")


def _task_slice(found: List[str], task_id: Optional[int]) -> List[str]:
    """The whole list, or the 1-based Nth of it for one SGE array task."""
    if task_id is None:
        return found
    return [found[task_id - 1]] if 1 <= task_id <= len(found) else []


def list_flat_records(store: str, stage: str,
                      task_id: Optional[int] = None) -> List[str]:
    """Identities in a store that keeps one record file per recording.

    The motion-correction store: ``<identity>.tif`` beside ``<identity>.json``.
    """
    if not os.path.isdir(store):
        return []
    found = [e[:-len(".json")] for e in sorted(os.listdir(store))
             if e.endswith(".json") and is_done(os.path.join(store, e), stage)]
    return _task_slice(found, task_id)


def list_dir_records(store: str, record_rel: str, stage: str,
                     task_id: Optional[int] = None) -> List[str]:
    """Identities in a store that keeps a directory per recording.

    Enumerating records rather than arrays is what keeps a killed run invisible:
    the probability map may be there and truncated, but without a record beside
    it the recording is simply not done.
    """
    if not os.path.isdir(store):
        return []
    found = [e for e in sorted(os.listdir(store))
             if os.path.isdir(os.path.join(store, e))
             and is_done(os.path.join(store, e, record_rel), stage)]
    return _task_slice(found, task_id)


# STAGE VERSIONS. Bump one when that stage computes something different from the
# same settings -- a changed algorithm, a changed default the config does not
# carry, a fixed defect. Nothing detects that you should have: a stage whose code
# changed without its version will reuse output it would no longer produce.
MOTION_VERSION = 1
INFER_VERSION = 1
SEGMENT_VERSION = 1
ACTIVITY_VERSION = 1
ANALYSIS_VERSION = 1

# What `infer` reads when `paths.pre_processed` holds movies directly rather than
# a store this pipeline wrote -- the documented route for recordings corrected
# elsewhere. It stands in for the settings that produced them, which are unknown
# by definition; the recordings themselves are still identified by content, so a
# swapped file is still noticed.
EXTERNAL = "external"

_MOVIE_EXTS = (".nd2", ".tif", ".tiff", ".npy")


@dataclass(frozen=True)
class Chain:
    """Every store one config names, and the model resolved on the way.

    Computed in one place because each stage's name contains the name of the
    stage above it: asking for the segment store is asking the whole question.
    """
    motion: str
    motion_dir: str
    infer: str
    segment: str
    activity: str
    analysis: str
    model: str
    model_digest: Optional[str]
    model_path: str


def motion_identity(cfg) -> str:
    mc = cfg.motion_correction
    return store_identity("mc", f"{mc.mode}.ms{mc.max_shift}",
                          {"stage": "motion_correction", "version": MOTION_VERSION,
                           "mode": mc.mode, "max_shift": mc.max_shift})


def infer_identity(cfg, model_identity: str, model_digest: Optional[str],
                   upstream: str) -> str:
    """Where the probability maps for one model and one movie store belong.

    Keyed on the model's *digest*, not the selector and not the identity. The
    selector is often ``latest``, which resolves to a different model over time
    and could never invalidate anything. The identity carries a timestamp, so the
    same model promoted twice would send the whole GPU stage over maps it had
    already computed. What decides the output is the content, and the content is
    the digest. A checkpoint carrying no digest falls back to its identity, which
    at worst recomputes rather than confusing two models for one.
    """
    sp = cfg.spatial
    name, tag = model_identity.split("_")[0], model_identity[-4:]
    return store_identity("inf", f"{name}.{tag}",
                          {"stage": "infer", "version": INFER_VERSION,
                           "upstream": upstream,
                           "model": model_digest or model_identity,
                           "resize_to": sp.resize_to,
                           "train_um_per_px": sp.train_um_per_px})


def segment_identity(cfg, upstream: str) -> str:
    sp = cfg.spatial
    return store_identity("seg", f"t{sp.threshold:g}.r{sp.min_radius:g}",
                          {"stage": "segment", "version": SEGMENT_VERSION,
                           "upstream": upstream, "threshold": sp.threshold,
                           "min_area": sp.min_area, "min_radius": sp.min_radius})


def activity_identity(cfg, upstream: str) -> str:
    b, d = cfg.baseline, cfg.deconvolution
    return store_identity("act", f"{b.method}.{d.method if d.enabled else 'none'}",
                          {"stage": "activity", "version": ACTIVITY_VERSION,
                           "upstream": upstream,
                           "baseline": _fields(b), "deconvolution": _fields(d),
                           "frame_rate": cfg.imaging.frame_rate,
                           "indicator": cfg.imaging.indicator})


def analysis_identity(cfg, upstream: str) -> str:
    return store_identity("ana", "", {"stage": "analysis", "version": ANALYSIS_VERSION,
                                      "upstream": upstream,
                                      "analysis": _fields(cfg.analysis),
                                      "frame_rate": cfg.imaging.frame_rate})


def holds_movies(d: str) -> bool:
    """Whether a directory holds movie files of its own, store folders aside."""
    return os.path.isdir(d) and any(
        f.lower().endswith(_MOVIE_EXTS) for f in os.listdir(d))


def motion_store(cfg) -> Tuple[str, str]:
    """``(directory, identity)`` of the corrected movies the stages below read.

    The store this config's motion-correction settings name, if it is there.
    Failing that, ``paths.pre_processed`` itself when it holds movies — the
    documented route for recordings corrected outside this pipeline. A store
    wins over loose movies beside it, so dropping a stray file into the root
    cannot divert a run that has a corrected cohort of its own.
    """
    root = cfg.paths.pre_processed
    ident = motion_identity(cfg)
    d = store_dir(root, ident)
    if os.path.isdir(d):
        return d, ident
    if holds_movies(root):
        return root, EXTERNAL
    raise ProvenanceError(
        f"{d} does not exist and {root} holds no movies: nothing to read. Run "
        f"motion_correction, or put movies corrected elsewhere in {root}.")


def movie_in_store(store: str, identity: str) -> Optional[str]:
    """The corrected movie a store holds for one identity, or None.

    Named by identity because the store already says what the file is; the
    ``<stem>_mc.tif`` convention it replaces had to encode that in every
    filename and be undone by string surgery at every reader.
    """
    for ext in (".tif", ".npy"):
        p = os.path.join(store, identity + ext)
        if os.path.isfile(p):
            return p
    return None


def chain(cfg, motion: Optional[Tuple[str, str]] = None) -> Chain:
    """Resolve every store name this config implies, top to bottom.

    ``motion`` overrides where the corrected movies are taken to be. Only
    ``status`` passes it, so it can describe a pipeline whose first stage has not
    run yet; a stage that is about to read those movies wants the refusal.
    """
    from orcann.pipeline.model_io import ModelStoreError, read_identity, resolve_model

    motion_dir, motion_id = motion if motion else motion_store(cfg)
    try:
        model, model_path = resolve_model(cfg.models.dir, cfg.models.spatial)
    except ModelStoreError as e:
        raise ProvenanceError(str(e))
    digest = read_identity(model_path).get("digest")
    inf = infer_identity(cfg, model, digest, motion_id)
    seg = segment_identity(cfg, inf)
    act = activity_identity(cfg, seg)
    return Chain(motion=motion_id, motion_dir=motion_dir, infer=inf, segment=seg,
                 activity=act, analysis=analysis_identity(cfg, act),
                 model=model, model_digest=digest, model_path=model_path)


def stage_inputs(cfg, stage: str) -> List[str]:
    """The recording identities a stage would work through, current or not.

    What sizes an SGE array. It is the enumeration the stage itself does, so a
    task index means the same recording in the submitter and in the worker.
    """
    from orcann.pipeline import inference as infer
    from orcann.pipeline.cli import identified_recordings, store_movies

    if stage in ("motion_correct", "motion_correction"):
        return [i for _, i in identified_recordings(cfg.paths.raw)]
    ch = chain(cfg)
    if stage == "infer":
        return [i for _, i in store_movies(ch.motion_dir, ch.motion)]
    if stage == "segment":
        return list_dir_records(require_store(cfg.paths.infer, ch.infer, "infer"),
                                infer.META_JSON, STAGE_INFER)
    if stage == "activity":
        return list_dir_records(
            require_store(cfg.paths.spatial, ch.segment, "segment"),
            os.path.join(infer.DATA_DIRNAME, infer.META_JSON), STAGE_SEGMENT)
    if stage == "analysis":
        return list_dir_records(
            require_store(cfg.paths.activity, ch.activity, "activity"),
            RUN_INFO, STAGE_ACTIVITY)
    raise ProvenanceError(f"no such stage: {stage!r}")


def _fields(section) -> Dict[str, object]:
    """Every value in a config section, by name.

    A whole section rather than a chosen few: a setting added to the section
    later is then in the key from the moment it exists, where a hand-written list
    would silently leave it out and let a changed setting reuse an old folder.
    """
    return {k: v for k, v in sorted(vars(section).items())
            if not k.startswith("_")}
