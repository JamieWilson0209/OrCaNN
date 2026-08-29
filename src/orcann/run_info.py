"""The per-recording ``run_info.json`` contract: one writer, one reader, one schema.

Every stage that records what it did, and every stage that later asks, goes
through this module. That is the whole point of it. Before it existed the file
was assembled inline by ``run_activity`` and interrogated by four separate
``.get()`` chains scattered across ``analysis/``, each guessing at a slightly
different shape, and the guesses drifted apart: the analysis stage spent months
reading the frame rate from ``info['config']['frame_rate']``, a nesting the
writer had stopped producing, and silently substituted 2.0 Hz for every
recording without one line of evidence in any output.

Two rules keep that from recurring.

**The file declares its schema.** ``schema_version`` is written at the top and
checked on read; a file that does not declare one, or declares a version this
code does not know, is refused by name rather than parsed on the assumption that
its keys mean what today's keys mean. Six mutually incompatible generations of
this file exist in the results directories on disk, distinguishable only by
guessing from which keys happen to be present. Refusing them is the point:
reading an old file with new assumptions is how a pipeline reports a number that
is wrong rather than absent. This follows the same rule as BIDS, whose readers
check ``BIDSVersion`` before trusting a dataset.

**The file is always valid JSON.** ``NaN`` and ``Infinity`` are not JSON values
(RFC 8259 admits no such literals), but Python's ``json`` module emits them as
bare tokens by default, producing a file that Python alone can read back and
that ``jq``, ``JSON.parse`` and R's ``jsonlite`` all reject. Non-finite values
are converted to ``null`` on the way out — a measurement that could not be made
is absent, which is what ``null`` means — and ``allow_nan=False`` makes any that
escape the conversion raise at the point of writing rather than producing a file
that fails somewhere else, later, in another language.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Bumped when the meaning of an existing key changes or a required key is added
# or removed. Readers refuse anything they do not recognise, so a bump is a
# deliberate break: it makes every older recording re-run rather than silently
# reinterpreted.
SCHEMA_VERSION = 2

FILENAME = "run_info.json"


class RunInfoError(Exception):
    """A recording's run_info.json is missing, unreadable, or not this schema."""


def json_safe(value: Any) -> Any:
    """Recursively convert a value into something ``json.dump`` can write.

    Non-finite floats become ``None``: JSON has no NaN or Infinity, and a
    measurement that could not be made is absent rather than zero. NumPy scalars
    and arrays become their Python equivalents, so a stray ``np.float32`` does
    not reach the encoder as an unserialisable object.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, bool):          # before int: bool is a subclass of int
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if value is None or isinstance(value, str):
        return value
    # NumPy scalars and arrays, without importing numpy at module scope.
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return json_safe(value.item())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return value


def dump(payload: Dict[str, Any], fh) -> None:
    """Write a payload as strictly valid JSON.

    ``allow_nan=False`` is the safety net rather than the mechanism: values are
    already sanitised, so this raises only if something reached here by a path
    ``json_safe`` does not cover — which is worth an exception at the point of
    writing, not a file that fails to parse elsewhere.
    """
    json.dump(json_safe(payload), fh, indent=2, allow_nan=False)


def write_record(path: str, stage: str, payload: Dict[str, Any]) -> str:
    """Write one stage's record of what it did to one recording.

    Every stage's record opens the same two keys — ``schema_version`` then
    ``stage`` — whatever the file is named and wherever it sits, so a reader can
    establish what it is holding before interpreting anything else. The files
    keep their existing names because each lives beside the data it describes:
    ``<stem>_mc.json`` next to the corrected movie, ``meta.json`` in the spatial
    results, ``run_info.json`` in the activity results.
    """
    body = {"schema_version": SCHEMA_VERSION, "stage": stage}
    body.update(payload)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        dump(body, fh)
    return path


def write(out_dir: str, payload: Dict[str, Any], stage: str = "activity") -> str:
    """The activity stage's record, under the conventional ``run_info.json``."""
    return write_record(os.path.join(out_dir, FILENAME), stage, payload)


def read_record(path, *, expect_stage: Optional[str] = None) -> Dict[str, Any]:
    """Parse and version-check any stage's record, returning the raw document.

    Raises ``RunInfoError`` naming the file and the reason. ``expect_stage``
    additionally rejects a record written by a different stage, which catches a
    path built wrongly rather than letting its keys be read as though they meant
    something they do not.
    """
    path = Path(path)
    if not path.exists():
        raise RunInfoError(f"{path}: no such record")

    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise RunInfoError(f"{path}: could not be read ({e})") from e

    if not isinstance(doc, dict):
        raise RunInfoError(f"{path}: not a JSON object")

    version = doc.get("schema_version")
    if version is None:
        raise RunInfoError(
            f"{path}: no schema_version, so it was written by an older pipeline "
            f"whose keys do not mean what this reader would take them to mean. "
            f"Re-run the stage that writes it.")
    if version != SCHEMA_VERSION:
        raise RunInfoError(
            f"{path}: schema_version {version!r}, but this code reads "
            f"{SCHEMA_VERSION}. Re-run the stage that writes it.")
    if expect_stage is not None and doc.get("stage") != expect_stage:
        raise RunInfoError(
            f"{path}: written by stage {doc.get('stage')!r}, not "
            f"{expect_stage!r}")
    return doc


@dataclass(frozen=True)
class RunInfo:
    """What a recording recorded about how it was produced.

    Named fields rather than a bare dict, so a reader states what it needs and
    gets a clear failure when the file does not carry it. ``raw`` is the parsed
    document for the occasional caller that needs a key not promoted here.
    """

    recording_id: str
    frame_rate: float
    dims: Optional[Tuple[int, int]]
    n_roi: int
    n_frames: int
    amplitude_method: str
    source: str
    deconv_method_used: str
    n_traces_rejected: int
    motion: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


def read(recording_dir) -> RunInfo:
    """Parse and validate one recording's run_info.json.

    Raises ``RunInfoError``, naming the recording and the reason, when the file
    is absent, unparseable, or not this schema version. Callers decide whether
    that costs one recording or the run; none of them has to guess what an
    unlabelled document meant.
    """
    doc = read_record(Path(recording_dir) / FILENAME, expect_stage="activity")

    dims = doc.get("dims")
    dims = (int(dims[0]), int(dims[1])) if dims else None
    deconv = doc.get("deconvolution") or {}

    return RunInfo(
        recording_id=str(doc.get("recording_id", "")),
        frame_rate=float(doc.get("frame_rate", 0.0)),
        dims=dims,
        n_roi=int(doc.get("n_roi", 0)),
        n_frames=int(doc.get("n_frames", 0)),
        amplitude_method=str(doc.get("amplitude_method", "")),
        source=str(doc.get("source") or ""),
        # The detector that reported having run. A configured value is intent,
        # not evidence of what produced the traces, so it is never read here.
        deconv_method_used=str(deconv.get("method_used") or ""),
        n_traces_rejected=int(deconv.get("n_traces_rejected") or 0),
        motion=doc.get("motion_correction") or {},
        raw=doc,
    )
