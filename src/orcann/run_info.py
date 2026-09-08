"""The per-recording record every stage writes, and the one reader for it.

Every record opens with ``schema_version`` then ``stage``, so a reader
establishes what it is holding before interpreting anything else. The files keep
their own names because each lives beside the data it describes.

Two rules hold the format together.

**A record declares its schema.** ``read_record`` refuses a file that carries no
version, or a version it does not know, rather than reading its keys as though
they meant what today's keys mean. Several incompatible generations of these
files exist in the results directories, distinguishable only by guessing from
which keys happen to be present; refusing them is the point, and the message
says to re-run the stage that writes it.

**A record carries the key it was written under.** ``key`` holds the settings
the stage ran under, its link to the record above, and the recording's content
tag; ``key_digest`` is that key in one value, for the record below to link to.
Currency is decided by comparing those fields, so the key is a required argument
rather than something a call site can leave out and be found to have left out
later. The package version beside it is provenance only and is deliberately not
in the key: a comment change must not invalidate a cohort of probability maps.

**A record is always valid JSON.** NaN and Infinity are not JSON values (RFC
8259 admits no such literals) but Python's ``json`` emits them as bare tokens,
giving a file only Python can read back. Non-finite values are written as
``null`` — a measurement that could not be made is absent — and
``allow_nan=False`` turns anything the sanitiser misses into an error here
rather than a parse failure elsewhere.
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
# reinterpreted, which is also the whole of the migration story for a change to
# how records are shaped.
SCHEMA_VERSION = 4

FILENAME = "run_info.json"

_SOFTWARE: Optional[Dict[str, Any]] = None


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


def _software() -> Dict[str, Any]:
    """What produced this record: the installed package, and the commit if any.

    Recorded on every record and compared by nothing. It answers "which code
    made this number" after the fact, which is the question a person asks; what
    a stage may reuse is decided by its ``stage_version``, which a person bumps
    deliberately.
    """
    global _SOFTWARE
    if _SOFTWARE is None:
        try:
            from importlib.metadata import version
            pkg = version("orcann")
        except Exception:
            pkg = None
        _SOFTWARE = {"package": pkg, "commit": _commit()}
    return _SOFTWARE


def _commit() -> Optional[str]:
    """The checkout's commit, or None when this is not run from one."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", here, "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def write_record(path: str, stage: str, key: Dict[str, Any], run_key: str,
                 payload: Dict[str, Any]) -> str:
    """Write one stage's record of what it did to one recording.

    Every record opens the same keys — ``schema_version``, ``stage``,
    ``run_key``, ``key``, ``key_digest`` — whatever the file is named and
    wherever it sits, so a reader can establish what it is holding and whether
    it is still current before interpreting anything else. The files keep their
    existing names because each lives beside the data it describes:
    ``<recording>.json`` next to the corrected movie, ``meta.json`` in the
    spatial results, ``run_info.json`` in the activity results.
    """
    from orcann.pipeline.provenance import key_digest

    body = {"schema_version": SCHEMA_VERSION, "stage": stage,
            "run_key": run_key, "key": key, "key_digest": key_digest(key),
            "software": _software()}
    body.update(payload)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        dump(body, fh)
    return path


def write(out_dir: str, payload: Dict[str, Any], key: Dict[str, Any],
          run_key: str, stage: str = "activity") -> str:
    """The activity stage's record, under the conventional ``run_info.json``."""
    return write_record(os.path.join(out_dir, FILENAME), stage, key, run_key,
                        payload)


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
