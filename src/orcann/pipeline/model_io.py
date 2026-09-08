"""Stable model persistence, and the identified store the pipeline selects from.

Saves a model as a plain dict rather than pickling the object, so the file
contains only tensors, plain Python data and string tags and loads correctly
after a rename or a refactor. Whole-object pickles break on both, which is why
this layer exists.

**A finished model is written once, under a name derived from what it is.**
``<name>_<UTC timestamp>_<4 hex>`` — the name and timestamp come from the run,
the four characters from the content digest. Two runs never collide because the
timestamp differs; the tag says at a glance which content you are looking at.
The *full* digest is stored inside the file and is what verification and cache
invalidation compare, because four characters is a label, not a checksum.

Two locations, deliberately distinct:

* ``train_spatial.out`` is where training leaves what it produced.
* ``models.dir`` is what the pipeline runs. Nothing arrives there by finishing a
  training run; a model is promoted into it, which is the act of choosing it.

``models.spatial`` selects from the store by identity, or ``latest`` for the
newest by timestamp. ``latest`` is resolved fresh on every run and the resolved
identity is what gets recorded, never the word ``latest`` — a record naming an
alias would compare equal to itself forever and could never detect a change.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
from typing import Dict, Optional, Tuple

import torch

from orcann.spatial.detection.segmenter import SpatialSegmenter

_REGISTRY = {
    SpatialSegmenter.KIND: SpatialSegmenter,
}

MODEL_FILENAME = "segmenter.pt"
REPORT_FILENAME = "train_report.json"
LATEST = "latest"

# <name>_<YYYYmmddTHHMMSSZ>_<4 hex>. UTC and colon-free so it is filesystem-safe
# everywhere and sorts chronologically as plain text.
_IDENTITY_RE = re.compile(r"^(?P<name>.+)_(?P<stamp>\d{8}T\d{6}Z)_(?P<tag>[0-9a-f]{4})$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")


class ModelStoreError(Exception):
    """A model could not be selected, found, or trusted."""


def _payload(model) -> Dict:
    return {"kind": model.KIND, "config": model.config,
            "state_dict": model.state_dict()}


def digest_of(payload: Dict) -> str:
    """sha256 over the three keys that decide what the model computes.

    ``config`` is hashed as well as the weights: it carries ``train_hw``, which
    drives ``resample_for_model``, so two models with identical weights and
    different frame-size targets are not the same model.
    """
    h = hashlib.sha256()
    h.update(str(payload["kind"]).encode())
    for k in sorted(payload["config"]):
        h.update(f"{k}={payload['config'][k]!r}".encode())
    for k, v in sorted(payload["state_dict"].items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_model(model, path: str) -> None:
    """Write a model to an exact path, unidentified.

    Used for the per-epoch checkpoint, which is scratch: it is overwritten every
    epoch and carries no digest, because a half-trained model is not a thing
    anything should be able to select.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(_payload(model), path)


def save_trained_model(model, out_dir: str, name: str, run_key: str,
                       report: Optional[Dict] = None) -> str:
    """Write a finished model under its own identity; return that identity.

    ``report`` is written beside the model rather than to a path of its own, so
    the record of a run cannot be separated from, or overwritten independently
    of, what the run produced.
    """
    if not _NAME_RE.match(name or ""):
        raise ModelStoreError(
            f"train_spatial.name {name!r} is not usable as a filename component: "
            f"it must start alphanumeric and hold only letters, digits, '.' or '-'.")
    payload = _payload(model)
    full = digest_of(payload)
    identity = f"{name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{full[:4]}"
    d = os.path.join(out_dir, identity)
    os.makedirs(d, exist_ok=True)
    torch.save(dict(payload, digest=full, identity=identity,
                    saved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
               os.path.join(d, MODEL_FILENAME))
    if report is not None:
        from orcann.run_info import write_record
        # The training report is a record of a model, not of a stage's output
        # for a recording: nothing tests it for currency, because a model is
        # chosen by being promoted. Its key is therefore what identifies the
        # model, which is the digest of its weights and configuration.
        write_record(os.path.join(d, REPORT_FILENAME), "train_spatial",
                     {"stage": "train_spatial", "model_digest": full}, run_key,
                     dict(report, identity=identity, digest=full))
    return identity


def list_models(models_dir: str) -> list:
    """Every identity in the store, oldest first. Anything not named like an
    identity is ignored rather than guessed at."""
    if not os.path.isdir(models_dir):
        return []
    found = []
    for entry in os.listdir(models_dir):
        m = _IDENTITY_RE.match(entry)
        if m and os.path.isfile(os.path.join(models_dir, entry, MODEL_FILENAME)):
            found.append((m.group("stamp"), entry))
    return [e for _, e in sorted(found)]


def resolve_model(models_dir: str, selector: str) -> Tuple[str, str]:
    """``(identity, path)`` for a selector: an exact identity, or ``latest``.

    ``latest`` is the newest by the timestamp in the name, never by mtime, which
    stops reflecting when the file was made the first time it is copied off the
    cluster.
    """
    if not selector:
        raise ModelStoreError("models.spatial is unset: name a model identity "
                              f"from {models_dir}, or 'latest'.")
    available = list_models(models_dir)
    if selector == LATEST:
        if not available:
            raise ModelStoreError(
                f"models.spatial is 'latest' but {models_dir} holds no model. "
                f"Promote one from train_spatial.out first.")
        identity = available[-1]
    else:
        if selector not in available:
            known = ", ".join(available[-5:]) or "nothing"
            raise ModelStoreError(
                f"models.spatial {selector!r} is not in {models_dir} "
                f"(it holds: {known}).")
        identity = selector
    return identity, os.path.join(models_dir, identity, MODEL_FILENAME)


def read_identity(path: str) -> Dict:
    """The identity keys of a checkpoint, without building the model.

    Loads the whole file — a checkpoint is small and torch has no partial read —
    but skips constructing the network, which is what the skip check wants.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        return {}
    return {k: obj[k] for k in ("identity", "digest", "saved_at") if k in obj}


def promote(src_dir: str, models_dir: str, identity: str) -> str:
    """Copy one finished model from the training output into the store.

    Promotion is a separate act from training on purpose: finishing a run does
    not put a model into service, choosing it does.
    """
    src = os.path.join(src_dir, identity)
    if not os.path.isfile(os.path.join(src, MODEL_FILENAME)):
        raise ModelStoreError(f"{identity} is not in {src_dir}.")
    dst = os.path.join(models_dir, identity)
    if os.path.exists(dst):
        raise ModelStoreError(f"{identity} is already in {models_dir}.")
    os.makedirs(models_dir, exist_ok=True)
    shutil.copytree(src, dst)
    return dst


def load_model(path: str, map_location="cpu", verify: bool = True):
    """Rebuild a model from a checkpoint, refusing one whose contents have moved.

    ``verify`` recomputes the digest and compares it against the one stored in
    the file. A checkpoint written before identities existed carries no digest
    and is loaded without the check, so old files still work.
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and "kind" in obj and "state_dict" in obj:
        cls = _REGISTRY.get(obj["kind"])
        if cls is None:
            raise ValueError(f"unknown model kind '{obj['kind']}' "
                             f"(known: {sorted(_REGISTRY)})")
        if verify and obj.get("digest"):
            found = digest_of(obj)
            if found != obj["digest"]:
                raise ModelStoreError(
                    f"{path}: contents do not match the digest recorded in the "
                    f"file (recorded {obj['digest'][:12]}, found {found[:12]}). "
                    f"The file has been altered or truncated since it was written.")
        model = cls(**obj["config"])
        model.load_state_dict(obj["state_dict"])
        return model.eval()
    # legacy whole-object pickle (pre-stability): use as-is
    return obj.eval() if hasattr(obj, "eval") else obj
