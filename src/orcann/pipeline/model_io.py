"""Stable model persistence.

Saves a model as a plain dict — ``{kind, config, state_dict}`` — rather than
pickling the object. The file therefore contains only tensors, plain Python
data, and a string tag, so it loads correctly even after the package is renamed
or a class is refactored (a new constructor argument with a default is filled in
automatically). Whole-object pickles (``torch.save(model)``) break on both, which
is why this layer exists.

Each savable model declares a ``KIND`` string and a ``config`` dict of its
constructor arguments; ``load_model`` rebuilds via ``cls(**config)`` then loads
the weights. Legacy whole-object pickles are still accepted as a fallback.
"""
from __future__ import annotations

import torch

from orcann.temporal.detector import TemporalRateModel
from orcann.spatial.detection.segmenter import SpatialSegmenter

_REGISTRY = {
    TemporalRateModel.KIND: TemporalRateModel,
    SpatialSegmenter.KIND: SpatialSegmenter,
}


def save_model(model, path: str) -> None:
    torch.save({"kind": model.KIND, "config": model.config,
                "state_dict": model.state_dict()}, path)


def load_model(path: str, map_location="cpu"):
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and "kind" in obj and "state_dict" in obj:
        cls = _REGISTRY.get(obj["kind"])
        if cls is None:
            raise ValueError(f"unknown model kind '{obj['kind']}' "
                             f"(known: {sorted(_REGISTRY)})")
        model = cls(**obj["config"])
        model.load_state_dict(obj["state_dict"])
        return model.eval()
    # legacy whole-object pickle (pre-stability): use as-is
    return obj.eval() if hasattr(obj, "eval") else obj
