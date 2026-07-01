"""Single source of truth for the pipeline: one YAML file drives every run.

The workspace layout (paths) and the trained models live here alongside every
tuning knob, so each subcommand reads its section instead of taking flags. Edit a
YAML file and pass ``--config``; generate a fully-commented starting file with
``orcann <stage> --dump-config config.yaml``. ``--set section.key=value`` is a
convenience for one-off overrides. The dataclass defaults below are authoritative,
so a YAML file may be partial. See docs/configuration.md.

Path resolution. Every path-valued field is stored relative in the YAML but
resolved to an absolute path against the *config file's own directory* (the
workspace root) by ``resolve_paths``, called once after the file loads and any
``--set`` overrides are applied. The config file is therefore the sole anchor:
the repo extracts anywhere and ``data/``, ``models/``, ``results/`` resolve
correctly with no setup step and no dependence on the current directory. An
absolute path in the YAML passes through untouched, so inputs or outputs can
point at scratch or group storage while the code tree lives elsewhere. There is
no current-directory fallback: a relative path with no loaded config to anchor it
is an error (only ``--synthetic`` self-tests run without a config, and they touch
no workspace paths).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional, Tuple


# --- workspace layout + trained models ---------------------------------------
@dataclass
class Paths:
    raw: str = "data/raw"
    pre_processed: str = "data/pre_processed"
    infer: str = "results/infer"         # cached probability maps (infer stage output)
    spatial: str = "results/spatial"
    transients: str = "results/transients"
    analysis: str = "results/analysis"   # analysis stage output (group figures + tables)


@dataclass
class Models:
    spatial: Optional[str] = "models/seg_final/segmenter.pt"
    temporal: Optional[str] = "models/temporal/rate_model.pt"


# --- detection stages ---------------------------------------------------------
@dataclass
class SpatialParams:
    threshold: float = 0.5
    watershed: bool = False
    min_distance: int = 4
    min_area: int = 4
    min_radius: float = 0.0
    resize_to: int = 0
    train_um_per_px: Optional[float] = None


@dataclass
class TemporalParams:
    frame_rate: float = 2.0
    min_prominence: float = 0.5
    floor_pct: float = 25.0
    min_isi_s: float = 1.0

    def detection(self) -> dict:
        return {"min_prominence": self.min_prominence,
                "floor_pct": self.floor_pct, "min_isi_s": self.min_isi_s}


@dataclass
class MotionCorrectionParams:
    mode: str = "auto"
    max_shift: int = 20


@dataclass
class FigureParams:
    enabled: bool = True
    max_roi_figures: int = 0


# --- training -----------------------------------------------------------------
@dataclass
class TrainSpatialParams:
    movies: Optional[str] = "data/annotated/movies"
    masks: Optional[str] = "data/annotated/masks"
    out: Optional[str] = "models/seg_final"
    report: Optional[str] = "results/spatial_eval/report.json"
    channels: Tuple[str, ...] = ("structural", "max", "variance")
    radii: Tuple[float, ...] = (3.0, 3.7, 4.5, 5.5, 6.7, 8.2, 10.0)
    min_cell_area: int = 0
    pixel_um: Optional[float] = None
    patch: int = 128
    epochs: int = 30
    val_frac: float = 0.2
    holdout: bool = True


@dataclass
class TrainTemporalParams:
    gt_dir: Optional[str] = "data/public_gt"
    indicator_map: Optional[str] = "data/public_gt/indicator_map.json"
    report: Optional[str] = "results/loio/report.json"
    save_final: Optional[str] = "models/temporal/rate_model.pt"
    target_fs: float = 2.0
    epochs: int = 25
    scale_dropout: float = 0.0
    exclude: Tuple[str, ...] = ()


# --- trace inspection figure (scripts/visualize_transients.py) ----------------
@dataclass
class VizParams:
    traces: Optional[str] = None
    model: Optional[str] = "models/temporal/rate_model.pt"
    out: str = "transients.png"


@dataclass
class AnalysisParams:
    # recording_id -> metadata parsing (checked via recording_metrics.csv)
    day_regex: str = "D([0-9]+)"              # capture group 1 -> developmental day (int)
    line_regex: str = "D[0-9]+[_-]([^_]+)"    # capture group 1 -> line/clone token
    control_prefix: str = "3"                 # line tokens starting with this are Control, else Mutant
    rois: str = "auto"
    max_rois: int = 6


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    models: Models = field(default_factory=Models)
    spatial: SpatialParams = field(default_factory=SpatialParams)
    temporal: TemporalParams = field(default_factory=TemporalParams)
    motion_correction: MotionCorrectionParams = field(default_factory=MotionCorrectionParams)
    figures: FigureParams = field(default_factory=FigureParams)
    train_spatial: TrainSpatialParams = field(default_factory=TrainSpatialParams)
    train_temporal: TrainTemporalParams = field(default_factory=TrainTemporalParams)
    viz: VizParams = field(default_factory=VizParams)
    analysis: AnalysisParams = field(default_factory=AnalysisParams)

    # Workspace root: the directory of the config file, set by load(). Not a
    # dataclass field, so it never appears in to_dict()/dump() output. None until
    # a file is loaded (a fresh Config() used only for --dump-config has no root).
    root = None

    # Path-valued fields, resolved against root by resolve_paths().
    _PATH_FIELDS = {
        "paths": ("raw", "pre_processed", "infer", "spatial", "transients", "analysis"),
        "models": ("spatial", "temporal"),
        "train_spatial": ("movies", "masks", "out", "report"),
        "train_temporal": ("gt_dir", "indicator_map", "report", "save_final"),
        "viz": ("traces", "model", "out"),
    }

    @classmethod
    def load(cls, path: Optional[str]) -> "Config":
        """Defaults, with a YAML/JSON file overlaid on top if ``path`` is given.

        Records the config file's directory as the workspace root used by
        ``resolve_paths``. Resolution is deferred so that ``--set`` overrides
        (applied after load) are anchored too; the caller invokes
        ``resolve_paths`` once both are in place.
        """
        cfg = cls()
        cfg.root = os.path.dirname(os.path.abspath(path)) if path else None
        if path:
            import yaml
            with open(path) as f:
                data = json.load(f) if path.endswith(".json") else yaml.safe_load(f)
            cfg._overlay(data or {})
        return cfg

    def resolve(self, p: Optional[str]) -> Optional[str]:
        """Make one config path absolute against the workspace root.

        ``None`` and already-absolute paths pass through unchanged. A relative
        path is joined onto root; with no root set (no config loaded) that is an
        error rather than a silent current-directory fallback.
        """
        if p is None or os.path.isabs(p):
            return p
        if self.root is None:
            raise ValueError(
                f"cannot resolve relative path {p!r}: no config file loaded to "
                f"anchor it (pass --config, or use an absolute path)")
        return os.path.normpath(os.path.join(self.root, p))

    def resolve_paths(self) -> "Config":
        """Rewrite every path-valued field to an absolute path, in place.

        No-op when no config was loaded (root is None): only ``--synthetic``
        self-tests run without a config, and they consume no workspace paths.
        Call once, after load() and apply_overrides().
        """
        if self.root is None:
            return self
        for section, keys in self._PATH_FIELDS.items():
            sub = getattr(self, section)
            for k in keys:
                setattr(sub, k, self.resolve(getattr(sub, k)))
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self, path: str) -> None:
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(json.loads(json.dumps(self.to_dict())), f, indent=2)
            return
        with open(path, "w") as f:
            f.write(self._to_commented_yaml())

    def apply_overrides(self, items) -> "Config":
        """Apply a list of ``section.key=value`` strings (the shared ``--set``)."""
        for it in items or []:
            head = it.split("=", 1)[0]
            if "=" not in it or "." not in head:
                raise ValueError(f"--set expects section.key=value, got {it!r}")
            dotted, raw = it.split("=", 1)
            section, key = dotted.split(".", 1)
            self._set_one(section, key, raw)
        return self

    # ---- internals --------------------------------------------------------
    def _overlay(self, data: dict) -> None:
        for section, vals in (data or {}).items():
            for key, v in (vals or {}).items():
                self._set_one(section, key, v)

    def _set_one(self, section: str, key: str, value: Any) -> None:
        sub = getattr(self, section, None)
        if not is_dataclass(sub):
            raise KeyError(f"unknown config section: {section!r}")
        if key not in {f.name for f in fields(sub)}:
            raise KeyError(f"unknown config key: {section}.{key}")
        setattr(sub, key, _coerce(value, getattr(sub, key)))

    def _to_commented_yaml(self) -> str:
        lines = [
            "# OrCaNN configuration - the single source of truth for a run.",
            "# Edit values and run, e.g.:  orcann run_pipeline --config config.yaml",
            "# Paths are relative to THIS file's directory (the workspace root), so the",
            "# repo works wherever it is extracted; an absolute path is used as-is.",
            "# null = unset / use the default; lists use [a, b].",
        ]
        for section in (f.name for f in fields(self)):
            sub = getattr(self, section)
            rows = [(k, _fmt(getattr(sub, k))) for k in (f.name for f in fields(sub))]
            width = max((len(f"  {k}: {v}") for k, v in rows), default=0)
            lines.append("")
            lines.append(f"# {_SECTION_DOC.get(section, '')}".rstrip())
            lines.append(f"{section}:")
            for k, v in rows:
                base = f"  {k}: {v}"
                doc = _FIELD_DOC.get(f"{section}.{k}", "")
                lines.append(base + (" " * (width - len(base) + 2) + f"# {doc}"
                                     if doc else ""))
        return "\n".join(lines) + "\n"


def _coerce(value: Any, current: Any) -> Any:
    """Cast a YAML/string value to the type of the field's current value."""
    if value is None:                                    # YAML null -> None
        return None
    if isinstance(current, tuple):
        parts = ([p for p in re.split(r"[,:\s]+", value.strip()) if p]
                 if isinstance(value, str) else list(value))
        elem = type(current[0]) if current else str
        return tuple(elem(p) for p in parts)
    if isinstance(current, bool):
        return value.strip().lower() in ("1", "true", "yes", "on") \
            if isinstance(value, str) else bool(value)
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if current is None:
        if value in (None, "", "none", "None", "null"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, str) and value.strip().lower() in ("none", "null"):
        return None                                      # --set field=null -> None
    return type(current)(value)


def _fmt(v: Any) -> str:
    """Render a value as a YAML scalar / flow-list for the commented dump."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(e) for e in v) + "]"
    return json.dumps(str(v))


_SECTION_DOC = {
    "paths": "Workspace layout: where recordings live and where results are written",
    "models": "Trained models used by the detection stages",
    "spatial": "Spatial stage (segmentation) knobs - used by segment",
    "temporal": "Temporal stage (transient detection) knobs - detect_transients",
    "motion_correction": "Motion correction (caiman env) - motion_correction stage",
    "figures": "QC figures written by the detection stages",
    "train_spatial": "train_spatial - fit the segmenter",
    "train_temporal": "train_temporal - fit/evaluate the temporal head",
    "viz": "scripts/visualize_transients.py - per-ROI inspection figure",
}

_FIELD_DOC = {
    "paths.raw": "recordings to process (.nd2 / .tif / .npy)",
    "paths.pre_processed": "motion-corrected movies (motion_correction output, infer input)",
    "paths.infer": "cached probability maps (infer output, segment input)",
    "paths.spatial": "segment output: <spatial>/<recording_id>/",
    "paths.transients": "detect_transients output: <transients>/<recording_id>/",
    "paths.analysis": "analysis stage output (group figures + tables)",
    "models.spatial": "trained segmenter (.pt)",
    "models.temporal": "trained temporal rate model (.pt)",
    "spatial.threshold": "soma-probability cut, ~0.5-0.6",
    "spatial.watershed": "split touching cells (false = connected components, which merge them)",
    "spatial.min_distance": "min peak separation in px for watershed seeding",
    "spatial.min_area": "drop detected regions smaller than this many px (0 disables)",
    "spatial.min_radius": "or drop regions below this equivalent radius in px",
    "spatial.resize_to": "force each frame to NxN when pixel size is unknown (0 = off)",
    "spatial.train_um_per_px": "override the model's recorded training pixel size (null = use model's)",
    "temporal.frame_rate": "recording frame rate in Hz",
    "temporal.min_prominence": "rate-units prominence a transient must clear (main sensitivity knob)",
    "temporal.floor_pct": "height floor = this percentile of the rate (gates quiet baseline)",
    "temporal.min_isi_s": "minimum separation between transients, in seconds",
    "motion_correction.mode": "rigid | piecewise_rigid | auto",
    "motion_correction.max_shift": "maximum shift in px",
    "figures.enabled": "write QC figures (overlay + per-ROI panels)",
    "figures.max_roi_figures": "cap per-ROI panels, keeping the most active (0 = all)",
    "train_spatial.movies": "dir of training movies (<stem>.tif)",
    "train_spatial.masks": "dir of instance-label masks (<stem>.npy) or ImageJ ROI sets",
    "train_spatial.out": "output dir for the trained segmenter.pt",
    "train_spatial.report": "optional JSON metrics path (null to skip)",
    "train_spatial.channels": "energy channels: any of structural, max, variance, correlation",
    "train_spatial.radii": "LoG scale bank, cell radii in px",
    "train_spatial.min_cell_area": "strip ROIs smaller than this from training masks (0 = keep all)",
    "train_spatial.pixel_um": "training movies' um/px, recorded in the model (null if unknown)",
    "train_spatial.patch": "training patch size in px",
    "train_spatial.epochs": "training epochs",
    "train_spatial.val_frac": "fraction of recordings held out for validation",
    "train_spatial.holdout": "hold out val_frac to evaluate; false = train final model on all data",
    "train_temporal.gt_dir": "dir of CASCADE ground-truth .mat files",
    "train_temporal.indicator_map": "JSON mapping each .mat filename to an indicator label",
    "train_temporal.report": "JSON path for the LOIO validation table (null to skip validation)",
    "train_temporal.save_final": "path to save the model fit on all data, loaded by detect_transients (null to skip)",
    "train_temporal.target_fs": "resample public ground truth to this rate, Hz",
    "train_temporal.epochs": "training epochs",
    "train_temporal.scale_dropout": "fraction of wavelet-scale channels dropped in training (0 = off)",
    "train_temporal.exclude": "drop indicator groups whose label contains any of these substrings",
    "viz.traces": "(n_roi, T) fluorescence traces (.npy)",
    "viz.model": "trained temporal model (.pt)",
    "viz.out": "output figure path (.png)",
    "viz.rois": "comma-separated ROI indices, or 'auto' for the most active",
    "viz.max_rois": "max number of ROIs to draw",
    "analysis.day_regex": "recording_id regex; capture group 1 -> developmental day (int)",
    "analysis.line_regex": "recording_id regex; capture group 1 -> line/clone token",
    "analysis.control_prefix": "line tokens starting with this are Control, else Mutant",
}
