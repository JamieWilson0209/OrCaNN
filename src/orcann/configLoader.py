"""Single source of truth for the pipeline: one YAML file drives every run.

The workspace layout (paths) and the trained model live here alongside every
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


# --- workspace layout + trained model ----------------------------------------
@dataclass
class Paths:
    raw: str = "data/raw"
    pre_processed: str = "data/pre_processed"
    infer: str = "results/infer"         # cached probability maps (infer stage output)
    spatial: str = "results/spatial"     # segment output: <spatial>/<recording_id>/
    activity: str = "results/activity"   # activity output: <activity>/<recording_id>/ (calcium-format)
    analysis: str = "results/analysis"   # analysis stage output (group figures + tables)


@dataclass
class Models:
    spatial: Optional[str] = "models/seg_final/segmenter.pt"


# --- recording-wide imaging metadata -----------------------------------------
@dataclass
class Imaging:
    frame_rate: float = 2.0
    indicator: str = "fluo4"          # resolves a decay time when deconvolution.decay_time is null


# --- spatial detection stage --------------------------------------------------
@dataclass
class SpatialParams:
    threshold: float = 0.5
    watershed: bool = False
    min_distance: int = 4
    min_area: int = 4
    min_radius: float = 0.0
    resize_to: int = 0
    train_um_per_px: Optional[float] = None


# --- activity stage: baseline + deconvolution (calcium bridge) ---------------
@dataclass
class BaselineParams:
    method: str = "global_dff"        # direct | global_dff | local_background
    percentile: float = 8.0
    window_fraction: float = 0.25
    min_window: int = 50
    max_window: int = 500
    presmooth_sigma: float = 0.0      # Gaussian smoothing (frames) for F0 estimation only; 0 = off


@dataclass
class DeconvolutionParams:
    enabled: bool = True
    method: str = "oasis"             # oasis | threshold | robust
    decay_time: Optional[float] = None  # seconds; null resolves from imaging.indicator
    optimize_g: bool = True
    penalty: float = 0.0              # L1 sparsity; 0 = auto-tune (recommended for OASIS)
    noise_method: str = "mean"        # mean | median | logmexp
    s_min: float = 0.1                # min spike amplitude in dF/F0 (OASIS suppresses below this)
    noise_gate_sigma: float = 3.5     # keep only spikes above this multiple of the trace noise floor
    robust_safety_net: bool = True    # on the oasis path, backfill obvious transients OASIS missed
    robust_k_onset: float = 3.0       # robust detector: event onset threshold (x noise)
    robust_k_peak: float = 5.0        # robust detector: required peak height (x noise)
    robust_min_duration_s: float = 0.5  # robust detector: minimum event duration (seconds)


@dataclass
class MotionCorrectionParams:
    mode: str = "auto"                # rigid | piecewise_rigid | auto
    max_shift: int = 20


@dataclass
class FigureParams:
    enabled: bool = True              # write the spatial QC overlay in segment


@dataclass
class GalleryParams:
    interactive: bool = True          # per-recording interactive HTML gallery
    movie: bool = False               # full-movie HTML viewer (large; off by default)
    movie_subsample: int = 1          # keep every Nth frame in the movie gallery
    max_rois: int = 500               # cap ROIs drawn in the gallery


# --- training (spatial only) --------------------------------------------------
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


# --- group analysis (calcium) -------------------------------------------------
@dataclass
class AnalysisParams:
    mutant_label: str = "CEP41 R242H"     # legend label for the non-control genotype
    min_roi_distance: float = 15.0        # dedupe ROIs whose centroids are closer than this (px)
    motion_max_threshold: float = 15.0    # QC: max motion shift (px) tolerated per recording
    motion_residual_threshold: float = 2.0  # QC: residual motion (px) tolerated per recording
    drift_threshold: float = 1.0          # QC: baseline drift tolerated per recording
    roi_peak_figures: bool = False        # also render per-ROI peak montages (slow)
    inactive_file: Optional[str] = None   # text file of recording ids to mark inactive (one per line)


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    models: Models = field(default_factory=Models)
    imaging: Imaging = field(default_factory=Imaging)
    spatial: SpatialParams = field(default_factory=SpatialParams)
    baseline: BaselineParams = field(default_factory=BaselineParams)
    deconvolution: DeconvolutionParams = field(default_factory=DeconvolutionParams)
    motion_correction: MotionCorrectionParams = field(default_factory=MotionCorrectionParams)
    figures: FigureParams = field(default_factory=FigureParams)
    gallery: GalleryParams = field(default_factory=GalleryParams)
    train_spatial: TrainSpatialParams = field(default_factory=TrainSpatialParams)
    analysis: AnalysisParams = field(default_factory=AnalysisParams)

    # Workspace root: the directory of the config file, set by load(). Not a
    # dataclass field, so it never appears in to_dict()/dump() output. None until
    # a file is loaded (a fresh Config() used only for --dump-config has no root).
    root = None

    # Path-valued fields, resolved against root by resolve_paths().
    _PATH_FIELDS = {
        "paths": ("raw", "pre_processed", "infer", "spatial", "activity", "analysis"),
        "models": ("spatial",),
        "train_spatial": ("movies", "masks", "out", "report"),
        "analysis": ("inactive_file",),
    }

    # ---- indicator -> decay time (s), used when deconvolution.decay_time is null
    INDICATOR_DECAY = {
        "gcamp6f": 0.4, "gcamp6s": 2.0, "jgcamp7f": 0.5, "jgcamp8f": 0.3,
        "jgcamp8m": 0.5, "jgcamp8s": 1.0, "fluo4": 0.4, "fluo-4": 0.4,
        "ogb1": 0.7, "ogb-1": 0.7, "jrgeco1a": 0.7,
    }

    def decay_time(self) -> float:
        """Resolved indicator decay time in seconds (explicit override wins)."""
        if self.deconvolution.decay_time is not None:
            return float(self.deconvolution.decay_time)
        key = str(self.imaging.indicator).strip().lower()
        return float(self.INDICATOR_DECAY.get(key, 0.4))

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
    "models": "Trained segmenter used by the spatial detection stages",
    "imaging": "Recording-wide imaging metadata",
    "spatial": "Spatial detection (segmentation) knobs - used by segment",
    "baseline": "Baseline correction (dF/F0) - first half of the activity stage",
    "deconvolution": "Spike deconvolution (OASIS) - second half of the activity stage",
    "motion_correction": "Motion correction (caiman env) - motion_correction stage",
    "figures": "Spatial QC figure written by segment",
    "gallery": "Per-recording HTML galleries written by the activity stage",
    "train_spatial": "train_spatial - fit the segmenter",
    "analysis": "Group analysis - cross-recording statistics + genotype/day figures",
}

_FIELD_DOC = {
    "paths.raw": "recordings to process (.nd2 / .tif / .npy)",
    "paths.pre_processed": "motion-corrected movies (motion_correction output, infer input)",
    "paths.infer": "cached probability maps (infer output, segment input)",
    "paths.spatial": "segment output: <spatial>/<recording_id>/",
    "paths.activity": "activity output: <activity>/<recording_id>/ (calcium-format, analysis input)",
    "paths.analysis": "analysis stage output (group figures + tables)",
    "models.spatial": "trained segmenter (.pt)",
    "imaging.frame_rate": "recording frame rate in Hz",
    "imaging.indicator": "calcium indicator; resolves a decay time when deconvolution.decay_time is null",
    "spatial.threshold": "soma-probability cut, ~0.5-0.6",
    "spatial.watershed": "split touching cells (false = connected components, which merge them)",
    "spatial.min_distance": "min peak separation in px for watershed seeding",
    "spatial.min_area": "drop detected regions smaller than this many px (0 disables)",
    "spatial.min_radius": "or drop regions below this equivalent radius in px",
    "spatial.resize_to": "force each frame to NxN when pixel size is unknown (0 = off)",
    "spatial.train_um_per_px": "override the model's recorded training pixel size (null = use model's)",
    "baseline.method": "direct | global_dff (per-trace rolling percentile) | local_background (tissue-masked)",
    "baseline.percentile": "baseline percentile for global_dff",
    "baseline.window_fraction": "rolling-baseline window as a fraction of trace length",
    "baseline.min_window": "minimum rolling-baseline window (frames)",
    "baseline.max_window": "maximum rolling-baseline window (frames)",
    "baseline.presmooth_sigma": "Gaussian smoothing (frames) for F0 estimation only; lifts F0 to the true resting level on noisy traces; 0 = off",
    "deconvolution.enabled": "run OASIS spike inference (false = skip; analysis then has no spikes)",
    "deconvolution.decay_time": "indicator decay time in s (null = resolve from imaging.indicator)",
    "deconvolution.optimize_g": "let OASIS fit the AR coefficient from data",
    "deconvolution.penalty": "L1 sparsity penalty; 0 = auto-tune (recommended for OASIS)",
    "deconvolution.noise_method": "OASIS noise estimator: mean | median | logmexp",
    "deconvolution.s_min": "min spike amplitude in dF/F0; OASIS discards events below this (0 = let OASIS decide)",
    "deconvolution.noise_gate_sigma": "keep only spikes exceeding this multiple of the trace noise floor (0 = no gate)",
    "deconvolution.method": "oasis (AR deconvolution) | threshold (peak detection) | robust (deterministic transient detector)",
    "deconvolution.robust_safety_net": "on the oasis path, backfill clear transients OASIS missed (true recommended)",
    "deconvolution.robust_k_onset": "robust detector: event onset threshold as a multiple of noise",
    "deconvolution.robust_k_peak": "robust detector: required peak height as a multiple of noise (main precision knob)",
    "deconvolution.robust_min_duration_s": "robust detector: minimum event duration in seconds (rejects single-sample noise)",
    "motion_correction.mode": "rigid | piecewise_rigid | auto",
    "motion_correction.max_shift": "maximum shift in px",
    "figures.enabled": "write the spatial QC overlay in segment",
    "gallery.interactive": "per-recording interactive HTML gallery (gallery.html)",
    "gallery.movie": "full-movie HTML viewer (large file; off by default)",
    "gallery.movie_subsample": "keep every Nth frame in the movie gallery",
    "gallery.max_rois": "cap the number of ROIs drawn in the gallery",
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
    "analysis.mutant_label": "legend label for the non-control genotype",
    "analysis.min_roi_distance": "dedupe ROIs whose centroids are closer than this (px)",
    "analysis.motion_max_threshold": "QC: max motion shift (px) tolerated per recording",
    "analysis.motion_residual_threshold": "QC: residual motion (px) tolerated per recording",
    "analysis.drift_threshold": "QC: baseline drift tolerated per recording",
    "analysis.roi_peak_figures": "also render per-ROI peak montages (slow)",
    "analysis.inactive_file": "text file of recording ids to mark inactive (one per line)",
}
