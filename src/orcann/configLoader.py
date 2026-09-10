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


# --- the run this config describes -------------------------------------------
@dataclass
class Run:
    # Names every output this config writes: <stage>_outputs_<key>. Chosen by a
    # person, so a result can be found by the name of the experiment that made
    # it. What a stage may reuse is decided by the key recorded beside each
    # output, not by this string -- see orcann/pipeline/provenance.py.
    key: str = "main"
    # Where hpc/run_all.sh asks SGE to mail when the chain's last stage ends.
    # Only that job carries the request: on an array, -m mails once per task.
    notify_email: Optional[str] = None


# --- workspace layout + trained model ----------------------------------------
@dataclass
class Paths:
    # Roots, not output folders: each stage writes into a folder under its root
    # named for the stage and the run key -- see orcann/pipeline/provenance.py.
    raw: str = "data/raw"
    pre_processed: str = "data/pre_processed"   # <root>/motion_correction_outputs_<key>/<recording>.tif
    infer: str = "results/infer"                # <root>/infer_outputs_<key>/<recording>/
    spatial: str = "results/spatial"            # <root>/segment_outputs_<key>/<recording>/
    activity: str = "results/activity"          # <root>/activity_outputs_<key>/<recording>/
    analysis: str = "results/analysis"          # <root>/analysis_outputs_<key>/
    runs: str = "results/runs"                  # <root>/<key>/config.yaml, the run's own snapshot


@dataclass
class Models:
    dir: Optional[str] = "models/in_use"
    # An identity from that directory, or "latest" for the newest by timestamp.
    # Not a path: a model reaches the store by being promoted into it, and
    # naming a file directly would let an unpromoted one into production.
    spatial: Optional[str] = "latest"


# --- recording-wide imaging metadata -----------------------------------------
@dataclass
class Imaging:
    frame_rate: float = 2.0
    indicator: str = "fluo4"          # resolves a decay time when deconvolution.decay_time is null


# --- spatial detection stage --------------------------------------------------
@dataclass
class SpatialParams:
    threshold: float = 0.5
    min_area: int = 4
    min_radius: float = 0.0
    resize_to: int = 0


# --- activity stage: baseline + deconvolution (calcium bridge) ---------------
@dataclass
class BaselineParams:
    method: str = "global_dff"        # global_dff (per-trace rolling percentile)
    percentile: float = 8.0
    window_fraction: float = 0.25
    min_window: int = 50
    max_window: int = 500
    presmooth_sigma: float = 0.0      # Gaussian smoothing (frames) for F0 estimation only; 0 = off


@dataclass
class DeconvolutionParams:
    enabled: bool = True
    method: str = "oasis"             # oasis | robust
    # Seeds OASIS's AR coefficient; it is NOT a measurement of how long these
    # transients last. With optimize_g the solver refits per trace, and on this
    # preparation it lands near 1.7 s against a 0.4 s indicator constant —
    # observable fluorescence decay is the indicator convolved with a slower
    # biological calcium decay.
    decay_initialisation: Optional[float] = None  # seconds; null resolves from imaging.indicator
    optimize_g: bool = True
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
    max_rois: int = 500               # cap ROIs drawn in the gallery


# --- training (spatial only) --------------------------------------------------
@dataclass
class TrainSpatialParams:
    movies: Optional[str] = "data/annotated/movies"
    masks: Optional[str] = "data/annotated/masks"
    name: str = "segmenter"
    out: Optional[str] = "models/trained"
    checkpoint: Optional[str] = "models/trained/_in_progress.pt"
    channels: Tuple[str, ...] = ("structural", "max", "variance")
    radii: Tuple[float, ...] = (3.0, 3.7, 4.5, 5.5, 6.7, 8.2, 10.0)
    min_cell_area: int = 0
    patch: int = 128
    # Sampling. How much of each recording one epoch sees; the defaults bound
    # compute rather than express anything measured about the data, and
    # n_energy_frames is stamped into the checkpoint and reused by infer.
    n_patch: int = 6
    fg_frac: float = 0.75
    n_energy_frames: Optional[int] = 64
    epochs: int = 30
    val_frac: float = 0.2
    holdout: bool = True
    # How the edge of a frame is read; travels with the model into infer.
    edge_correction: str = "zeros"
    # Write a prediction overlay on one training and one held-out recording at a
    # tapering epoch schedule, for watching the fit afterwards as a series.
    progress_frames: bool = False


# --- group analysis (calcium) -------------------------------------------------
@dataclass
class AnalysisParams:
    mutant_label: str = "CEP41 R242H"     # legend label for the non-control genotype
    # Line-field prefixes that mean control; every other prefix is the mutant.
    control_line_prefixes: Tuple[str, ...] = ("3",)
    min_roi_distance: float = 15.0        # dedupe ROIs whose centroids are closer than this (px)
    motion_max_threshold: float = 15.0    # QC: max motion shift (px) tolerated per recording
    motion_residual_threshold: float = 2.0  # QC: residual motion (px) tolerated per recording
    drift_threshold: float = 1.0          # QC: baseline drift tolerated per recording
    roi_peak_figures: bool = False        # also render per-ROI peak montages (slow)
    dev: bool = False                     # run orcann.dev analyses (unsupported, may change)


@dataclass
class Config:
    run: Run = field(default_factory=Run)
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
        "paths": ("raw", "pre_processed", "infer", "spatial", "activity", "analysis",
                  "runs"),
        "models": ("dir",),
        "train_spatial": ("movies", "masks", "out", "checkpoint"),
    }

    # ---- indicator -> decay constant (s), used when
    # deconvolution.decay_initialisation is null
    INDICATOR_DECAY = {
        "gcamp6f": 0.4, "gcamp6s": 2.0, "jgcamp7f": 0.5, "jgcamp8f": 0.3,
        "jgcamp8m": 0.5, "jgcamp8s": 1.0, "fluo4": 0.4, "fluo-4": 0.4,
        "ogb1": 0.7, "ogb-1": 0.7, "jrgeco1a": 0.7,
    }

    def decay_initialisation(self) -> float:
        """Resolved AR seed in seconds (explicit override wins).

        The indicator's own decay constant, used to initialise OASIS's g. Not
        the decay of the transients in the data — see DeconvolutionParams.
        """
        if self.deconvolution.decay_initialisation is not None:
            return float(self.deconvolution.decay_initialisation)
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
            # Same writer as every stage record, so an exported config is
            # strictly valid JSON by the same rule and not a round-trip through
            # dumps/loads that happens to normalise most things.
            from orcann.run_info import dump as dump_json
            with open(path, "w") as f:
                dump_json(self.to_dict(), f)
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
        fld = next((f for f in fields(sub) if f.name == key), None)
        if fld is None:
            raise KeyError(f"unknown config key: {section}.{key}")
        setattr(sub, key, _coerce(value, getattr(sub, key), _accepts_none(fld)))

    def _to_commented_yaml(self) -> str:
        lines = [
            "# OrCaNN configuration - the single source of truth for a run.",
            "# Edit values and run, e.g.:  orcann segment --config config.yaml",
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


def _accepts_none(fld) -> bool:
    """Whether a field's annotation admits None. Read as text, because a module
    may have its annotations stringified, and because the answer is only used to
    decide whether `--set key=null` clears the field or is a type error."""
    return "Optional[" in str(fld.type) or "None" in str(fld.type)


def _coerce(value: Any, current: Any, accepts_none: bool = True) -> Any:
    """Cast a YAML/string value to the type of the field's current value."""
    if value is None:                                    # YAML null -> None
        return None
    if accepts_none and isinstance(value, str) \
            and value.strip().lower() in ("none", "null"):
        # Reached before the casts below, so `--set n_energy_frames=null` clears
        # an optional number instead of failing int(); a field that may not be
        # null still gets the cast, and its type error.
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
    "run": "The run this config describes - names every output it writes",
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
    "run.key": "names every output of this run: <stage>_outputs_<key>",
    "run.notify_email": "address SGE mails when a run_all chain's last stage ends or aborts; null asks for no mail",
    "paths.raw": "recordings to process (.nd2 / .tif / .npy)",
    "paths.pre_processed": "motion-corrected movies (motion_correction output, infer input)",
    "paths.infer": "cached probability maps (infer output, segment input)",
    "paths.spatial": "segment output: <spatial>/segment_outputs_<key>/<recording>/",
    "paths.activity": "activity output: <activity>/activity_outputs_<key>/<recording>/ (calcium-format, analysis input)",
    "paths.analysis": "analysis stage output (group figures + tables)",
    "paths.runs": "per-run directory: <runs>/<run.key>/config.yaml, the config a submitted run reads",
    "imaging.frame_rate": "recording frame rate in Hz",
    "imaging.indicator": "calcium indicator; resolves the AR seed when deconvolution.decay_initialisation is null",
    "models.dir": "directory of promoted, in-use models",
    "models.spatial": "model identity to run, or 'latest' for the newest promoted one",
    "spatial.threshold": "soma-probability cut, ~0.5-0.6",
    "spatial.min_area": "drop detected regions smaller than this many px (0 disables)",
    "spatial.min_radius": "or drop regions below this equivalent radius in px",
    "spatial.resize_to": "force each frame to NxN instead of the model's training frame size (0 = off)",
    "baseline.method": "global_dff (per-trace rolling percentile); the only supported method",
    "baseline.percentile": "baseline percentile for global_dff",
    "baseline.window_fraction": "rolling-baseline window as a fraction of trace length",
    "baseline.min_window": "minimum rolling-baseline window (frames)",
    "baseline.max_window": "maximum rolling-baseline window (frames)",
    "baseline.presmooth_sigma": "Gaussian smoothing (frames) for F0 estimation only; lifts F0 to the true resting level on noisy traces; 0 = off",
    "deconvolution.enabled": "run OASIS spike inference (false = skip; analysis then has no spikes)",
    "deconvolution.decay_initialisation": "indicator decay constant in s, seeding OASIS's AR coefficient; NOT a measured transient decay (null = resolve from imaging.indicator)",
    "deconvolution.optimize_g": "let OASIS fit the AR coefficient from data",
    "deconvolution.noise_method": "OASIS noise estimator: mean | median | logmexp",
    "deconvolution.s_min": "min spike amplitude in dF/F0; OASIS discards events below this (0 = let OASIS decide)",
    "deconvolution.noise_gate_sigma": "keep only spikes exceeding this multiple of the trace noise floor (0 = no gate)",
    "deconvolution.method": "oasis (AR deconvolution) | robust (deterministic transient detector); no fallback between them",
    "deconvolution.robust_safety_net": "on the oasis path, backfill clear transients OASIS missed (true recommended)",
    "deconvolution.robust_k_onset": "robust detector: event onset threshold as a multiple of noise",
    "deconvolution.robust_k_peak": "robust detector: required peak height as a multiple of noise (main precision knob)",
    "deconvolution.robust_min_duration_s": "robust detector: minimum event duration in seconds (rejects single-sample noise)",
    "motion_correction.mode": "rigid | piecewise_rigid | auto",
    "motion_correction.max_shift": "maximum shift in px",
    "figures.enabled": "write the spatial QC overlay in segment",
    "gallery.interactive": "per-recording interactive HTML gallery (gallery.html)",
    "gallery.max_rois": "cap the number of ROIs drawn in the gallery",
    "train_spatial.movies": "dir of training movies (<stem>.tif)",
    "train_spatial.masks": "dir of masks (<stem>.npy) or ImageJ ROI sets (<stem>.zip); <stem> must equal the movie's exactly",
    "train_spatial.out": "output dir for the trained segmenter.pt",
    "train_spatial.name": "name component of the trained model's identity",
    "train_spatial.checkpoint": "per-epoch checkpoint, overwritten; scratch, not selectable",
    "train_spatial.channels": "energy channels: any of structural, max, variance, correlation",
    "train_spatial.radii": "LoG scale bank, cell radii in px",
    "train_spatial.min_cell_area": "strip ROIs smaller than this from training masks (0 = keep all)",
    "train_spatial.patch": "training patch size in px (must be smaller than the frame)",
    "train_spatial.n_patch": "patches sampled per recording per epoch -- advanced, change only with a measured reason",
    "train_spatial.fg_frac": "fraction of those centred on an annotated cell -- advanced, change only with a measured reason",
    "train_spatial.n_energy_frames": "frames pooled per forward pass, null = every frame; travels with the model into infer -- advanced, change only with a measured reason",
    "train_spatial.edge_correction": "how the frame edge is read: zeros | mean_fill | mean_fill_gain; travels with the model into infer -- advanced, change only with a measured reason",
    "train_spatial.epochs": "training epochs",
    "train_spatial.progress_frames": "write a prediction overlay per tapering epoch schedule into <out>/<name>_progress, for assembling into a series afterwards",
    "train_spatial.val_frac": "fraction of recordings held out for validation",
    "train_spatial.holdout": "hold out val_frac to evaluate; false = train final model on all data",
    "analysis.control_line_prefixes": "cell-line prefixes that are control; any other prefix is the mutant genotype",
    "analysis.mutant_label": "legend label for the non-control genotype",
    "analysis.min_roi_distance": "dedupe ROIs whose centroids are closer than this (px)",
    "analysis.motion_max_threshold": "QC: max motion shift (px) tolerated per recording",
    "analysis.motion_residual_threshold": "QC: residual motion (px) tolerated per recording",
    "analysis.drift_threshold": "QC: baseline drift tolerated per recording",
    "analysis.roi_peak_figures": "also render per-ROI peak montages (slow)",
    "analysis.dev": "run the unsupported orcann.dev analyses (off by default)",
}
