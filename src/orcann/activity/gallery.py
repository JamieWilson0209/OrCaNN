"""The interactive per-recording ROI viewer, as one self-contained HTML file.

``gallery.html`` is a *view* of a recording, not a second copy of it. The arrays
it draws from — ``temporal_traces.npy``, ``traces_denoised.npy``,
``spike_trains.npy`` — sit beside it in the same output directory and stay the
reference. That is what licenses the two reductions below; neither changes what
the page shows.

Traces are stored as a **min/max envelope** at the width the trace card actually
draws (``TRACE_SAMPLES``), quantised to int16 at a step set by each ROI's own
noise, and the whole set deflated into one stream. The envelope keeps both
extremes of every display column, in the order they occur, so the drawn polyline
reduces to the same column-by-column extent as one drawn from every frame;
striding instead would drop transient peaks and make the plot understate the
data. The step comes from :func:`_noise_sigma` per ROI, so the rounding error
sits an order of magnitude below the noise the trace already carries rather than
at a fixed number someone picked.

The page itself is ``gallery.html`` beside this module — shell, style and
behaviour — with the recording substituted into it as one JSON payload. Layout,
palette and interaction follow the OrCaNN ROI viewer design that the desktop app
implements in ``orcann.app.roi_viewer``.

A cell counts as active here when its event rate exceeds zero, the same test
``analysis/loading.py`` applies. The two are still counts over different sets:
the analysis counts within its quality-selected neurons, this counts every
detection in the recording.
"""

import base64
import html as _html
import json
import logging
import zlib
from importlib import resources
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Samples kept per trace. The trace card is ~316 CSS px inside a fixed 340px
# rail and renders at 2x device pixels, so 600 columns of min/max is the width
# the plot can actually resolve. A recording shorter than this is stored whole.
TRACE_SAMPLES = 1200

# The background layers, in the order the rail lists them. The description is
# what the reader sees under the name; it is the whole point of the row.
LAYER_DESCRIPTIONS = [
    ("max", "Max projection", "Brightest value per pixel — best for spotting cells"),
    ("mean", "Mean projection", "Time average — stable structure"),
    ("corr", "Correlation", "Local co-activity between neighbours"),
    ("std", "Std projection", "Variability over time"),
    ("baseline", "Baseline frame", "A quiet reference frame"),
    ("peak", "Peak frame", "The single brightest event frame"),
]


def _grey_png(arr: np.ndarray) -> str:
    """A percentile-windowed greyscale PNG of `arr`, as a data URI.

    Single-channel: these are greyscale images, and routing them through a
    colormap to RGB tripled the bytes for three identical channels.
    """
    from PIL import Image

    a = np.asarray(arr, dtype=np.float32)
    # nanpercentile, then fill: np.percentile returns NaN if any element is NaN,
    # which would make the whole frame black with ROI outlines drawn over it.
    finite = np.isfinite(a)
    vmin = float(np.nanpercentile(a, 1)) if finite.any() else 0.0
    vmax = float(np.nanpercentile(a, 99)) if finite.any() else 1.0

    norm = np.clip((a - vmin) / (vmax - vmin + 1e-10), 0, 1)
    norm[~finite] = 0.0
    u8 = (norm * 255).astype(np.uint8)

    buf = BytesIO()
    Image.fromarray(u8, "L").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _noise_sigma(rows: np.ndarray) -> np.ndarray:
    """Per-row noise level, from the MAD of the first difference.

    Differencing removes the transient's own slow shape and leaves the
    frame-to-frame noise; taking the MAD of that rather than a standard
    deviation keeps the spikes themselves out of the estimate. 1.4826 converts a
    MAD to a Gaussian sigma, and the sqrt(2) undoes the differencing.
    """
    d = np.diff(np.asarray(rows, np.float64), axis=1)
    if d.shape[1] == 0:
        return np.zeros(d.shape[0])
    med = np.median(d, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(d - med), axis=1) / np.sqrt(2.0)


def _envelope(rows: np.ndarray, n_samples: int) -> np.ndarray:
    """Reduce each row to `n_samples` points, keeping every column's extremes.

    Each output column contributes its minimum and its maximum, emitted in the
    order they occur in the recording so the polyline keeps the direction of the
    trace through that column as well as its extent.
    """
    a = np.asarray(rows, np.float32)
    n, T = a.shape
    cols = n_samples // 2
    if T <= n_samples or cols < 1:
        return a

    # reduceat over uneven bins rather than a reshape: a reshape needs T to be a
    # multiple of the column count, and padding it up to one puts the padding
    # across whole columns at the end of the trace, which flattens them.
    edges = np.linspace(0, T, cols + 1).astype(int)[:-1]
    lo = np.minimum.reduceat(a, edges, axis=1)
    hi = np.maximum.reduceat(a, edges, axis=1)

    # Which extreme came first, from the same reduction over half-columns. The
    # pair is emitted in that order so the polyline keeps the trace's direction
    # through the column instead of always rising.
    half = np.linspace(0, T, 2 * cols + 1).astype(int)[:-1]
    lo_half = np.minimum.reduceat(a, half, axis=1)
    lo_first = lo_half[:, 0::2] <= lo_half[:, 1::2]

    out = np.empty((n, 2 * cols), np.float32)
    out[:, 0::2] = np.where(lo_first, lo, hi)
    out[:, 1::2] = np.where(lo_first, hi, lo)
    return out


def _quantise(rows: np.ndarray, sigma: np.ndarray) -> tuple:
    """int16 codes for each row, and the step each row was coded at.

    The step is a tenth of that row's noise, widened only where the row's range
    would not otherwise fit in an int16 — so a trace whose peak is thousands of
    times its noise is coded coarsely rather than wrapping around.
    """
    a = np.asarray(rows, np.float32)
    max_abs = np.abs(a).max(axis=1) if a.shape[1] else np.zeros(a.shape[0])
    step = np.maximum(sigma / 10.0, max_abs / 32767.0)
    step[~np.isfinite(step) | (step <= 0)] = 1.0
    return np.round(a / step[:, None]).astype(np.int16), step


class _Blob:
    """The int16 arrays every ROI reads, concatenated into one deflate stream.

    One stream for the whole recording rather than one per ROI: the denoised
    traces are near-identical stretches of zero, and deflate can only exploit
    that across ROIs if it sees them together.
    """

    def __init__(self):
        self._parts: List[np.ndarray] = []
        self._offset = 0

    def add(self, arr: np.ndarray) -> List[int]:
        """Append an int16 array; return its [offset, length] in int16 units."""
        flat = np.ascontiguousarray(arr, dtype=np.int16).ravel()
        self._parts.append(flat)
        span = [self._offset, int(flat.size)]
        self._offset += int(flat.size)
        return span

    def encode(self) -> str:
        if not self._parts:
            return ""
        raw = np.concatenate(self._parts).tobytes()
        return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def _roi_records(seeds, blob: _Blob, traces_dff: np.ndarray, max_rois: int,
                 frame_rate: float, traces_denoised: Optional[np.ndarray],
                 spike_trains: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    """One record per ROI: geometry, activity metrics, and spans into `blob`."""
    n = min(int(seeds.n_seeds), int(max_rois))

    dff = np.asarray(traces_dff[:n], np.float32) * 100.0      # percent, for display
    den = (np.asarray(traces_denoised[:n], np.float32) * 100.0
           if traces_denoised is not None else None)

    # One step per ROI for both traces: they are the same quantity in the same
    # units, and the denoised trace of a silent cell is all zeros, which carries
    # no noise estimate of its own.
    sigma = _noise_sigma(dff)
    dff_q, step = _quantise(_envelope(dff, TRACE_SAMPLES), sigma)
    den_q = _quantise(_envelope(den, TRACE_SAMPLES), sigma)[0] if den is not None else None

    duration_s = dff.shape[1] / frame_rate
    # Matches analysis/loading.py: events per 10 s over the recording's duration,
    # so the two cannot report different activity for the same recording.
    if spike_trains is not None:
        counts = (np.asarray(spike_trains[:n]) > 0).sum(axis=1)
        rates = counts / duration_s * 10.0
    else:
        counts = np.zeros(n, int)
        rates = np.zeros(n)

    # Peak amplitude off the full trace, not the envelope, and off the denoised
    # trace where there is one. Without deconvolution the dF/F0 peak stands in,
    # which is what the analysis would also be left with.
    peaks = (den if den is not None else dff).max(axis=1) if n else np.zeros(0)

    rois: List[Dict[str, Any]] = []
    no_outline = 0
    for i in range(n):
        y, x = seeds.centers[i]

        contour_span = None
        circularity = None
        if seeds.contour_success[i] and seeds.contours[i] is not None:
            poly = np.asarray(seeds.contours[i].contour).squeeze()
            circularity = float(seeds.contours[i].circularity)
            # find_contours returns half-pixel coordinates, so doubling them is
            # exact rather than a rounding. Under 3 points is not a closed shape.
            if poly.ndim == 2 and poly.shape[0] >= 3:
                contour_span = blob.add(np.round(poly * 2.0))
        if contour_span is None:
            no_outline += 1

        rois.append({
            # 1-based, so a number on screen is the label id in labels.npy and
            # row i-1 of the trace arrays; those arrays are the reference.
            "id": i + 1,
            "cy": float(y), "cx": float(x),
            "r": float(seeds.radii[i]),
            "circ": circularity,
            "intensity": float(seeds.intensities[i]),
            "rate": float(rates[i]),
            "amp": float(peaks[i]),
            "n_ev": int(counts[i]),
            "step": float(step[i]),
            "c": contour_span,
            "t": blob.add(dff_q[i]),
            "d": blob.add(den_q[i]) if den_q is not None else None,
            "ev": (np.flatnonzero(np.asarray(spike_trains[i]) > 0).tolist()
                   if spike_trains is not None else []),
        })

    if no_outline:
        # These stay in the list, the statistics and the arrow-key walk — they
        # have real traces — so this is the one place the canvas and the list
        # legitimately hold different sets, and it should not be silent.
        logger.warning(f"  {no_outline} ROIs have no drawable outline; they are "
                       f"listed and selectable but not drawn on the canvas")
    return rois


def generate_interactive_gallery(
    seeds,                       # ContourSeedResult / roi_adapter.SeedView
    projections,                 # ProjectionSet
    movie: Optional[np.ndarray],
    output_path: str,
    frame_rate: float,
    title: str = "ROI Gallery",
    max_rois: int = 500,
    traces_denoised: Optional[np.ndarray] = None,
    spike_trains: Optional[np.ndarray] = None,
    pipeline_traces_dff: Optional[np.ndarray] = None,
) -> str:
    """Write the recording's interactive gallery to `output_path`.

    Parameters
    ----------
    seeds, projections
        Per-ROI geometry and the projection images, as `roi_adapter` builds them.
    movie : (T, Y, X) or None
        Read only for the two frame-derived background layers. Without it the
        gallery still builds, from whichever projections `projections` carries.
    frame_rate : float
        Hz, from config. Required: an events/10s computed against a guessed
        frame rate would look right and be wrong.
    traces_denoised, spike_trains : (N, T), optional
        Deconvolution output. Absent, the page drops its activity sorting rather
        than ordering every cell by a rate of zero.
    pipeline_traces_dff : (N, T)
        The corrected dF/F0 the detector actually saw.
    """
    if pipeline_traces_dff is None:
        raise ValueError(
            "generate_interactive_gallery requires pipeline_traces_dff. It used "
            "to fall back to a square bounding-box mean of the movie — a "
            "neuropil-contaminated quantity computed over a box rather than the "
            "ROI footprint — and label the panel the same either way, so the "
            "viewer could not tell which they were looking at.")
    if not frame_rate > 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate!r}. The "
                         f"event rate is events per 10 s and has no meaning "
                         f"without it.")

    n_traces = int(pipeline_traces_dff.shape[0])
    if n_traces < seeds.n_seeds:
        raise ValueError(
            f"pipeline_traces_dff covers {n_traces} ROIs but the label image has "
            f"{seeds.n_seeds}. These come from different files "
            f"(traces.npy vs labels.npy); a mismatch means one is stale. "
            f"Refusing rather than showing bounding-box means for the "
            f"remainder — the gallery is wrapped by the caller, so this costs "
            f"the gallery, not the recording's data.")

    n_shown = min(int(seeds.n_seeds), int(max_rois))
    logger.info(f"Generating interactive gallery with {n_shown} ROIs...")
    if spike_trains is None:
        logger.info("  no spike trains: amplitude falls back to the dF/F0 peak "
                    "and the activity sorts are dropped")

    logger.info("  Encoding projection images...")
    sources = {"max": projections.max_proj,
               "mean": getattr(projections, "mean_proj", None),
               "corr": getattr(projections, "correlation", None),
               "std": getattr(projections, "std_proj", None)}
    peak_idx = None
    if movie is not None:
        frame_means = np.mean(movie, axis=(1, 2))
        peak_idx = int(np.argmax(frame_means))
        sources["baseline"] = np.mean(movie[:min(50, movie.shape[0])], axis=0)
        sources["peak"] = movie[peak_idx].astype(np.float32)
        logger.info(f"  Peak frame: #{peak_idx} (mean={frame_means[peak_idx]:.1f})")

    # A layer the recording did not write is left out of the rail rather than
    # shown greyed: an empty control invites the reader to wonder what they are
    # missing, and the rail is meant to list what this recording actually has.
    layers = [{"key": k, "name": name,
               "desc": desc + (f" (#{peak_idx})" if k == "peak" else ""),
               "src": _grey_png(sources[k])}
              for k, name, desc in LAYER_DESCRIPTIONS if sources.get(k) is not None]
    if not layers:
        raise ValueError("no projection layers to draw: the gallery needs at "
                         "least a max projection to put the outlines over.")
    missing = [k for k, _, _ in LAYER_DESCRIPTIONS if sources.get(k) is None]
    if missing:
        logger.info(f"  {len(layers)} background layers; not written by this "
                    f"recording: {', '.join(missing)}")

    logger.info("  Computing ROI diagnostics...")
    blob = _Blob()
    rois = _roi_records(seeds, blob, pipeline_traces_dff, max_rois, frame_rate,
                        traces_denoised, spike_trains)

    d1, d2 = projections.max_proj.shape
    has_radii = seeds.radii.size > 0
    payload = {
        "title": title,
        "width": int(d2), "height": int(d1),
        "frame_rate": float(frame_rate),
        "n_frames": int(pipeline_traces_dff.shape[1]),
        "samples": int(rois[0]["t"][1]) if rois else 0,
        "has_events": spike_trains is not None,
        "layers": layers,
        "stats": {
            "total": int(seeds.n_seeds),
            "shown": len(rois),                  # max_rois can truncate; say so
            "active": int(sum(1 for r in rois if r["rate"] > 0)),
            # Guarded: reductions over an empty array raise, and a recording
            # where segmentation found nothing would lose its whole gallery.
            "median_radius": float(np.median(seeds.radii)) if has_radii else 0.0,
        },
        "rois": rois,
        "blob": blob.encode(),
    }

    template = (resources.files("orcann.activity")
                .joinpath("gallery.html").read_text(encoding="utf-8"))
    # The payload sits inside a <script> block, where the only sequence that can
    # end it early is "</".
    body = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = (template
            .replace("__TITLE__", _html.escape(str(title), quote=True))
            .replace("__PAYLOAD__", body))

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    size_mb = len(html.encode("utf-8")) / 1e6
    logger.info(f"  gallery.html written ({size_mb:.1f} MB)")
    return output_path
