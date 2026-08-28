"""Transient duration by genotype and organoid age. DEV ONLY.

Measures, for every detected transient, how long it stays above the detector's
own onset threshold — ``baseline + k x noise`` — from the frame the excursion
starts on to the first frame back below it. Two figures come out of the same
measurement: durations against developmental day by genotype, and the shape of
the pooled duration distribution, where any clustering would live.

Scope: the ``robust`` deconvolution method ONLY
---------------------------------------------
This is deliberate and the module refuses to run otherwise. The robust detector
defines an event as a contiguous excursion above ``baseline + robust_k_onset x
noise`` (``activity/deconvolution.py::_robust_events``), so the excursion
measured here is the same object the detector used to decide the event existed
— it walks the same boundary, at both ends rather than one. It reports the
event the detector found, not a different event reconstructed post hoc.

On the OASIS path the same measurement would not mean the same thing: OASIS
returns a model reconstruction whose shape is imposed by a single fitted AR(1)
coefficient per ROI, so every event in a trace decays at the same rate by
construction and the measured spread would be an artefact of the fit rather than
a property of the transients. The robust path does no reconstruction at all —
``C_denoised`` is a pass-through copy of the input ΔF/F₀ — so the trace measured
here is the real signal.

What the number is, and is not
------------------------------
- It is a **threshold-crossing width**, not tau, not FWHM and not a decay time.
  It grows with the amplitude of the event at fixed kinetics (a larger transient
  crosses the threshold earlier on the way up and later on the way down) and
  with the noise estimate, which sets where the threshold sits. Comparisons are
  meaningful across groups measured the same way; the absolute seconds are not a
  kinetic constant.
- **Its left edge is a config artefact.** ``robust_min_duration_s`` (default
  0.5 s) censors everything shorter at detection time, so the shortest durations
  here are the detector's floor rather than the shortest transients present.
- Events whose extent runs past what the data can show are **censored**, not
  truncated to a number, and are counted by reason: already above threshold at
  frame 0 (``trace_start``, so the event began before the recording did), merged
  with the previous or next detected event, or still elevated at the trace end.
  Including them would pile a spike of identical wrong values at whatever
  cut-off was chosen.
- A duration measured from the **onset** rather than the peak is the extent of
  one excursion, so two transients riding on one excursion are not two short
  events but one censored long one. That is the honest reading of a merge, and
  it is why the merge counts are on the figure.

Why example traces sit in the same figure
-----------------------------------------
A duration distribution cannot say what its own tail is made of. A 40 s event is
equally consistent with one genuinely slow transient, two transients that never
fell back below threshold between them, and a baseline that wandered upward
under an ordinary event — and the summary statistics are identical in all three
cases. Panels E-G therefore draw the whole trace behind one event at each end of
the measured range, so the reader can see which of those the extremes actually
are before reading anything into panels A-D.

Outputs (under ``<analysis>/figures/dev - Transient Duration/``)
    transient_duration_overview.png    per-event detail by timepoint, pooled
                                       ECDF, frame quantisation, median age
                                       trend, and nine whole example traces
    transient_duration_clustering.png  the pooled distribution on its own, at
                                       frame resolution, for the question of
                                       whether durations cluster at particular
                                       values rather than spreading smoothly
    transient_duration_events.csv      every measured event, including censored

Timepoints are plotted on evenly spaced slots, not on a real-day axis. The days
sampled here cluster (96-99, then 109, 115, 129-130), so a linear day axis spends
most of its width on empty gaps and collides the adjacent labels; nothing in the
analysis reads a rate off that axis. Every panel using it says so, because
evenly spaced days are no longer a timeline and must not be read as one.
"""
import csv
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory as _blend

from orcann.figures._style import GENOTYPE_COLOURS, _save

logger = logging.getLogger(__name__)

# Marker shapes. Outliers get their own shape (not just a colour) so they stay
# distinguishable in greyscale print and for colourblind readers, who cannot
# rely on the genotype hue to carry a second meaning as well.
_MARKER_NORMAL = "o"
_MARKER_OUTLIER = "^"

# Why an event has no duration, in the order a single cause is chosen when more
# than one applies: an event whose start is missing has no duration to be
# uncertain about, so the onset reasons come first.
_PERMUTATIONS = 400

def _tick_fmt() -> FuncFormatter:
    """Log axes here carry seconds a reader thinks in: 0.5, 2, 20 — not 10^0.

    A new instance per axis: a Formatter is bound to the axis it is set on, so
    one shared instance would leave every axis but the last holding a formatter
    that points at someone else's.
    """
    return FuncFormatter(lambda v, _pos: f"{v:g}")

_CENSOR_REASONS = ("trace_start", "prev_event", "next_event", "trace_end")

# An IQR fence needs enough points to mean anything; below this a cell is left
# unflagged rather than declaring one of three points an outlier.
_MIN_FOR_OUTLIERS = 8

# A median and an IQR need fewer points than an outlier fence, but not one.
_MIN_FOR_TREND = 5

# Example traces per role (shortest / median / longest). One example of a
# duration shared by hundreds of events is an anecdote; three that span the
# amplitude range show whether the duration is a property of the event or of
# the ROI it was measured in.
_EXAMPLES_PER_ROLE = 3

# Chart furniture. Text wears ink, never a series colour, so genotype hue never
# has to compete with a label for meaning.
_INK, _MUTED, _GRID = "#333333", "#777777", "#DDDDDD"

# Genotype offset either side of a timepoint slot, and where the per-cell counts
# sit (axes fraction, below the tick labels).
_SLOT_OFFSET = 0.19
_COUNT_Y = -0.115


def _run_info(ds) -> Dict:
    """A recording's ``run_info.json`` as a dict, or ``{}`` if it is unusable."""
    p = os.path.join(str(getattr(ds, "filepath", "") or ""), "run_info.json")
    try:
        with open(p) as fh:
            info = json.load(fh)
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def _frame_rate(ds, default: float) -> float:
    """The recording's own frame rate, from ``run_info.json`` where it exists.

    ``ds.frame_rate`` cannot be trusted for this: ``loading.load_dataset_metrics``
    reads the rate from a ``config`` sub-dict of run_info.json, and
    ``run_activity`` writes ``frame_rate`` at the top level of that file, so the
    lookup misses and every dataset carries the global override instead. Every
    number this module reports is a count of frames divided by this value, so a
    recording imaged at a different rate would be wrong end to end — durations,
    onsets and peak times alike — with nothing on the figure to show it.
    """
    v = _run_info(ds).get("frame_rate")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        return float(v)
    return float(getattr(ds, "frame_rate", 0) or default)


def _extract_day(name: str) -> Optional[int]:
    """Developmental day as an int from a recording id ('D109_3-63_...' -> 109).

    ``loading._extract_organoid_id`` returns the 'D109' token as a string, which
    sorts lexically ('D96' after 'D109') and cannot be a numeric axis. This is
    the numeric twin, and returns None rather than guessing when the id does not
    start with the D<number> convention.
    """
    m = re.match(r"^[Dd](\d+)", str(name).strip())
    return int(m.group(1)) if m else None


def _mad_noise(trace: np.ndarray) -> float:
    """Robust noise from successive differences.

    Same estimator as the robust detector uses, so the threshold measured
    against here is the threshold the detector actually applied. Kept as a local
    copy rather than imported from ``activity.deconvolution`` to keep this dev
    module from pinning anything in the supported path.
    """
    diff = np.diff(trace)
    return float(1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2))


def measure_durations(traces: np.ndarray, spikes: np.ndarray, frame_rate: float,
                      k_onset: float = 3.0) -> Dict[str, np.ndarray]:
    """Above-threshold duration of every detected event in one recording.

    An event's extent is the contiguous excursion above ``median + k_onset x
    noise`` containing the detected peak: from the first frame of that excursion
    to the first frame back below it. That is the boundary ``_robust_events``
    itself used to decide the event existed, applied at both ends instead of
    one, so the number reports the event the detector found.

    Parameters
    ----------
    traces : (n_neuron, T) — the ΔF/F₀ traces the detector ran on.
    spikes : (n_neuron, T) — spike train; non-zero marks an event peak.
    frame_rate : Hz.
    k_onset : the detector's onset multiplier; the excursion is bounded by
        ``median(trace) + k_onset x noise`` at both ends.

    Returns
    -------
    dict of parallel arrays, one entry per detected event:
        neuron, onset_frame, peak_frame, duration_s, amplitude,
        censored (bool), censor_reason:

        ``''``            bounded at both ends, so measured
        ``trace_start``   above threshold from frame 0 — the event was already
                          under way when the recording began, so its onset does
                          not exist in the data
        ``prev_event``    the excursion also holds the previous detected peak
        ``next_event``    the excursion also holds the next detected peak
        ``trace_end``     still above threshold at the last frame

    A censored event is reported, not guessed at: the reason is recorded and the
    duration left as NaN. Where both ends fail, the onset is named, because an
    event of unknown start has no duration to be uncertain about in the first
    place.
    """
    neuron, onset_f, peak_f, dur_s = [], [], [], []
    amp, censored, reason = [], [], []
    n, T = traces.shape

    for i in range(n):
        trace = np.asarray(traces[i], dtype=np.float64)
        noise = _mad_noise(trace)
        if not np.isfinite(noise) or noise <= 0:
            continue
        base = float(np.median(trace))
        threshold = base + k_onset * noise

        peaks = np.flatnonzero(np.asarray(spikes[i]) > 0)
        for j, pk in enumerate(peaks):
            pk = int(pk)
            # Both searches stop at the neighbouring peak: past it the trace
            # belongs to that event, and running on would measure the pair as
            # one transient.
            prev_pk = int(peaks[j - 1]) if j else -1
            stop = int(peaks[j + 1]) if j + 1 < len(peaks) else T

            before = trace[prev_pk + 1:pk]
            back = np.flatnonzero(before <= threshold)
            if back.size:
                # The excursion starts on the frame after the last one below.
                onset = prev_pk + 1 + int(back[-1]) + 1
                why = ""
            else:
                onset = -1
                why = "prev_event" if prev_pk >= 0 else "trace_start"

            after = trace[pk + 1:stop]
            fwd = np.flatnonzero(after <= threshold)
            if fwd.size:
                end = pk + 1 + int(fwd[0])
            else:
                end = -1
                why = why or ("trace_end" if stop == T else "next_event")

            neuron.append(i)
            onset_f.append(onset)
            peak_f.append(pk)
            amp.append(float(trace[pk] - base))
            if why:
                dur_s.append(np.nan)
                censored.append(True)
                reason.append(why)
            else:
                dur_s.append(float((end - onset) / frame_rate))
                censored.append(False)
                reason.append("")

    return {
        "neuron": np.asarray(neuron, dtype=np.int32),
        "onset_frame": np.asarray(onset_f, dtype=np.int32),
        "peak_frame": np.asarray(peak_f, dtype=np.int32),
        "duration_s": np.asarray(dur_s, dtype=np.float64),
        "amplitude": np.asarray(amp, dtype=np.float64),
        "censored": np.asarray(censored, dtype=bool),
        "censor_reason": np.asarray(reason, dtype=object),
    }


def _flag_outliers(values: np.ndarray) -> np.ndarray:
    """1.5xIQR fence on log10(duration), within one (day, genotype) cell.

    On log values because durations are right-skewed: a raw-seconds fence
    flags most of the long tail of a perfectly ordinary distribution. Cells
    smaller than ``_MIN_FOR_OUTLIERS`` are never flagged.
    """
    out = np.zeros(values.shape, dtype=bool)
    good = np.isfinite(values) & (values > 0)
    if good.sum() < _MIN_FOR_OUTLIERS:
        return out
    lv = np.log10(values[good])
    q1, q3 = np.percentile(lv, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return out
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    idx = np.flatnonzero(good)
    out[idx[(lv < lo) | (lv > hi)]] = True
    return out


def _log_limits(rows: List[Dict]) -> Tuple[float, float]:
    """Padded log-axis limits covering every measured duration in ``rows``."""
    vals = [r["duration_s"] for r in rows if r.get("duration_s") is not None]
    if not vals:
        return .5, 10.
    return float(min(vals)) * .7, float(max(vals)) * 1.4


def _log_ticks(lo: float, hi: float) -> List[float]:
    """The 1-2-5 decade ticks that fall inside a range."""
    return [t for t in (.2, .5, 1, 2, 5, 10, 20, 50, 100, 200, 500)
            if lo <= t <= hi]


def _load_selected(ds) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """The selected ΔF/F₀ traces and spike trains for one recording.

    ``load_dataset_metrics`` drops the three (n_selected, T) matrices before it
    returns — the "Free large trace arrays" block at the end of
    ``analysis/loading.py`` — so *every* dataset handed to an analysis module
    has ``selected_traces is None``. Reading the attribute is therefore not
    enough; reload from disk and re-slice, the way
    ``figures.overview.fig_selected_traces`` already does, reproducing exactly
    what loading held: ``C_denoised[selected_roi_indices]`` and
    ``S[selected_roi_indices]``.

    ``traces_denoised.npy`` is the right file to reload even though this module
    wants the real signal rather than a reconstruction: on the robust path it
    is a pass-through copy of the input ΔF/F₀ (see the module docstring), and
    that path is the only one this module runs on.

    The in-memory arrays are still preferred when they are present, so a caller
    that builds datasets itself — a test, a notebook — needs nothing on disk.

    Rows come back in selected-neuron order, which makes the ``neuron`` column
    of the events CSV the row index into that recording's block of
    ``data/selected_rois.csv``. A recording whose stored indices do not fit the
    arrays on disk is skipped whole rather than in part, since dropping rows
    would shift that correspondence silently.
    """
    traces = getattr(ds, "selected_traces", None)
    spikes = getattr(ds, "selected_spikes", None)
    if traces is not None and spikes is not None and len(traces):
        # A caller that builds datasets itself can hand over mismatched arrays;
        # rejecting the pair here produces the documented "skipped" path instead
        # of an IndexError from inside a figure, half a run later.
        if len(spikes) != len(traces):
            logger.warning(f"  {ds.name}: {len(traces)} traces against "
                           f"{len(spikes)} spike rows — skipped")
            return None, None
        return np.asarray(traces), np.asarray(spikes)

    roi_idx = getattr(ds, "selected_roi_indices", None)
    if roi_idx is None or len(roi_idx) == 0:
        logger.warning(f"  {ds.name}: no selected_roi_indices — skipped")
        return None, None

    data_dir = os.path.join(str(getattr(ds, "filepath", "") or ""), "data")
    t_path = os.path.join(data_dir, "traces_denoised.npy")
    s_path = os.path.join(data_dir, "spike_trains.npy")
    if not (os.path.isfile(t_path) and os.path.isfile(s_path)):
        logger.warning(f"  {ds.name}: traces_denoised.npy / spike_trains.npy "
                       f"not found under {data_dir} — skipped")
        return None, None

    C = np.load(t_path)
    S = np.load(s_path)
    roi_idx = np.asarray(roi_idx, dtype=int)
    n_rows = min(C.shape[0], S.shape[0])
    if int(roi_idx.max()) >= n_rows:
        logger.warning(f"  {ds.name}: selected ROI index {int(roi_idx.max())} "
                       f"out of range for the arrays on disk "
                       f"({C.shape[0]} denoised, {S.shape[0]} spike rows) — "
                       f"skipped")
        return None, None
    return C[roi_idx], S[roi_idx]


def _collect(datasets: List, frame_rate_default: float, k_onset: float
             ) -> Tuple[List[Dict], Dict, Dict[str, float]]:
    """Measure every dataset, returning per-event rows plus a censoring tally.

    The third return is the frame rate each recording was measured at, kept
    because the example-trace panels need it to convert a stored frame back to
    a time and it is not in the events CSV.
    """
    rows: List[Dict] = []
    tally = {"events": 0, "recordings": 0, "no_day": 0, "no_traces": 0,
             "censored": {r: 0 for r in _CENSOR_REASONS}}
    rates: Dict[str, float] = {}

    for ds in datasets:
        traces, spikes = _load_selected(ds)
        if traces is None:
            tally["no_traces"] += 1
            continue
        day = _extract_day(ds.name)
        if day is None:
            tally["no_day"] += 1
            logger.warning(f"  {ds.name}: no D<number> day in the id — skipped")
            continue

        fr = _frame_rate(ds, frame_rate_default)
        rates[ds.name] = fr
        m = measure_durations(traces, spikes, fr, k_onset)
        tally["recordings"] += 1
        tally["events"] += len(m["duration_s"])
        for r in _CENSOR_REASONS:
            tally["censored"][r] += int((m["censor_reason"] == r).sum())

        geno = getattr(ds, "genotype", "") or "Unknown"
        for j in range(len(m["duration_s"])):
            onset = int(m["onset_frame"][j])
            rows.append({
                "recording": ds.name,
                "day": day,
                "genotype": geno,
                "line_id": getattr(ds, "line_id", "") or "",
                "neuron": int(m["neuron"][j]),
                "onset_frame": onset if onset >= 0 else None,
                "onset_time_s": round(onset / fr, 4) if onset >= 0 else None,
                "peak_frame": int(m["peak_frame"][j]),
                "peak_time_s": round(float(m["peak_frame"][j]) / fr, 4),
                "duration_s": (None if m["censored"][j]
                               else round(float(m["duration_s"][j]), 4)),
                "amplitude_dff": round(float(m["amplitude"][j]), 5),
                "censored": bool(m["censored"][j]),
                "censor_reason": str(m["censor_reason"][j]),
                "outlier": False,          # filled in per (day, genotype) below
            })
    return rows, tally, rates


def _mark_outliers(rows: List[Dict]) -> None:
    """Flag outliers in place, independently within each (day, genotype) cell."""
    cells: Dict[Tuple[int, str], List[int]] = {}
    for i, r in enumerate(rows):
        if r["duration_s"] is not None:
            cells.setdefault((r["day"], r["genotype"]), []).append(i)
    for idx in cells.values():
        vals = np.array([rows[i]["duration_s"] for i in idx], dtype=float)
        for i, flag in zip(idx, _flag_outliers(vals)):
            rows[i]["outlier"] = bool(flag)


def _write_csv(rows: List[Dict], path: str) -> None:
    fields = ["recording", "day", "genotype", "line_id", "neuron", "onset_frame",
              "onset_time_s", "peak_frame", "peak_time_s", "duration_s",
              "amplitude_dff", "censored", "censor_reason", "outlier"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"  wrote {len(rows)} events -> {path}")


def _spread(cands: List[Dict], n: int) -> List[Dict]:
    """``n`` events spanning the amplitude range of a set tied on duration.

    Ties are the rule here, not the exception: duration is quantised to the frame,
    so one duration is shared by hundreds of events and "the shortest event" is
    not a single trace. Sorting the tied set by amplitude and sampling it evenly
    shows that duration at a small, a typical and a large event instead of three
    interchangeable copies of one — and amplitude is the variable the
    measurement is known to depend on (a taller transient starts further above
    the threshold), so it is the one worth spanning. The rest of the sort key
    only makes the choice reproducible across runs.

    Recordings are deliberately *not* spread over: three long events that all
    come from one recording is a fact about the tail worth seeing, and forcing
    variety would hide it. The subtitles name the recording in every panel.
    """
    ranked = sorted(cands, key=lambda r: (r["amplitude_dff"], r["recording"],
                                          r["neuron"], r["peak_frame"]))
    if len(ranked) <= n:
        return ranked
    if n == 1:
        return [ranked[len(ranked) // 2]]
    idx = sorted({round(i * (len(ranked) - 1) / (n - 1)) for i in range(n)})
    return [ranked[i] for i in idx]


def _pick_exemplars(rows: List[Dict],
                    per_role: int = _EXAMPLES_PER_ROLE) -> List[Dict]:
    """Example events at the short end, the middle and the long end.

    Short and median are drawn from the events sharing that exact duration,
    because hundreds do; long is simply the longest few, which are distinct
    durations rather than a tie. Censored events are never candidates: their
    duration was never measured, so there is no sense in which one of them is
    the longest — and that is the panel a censored event would be most tempting
    to fill.
    """
    measured = [r for r in rows if r["duration_s"] is not None]
    if not measured:
        return []
    vals = np.array([r["duration_s"] for r in measured], dtype=float)

    picks: List[Dict] = []
    for letter, role, target in (("E", "Shortest", float(vals.min())),
                                 ("F", "Median", float(np.median(vals)))):
        # The median of a quantised quantity need not be a value that occurs;
        # the tie group is the observed value nearest to it.
        value = float(vals[np.argmin(np.abs(vals - target))])
        tied = [r for r in measured if r["duration_s"] == value]
        chosen = _spread(tied, per_role)
        picks += [{"label": f"{letter}{k + 1}", "role": role, "row": r}
                  for k, r in enumerate(chosen)]

    longest = sorted(measured, key=lambda r: (-r["duration_s"], r["recording"],
                                              r["neuron"], r["peak_frame"]))
    picks += [{"label": f"G{k + 1}", "role": "Longest", "row": r}
              for k, r in enumerate(longest[:per_role])]
    return picks


def _exemplar_traces(picks: List[Dict], datasets: List,
                     rates: Dict[str, float], k_onset: float,
                     frame_rate_default: float) -> List[Dict]:
    """Reload each exemplar's whole trace, and locate its event within it.

    Reloading is the price of not holding every recording's traces in memory for
    the whole run: a handful of recordings are read a second time, with the same
    call ``_collect`` already used, and each is read once however many exemplars
    it supplies. The threshold is recomputed from the reloaded trace rather than
    stored, so the line a panel draws is the line its measurement was made
    against, by construction.

    A panel whose trace cannot be recovered carries the reason instead of a
    plot. It is not silently replaced with the next candidate: that would show
    an event other than the one the title claims.
    """
    by_name = {str(getattr(d, "name", "")): d for d in datasets}
    cache: Dict[str, Tuple] = {}
    out: List[Dict] = []

    for pick in picks:
        row = pick["row"]
        ex: Dict = dict(pick, trace=None, reason="")
        rec = str(row["recording"])
        if rec not in cache:
            ds = by_name.get(rec)
            cache[rec] = _load_selected(ds) if ds is not None else (None, None)
        traces, spikes = cache[rec]

        if traces is None:
            ex["reason"] = f"{rec}:\ntraces could not be reloaded"
        elif int(row["neuron"]) >= len(traces):
            ex["reason"] = (f"{rec}:\nneuron {row['neuron']} is not in the "
                            f"reloaded traces")
        else:
            i = int(row["neuron"])
            fr = float(rates.get(rec, frame_rate_default))
            trace = np.asarray(traces[i], dtype=np.float64)
            base = float(np.median(trace))
            onset = int(row["onset_frame"])
            ex.update(trace=trace, fr=fr, base=base,
                      threshold=base + k_onset * _mad_noise(trace),
                      peak=int(row["peak_frame"]), onset=onset,
                      end=onset + int(round(float(row["duration_s"]) * fr)),
                      peaks=np.flatnonzero(np.asarray(spikes[i]) > 0))
        out.append(ex)

    return out


def _panel_box_strip(ax, rows, days, genos, colours, labels) -> None:
    """A: every event, boxed by (timepoint x genotype), on evenly spaced slots.

    A timepoint holding only one genotype supports no genotype comparison at
    all, and three of these eight do. That is left to the per-cell counts under
    the axis, which say "not sampled" in as many words.

    Slots are ``arange(len(days))``, not the day numbers. The real-day axis this
    replaces spent most of its width on the 99->109 and 115->129 gaps and
    collided the D129/D130 tick labels; nothing in the analysis reads a rate off
    the x axis, so equal width per sampling occasion is the honest trade. The
    axis label says so, because evenly spaced days are no longer a timeline.

    y is log: the distribution is right-skewed (median 2 s against a max near
    80 s), so on a linear axis four fifths of the events pile into the bottom
    tenth of the panel. The module already treats the quantity as log-scaled —
    ``_flag_outliers`` fences on log10 — so the axis now matches the statistic.
    """
    tf = _blend(ax.transData, ax.transAxes)
    x = np.arange(len(days), dtype=float)

    for gi, geno in enumerate(genos):
        colour = colours[geno]
        for di, day in enumerate(days):
            sel = [r for r in rows if r["day"] == day and r["genotype"] == geno]
            pos = x[di] + (_SLOT_OFFSET if gi else -_SLOT_OFFSET)
            if not sel:
                ax.text(pos, _COUNT_Y, "not\nsampled", ha="center", va="top",
                        fontsize=6.5, color=_MUTED, linespacing=1.15, transform=tf)
                continue
            vals = np.array([r["duration_s"] for r in sel], dtype=float)
            rng = np.random.default_rng(0)     # fixed: same data plots the same
            for flagged, marker, alpha, size in ((False, _MARKER_NORMAL, .28, 7),
                                                 (True, _MARKER_OUTLIER, .95, 26)):
                v = np.array([r["duration_s"] for r in sel
                              if r["outlier"] is flagged], dtype=float)
                if not v.size:
                    continue
                ax.scatter(pos + rng.uniform(-.075, .075, v.size), v, s=size,
                           marker=marker, alpha=alpha, zorder=4 if flagged else 2,
                           facecolors="none" if flagged else colour,
                           edgecolors=colour, linewidths=.9 if flagged else 0)
            # Unfilled box: a white face hides the strip over the IQR, which is
            # exactly the range the strip exists to show.
            ax.boxplot([vals], positions=[pos], widths=.28, showfliers=False,
                       patch_artist=True, zorder=3,
                       medianprops=dict(color=colour, lw=2.2),
                       boxprops=dict(facecolor="none", edgecolor=colour, lw=1.1),
                       whiskerprops=dict(color=colour, lw=1.1),
                       capprops=dict(color=colour, lw=1.1))
            n_rec = len({r["recording"] for r in sel})
            ax.text(pos, _COUNT_Y, f"{vals.size}\n{n_rec} rec", ha="center",
                    va="top", fontsize=6.5, color=_MUTED, linespacing=1.15,
                    transform=tf)

    ax.set_yscale("log")
    lim = _log_limits(rows)
    ax.set_ylim(*lim)
    # A narrow range can hold no 1-2-5 tick at all; matplotlib's own locator is
    # a better answer there than an axis with no labels on it.
    if _log_ticks(*lim):
        ax.set_yticks(_log_ticks(*lim))
        ax.get_yaxis().set_major_formatter(_tick_fmt())
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}" for d in days])
    ax.set_xlim(-.6, len(days) - .4)
    ax.set_ylabel("Duration above threshold (s, log)", fontsize=10)
    ax.set_xlabel("Timepoint \u2014 evenly spaced, NOT a real-time axis.   "
                  "A cell reading \u201cnot sampled\u201d has no genotype "
                  "comparison at that timepoint.",
                  fontsize=9, color=_MUTED, labelpad=34)
    ax.set_title("A   Every event, by timepoint and genotype", fontsize=11,
                 fontweight="bold", loc="left", color=_INK)
    for geno in genos:
        ax.scatter([], [], s=28, color=colours[geno], label=labels[geno])
    ax.scatter([], [], s=26, marker=_MARKER_OUTLIER, facecolors="none",
               edgecolors=_MUTED, label="outlier (1.5xIQR on log10, within cell)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3)


def _panel_ecdf(ax, pooled, genos, colours, labels) -> None:
    """B: the two distributions with no binning choice in the way.

    An ECDF is the right form for "are these two the same shape": it uses every
    event, invents no bin edges, and its stair-steps make the frame quantisation
    visible rather than smoothing it away.
    """
    for geno in genos:
        v = np.sort(pooled[geno])
        if not v.size:
            continue
        ax.step(v, np.arange(1, v.size + 1) / v.size, where="post",
                color=colours[geno], lw=1.8, label=f"{labels[geno]}  n={v.size}")
    ax.set_xscale("log")
    lim = _log_limits([{"duration_s": v} for g in genos for v in pooled[g]])
    ax.set_xlim(*lim)
    if _log_ticks(*lim):
        ax.set_xticks(_log_ticks(*lim))
        ax.get_xaxis().set_major_formatter(_tick_fmt())
    ax.axhline(.5, color=_GRID, lw=.8, zorder=0)
    ax.set_ylabel("Cumulative fraction", fontsize=9.5)
    ax.set_xlabel("Duration (s, log)", fontsize=9.5)
    ax.set_title("B   Pooled distribution", fontsize=10.5, fontweight="bold",
                 loc="left", color=_INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")


def _panel_quantisation(ax, pooled, genos, colours, labels, quantum, *,
                        one_grid: bool = True) -> None:
    """C: the measurement's real resolution, stated rather than implied.

    Decay is a count of frames until the trace falls back under the threshold,
    so it can only ever be a multiple of one frame. Panels A/B render it as a
    continuous quantity; this one says out loud how coarse the underlying grid
    is, which is what stops a 1.5 s vs 2.0 s difference being over-read.
    """
    n_bin = 16
    edges = np.arange(0, n_bin + 2) * quantum
    for gi, geno in enumerate(genos):
        v = pooled[geno]
        if not v.size:
            continue
        h, _ = np.histogram(np.clip(v, 0, n_bin * quantum), bins=edges)
        ax.bar(edges[:-1] + quantum * (.12 + .38 * gi), h / h.sum(),
               width=quantum * .38, color=colours[geno], align="edge",
               label=labels[geno], linewidth=0)
    top = n_bin * quantum
    ax.set_xticks([0, top / 4, top / 2, top * 3 / 4, top])
    ax.set_xticklabels(["0", f"{top/4:g}", f"{top/2:g}", f"{top*3/4:g}", f"\u2265{top:g}"])
    ax.set_xlabel(f"Duration (s) \u2014 bin = 1 frame = {quantum:g} s" if one_grid
                  else f"Duration (s) \u2014 bin = {quantum:g} s, the smallest "
                       f"step seen; recordings differ in frame rate",
                  fontsize=9.5)
    ax.set_ylabel("Fraction of events", fontsize=9.5)
    ax.set_title("C   Resolution is one frame", fontsize=10.5, fontweight="bold",
                 loc="left", color=_INK)
    n_distinct = len(np.unique(np.concatenate([pooled[g] for g in genos
                                               if pooled[g].size])))
    # Left of the last bar, which the right edge of this note used to sit on.
    ax.text(.90, .62, f"{n_distinct} distinct values exist\nin the whole dataset\n"
            f"last bar pools \u2265{top:g} s", transform=ax.transAxes, ha="right",
            va="top", fontsize=7.5, color=_MUTED, linespacing=1.4)
    ax.legend(fontsize=8, frameon=False, loc="upper right")


def _panel_trend(ax, rows, days, genos, colours, labels) -> None:
    """D: the age trend as median + IQR, on the same evenly spaced slots.

    The line is broken with NaN wherever a genotype has no cell at a timepoint.
    Joining D109 to D129 across a D115 the Control was never sampled at would
    draw a trend through data that does not exist — the single most misleading
    thing this panel could do, given three of eight timepoints hold one genotype.
    """
    x = np.arange(len(days), dtype=float)
    drawn: List[float] = []
    for gi, geno in enumerate(genos):
        dx = .055 if gi else -.055
        med = np.full(len(days), np.nan)
        for di, day in enumerate(days):
            v = np.array([r["duration_s"] for r in rows
                          if r["day"] == day and r["genotype"] == geno], dtype=float)
            if v.size < _MIN_FOR_TREND:
                continue
            lo, hi = np.percentile(v, [25, 75])
            med[di] = np.median(v)
            drawn += [float(lo), float(hi)]
            ax.plot([x[di] + dx] * 2, [lo, hi], color=colours[geno], lw=1.2,
                    alpha=.55, solid_capstyle="butt", zorder=2)
        ax.plot(x + dx, med, color=colours[geno], lw=1.8, zorder=3,
                marker="o" if gi == 0 else "s", ms=5, label=labels[geno],
                markeredgecolor="white", markeredgewidth=.8)

    # From what the panel draws. A fixed ceiling would push a slow cell's whisker
    # off the top, and a break in this line already means "not sampled" — the
    # last thing that should also be able to mean "off the scale".
    ax.set_yscale("log")
    lim = ((min(drawn) * .8, max(drawn) * 1.25) if drawn else (.6, 16))
    ax.set_ylim(*lim)
    ticks = _log_ticks(*lim)
    if ticks:
        ax.set_yticks(ticks)
        ax.get_yaxis().set_major_formatter(_tick_fmt())
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}" for d in days], fontsize=8)
    ax.set_xlim(-.55, len(days) - .45)
    ax.set_ylabel("Median duration (s, log)", fontsize=9.5)
    ax.set_xlabel("Timepoint — evenly spaced; breaks = not sampled",
                  fontsize=8.5, color=_MUTED)
    ax.set_title("D   Median with IQR", fontsize=10.5, fontweight="bold",
                 loc="left", color=_INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def _panel_trace(ax, ex: Dict, colours: Dict, labels: Dict, *,
                 show_x: bool = True, show_y: bool = True) -> None:
    """One example event, drawn in the whole trace it was measured in.

    The whole recording is shown rather than a window around the event, because
    the questions an example is being asked are about context: is the baseline
    this event sits on flat, is this ROI firing constantly or once, is the
    measured event typical of the trace it came from. A cropped window answers
    none of them, and a flat-looking crop can come from a trace that drifts.

    The cost is that a one-frame event is a hairline at this scale, so the
    measured interval is shaded *and* labelled with a leader down to the peak;
    the shading keeps the true width, the label makes it findable. Every other
    detected peak in the trace is a tick on the rug along the bottom, so the
    reader can see whether the event stands alone or sits in a burst.
    """
    row = ex["row"]
    title = f"{ex['label']}   {ex['role']} \u2014 {float(row['duration_s']):g} s"

    if ex["trace"] is None:
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left",
                     color=_INK)
        ax.text(.5, .5, ex["reason"], transform=ax.transAxes, ha="center",
                va="center", fontsize=8, color=_MUTED, linespacing=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return

    colour = colours.get(row["genotype"], GENOTYPE_COLOURS["Unknown"])
    fr, peak, thr, base = ex["fr"], ex["peak"], ex["threshold"], ex["base"]
    trace = ex["trace"]
    dur = float(row["duration_s"])
    t = np.arange(trace.size) / fr

    ax.axvspan(ex["onset"] / fr, ex["end"] / fr, color=colour, alpha=.30,
               linewidth=0, zorder=2)
    ax.axhline(base, color=_MUTED, lw=.7, ls=(0, (5, 4)), zorder=3)
    ax.axhline(thr, color=colour, lw=.8, ls=(0, (5, 4)), zorder=3)
    ax.plot(t, trace, color=_INK, lw=.65, zorder=4)

    # Headroom above the trace for the label, and room under it for the rug.
    ymin, ymax = float(min(trace.min(), base)), float(max(trace.max(), thr))
    span = (ymax - ymin) or 1.0
    ax.set_ylim(ymin - .17 * span, ymax + .30 * span)

    rug = [p / fr for p in ex["peaks"] if int(p) != peak]
    if rug:
        ax.plot(rug, [.045] * len(rug), marker="|", ms=4.5, ls="none",
                color=_MUTED, alpha=.55, mew=.8, zorder=3,
                transform=_blend(ax.transData, ax.transAxes))

    at_edge = peak / max(trace.size - 1, 1)
    dx = 20 if at_edge < .07 else (-20 if at_edge > .93 else 0)
    ax.annotate(f"{dur:g} s", xy=(peak / fr, trace[peak]),
                xytext=(dx, 16), textcoords="offset points",
                ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                va="bottom", fontsize=7.5, fontweight="bold", color=colour,
                zorder=6,
                bbox=dict(boxstyle="round,pad=.16", fc="white", ec="none",
                          alpha=.85),
                arrowprops=dict(arrowstyle="-", color=colour, lw=.8,
                                shrinkA=1, shrinkB=2))

    ax.set_xlim(0, t[-1])
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left",
                 color=_INK, pad=15)
    meta = (f"{row['recording']}  \u00b7  "
            f"{labels.get(row['genotype'], row['genotype'])}"
            f"  \u00b7  neuron {row['neuron']}  \u00b7  peak "
            f"{row['peak_time_s']:g} s  \u00b7  amp "
            f"{row['amplitude_dff']:.3g}"
            + ("  \u00b7  outlier" if row["outlier"] else ""))
    ax.text(0, 1.03, meta, transform=ax.transAxes, fontsize=6.8, color=_MUTED,
            ha="left", va="bottom")
    if show_x:
        ax.set_xlabel("Time in recording (s)", fontsize=8.5)
    if show_y:
        ax.set_ylabel("\u0394F/F\u2080", fontsize=9)


def _figure(rows: List[Dict], tally: Dict, path: str,
            exemplars: Optional[List[Dict]] = None,
            mutant_label: str = "Mutant", k_onset: float = 3.0,
            rates: Optional[Dict[str, float]] = None) -> None:
    """Four summaries and nine examples, because no one view is sufficient.

    A is the detail (every event, both genotypes, per timepoint), B compares the
    two distributions without binning them, C states the measurement resolution,
    and D is the age trend. A and D share evenly spaced timepoint slots.

    E, F and G are a row each — shortest, median and longest — of three whole
    traces. They are the only panels that can distinguish a slow transient from
    a merged pair, or a clean measurement from one made on a drifting baseline,
    and they are placed under the distributions they are the extremes of.
    """
    plotted = [r for r in rows if r["duration_s"] is not None]
    if not plotted:
        logger.warning("  no uncensored events to plot — figure skipped")
        return

    days = sorted({r["day"] for r in plotted})
    genos = sorted({r["genotype"] for r in plotted})
    colours = {g: GENOTYPE_COLOURS.get(g, GENOTYPE_COLOURS["Unknown"]) for g in genos}
    labels = {g: (mutant_label if g == "Mutant" else g) for g in genos}
    pooled = {g: np.array([r["duration_s"] for r in plotted if r["genotype"] == g],
                          dtype=float) for g in genos}
    # One frame. Taken from the frame rate when every recording shares one,
    # since then it is known exactly rather than inferred; otherwise from the
    # spacing between the values that occur. Never from the smallest value: the
    # detector rejects anything under ``robust_min_duration_s``, so the shortest
    # measured event is the floor that rule imposes (two frames here), and
    # reading the frame off it would state the resolution as twice its real size
    # in the one panel whose whole job is to state it correctly.
    frames = sorted({round(1 / fr, 6) for fr in (rates or {}).values() if fr > 0})
    if len(frames) == 1:
        quantum, one_grid = frames[0], True
    else:
        distinct = np.unique([r["duration_s"] for r in plotted])
        quantum = (float(np.diff(distinct).min()) if distinct.size > 1
                   else float(distinct[0]))
        one_grid = not frames

    exemplars = exemplars or []
    role_row = {}
    for ex in exemplars:
        role_row.setdefault(ex["label"][0], len(role_row))
    rows_of_traces = len(role_row)

    # Laid out in inches and converted, because the two halves of this figure
    # want different spacing and a single GridSpec can only have one: the
    # summaries carry a y label each and need room between them, the trace grid
    # labels only its outer edge and would waste that room. Fractions here would
    # have to be recomputed by hand every time a row is added.
    # ``block`` is the gap under the summaries, which has to clear their x
    # labels before the first trace title; ``gap`` is the smaller one between
    # trace rows, which clears a title and a subtitle only.
    m_top, h_sum, block, gap, h_row, m_bot = 2.0, 6.6, 1.2, .62, 2.35, .6
    height = (m_top + h_sum + m_bot
              + (block + rows_of_traces * h_row if rows_of_traces else 0))
    fig = plt.figure(figsize=(15, height))
    edges = dict(left=.058, right=.982)

    gs_sum = GridSpec(2, 6, figure=fig, height_ratios=[1.3, 1], hspace=.5,
                      wspace=.75, top=1 - m_top / height,
                      bottom=1 - (m_top + h_sum) / height, **edges)
    ax_a = fig.add_subplot(gs_sum[0, :])
    ax_b = fig.add_subplot(gs_sum[1, :2])
    ax_c = fig.add_subplot(gs_sum[1, 2:4])
    ax_d = fig.add_subplot(gs_sum[1, 4:])

    # Placed by label, not by running order: a role that yields fewer than
    # _EXAMPLES_PER_ROLE examples would otherwise pull the next role up into its
    # row, and the caption promising one role per row would be false.
    slots: Dict[Tuple[int, int], Dict] = {}
    if rows_of_traces:
        gs_ex = GridSpec(rows_of_traces, _EXAMPLES_PER_ROLE, figure=fig,
                         hspace=gap / (h_row - gap), wspace=.22,
                         top=1 - (m_top + h_sum + block) / height,
                         bottom=m_bot / height, **edges)
        for ex in exemplars:
            slots[(role_row[ex["label"][0]], int(ex["label"][1:]) - 1)] = ex

    _panel_box_strip(ax_a, plotted, days, genos, colours, labels)
    _panel_ecdf(ax_b, pooled, genos, colours, labels)
    _panel_quantisation(ax_c, pooled, genos, colours, labels, quantum,
                        one_grid=one_grid)
    _panel_trend(ax_d, plotted, days, genos, colours, labels)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.grid(axis="y", color=_GRID, lw=.6, alpha=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=_MUTED, labelsize=8.5)

    # Trace panels get no grid: gridlines here would be read as levels of the
    # signal, and the two lines that are levels of the signal are already drawn.
    # Axis labels go on the outer edge of the grid only — nine copies of
    # "Time in recording (s)" is furniture, not information — but every panel
    # keeps its own ticks, since the traces differ in length and in scale. The
    # x label goes to whichever panel is last in its column, which is not the
    # bottom row when a role came up short.
    for (r, c), ex in sorted(slots.items()):
        ax = fig.add_subplot(gs_ex[r, c])
        _panel_trace(ax, ex, colours, labels,
                     show_x=(r + 1, c) not in slots, show_y=(c == 0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=_MUTED, labelsize=7.5)

    cens = tally["censored"]
    censored = sum(cens.values())
    fig.suptitle("Transient duration  [dev]", fontsize=14,
                 fontweight="bold", x=.009, ha="left",
                 y=1 - .28 / height, color=_INK)
    fig.text(.009, 1 - .62 / height, "\n".join([
        f"{len(plotted)} events from {tally['recordings']} recordings; "
        f"{censored} censored and not shown \u2014 "
        f"{cens['prev_event'] + cens['next_event']} merged with a neighbouring "
        f"event, {cens['trace_start']} already above threshold at frame 0, "
        f"{cens['trace_end']} still elevated at the trace end.",
        f"Duration is the excursion above median + {k_onset:g}\u00d7 noise: "
        f"onset frame to the first frame back below it. D omits cells with "
        f"n<{_MIN_FOR_TREND}.",
        f"E, F and G are a row each \u2014 shortest, median, longest \u2014 of "
        f"three whole traces: x is seconds of recording, y is \u0394F/F\u2080. "
        f"Single events, not averages, chosen across the amplitude range of "
        f"their duration.",
        f"Shading is the measured interval \u2014 the whole excursion above "
        f"the dashed onset threshold, one frame wide at the short end. The "
        f"fainter dashed line is the trace median.",
        f"Ticks along the bottom of a trace mark every other detected peak in "
        f"it: the reason an excursion can be cut short, and the reason a long "
        f"one may be a merged pair rather than one slow transient.",
    ]), fontsize=8.5, color=_MUTED, ha="left", va="top", linespacing=1.6)

    _save(fig, path)
    logger.info(f"  wrote {path}")


def _indicator_tau(datasets: List) -> Optional[float]:
    """The indicator decay time the activity stage actually used, or None.

    Read from each recording's ``run_info.json`` rather than taken as an
    argument: the config's ``deconvolution.decay_initialisation`` is usually null and is
    resolved from ``imaging.indicator`` downstream, so the resolved value is the
    only one that describes what the detector ran with. Recordings that disagree
    return None — a single reference line drawn across a mixture of indicators
    would be wrong for some of the events under it.
    """
    seen = set()
    for ds in datasets:
        _ri = _run_info(ds)
        v = _ri.get("decay_initialisation_s", _ri.get("decay_time_s"))
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            seen.add(round(float(v), 6))
    if len(seen) == 1:
        return seen.pop()
    if len(seen) > 1:
        logger.info(f"  indicator decay time differs across recordings "
                    f"({sorted(seen)}) \u2014 no reference line drawn")
    return None


def _by_genotype(rows: List[Dict]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Measured durations per genotype, and the genotypes in a fixed order."""
    measured = [r for r in rows if r["duration_s"] is not None]
    genos = sorted({r["genotype"] for r in measured})
    return genos, {g: np.sort(np.array([r["duration_s"] for r in measured
                                        if r["genotype"] == g], dtype=float))
                   for g in genos}


def _ecdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(values, grid, side="right") / max(values.size, 1)


def _compared(genos: List[str], pooled: Dict[str, np.ndarray]) -> List[str]:
    """The two genotypes the comparison panel uses: the two with the most events.

    ``loading._extract_genotype`` returns ``Unknown`` for any recording whose
    name does not carry a genotype, so a single stray folder adds a third group
    with one recording in it. Gating the comparison on *every* group present
    would let that one recording suppress a comparison between two arms of
    twenty. Ordered so the difference reads mutant minus control.
    """
    ranked = sorted(genos, key=lambda g: (-pooled[g].size, g))
    return sorted(ranked[:2])


def _null_band(rows: List[Dict], pair: List[str], grid: np.ndarray,
               n_perm: int = _PERMUTATIONS, seed: int = 0
               ) -> Optional[np.ndarray]:
    """The ECDF difference that shuffling genotype labels between recordings can
    manufacture on its own.

    Whole recordings keep their events, their ROI structure and their size; only
    the labels move. The band is therefore what this comparison does when there
    is nothing to find, and a curve leaving it is a difference larger than
    relabelling produces.

    Resampling *within* each genotype instead — the obvious bootstrap — would
    give a percentile interval centred on the observed difference, which the
    observed difference then lies inside by construction: a band that cannot
    fail, and so cannot inform. Recordings are the unit either way, because
    events are nested in them: one recording contributes hundreds, and treating
    those as independent would call almost any difference conclusive.

    Returns ``(2, len(grid))`` percentiles, or None when either arm has too few
    recordings for a permutation to mean anything.
    """
    per_rec: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        if r["duration_s"] is not None and r["genotype"] in pair:
            per_rec.setdefault((r["genotype"], r["recording"]), []).append(
                r["duration_s"])
    pools = {g: [np.array(v, dtype=float) for (gg, _rec), v in per_rec.items()
                 if gg == g] for g in pair}
    if any(len(pools[g]) < 3 for g in pair):
        return None

    every = pools[pair[0]] + pools[pair[1]]
    n_first = len(pools[pair[0]])
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, grid.size), dtype=float)
    for p in range(n_perm):
        order = rng.permutation(len(every))
        a = np.sort(np.concatenate([every[i] for i in order[:n_first]]))
        b = np.sort(np.concatenate([every[i] for i in order[n_first:]]))
        null[p] = _ecdf(b, grid) - _ecdf(a, grid)
    return np.percentile(null, [2.5, 97.5], axis=0)


def _panel_spectrum(ax, genos, pooled, colours, labels, tau) -> None:
    """A: how often each measurable duration actually occurs.

    Duration is a count of frames, so the quantity has a grid and only the
    values on it can occur. Plotting the frequency *at* each of those values —
    rather than binning them into something wider — is the only view in which
    clustering could be seen at all: a bin two frames wide would average away
    any preference for one frame over the next, which is precisely the pattern
    being looked for.

    Both axes are log. x because the range spans two decades; y because the
    interesting structure at typical durations sits three decades above the
    tail, and on a linear axis the tail is a flat line on the floor.
    """
    for geno in genos:
        v = pooled[geno]
        if not v.size:
            continue
        vals, counts = np.unique(v, return_counts=True)
        ax.plot(vals, counts / v.size, marker="o", ms=3.2, lw=1.1,
                color=colours[geno], alpha=.9,
                label=f"{labels[geno]}  n={v.size}")
    _mark_reference(ax, tau, pooled, genos, colours)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Duration (s, log) \u2014 only frame multiples can occur",
                  fontsize=9.5)
    ax.set_ylabel("Fraction of that genotype's events", fontsize=9.5)
    ax.set_title("A   Frequency of every duration that occurs", fontsize=10.5,
                 fontweight="bold", loc="left", color=_INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left")


def _mark_reference(ax, tau, pooled, genos, colours) -> None:
    """The two durations that are set by configuration, not by the biology.

    Without them a hump at the short end reads as a population of fast
    transients, when the detector's floor is the reason nothing shorter exists
    and the indicator's tau is the scale everything is expected to sit near.
    """
    floor = min((pooled[g].min() for g in genos if pooled[g].size), default=None)
    tf = _blend(ax.transData, ax.transAxes)
    for x, colour, text in ((floor, _MUTED, f"detector floor {floor:g} s"
                             if floor is not None else ""),
                            (tau, _INK, f"indicator \u03c4 {tau:g} s" if tau else "")):
        if not x:
            continue
        ax.axvline(x, color=colour, lw=.9, ls=(0, (3, 3)), zorder=1)
        ax.text(x, .985, f"  {text}", transform=tf, rotation=90, fontsize=6.8,
                color=colour, ha="left", va="top")

    # The floor of the panel is one event, not zero: below it a value simply did
    # not occur, and the jitter along the bottom is single events, not a rate.
    # One line per genotype, because the fractions are per genotype — a single
    # line at the mean n would sit above every singleton in the larger group and
    # below every singleton in the smaller one.
    tfa = _blend(ax.transAxes, ax.transData)
    sizes = sorted({pooled[g].size for g in genos if pooled[g].size})
    for g in genos:
        if pooled[g].size:
            ax.axhline(1 / pooled[g].size, color=colours[g], lw=.8, alpha=.45,
                       zorder=1)
    if sizes:
        ax.text(.995, 1 / sizes[0], "one event ", transform=tfa, fontsize=6.8,
                color=_MUTED, ha="right", va="bottom")


def _panel_sorted(ax, genos, pooled, colours, labels) -> None:
    """B: every event, sorted — the plateaus are the clusters.

    The quantile function of the same data as A, and the form in which
    clustering is hardest to argue with: a flat run means a large share of
    events share one duration, and its width is that share. Nothing is binned,
    smoothed or fitted, and every event is on the axes.
    """
    for geno in genos:
        v = pooled[geno]
        if not v.size:
            continue
        ax.plot(np.arange(v.size) / max(v.size - 1, 1) * 100, v, lw=1.8,
                color=colours[geno], label=f"{labels[geno]}  n={v.size}")
    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile of events", fontsize=9.5)
    ax.set_ylabel("Duration (s, log)", fontsize=9.5)
    ax.set_title("B   Every event, sorted", fontsize=10.5, fontweight="bold",
                 loc="left", color=_INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def _panel_difference(ax, rows, genos, pooled, colours, labels) -> None:
    """C: whether the gap between the two curves is bigger than chance produces.

    A and B are read side by side and any separation looks like a finding. This
    panel is the control on that reading: the band is the difference that
    shuffling genotype labels between whole recordings produces on its own, so a
    curve inside it is a gap that relabelling would have given anyway.
    """
    ax.set_title("C   Difference between genotypes", fontsize=10.5,
                 fontweight="bold", loc="left", color=_INK)
    if len(genos) < 2:
        ax.text(.5, .5, "only one genotype in this dataset\nno comparison to draw",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
                color=_MUTED, linespacing=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return

    pair = _compared(genos, pooled)
    grid = np.unique(np.concatenate([pooled[g] for g in pair]))
    diff = _ecdf(pooled[pair[1]], grid) - _ecdf(pooled[pair[0]], grid)
    band = _null_band(rows, pair, grid)

    if band is not None:
        ax.fill_between(grid, band[0], band[1], color=_GRID, alpha=.9,
                        linewidth=0, zorder=1,
                        label=f"95% of {_PERMUTATIONS} label permutations")
    ax.axhline(0, color=_MUTED, lw=.8, zorder=2)
    ax.plot(grid, diff, color=colours[pair[1]], lw=1.8, zorder=3,
            label=f"{labels[pair[1]]} \u2212 {labels[pair[0]]}")
    ax.set_xscale("log")
    ax.set_xlabel("Duration (s, log)", fontsize=9.5)
    ax.set_ylabel("Difference in cumulative fraction", fontsize=9.5)

    notes = []
    if band is None:
        notes.append("too few recordings per genotype to permute")
    else:
        # The band is pointwise, so about 5% of durations leave it under the
        # null anyway, and neighbouring points on an ECDF are far from
        # independent. The expected count is stated next to the observed one so
        # a handful of excursions is not read as a result on its own.
        outside = int(np.sum((diff < band[0]) | (diff > band[1])))
        notes.append(
            f"{outside} of {grid.size} durations leave the band "
            f"(\u2248{.05 * grid.size:.0f} expected under the null: the band "
            f"is pointwise)")
    others = [g for g in genos if g not in pair]
    if others:
        notes.append("not compared: " + ", ".join(
            f"{labels.get(g, g)} (n={pooled[g].size})" for g in others))
    ax.text(.98, .04, "\n".join(notes), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7.5, color=_MUTED, linespacing=1.4)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def _panel_amplitude(ax, rows, genos, colours, labels) -> None:
    """D: whether a duration cluster is really an amplitude cluster.

    The measurement is a threshold crossing, so it grows with amplitude at
    fixed kinetics: a taller transient crosses the threshold earlier on the way
    up and later on the way down. Any horizontal band in A or B therefore has a
    rival explanation — a group of similarly sized events — and this is where
    the two are told apart. Points are one event each; the heavy line is the
    median duration per amplitude decile, which is the trend the cloud hides.
    """
    for geno in genos:
        pts = [(r["amplitude_dff"], r["duration_s"]) for r in rows
               if r["genotype"] == geno and r["duration_s"] is not None
               and r["amplitude_dff"] > 0]
        if not pts:
            continue
        a = np.array([p[0] for p in pts], dtype=float)
        d = np.array([p[1] for p in pts], dtype=float)
        ax.scatter(a, d, s=3.5, alpha=.13, linewidths=0, color=colours[geno],
                   zorder=2)
        # Unique edges, then one bin per event: amplitude ties can collapse two
        # deciles onto the same edge, and bins sharing a boundary would count
        # the events on it twice.
        edges = np.unique(np.percentile(a, np.arange(0, 101, 10)))
        which = np.clip(np.searchsorted(edges, a, side="right") - 1,
                        0, max(edges.size - 2, 0))
        mids, meds = [], []
        for b in range(max(edges.size - 1, 0)):
            sel = d[which == b]
            if sel.size >= _MIN_FOR_TREND:
                mids.append(float(np.sqrt(edges[b] * edges[b + 1])))
                meds.append(float(np.median(sel)))
        if mids:
            ax.plot(mids, meds, color=colours[geno], lw=2.0, zorder=4,
                    marker="o", ms=4, markeredgecolor="white", markeredgewidth=.8,
                    label=f"{labels[geno]} median per decile")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Event amplitude (\u0394F/F\u2080, log)", fontsize=9.5)
    ax.set_ylabel("Duration (s, log)", fontsize=9.5)
    ax.set_title("D   Duration against amplitude", fontsize=10.5,
                 fontweight="bold", loc="left", color=_INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def _figure_clustering(rows: List[Dict], path: str, *,
                       mutant_label: str = "Mutant", k_onset: float = 3.0,
                       decay_time: Optional[float] = None) -> Dict:
    """The pooled duration distribution on its own, at frame resolution.

    The overview figure answers "how long are transients, and does that change
    with age"; this one answers "does duration take preferred values", which
    needs the distribution drawn at the resolution the measurement actually has
    and nothing else competing for the space.

    Returns a small summary of what the panels show, or ``{}`` if there was
    nothing to draw.
    """
    genos, pooled = _by_genotype(rows)
    if not genos or all(not pooled[g].size for g in genos):
        logger.warning("  no measured events — clustering figure skipped")
        return {}

    colours = {g: GENOTYPE_COLOURS.get(g, GENOTYPE_COLOURS["Unknown"])
               for g in genos}
    labels = {g: (mutant_label if g == "Mutant" else g) for g in genos}

    m_top, h_row, gap, m_bot = 1.78, 3.7, 1.0, .62
    height = m_top + 2 * h_row + gap + m_bot
    fig = plt.figure(figsize=(13.5, height))
    gs = GridSpec(2, 2, figure=fig, hspace=gap / h_row, wspace=.24,
                  top=1 - m_top / height, bottom=m_bot / height,
                  left=.068, right=.98)
    axes = [fig.add_subplot(gs[k // 2, k % 2]) for k in range(4)]

    _panel_spectrum(axes[0], genos, pooled, colours, labels, decay_time)
    _panel_sorted(axes[1], genos, pooled, colours, labels)
    _panel_difference(axes[2], rows, genos, pooled, colours, labels)
    _panel_amplitude(axes[3], rows, genos, colours, labels)

    for ax in axes:
        ax.grid(axis="y", color=_GRID, lw=.6, alpha=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=_MUTED, labelsize=8.5)

    n_vals = len(np.unique(np.concatenate([pooled[g] for g in genos
                                           if pooled[g].size])))
    fig.suptitle("Transient duration \u2014 does it cluster?  [dev]",
                 fontsize=14, fontweight="bold", x=.009, ha="left",
                 y=1 - .28 / height, color=_INK)
    fig.text(.009, 1 - .62 / height, "\n".join([
        f"Every measured event, pooled over age and recording, by genotype. "
        f"Duration is the excursion above median + {k_onset:g}\u00d7 noise, so it "
        f"is a count of frames:",
        f"{n_vals} distinct values exist in the whole dataset and nothing "
        f"between them can occur. Censored events are excluded \u2014 an event "
        f"of unknown extent cannot be clustered.",
        f"A cluster sitting on the detector floor, or one frame wide anywhere, "
        f"is a property of the measurement rather than of the transients.",
    ]), fontsize=8.5, color=_MUTED, ha="left", va="top", linespacing=1.6)

    _save(fig, path)
    logger.info(f"  wrote {path}")

    summary = {"n_distinct_durations": int(n_vals)}
    for g in genos:
        v = pooled[g]
        if v.size:
            vals, counts = np.unique(v, return_counts=True)
            summary[f"mode_s_{g.lower()}"] = float(vals[int(np.argmax(counts))])
            summary[f"mode_share_{g.lower()}"] = float(counts.max() / v.size)
            summary[f"median_s_{g.lower()}"] = float(np.median(v))
    return summary


def run_transient_decay(datasets: List, output_dir: str, *,
                        deconv_method: str = "oasis",
                        k_onset: float = 3.0,
                        frame_rate: float = 2.0,
                        mutant_label: str = "Mutant") -> Dict:
    """Entry point. Returns a summary dict, or ``{'skipped': reason}``.

    The summary names the events drawn as example traces, so a reader of
    ``run_info.json`` can go straight to those recordings without opening the
    figure to find out which they were, and carries the modal duration per
    genotype from the clustering figure.

    Refuses to run on anything but the robust detector — see the module
    docstring for why the same measurement on an OASIS reconstruction would
    report the fit rather than the data.
    """
    if str(deconv_method).lower() != "robust":
        msg = (f"transient duration [dev] needs deconvolution.method='robust', "
               f"got '{deconv_method}' — skipped")
        logger.info(f"  {msg}")
        return {"skipped": msg}

    fig_dir = os.path.join(output_dir, "figures", "dev - Transient Duration")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    rows, tally, rates = _collect(datasets, frame_rate, k_onset)
    if not rows:
        # Name the cause. The counts distinguish "the traces never arrived"
        # from "the ids do not carry a day" from "the detector found nothing",
        # which a single combined message cannot.
        msg = (f"no measurable events from {len(datasets)} dataset(s): "
               f"{tally['recordings']} measured, "
               f"{tally['no_traces']} with no loadable traces, "
               f"{tally['no_day']} with no parsable day")
        logger.warning(f"  transient duration [dev]: {msg}")
        return {"skipped": msg}

    # Outliers first: the exemplar panels report whether the event they show was
    # flagged, which is only known once the fences have been drawn.
    _mark_outliers(rows)
    _write_csv(rows, os.path.join(data_dir, "transient_duration_events.csv"))

    picks = _pick_exemplars(rows)
    exemplars = _exemplar_traces(picks, datasets, rates, k_onset, frame_rate)
    for ex in exemplars:
        if ex["trace"] is None:
            logger.warning(f"  example {ex['label']} ({ex['role'].lower()}) "
                           f"unavailable: {ex['reason']}")

    _figure(rows, tally, os.path.join(fig_dir, "transient_duration_overview.png"),
            exemplars=exemplars, mutant_label=mutant_label, k_onset=k_onset,
            rates=rates)
    clustering = _figure_clustering(
        rows, os.path.join(fig_dir, "transient_duration_clustering.png"),
        mutant_label=mutant_label, k_onset=k_onset,
        decay_time=_indicator_tau(datasets))

    good = [r["duration_s"] for r in rows if r["duration_s"] is not None]
    return {
        "n_recordings": tally["recordings"],
        "n_events": tally["events"],
        "n_measured": len(good),
        "n_censored": {k: v for k, v in tally["censored"].items()},
        "n_outliers": sum(1 for r in rows if r["outlier"]),
        "median_duration_s": float(np.median(good)) if good else None,
        "clustering": clustering,
        "k_onset": float(k_onset),
        "definition": ("onset to first return below median(trace) + k_onset * "
                       "MAD noise; the extent of one excursion above it"),
        "exemplars": [{
            "panel": ex["label"],
            "role": ex["role"].lower(),
            "recording": ex["row"]["recording"],
            "neuron": ex["row"]["neuron"],
            "peak_time_s": ex["row"]["peak_time_s"],
            "duration_s": ex["row"]["duration_s"],
            "amplitude_dff": ex["row"]["amplitude_dff"],
            "outlier": ex["row"]["outlier"],
            "plotted": ex["trace"] is not None,
        } for ex in exemplars],
    }
