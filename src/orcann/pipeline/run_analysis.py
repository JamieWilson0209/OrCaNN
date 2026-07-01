"""Analysis stage: cross-recording group analysis from the transient outputs.

Reads results/transients/<rec>/data/{events.npz, meta.json} for every recording,
extracts per-ROI event frequency and per-event timescale (the geometric event
duration), summarises each recording, and compares across genotype and across
developmental day. Unlike the per-recording stages this is a single aggregate run
over all transient outputs -> results/analysis/.

Metrics per recording:
  - active_fraction    : fraction of ROIs with >= 1 detected transient
  - frequency          : events per minute, per ROI (distribution + median over active ROIs)
  - timescale          : event duration in seconds, per event (distribution + median)

Recording metadata (developmental day, genotype) is parsed from the recording_id
by the configurable regexes in the analysis config; the parsed values are written
to recording_metrics.csv so they can be checked and the regexes corrected. The
defaults match ids like D130_1-18_070226_R6: day from D<n>, line token the field
after it, genotype Control iff that token starts with analysis.control_prefix.

Outputs (results/analysis/):
  recording_metrics.csv                  one row per recording (parsed meta + summary)
  within_recording_distributions.png     per-recording frequency + timescale distributions
  genotype_comparison.png                summary metrics Control vs Mutant (+ Mann-Whitney)
  longitudinal_by_day.png                summary metrics vs developmental day, by genotype
  summary.json                           aggregate stats by genotype and by day
"""
import csv
import json
import os
import re
from collections import Counter

import numpy as np

GENO_COLORS = {"Control": "#2980b9", "Mutant": "#c0392b", "unknown": "#7f8c8d"}
_METRICS = [("active_fraction", "active fraction"),
            ("median_freq_per_min", "median freq (events/min)"),
            ("median_timescale_s", "median timescale (s)")]
_CSV_FIELDS = ["recording_id", "day", "line", "genotype", "n_roi", "n_events",
               "duration_min", "active_fraction", "median_freq_per_min",
               "mean_freq_per_min", "median_timescale_s"]


def _parse_meta(rec_id, ap):
    """recording_id -> (day:int|None, line:str|None, genotype)."""
    day = None
    m = re.search(ap.day_regex, rec_id)
    if m:
        try:
            day = int(m.group(1))
        except (ValueError, IndexError):
            day = None
    line, geno = None, "unknown"
    m = re.search(ap.line_regex, rec_id)
    if m:
        line = m.group(1)
        geno = "Control" if line.startswith(ap.control_prefix) else "Mutant"
    return day, line, geno


def _list_recordings(tdir):
    if not os.path.isdir(tdir):
        return []
    return sorted(d for d in os.listdir(tdir)
                  if os.path.isfile(os.path.join(tdir, d, "data", "events.npz")))


def _recording_metrics(tdir, rec_id, ap):
    ddir = os.path.join(tdir, rec_id, "data")
    with open(os.path.join(ddir, "meta.json")) as fh:
        meta = json.load(fh)
    ev = np.load(os.path.join(ddir, "events.npz"))
    roi = np.asarray(ev["roi"]).ravel()
    dur = np.asarray(ev["duration_s"], float).ravel()

    n_roi = int(meta.get("n_roi", 0))
    fr = float(meta.get("frame_rate", 0) or 0)
    nfr = int(meta.get("n_frames", 0))
    duration_min = (nfr / fr / 60.0) if fr > 0 else 0.0

    counts = (np.bincount(roi, minlength=n_roi).astype(float)
              if roi.size and n_roi else np.zeros(max(n_roi, 0)))
    freq = counts / duration_min if duration_min > 0 else np.zeros_like(counts)
    active = counts > 0

    day, line, geno = _parse_meta(rec_id, ap)
    return {
        "recording_id": rec_id, "day": day, "line": line, "genotype": geno,
        "n_roi": n_roi, "n_events": int(roi.size),
        "duration_min": round(duration_min, 3),
        "active_fraction": round(float(active.mean()) if n_roi else 0.0, 4),
        "median_freq_per_min": round(float(np.median(freq[active])) if active.any() else 0.0, 4),
        "mean_freq_per_min": round(float(freq.mean()) if n_roi else 0.0, 4),
        "median_timescale_s": round(float(np.median(dur)), 4) if dur.size else None,
        # arrays kept in-memory for the distribution figures (not written to CSV)
        "_freq_active": freq[active], "_dur": dur,
    }


def run(cfg, force=False):
    tdir, out = cfg.paths.transients, cfg.paths.analysis
    ap = cfg.analysis
    recs = _list_recordings(tdir)
    if not recs:
        print(f"analysis: no recordings with events.npz in {tdir} "
              f"(run detect_transients first)")
        return
    os.makedirs(out, exist_ok=True)

    M = [_recording_metrics(tdir, r, ap) for r in recs]
    gc = Counter(m["genotype"] for m in M)
    days = sorted({m["day"] for m in M if m["day"] is not None})
    print(f"analysis: {len(M)} recordings | genotype {dict(gc)} | days {days}")
    if "unknown" in gc:
        print(f"  note: {gc['unknown']} recording(s) had an unparseable genotype; "
              f"check line_regex/control_prefix against recording_metrics.csv")

    _write_metrics_csv(os.path.join(out, "recording_metrics.csv"), M)
    _fig_within(os.path.join(out, "within_recording_distributions.png"), M)
    _fig_genotype(os.path.join(out, "genotype_comparison.png"), M)
    _fig_longitudinal(os.path.join(out, "longitudinal_by_day.png"), M)
    _write_summary(os.path.join(out, "summary.json"), M)
    print(f"analysis outputs -> {out}/")


def _write_metrics_csv(path, M):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for m in M:
            w.writerow({k: ("" if m.get(k) is None else m.get(k)) for k in _CSV_FIELDS})


def _fig_within(path, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    freq_all = np.concatenate([m["_freq_active"] for m in M if m["_freq_active"].size]) \
        if any(m["_freq_active"].size for m in M) else np.array([1.0])
    dur_all = np.concatenate([m["_dur"] for m in M if m["_dur"].size]) \
        if any(m["_dur"].size for m in M) else np.array([1.0])
    fbins = np.linspace(0, max(np.percentile(freq_all, 99), 1e-6), 30)
    dbins = np.linspace(0, max(np.percentile(dur_all, 99), 1e-6), 30)

    for m in M:
        col = GENO_COLORS.get(m["genotype"], "#777")
        f = m["_freq_active"]
        if f.size >= 3:
            h, e = np.histogram(f, bins=fbins, density=True)
            a1.plot((e[:-1] + e[1:]) / 2, h, color=col, alpha=0.30, lw=1.0)
        d = m["_dur"]
        if d.size >= 3:
            h, e = np.histogram(d, bins=dbins, density=True)
            a2.plot((e[:-1] + e[1:]) / 2, h, color=col, alpha=0.30, lw=1.0)

    a1.set_xlabel("event frequency (events/min, active ROIs)"); a1.set_ylabel("density")
    a1.set_title("within-recording frequency distributions")
    a2.set_xlabel("event timescale (duration, s)"); a2.set_ylabel("density")
    a2.set_title("within-recording timescale distributions")
    handles = [mlines.Line2D([], [], color=c, label=g) for g, c in GENO_COLORS.items()]
    a1.legend(handles=handles, fontsize=8, title="one line = one recording")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_genotype(path, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    rng = np.random.default_rng(0)
    genos = ["Control", "Mutant"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (key, label) in zip(axes, _METRICS):
        data = {g: [m[key] for m in M if m["genotype"] == g and m[key] is not None]
                for g in genos}
        for i, g in enumerate(genos):
            v = np.asarray(data[g], float)
            if v.size:
                ax.scatter(i + rng.uniform(-0.08, 0.08, v.size), v,
                           color=GENO_COLORS[g], alpha=0.7, s=22, edgecolors="none")
                ax.hlines(np.median(v), i - 0.22, i + 0.22, color="k", lw=2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(genos); ax.set_ylabel(label)
        nC, nM = len(data["Control"]), len(data["Mutant"])
        if nC >= 3 and nM >= 3:
            try:
                _, p = stats.mannwhitneyu(data["Control"], data["Mutant"],
                                          alternative="two-sided")
                ax.set_title(f"{label}\nMann-Whitney p={p:.3g}  (nC={nC}, nM={nM})",
                             fontsize=9)
            except ValueError:
                ax.set_title(f"{label}  (nC={nC}, nM={nM})", fontsize=9)
        else:
            ax.set_title(f"{label}\n(n too small to test: nC={nC}, nM={nM})", fontsize=9)
    fig.suptitle("genotype comparison  (one point = one recording; bar = median)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_longitudinal(path, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    genos = ["Control", "Mutant"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (key, label) in zip(axes, _METRICS):
        for g in genos:
            pts = [(m["day"], m[key]) for m in M
                   if m["genotype"] == g and m["day"] is not None and m[key] is not None]
            if not pts:
                continue
            days = sorted({d for d, _ in pts})
            mean = [np.mean([v for d, v in pts if d == dd]) for dd in days]
            sem = [np.std([v for d, v in pts if d == dd]) /
                   max(np.sqrt(sum(1 for d, _ in pts if d == dd)), 1.0) for dd in days]
            ax.scatter([d for d, _ in pts], [v for _, v in pts],
                       color=GENO_COLORS[g], alpha=0.30, s=16, edgecolors="none")
            ax.errorbar(days, mean, yerr=sem, color=GENO_COLORS[g], marker="o",
                        lw=2, capsize=3, label=g)
        ax.set_xlabel("developmental day"); ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle("longitudinal across developmental day  (mean +/- sem per day)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _agg(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": round(float(v.mean()), 4),
            "median": round(float(np.median(v)), 4), "sd": round(float(v.std()), 4)}


def _write_summary(path, M):
    from scipy import stats
    genos = ["Control", "Mutant"]
    summary = {"n_recordings": len(M),
               "by_genotype": {}, "genotype_tests": {}, "by_day": {}}
    for g in genos:
        sub = [m for m in M if m["genotype"] == g]
        summary["by_genotype"][g] = {"n_recordings": len(sub),
                                     **{k: _agg([m[k] for m in sub]) for k, _ in _METRICS}}
    for key, _ in _METRICS:
        c = [m[key] for m in M if m["genotype"] == "Control" and m[key] is not None]
        u = [m[key] for m in M if m["genotype"] == "Mutant" and m[key] is not None]
        rec = {"nC": len(c), "nM": len(u)}
        if len(c) >= 3 and len(u) >= 3:
            try:
                _, p = stats.mannwhitneyu(c, u, alternative="two-sided")
                rec["p_mannwhitney"] = round(float(p), 6)
            except ValueError:
                rec["p_mannwhitney"] = None
        summary["genotype_tests"][key] = rec
    for day in sorted({m["day"] for m in M if m["day"] is not None}):
        summary["by_day"][str(day)] = {
            g: {k: _agg([m[k] for m in M
                         if m["day"] == day and m["genotype"] == g]) for k, _ in _METRICS}
            for g in genos}
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
