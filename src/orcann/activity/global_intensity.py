"""Global intensity diagnostic: a per-recording QC on the movie the activity
stage actually consumes.

Widefield 1-photon Fluo-4 recordings bleach steadily over their length, which
the per-ROI rolling baseline absorbs without trouble. What the rolling baseline
does not handle cleanly is a sharp, whole-field intensity step (a dropped or
duplicated frame, a light-source flicker, a shutter or focus glitch). Such a
step lands on every ROI at the same frame and reads as a coincident transient,
which is dangerous here specifically because pairwise correlation and network
synchrony are the experimental readouts: a global step manufactures exactly the
co-activation the analysis is trying to measure.

This module does not correct anything. It measures the global mean-F trace,
flags the largest sharp step, and writes a figure plus a small metrics block so
the step can be found and its severity judged across the whole set from
run_info.json alone, without opening every figure.

Note on placement: the activity stage receives the motion-corrected movie, so
this runs on that movie rather than the raw ND2. That is the correct object to
QC, since it is what baseline correction and deconvolution downstream actually
see, and motion correction does not remove a global luminance step in any case.
"""
import logging
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def global_intensity_diagnostic(movie, out_dir, rec_id="", *, guard=2,
                                mad_k=6.0, level_win=25):
    """Measure the global mean-F trace, flag the largest sharp step, plot it.

    Parameters
    ----------
    movie : ndarray (T, H, W)
        The motion-corrected movie the activity stage has loaded.
    out_dir : str
        Per-recording output folder; the figure is written here as
        global_intensity.png.
    rec_id : str
        Recording id, used only for the figure title.
    guard : int
        Frames either side of the flagged step excluded when measuring the
        persistent level shift, so the single-frame excursion does not
        contaminate the before / after windows.
    mad_k : float
        Robust threshold in MAD units for flagging a frame-to-frame step. MAD
        rather than standard deviation so the outlier being hunted does not
        inflate its own threshold (same reasoning as the robust event detector
        in deconvolution).
    level_win : int
        Window length in frames used to measure baseline level on each side of
        the step, and to summarise start / end level for the bleach estimate.

    Returns
    -------
    dict
        Metrics block for run_info.json. Keys: n_frames, mean_F_start,
        mean_F_end, bleach_fraction, n_global_steps, step_frame, dip_magnitude,
        level_shift. step_frame is None when nothing crosses the threshold.
    """
    m = np.asarray(movie.mean(axis=(-2, -1)), dtype=float)
    n = m.size

    d = np.diff(m)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) * 1.4826 + 1e-12
    z = np.abs(d - med) / mad
    flagged = np.where(z > mad_k)[0] + 1          # frame indices of steps

    metrics = {
        "n_frames": int(n),
        "mean_F_start": float(np.median(m[:level_win])),
        "mean_F_end": float(np.median(m[-level_win:])),
        "bleach_fraction": float(1.0 - np.median(m[-level_win:]) / np.median(m[:level_win])),
        "n_global_steps": int(flagged.size),
        "step_frame": None,
        "dip_magnitude": 0.0,
        "level_shift": 0.0,
    }

    break_idx = None
    if flagged.size:
        # characterise the single largest excursion
        break_idx = int(flagged[np.argmax(z[flagged - 1])])
        metrics["step_frame"] = break_idx
        if 0 < break_idx < n - 1:
            neigh = 0.5 * (m[break_idx - 1] + m[break_idx + 1])
        else:
            neigh = m[break_idx]
        metrics["dip_magnitude"] = float(m[break_idx] - neigh)

        # persistent level shift: median level after minus before, skipping a
        # guard band so the single-frame dip is excluded from both windows
        a = m[max(0, break_idx - guard - level_win): max(0, break_idx - guard)]
        b = m[min(n, break_idx + guard): min(n, break_idx + guard + level_win)]
        if a.size and b.size:
            metrics["level_shift"] = float(np.median(b) - np.median(a))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(m, lw=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel("mean F")
    ax.set_title("global intensity" if not rec_id else f"global intensity: {rec_id}")

    if break_idx is not None:
        ax.axvline(break_idx, color="crimson", lw=0.8, alpha=0.7)
        ax.annotate(
            f"step at frame {break_idx}\n"
            f"dip {metrics['dip_magnitude']:.1f}, level shift {metrics['level_shift']:.1f}",
            xy=(break_idx, m[break_idx]),
            xytext=(0.55, 0.15), textcoords="axes fraction",
            fontsize=8, color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=0.6))

    fig.tight_layout()
    path = os.path.join(out_dir, "global_intensity.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  global_intensity.png written")
    return metrics
