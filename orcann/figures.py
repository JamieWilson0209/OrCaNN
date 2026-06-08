"""Figure rendering for the spatial+temporal pipeline.

Single source of truth for the per-ROI transient panels (so the batch job and
the standalone visualizer render identically) plus the max-projection overlay.
All functions are plotting-only; detection logic lives in temporal_dog.
"""
from __future__ import annotations

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orcann.temporal_dog import detect_transients


# =============================================================================
# PER-ROI TRANSIENT PANELS  (trace + scalogram + rate)
# =============================================================================

def draw_roi_panels(ax_tr, ax_sc, ax_rt, model, trace, frame_rate,
                    roi_label="", min_prominence=0.5, floor_pct=25.0,
                    min_isi_s=1.0, show_xlabel=True):
    """Draw the 3-panel view for one ROI onto the supplied axes.

    Top: standardized trace with detected transients (red) and their geometric
    durations (blue spans). Middle: wavelet scalogram |W(τ,t)|. Bottom: the
    per-bin rate with the detection floor.
    """
    det = detect_transients(model, trace, frame_rate, min_prominence=min_prominence,
                            floor_pct=floor_pct, min_isi_s=min_isi_s)
    x, rate = det["standardized"], det["rate"]
    peaks, durs, floor = det["peaks"], det["durations_s"], det["floor"]
    device = next(model.parameters()).device
    with torch.no_grad():
        W = model.response(torch.from_numpy(x)[None].to(device))[0].abs().cpu().numpy()
    ts = model.bank.timescales_s.detach().cpu().numpy()
    t = np.arange(len(x)) / frame_rate

    ax_tr.plot(t, x, lw=0.8, color="#222")
    for p, d in zip(peaks, durs):
        tp = p / frame_rate
        ax_tr.axvline(tp, color="#c0392b", lw=0.8, alpha=0.7)
        ax_tr.hlines(x.max() * 1.05, tp - d / 2, tp + d / 2,
                     color="#2980b9", lw=2.5, alpha=0.8)
    ax_tr.set_ylabel(f"{roi_label}\nΔF/F (SNR)" if roi_label else "ΔF/F (SNR)")
    ax_tr.set_title(f"{roi_label}: {len(peaks)} transients detected "
                    f"(blue = geometric duration)", fontsize=9, loc="left")
    ax_tr.set_xlim(t[0], t[-1])

    ax_sc.imshow(W, aspect="auto", extent=[t[0], t[-1], ts[-1], ts[0]],
                 cmap="magma", interpolation="nearest")
    ax_sc.set_yscale("log")
    ax_sc.set_ylabel("timescale τ (s)")
    ax_sc.set_xlim(t[0], t[-1])

    ax_rt.plot(t, rate, lw=0.9, color="#27ae60")
    ax_rt.fill_between(t, 0, rate, color="#27ae60", alpha=0.25)
    ax_rt.axhline(floor, color="#7f8c8d", lw=0.6, ls="--")
    ax_rt.set_ylabel("rate (a.u.)")
    ax_rt.set_xlim(t[0], t[-1])
    if show_xlabel:
        ax_rt.set_xlabel("time (s)")
    return len(peaks)


def roi_figure(path, model, trace, frame_rate, roi_label="", **det_kw):
    """Save a single-ROI 3-panel figure to ``path``."""
    fig, (ax_tr, ax_sc, ax_rt) = plt.subplots(
        3, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [2, 2, 1]})
    draw_roi_panels(ax_tr, ax_sc, ax_rt, model, trace, frame_rate,
                    roi_label=roi_label, **det_kw)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAX-PROJECTION OVERLAY  (detections on the summary image)
# =============================================================================

def max_projection_figure(path, max_proj, centroids, footprints=None,
                          contour_level=0.3):
    """Max-projection grayscale with detected ROIs overlaid.

    Footprint outlines (if given) are drawn as contours; centroids are marked
    and numbered. ``max_proj`` is (H, W); ``centroids`` is (N, 2) as (row, col).
    """
    H, W = max_proj.shape
    fig, ax = plt.subplots(figsize=(8, 8 * H / max(W, 1)))
    vmax = np.percentile(max_proj, 99.5)
    ax.imshow(max_proj, cmap="gray", vmin=float(max_proj.min()), vmax=float(vmax))

    if footprints is not None and len(footprints):
        for fp in footprints:
            if fp.max() <= 0:
                continue
            ax.contour(fp, levels=[contour_level * fp.max()],
                       colors="#f1c40f", linewidths=0.8, alpha=0.9)
    if len(centroids):
        cents = np.asarray(centroids)
        ax.scatter(cents[:, 1], cents[:, 0], s=14, facecolors="none",
                   edgecolors="#e74c3c", linewidths=1.0)
        for i, (r, c) in enumerate(cents):
            ax.text(c + 1.5, r - 1.5, str(i), color="#e74c3c", fontsize=6)

    ax.set_title(f"max projection + {len(centroids)} detected ROIs", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
