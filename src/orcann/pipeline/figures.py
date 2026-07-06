"""Spatial QC figures for the detection stages.

Plotting-only, no torch and no temporal detector: the infer stage's
probability-map overlay (`prob_overlay_figure`) and a max-projection detection
overlay (`max_projection_figure`, used by `scripts/check_annotation.py`). The
per-ROI trace view now lives in the activity stage's interactive HTML gallery,
so the old wavelet/rate panels are gone from here.
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# PROBABILITY-MAP OVERLAY  (infer QC: gamma-stretched prob over max projection)

def prob_overlay_figure(path, prob, max_proj, *, gamma=0.45, proj_alpha=0.4,
                        title=""):
    """Gamma-stretched soma-probability map (magma) over a translucent max proj.

    The infer stage's QC image. The probability is gamma-stretched (default 0.45,
    i.e. ``prob ** gamma``) to lift dim edge/soma structure a linear map would
    hide, drawn in magma with *per-pixel* alpha equal to that stretched value: so
    confident somata are opaque colour and low-probability background is
    transparent, letting the faint max projection (grey, alpha ``proj_alpha``)
    show through for anatomical context. On a black ground so magma reads true.
    ``prob`` and ``max_proj`` are both (H, W) at the model's working resolution.
    """
    H, W = prob.shape
    gp = np.clip(prob.astype(np.float32), 0.0, 1.0) ** float(gamma)
    lo, hi = np.percentile(max_proj, (1.0, 99.5))
    hi = hi if hi > lo else lo + 1e-6

    fig, ax = plt.subplots(figsize=(8, 8 * H / max(W, 1)))
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.imshow(max_proj, cmap="gray", vmin=lo, vmax=hi, alpha=proj_alpha,
              interpolation="nearest")
    ax.imshow(gp, cmap="magma", vmin=0.0, vmax=1.0, alpha=gp, interpolation="nearest")
    ax.set_title(title or "soma probability (gamma-stretched) over max projection",
                 fontsize=10, color="white")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)


# MAX-PROJECTION OVERLAY  (detections on the summary image)

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
