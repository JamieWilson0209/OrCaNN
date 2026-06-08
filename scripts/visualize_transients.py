#!/usr/bin/env python
"""Visual verification of the temporal stage on real ROI fluorescence traces.

INPUT IS A FLUORESCENCE TRACE, NOT A SPIKE TRAIN. The model finds calcium
transients in the continuous ΔF/F signal; feeding it deconvolved spikes is
meaningless. Use the per-ROI trace file (e.g. temporal_traces.npy), shape
(n_roi, T) or (T,). standardize_trace handles raw F or ΔF/F either way.

For each ROI it renders three stacked panels:
  1. standardized trace, with detected transients marked and their geometric
     duration drawn as a horizontal span;
  2. the wavelet scalogram |W(τ, t)| — the ∇²G response across timescales,
     i.e. "a transient of what duration, when";
  3. the predicted per-bin event rate.

    python scripts/visualize_transients.py --traces temporal_traces.npy \
        --model models/temporal/rate_model.pt --frame-rate 2.0 --max-rois 6 \
        --out results/inference/transients.png
"""
import argparse, os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from orcann.temporal_dog import standardize_trace
from orcann.figures import draw_roi_panels


def load_traces(path: str) -> np.ndarray:
    a = np.load(path)
    if a.ndim == 1:
        a = a[None]
    if a.ndim != 2:
        raise ValueError(f"expected (n_roi, T) or (T,), got {a.shape}")
    # orient as (n_roi, T): time should be the long axis
    if a.shape[0] > a.shape[1]:
        a = a.T
    return a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frame-rate", type=float, default=2.0)
    ap.add_argument("--rois", default="auto", help="comma-separated indices, or 'auto'")
    ap.add_argument("--max-rois", type=int, default=6)
    ap.add_argument("--min-prominence", type=float, default=0.5,
                    help="MAIN knob (s_min analogue): a transient's rate peak must "
                         "rise this far (rate units) above its local surroundings. "
                         "Raise to mark fewer/cleaner events; tune from the figure.")
    ap.add_argument("--floor-pct", type=float, default=25.0,
                    help="height floor = this percentile of the ROI's rate (gates "
                         "out quiet periods / noise baseline)")
    ap.add_argument("--min-isi-s", type=float, default=1.0,
                    help="minimum separation between marked transients (s)")
    ap.add_argument("--out", default="transients.png")
    a = ap.parse_args()

    traces = load_traces(a.traces)
    model = torch.load(a.model, weights_only=False, map_location="cpu").eval()
    fs = a.frame_rate

    if a.rois == "auto":
        # pick the most active ROIs (largest standardized variance) for a useful view
        score = [np.var(standardize_trace(traces[i], fs)) for i in range(len(traces))]
        rois = list(np.argsort(score)[::-1][:a.max_rois])
    else:
        rois = [int(x) for x in a.rois.split(",")][:a.max_rois]

    nr = len(rois)
    fig, axes = plt.subplots(nr * 3, 1, figsize=(12, 2.4 * nr * 3),
                             gridspec_kw={"height_ratios": [2, 2, 1] * nr})
    if nr == 1:
        axes = np.array(axes)

    for r, roi in enumerate(rois):
        draw_roi_panels(axes[3 * r], axes[3 * r + 1], axes[3 * r + 2],
                        model, traces[roi], fs, roi_label=f"ROI {roi}",
                        min_prominence=a.min_prominence, floor_pct=a.floor_pct,
                        min_isi_s=a.min_isi_s, show_xlabel=(r == nr - 1))

    fig.suptitle("Temporal stage — transient detection on real ROI traces",
                 fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.997])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"{len(rois)} ROIs -> {a.out}")
    print("durations are inflated multiples of the decay constant (ordering "
          "faithful, absolute seconds not).")


if __name__ == "__main__":
    main()
