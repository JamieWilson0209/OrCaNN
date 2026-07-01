#!/usr/bin/env python
"""Per-ROI inspection figure for the temporal stage, driven by a YAML config.

INPUT IS A FLUORESCENCE TRACE, NOT A SPIKE TRAIN. For each selected ROI it renders
the standardized trace with detected transients, the wavelet scalogram, and the
predicted rate. Set the `viz` section (traces, model, out, rois, max_rois) and the
`temporal` section in the config, then:

    python scripts/visualize_transients.py --config config.yaml

See docs/configuration.md.
"""
import argparse, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orcann.configLoader import Config                                       # noqa: E402
from orcann.pipeline.model_io import load_model                        # noqa: E402
from orcann.temporal import standardize_trace                          # noqa: E402
from orcann.pipeline.figures import draw_roi_panels                    # noqa: E402


def load_traces(path: str) -> np.ndarray:
    a = np.load(path)
    if a.ndim == 1:
        a = a[None]
    if a.ndim != 2:
        raise ValueError(f"expected (n_roi, T) or (T,), got {a.shape}")
    if a.shape[0] > a.shape[1]:
        a = a.T
    return a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Temporal-stage inspection figure (YAML config).")
    ap.add_argument("--config", default="config.yaml",
                    help="YAML config file (default: ./config.yaml)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="section.key=value", help="one-off config override (repeatable)")
    ap.add_argument("--dump-config", metavar="PATH",
                    help="write a commented config to PATH and exit")
    a = ap.parse_args()

    if a.dump_config:
        Config().apply_overrides(a.overrides).dump(a.dump_config)
        print(f"wrote config -> {a.dump_config}"); return
    if not os.path.exists(a.config):
        ap.error(f"config not found: {a.config} (create one with --dump-config {a.config})")

    cfg = Config.load(a.config).apply_overrides(a.overrides).resolve_paths()
    io, tp = cfg.viz, cfg.temporal
    for k in ("traces", "model"):
        if getattr(io, k) is None:
            ap.error(f"set viz.{k} in {a.config}")

    fs, det = tp.frame_rate, tp.detection()
    traces = load_traces(io.traces)
    model = load_model(io.model, map_location="cpu")

    if io.rois == "auto":
        score = [np.var(standardize_trace(traces[i], fs)) for i in range(len(traces))]
        rois = list(np.argsort(score)[::-1][:io.max_rois])
    else:
        rois = [int(x) for x in io.rois.split(",")][:io.max_rois]

    nr = len(rois)
    fig, axes = plt.subplots(nr * 3, 1, figsize=(12, 2.4 * nr * 3),
                             gridspec_kw={"height_ratios": [2, 2, 1] * nr})
    if nr == 1:
        axes = np.array(axes)
    for r, roi in enumerate(rois):
        draw_roi_panels(axes[3 * r], axes[3 * r + 1], axes[3 * r + 2], model,
                        traces[roi], fs, roi_label=f"ROI {roi}",
                        show_xlabel=(r == nr - 1), **det)
    fig.suptitle("Temporal stage - transient detection on real ROI traces",
                 fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.997])
    os.makedirs(os.path.dirname(io.out) or ".", exist_ok=True)
    fig.savefig(io.out, dpi=130, bbox_inches="tight")
    print(f"{len(rois)} ROIs -> {io.out}")


if __name__ == "__main__":
    main()
