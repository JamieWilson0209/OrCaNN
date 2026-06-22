#!/usr/bin/env python
"""Measure the cell-size distribution from instance masks and recommend a LoG
radius bank for the segmenter.

The model's --radii are cell radii in pixels (the LoG matches a blob of radius r
with sigma = r/sqrt2). The bank must span the real size range: scales below the
smallest cells over-fire on noise, and cells larger than the top scale are
missed. This reads your instance-label masks, computes each cell's equivalent
radius sqrt(area/pi), and prints the distribution plus a suggested bank.

    python scripts/measure_cell_sizes.py --masks data/annotated/masks
"""
import argparse, glob, os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="dir of instance-label .npy masks")
    ap.add_argument("--n-scales", type=int, default=7,
                    help="how many radii to suggest in the bank")
    ap.add_argument("--lo-pct", type=float, default=10.0,
                    help="lower percentile for the smallest scale")
    ap.add_argument("--hi-pct", type=float, default=99.0,
                    help="upper percentile for the largest scale (raise to reach "
                         "the large-cell tail)")
    ap.add_argument("--min-radius", type=float, default=2.0,
                    help="floor for the smallest scale; a LoG below ~2 px (sigma "
                         "~1.4) mostly responds to hotspots/noise, not somata")
    ap.add_argument("--min-area", type=int, default=2,
                    help="ignore regions smaller than this many pixels (noise)")
    ap.add_argument("--um-per-px", type=float, default=None,
                    help="training pixel size; if given, sizes are also reported in "
                         "microns so you can check them against real soma sizes")
    ap.add_argument("--soma-diam-um", default=None,
                    help="MIN,MAX soma diameter in microns; with --um-per-px this "
                         "builds the bank from biology instead of percentiles "
                         "(resolution-invariant)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.masks, "*.npy")))
    if not files:
        print(f"no .npy masks in {a.masks}"); return

    radii = []
    for f in files:
        lab = np.load(f)
        ids = np.unique(lab); ids = ids[ids != 0]
        for k in ids:
            area = int((lab == k).sum())
            if area >= a.min_area:
                radii.append(float(np.sqrt(area / np.pi)))
    radii = np.asarray(radii)
    if radii.size == 0:
        print("no cells found"); return

    pct = {p: float(np.percentile(radii, p)) for p in (5, 10, 25, 50, 75, 90, 95, 99)}
    print(f"{len(radii)} cells across {len(files)} recordings  (equivalent radius, px)")
    print(f"  min {radii.min():.2f}   p5 {pct[5]:.2f}   p10 {pct[10]:.2f}   "
          f"p25 {pct[25]:.2f}   median {pct[50]:.2f}")
    print(f"  p75 {pct[75]:.2f}   p90 {pct[90]:.2f}   p95 {pct[95]:.2f}   "
          f"p99 {pct[99]:.2f}   max {radii.max():.2f}")
    if a.um_per_px:
        u = a.um_per_px
        print(f"  soma DIAMETER in microns (x2 x {u:g} um/px): "
              f"p10 {2 * pct[10] * u:.1f}   median {2 * pct[50] * u:.1f}   "
              f"p90 {2 * pct[90] * u:.1f}   p99 {2 * pct[99] * u:.1f}   "
              f"max {2 * radii.max() * u:.1f}")

    # bank source: biology (soma diameter in um) if given, else floored percentiles
    if a.soma_diam_um and a.um_per_px:
        dmin, dmax = (float(x) for x in a.soma_diam_um.split(","))
        lo = (dmin / 2.0) / a.um_per_px
        hi = (dmax / 2.0) / a.um_per_px
        src = f"soma diameter {dmin:g}-{dmax:g} um @ {a.um_per_px:g} um/px"
    else:
        lo = max(float(np.percentile(radii, a.lo_pct)), a.min_radius)
        hi = max(float(np.percentile(radii, a.hi_pct)), lo + 1.0)
        src = f"p{a.lo_pct:g}..p{a.hi_pct:g}, floored at {a.min_radius:g} px"
    bank = np.geomspace(lo, hi, a.n_scales)
    bank = sorted({round(float(x), 1) for x in bank})
    above = int((radii > bank[-1]).sum())
    below = int((radii < bank[0]).sum())
    print(f"\nsuggested bank ({a.n_scales} geometric scales; {src}):")
    print("  --radii " + ",".join(f"{x:g}" for x in bank))
    if a.um_per_px:
        diam = ",".join(f"{2 * x * a.um_per_px:.1f}" for x in bank)
        print(f"  = soma diameters (um): {diam}")
    print(f"  annotated regions below the smallest scale ({bank[0]:g} px): {below} "
          f"({100 * below / len(radii):.1f}%)  <- likely fragments/noise")
    print(f"  cells above the top scale ({bank[-1]:g} px): {above} "
          f"({100 * above / len(radii):.1f}%)")
    print("\nnotes: defining the bank in microns makes it portable across "
          "resolutions; the empirical px distribution above is the cross-check. "
          "Clean residual specks at inference with --min-area, not a sub-soma scale.")


if __name__ == "__main__":
    main()
