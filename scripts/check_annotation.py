#!/usr/bin/env python
"""DATA-INTAKE pre-flight: overlay annotation ROIs on the max projection.

Run this on the FIRST real annotated recording before training. ImageJ stores
ROI coordinates as (x, y) while arrays are (row, col); a transpose is a silent
failure that ruins training. This overlays the loaded annotation centroids on
the movie's max projection so you can confirm they land on real somata.

    python scripts/check_annotation.py --movie rec.nd2 --annotation RoiSet.zip \
        --out check.png
"""
import argparse

import numpy as np

from orcann.extract import _load_movie
from orcann.train_spatial import _load_annotation
from orcann.figures import max_projection_figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--default-radius", type=float, default=6.0)
    ap.add_argument("--out", default="annotation_check.png")
    a = ap.parse_args()

    movie = _load_movie(a.movie)
    H, W = movie.shape[1:]
    cents, radii = _load_annotation(a.annotation, (H, W), a.default_radius)
    max_proj = movie.max(axis=0).astype(np.float32)
    max_projection_figure(a.out, max_proj, cents, footprints=None)

    print(f"movie {movie.shape}  annotation {len(cents)} ROIs")
    print(f"centroid row range [{cents[:,0].min():.0f}, {cents[:,0].max():.0f}] / 0..{H}")
    print(f"centroid col range [{cents[:,1].min():.0f}, {cents[:,1].max():.0f}] / 0..{W}")
    print(f"overlay -> {a.out}")
    print("CHECK: every red marker should sit on a visible soma. If they look "
          "transposed/mirrored, the annotation orientation is wrong — stop and fix "
          "before training.")


if __name__ == "__main__":
    main()
