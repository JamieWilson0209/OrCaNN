"""Segment stage (CPU): results/infer (cached prob) + movie -> results/spatial.

The parameter-dependent half of spatial detection, and model-free: it reads the
cached probability map, applies threshold / min_radius / watershed, extracts a
trace per ROI from the movie, and writes the canonical results/spatial/<rec>/
folder. No GPU, no model forward pass — so re-tuning is a cheap CPU job.

  - `--sweep "threshold=0.5,0.6,0.7" --sweep "min_radius=0,2"` previews the grid:
    one overlay montage + a comparison table per recording, no trace extraction
    (the slider replacement). Pick the winning values, set them in config, run
    segment for real.
  - curation: a per-recording results/spatial/<rec>/curate.json
        {"exclude_rois": [5, 37], "exclude_boxes": [[r0, c0, r1, c1]]}
    drops those ROIs as a post-extraction subset (no re-segmentation). ids are the
    numbers on figures/overlay.png; boxes are in the working/overlay resolution.

Recordings whose traces.npy exists are skipped unless force=True.
"""
import itertools
import json
import os

import numpy as np

from orcann.pipeline import inference as infer
from orcann.pipeline.cli import list_recordings, list_infer_recordings
from orcann.pipeline.postprocess import labels_from_prob, subset_rois


def _infer_recs(infer_dir, task_id=None):
    return list_infer_recordings(infer_dir, task_id)


def _movie_for(rec_id, pre):
    for f in list_recordings(pre):
        if infer.recording_id(f) == rec_id:
            return f
    return None


def _load_curation(spatial_dir, rec_id):
    p = os.path.join(spatial_dir, rec_id, "curate.json")
    if not os.path.isfile(p):
        return [], []
    with open(p) as fh:
        c = json.load(fh)
    return list(c.get("exclude_rois", [])), list(c.get("exclude_boxes", []))


def _coerce(v):
    v = v.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _expand_sweeps(sweeps):
    """['threshold=0.5,0.6', 'min_radius=0:2'] -> list of param-dict combinations.

    Values may be separated by comma, colon, or whitespace. Colon/space matter
    because ``qsub -v`` splits its variable list on commas: a comma inside the
    SWEEP value gets eaten, collapsing the sweep to its first value. So inside
    ``qsub -v ...,SWEEP='threshold=0.5:0.6:0.7'`` use colons; on the CLI
    (``--sweep threshold=0.5,0.6,0.7``) commas are fine.
    """
    import re
    axes = {}
    for s in sweeps:
        if "=" not in s:
            raise SystemExit(f"--sweep expects key=v1,v2,... (or v1:v2), got {s!r}")
        key, vals = s.split("=", 1)
        parts = [p for p in re.split(r"[,:\s]+", vals.strip()) if p]
        if not parts:
            raise SystemExit(f"--sweep {key.strip()} has no values: {s!r}")
        axes[key.strip()] = [_coerce(v) for v in parts]
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]


def run(cfg, task_id=None, force=False, sweeps=None):
    inf, pre, out = cfg.paths.infer, cfg.paths.pre_processed, cfg.paths.spatial
    sp, fg, tp = cfg.spatial, cfg.figures, cfg.temporal
    recs = _infer_recs(inf, task_id)
    if not recs:
        print(f"segment: no cached prob maps in {inf} (run infer first)"); return

    if sweeps:
        _run_sweep(cfg, recs, sweeps); return

    os.makedirs(out, exist_ok=True)
    models = {"spatial": os.path.abspath(cfg.models.spatial) if cfg.models.spatial else None}
    base = dict(threshold=sp.threshold, watershed=sp.watershed,
                min_distance=sp.min_distance, min_area=sp.min_area, min_radius=sp.min_radius)
    print(f"segment: {len(recs)} recording(s)  {inf} + {pre} -> {out}")
    for rec_id in recs:
        if os.path.exists(os.path.join(out, rec_id, "data", "traces.npy")) and not force:
            print(f"{rec_id:28s} (exists, skipped)")
            continue
        prob = np.load(os.path.join(inf, rec_id, infer.PROB_NPY))
        maxproj = np.load(os.path.join(inf, rec_id, infer.MAXPROJ_NPY))
        labels = labels_from_prob(prob, **base)

        mv = _movie_for(rec_id, pre)
        if mv is None:
            print(f"{rec_id:28s} ERROR: source movie not found in {pre}, skipping")
            continue
        from orcann.pipeline.extraction import _load_movie
        movie = infer.resample_to_shape(_load_movie(mv), prob.shape)
        traces, centroids = infer.traces_from_labels(movie, labels, weights=prob)

        ex_rois, ex_boxes = _load_curation(out, rec_id)
        cur_meta = None
        if ex_rois or ex_boxes:
            n0 = int(traces.shape[0])
            labels, centroids, traces = subset_rois(labels, centroids, traces, ex_rois, ex_boxes)
            cur_meta = {"curation": {"exclude_rois": ex_rois, "exclude_boxes": ex_boxes,
                                     "removed": n0 - int(traces.shape[0])}}

        rec_dir = infer.write_recording(
            out, rec_id, traces=traces, frame_rate=tp.frame_rate,
            detection=base, stage="segment (threshold + extract)", models=models,
            labels=labels, centroids=centroids, max_projection=maxproj,
            source=mv, extra_meta=cur_meta)
        if fg.enabled:
            infer.write_figures(rec_dir, None, traces, None, tp.frame_rate, {},
                                max_projection=maxproj, labels=labels, centroids=centroids)
        tag = f"  (curated -{cur_meta['curation']['removed']})" if cur_meta else ""
        print(f"{rec_id:28s} {int(traces.shape[0]):6d} cells{tag}")
    print(f"spatial outputs -> {out}/<recording_id>/  (now run: detect_transients)")


def _run_sweep(cfg, recs, sweeps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    sp, inf, out = cfg.spatial, cfg.paths.infer, cfg.paths.spatial
    grid = _expand_sweeps(sweeps)
    base = dict(threshold=sp.threshold, watershed=sp.watershed,
                min_distance=sp.min_distance, min_area=sp.min_area, min_radius=sp.min_radius)
    keys = list(grid[0].keys())
    ncol = min(4, len(grid))
    nrow = (len(grid) + ncol - 1) // ncol
    os.makedirs(out, exist_ok=True)
    print(f"segment sweep: {len(grid)} settings x {len(recs)} recording(s)")

    for rec_id in recs:
        prob = np.load(os.path.join(inf, rec_id, infer.PROB_NPY))
        maxproj = np.load(os.path.join(inf, rec_id, infer.MAXPROJ_NPY))
        lo, hi = np.percentile(maxproj, (1.0, 99.5))
        hi = hi if hi > lo else lo + 1e-6

        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 6 * nrow), squeeze=False)
        rows = []
        for k, combo in enumerate(grid):
            params = {**base, **combo}
            labels = labels_from_prob(prob, **params)
            n = int(labels.max())
            areas = np.bincount(labels.ravel())[1:] if n else np.array([0])
            rows.append({**combo, "n_roi": n, "median_area": float(np.median(areas))})
            ax = axes[k // ncol][k % ncol]
            ax.imshow(maxproj, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ov = np.zeros((*labels.shape, 4), np.float32)
            ov[find_boundaries(labels, mode="outer")] = (0.10, 0.95, 0.95, 1.0)
            ax.imshow(ov, interpolation="nearest")
            ax.set_title(", ".join(f"{kk}={combo[kk]}" for kk in keys) + f"  ->  {n}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        for j in range(len(grid), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")

        d = os.path.join(out, rec_id)
        os.makedirs(d, exist_ok=True)
        fig.suptitle(f"{rec_id} - segmentation sweep (n_roi per setting)", fontsize=12)
        fig.tight_layout()
        montage = os.path.join(d, "sweep_montage.png")
        fig.savefig(montage, dpi=150, bbox_inches="tight"); plt.close(fig)

        table = os.path.join(d, "sweep_table.csv")
        cols = keys + ["n_roi", "median_area"]
        with open(table, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(str(r[c]) for c in cols) + "\n")
        print(f"{rec_id:28s} -> {montage}  |  {table}")
    print("pick values from the montages, set spatial.* in config.yaml, then run "
          "segment (no --sweep) to extract.")
