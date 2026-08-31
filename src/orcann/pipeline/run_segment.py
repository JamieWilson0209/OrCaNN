"""Segment stage (CPU): results/infer (cached prob) + movie -> results/spatial.

The parameter-dependent half of spatial detection, and model-free: it reads the
cached probability map, applies threshold / min_radius, extracts a
trace per ROI from the movie, and writes the canonical results/spatial/<rec>/
folder. No GPU, no model forward pass — so re-tuning is a cheap CPU job.

  - `--sweep "threshold=0.5,0.6,0.7" --sweep "min_radius=0,2"` previews the grid:
    one overlay montage + a comparison table per recording, no trace extraction
    (the slider replacement). Pick the winning values, set them in config, run
    segment for real.

Recordings whose traces.npy exists are skipped unless force=True.
"""
import itertools
import os

import numpy as np

from orcann.pipeline import inference as infer
from orcann.pipeline.cli import list_recordings, list_infer_recordings
from orcann.pipeline.postprocess import labels_from_prob
from orcann.run_info import RunInfoError, read_record


def _infer_recs(infer_dir, task_id=None):
    return list_infer_recordings(infer_dir, task_id)


def _model_of(infer_dir, rec_id):
    """The model identity recorded by ``infer`` for this recording, or None.

    None means the map predates identities; it is reported as unknown rather
    than filled in from config, because an unmeasured provenance is not a
    provenance.
    """
    try:
        rec = read_record(os.path.join(infer_dir, rec_id, infer.META_JSON),
                          expect_stage="infer")
    except (RunInfoError, OSError, ValueError):
        return None
    return rec.get("model_identity")


def _movie_for(rec_id, pre):
    for f in list_recordings(pre):
        if infer.recording_id(f) == rec_id:
            return f
    return None


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
    sp, fg, frame_rate = cfg.spatial, cfg.figures, cfg.imaging.frame_rate
    recs = _infer_recs(inf, task_id)
    if not recs:
        print(f"segment: no cached prob maps in {inf} (run infer first)"); return

    if sweeps:
        _run_sweep(cfg, recs, sweeps); return

    os.makedirs(out, exist_ok=True)
    base = dict(threshold=sp.threshold, min_area=sp.min_area,
                min_radius=sp.min_radius)
    print(f"segment: {len(recs)} recording(s)  {inf} + {pre} -> {out}")
    for rec_id in recs:
        if os.path.exists(os.path.join(out, rec_id, "data", "traces.npy")) and not force:
            print(f"{rec_id:28s} (exists, skipped)")
            continue
        # The model that made this map, read from its own record. Stamping the
        # identity config names today would attribute these ROIs to whatever is
        # selected now, which is not necessarily what produced the map.
        models = {"spatial": _model_of(inf, rec_id)}
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

        rec_dir = infer.write_recording(
            out, rec_id, traces=traces, frame_rate=frame_rate,
            detection=base, stage="segment (threshold + extract)", models=models,
            labels=labels, centroids=centroids, max_projection=maxproj,
            source=mv)
        if fg.enabled:
            infer.write_figures(rec_dir, None, traces, None, frame_rate, {},
                                max_projection=maxproj, labels=labels)
        print(f"{rec_id:28s} {int(traces.shape[0]):6d} cells")
    print(f"spatial outputs -> {out}/<recording_id>/  (now run: activity)")


def _run_sweep(cfg, recs, sweeps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    sp, inf, out = cfg.spatial, cfg.paths.infer, cfg.paths.spatial
    grid = _expand_sweeps(sweeps)
    base = dict(threshold=sp.threshold, min_area=sp.min_area,
                min_radius=sp.min_radius)
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
