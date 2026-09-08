"""Segment stage (CPU): an infer store + the movies -> results/spatial/<store>/.

The parameter-dependent half of spatial detection, and model-free: it reads the
cached probability map, applies threshold / min_radius, extracts a trace per ROI
from the movie, and writes the canonical <store>/<recording>/ folder. No GPU, no
model forward pass — so re-tuning is a cheap CPU job.

  - `--sweep "threshold=0.5,0.6,0.7" --sweep "min_radius=0,2"` previews the grid:
    one overlay montage + a comparison table per recording, no trace extraction
    (the slider replacement). Pick the winning values, set them in config, run
    segment for real.

A recording whose record was written under the settings in this config is
skipped unless force=True.
"""
import itertools
import os

import numpy as np

from orcann.pipeline import inference as infer
from orcann.pipeline import provenance as prov
from orcann.pipeline.cli import store_movies
from orcann.pipeline.postprocess import labels_from_prob


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
    sp, fg, frame_rate = cfg.spatial, cfg.figures, cfg.imaging.frame_rate
    try:
        chain = prov.chain(cfg)
        keys = prov.stage_keys(cfg, chain)
        inf = prov.require_store(cfg.paths.infer, prov.STAGE_INFER, chain.run_key,
                                 keys[prov.STAGE_INFER], infer.META_JSON)
    except prov.ProvenanceError as e:
        raise SystemExit(str(e))
    found = prov.list_dir_records(inf, infer.META_JSON, prov.STAGE_INFER)
    recs, surplus = prov.for_task(found, task_id)
    if surplus:
        print(f"segment: task {task_id} is past the end of {len(found)} "
              f"recording(s); nothing to do")
        return True
    if not recs:
        print(f"segment: no probability maps in {inf} (run infer first)")
        return False

    seg_rel = os.path.join(infer.DATA_DIRNAME, infer.META_JSON)
    out = prov.output_store(cfg.paths.spatial, prov.STAGE_SEGMENT, chain.run_key,
                            keys[prov.STAGE_SEGMENT], seg_rel)
    if sweeps:
        _run_sweep(cfg, recs, sweeps, inf, out)
        return True

    movies = dict((i, f) for f, i, _ in store_movies(chain.motion_dir,
                                                     chain.motion_external))
    os.makedirs(out, exist_ok=True)
    base = dict(threshold=sp.threshold, min_area=sp.min_area,
                min_radius=sp.min_radius)
    print(f"segment: {len(recs)} recording(s)  {inf} + {chain.motion_dir} -> {out}")
    missing = False
    for rec_id in recs:
        # The tag the map upstream was made from, carried down rather than
        # re-derived: a raw recording replaced after inference must reach this
        # stage as a change, and the movie beside it is a corrected copy whose
        # own pixels are not the thing being identified.
        upstream = prov.record_of(inf, rec_id, infer.META_JSON, prov.STAGE_INFER)
        key = dict(keys[prov.STAGE_SEGMENT],
                   recording=upstream.get("key", {}).get("recording", {}))
        record = os.path.join(out, rec_id, seg_rel)
        ok, why = prov.is_current(record, prov.STAGE_SEGMENT, key)
        if ok and not force:
            print(f"{rec_id:32s} (current, skipped)")
            continue
        if not ok and why != "no record":
            print(f"{rec_id:32s} ({why})")
        mv = movies.get(rec_id)
        if mv is None:
            print(f"{rec_id:32s} ERROR: no movie for this recording in "
                  f"{chain.motion_dir}, skipping")
            missing = True
            continue
        prob = np.load(os.path.join(inf, rec_id, infer.PROB_NPY))
        maxproj = np.load(os.path.join(inf, rec_id, infer.MAXPROJ_NPY))
        labels = labels_from_prob(prob, **base)

        from orcann.pipeline.extraction import _load_movie
        movie = infer.resample_to_shape(_load_movie(mv), prob.shape)
        traces, centroids = infer.traces_from_labels(movie, labels, weights=prob)

        # The model identity comes from the map's own record, not from the chain.
        # The key is keyed on the model's digest, so the same weights promoted
        # under a second name reach these maps too -- and stamping the identity
        # the config selects today would then name a model that did not make
        # them. The digest is what the key establishes; the name is not.
        rec_dir = infer.write_recording(
            out, rec_id, traces=traces, frame_rate=frame_rate,
            detection=base, stage=prov.STAGE_SEGMENT,
            models={"spatial": upstream.get("model_identity")},
            key=key, run_key=chain.run_key,
            provenance={"upstream_store": inf},
            labels=labels, centroids=centroids, max_projection=maxproj,
            source=mv)
        if fg.enabled:
            infer.write_figures(rec_dir, None, traces, None, frame_rate, {},
                                max_projection=maxproj, labels=labels)
        print(f"{rec_id:32s} {int(traces.shape[0]):6d} cells")
    print(f"spatial outputs -> {out}/<recording>/  (now run: activity)")
    return not missing


def _run_sweep(cfg, recs, sweeps, inf, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    sp = cfg.spatial
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
        print(f"{rec_id:32s} -> {montage}  |  {table}")
    print("pick values from the montages, set spatial.* in config.yaml, then run "
          "segment (no --sweep) to extract.")
