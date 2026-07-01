"""Training stage for the temporal rate head: validate, and save a deployable model.

Reads the train_temporal section of the config (gt_dir, indicator_map, report,
save_final, and the training knobs). Two independent outputs, either or both per run:

  - LOIO table (validation): leave-one-indicator-out held-out correlations. Runs
    when ``report`` is set (printed to the log and written there as JSON).
  - final model (deployment): one model fit on ALL kept data and saved to
    ``save_final`` (this is what the detect_transients stage loads).

They no longer exclude each other, so a normal run validates AND saves a model.
Set ``report: null`` to skip validation (save only); set ``save_final: null`` to
skip the final fit (validate only). indicator_map.json maps each .mat filename to
an indicator label; grouping is by indicator and cell class. ``synthetic=True``
is a self-test: it runs LOIO on a synthetic bank and writes nothing to disk.
"""
import glob
import json
import os

from orcann.temporal import (
    load_cascade_mat, synthetic_indicator_bank, run_loio, train_final)
from orcann.pipeline.model_io import save_model


def run(cfg, synthetic=False):
    t = cfg.train_temporal
    excl = [s.strip().lower() for s in t.exclude if s.strip()]

    if synthetic:
        neurons = synthetic_indicator_bank(n_per_indicator=8)
    else:
        if t.gt_dir is None:
            raise SystemExit("set train_temporal.gt_dir in the config (or run --synthetic)")
        imap = json.load(open(t.indicator_map)) if t.indicator_map else {}
        neurons, n_files, n_skip, n_excl = [], 0, 0, 0
        for mat in sorted(glob.glob(os.path.join(t.gt_dir, "**", "*.mat"), recursive=True)):
            rel = os.path.relpath(mat, t.gt_dir)
            ind = imap.get(rel) or imap.get(os.path.basename(mat))
            if ind is None:
                n_skip += 1; continue
            if any(e in ind.lower() for e in excl):
                n_excl += 1; continue
            n_files += 1
            neurons += load_cascade_mat(mat, indicator=ind, dataset_id=rel)
        print(f"loaded {n_files} files ({n_skip} unlisted, {n_excl} excluded), "
              f"{len(neurons)} neurons")
    print(f"{len(neurons)} neurons across {len({n.indicator for n in neurons})} groups")

    # validation: leave-one-indicator-out held-out table. Run it when a report is
    # wanted, or when nothing else would run, so a bare run is never a no-op.
    if t.report or not t.save_final:
        report = run_loio(neurons, dst_fs=t.target_fs, epochs=t.epochs,
                          scale_dropout=t.scale_dropout)
        for r in report:
            print(f"  held-out {r['held_out']:>24}: heldout_corr {r['heldout_corr']} "
                  f"| gap {r['transfer_gap']} | theta {r['learned_theta']} "
                  f"| n_test {r['n_test_windows']}")
        if t.report and not synthetic:
            os.makedirs(os.path.dirname(t.report) or ".", exist_ok=True)
            json.dump(report, open(t.report, "w"), indent=2)
            print(f"wrote LOIO table -> {t.report}")

    # deployable model: fit on ALL data and save. Independent of the table above;
    # this is the model detect_transients loads. Skipped in the synthetic self-test
    # so it never overwrites a real model with one trained on synthetic data.
    if t.save_final and not synthetic:
        model = train_final(neurons, dst_fs=t.target_fs, epochs=max(t.epochs, 40),
                            scale_dropout=t.scale_dropout)
        os.makedirs(os.path.dirname(t.save_final) or ".", exist_ok=True)
        save_model(model, t.save_final)
        print(f"saved final temporal model -> {t.save_final}")
    elif synthetic and t.save_final:
        print("(--synthetic: ran the LOIO self-test only; not saving a model)")
