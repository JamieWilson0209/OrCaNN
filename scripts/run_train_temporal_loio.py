#!/usr/bin/env python
"""Leave-one-indicator-out training/eval of the temporal rate head.

Loads CASCADE-format ground-truth .mat files, labels each by indicator from a
JSON map (filename -> indicator[/cell-class]), and runs the LOIO transfer
table. Use --synthetic for an end-to-end self-test.

DATA-INTAKE: provide indicator_map.json as {"<filename.mat>": "GCaMP6f_exc", ...}.
Grouping by indicator AND cell class is recommended (interneuron vs pyramidal
kinetics differ).
"""
import argparse, glob, json, os

from orcann.train_loio import (
    load_cascade_mat, synthetic_indicator_bank, run_loio, train_final,
)
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir")
    ap.add_argument("--indicator-map")
    ap.add_argument("--target-fs", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--report", default=None)
    ap.add_argument("--exclude", default="",
                    help="comma-separated substrings; groups matching are dropped "
                         "(e.g. 'spinal-cord' to drop DS40/41)")
    ap.add_argument("--save-final", default=None,
                    help="train ONE model on all kept data and save it here "
                         "(skips the LOIO table)")
    ap.add_argument("--scale-dropout", type=float, default=0.0,
                    help="fraction of wavelet-scale channels to drop in training (0 = off)")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    excl = [s.strip().lower() for s in a.exclude.split(",") if s.strip()]
    if a.synthetic:
        neurons = synthetic_indicator_bank(n_per_indicator=8)
    else:
        imap = json.load(open(a.indicator_map)) if a.indicator_map else {}
        neurons, n_files, n_skip, n_excl = [], 0, 0, 0
        for mat in sorted(glob.glob(os.path.join(a.gt_dir, "**", "*.mat"), recursive=True)):
            rel = os.path.relpath(mat, a.gt_dir)
            ind = imap.get(rel) or imap.get(os.path.basename(mat))
            if ind is None:
                n_skip += 1
                continue
            if any(e in ind.lower() for e in excl):
                n_excl += 1
                continue
            n_files += 1
            neurons += load_cascade_mat(mat, indicator=ind, dataset_id=rel)
        print(f"loaded {n_files} files ({n_skip} unlisted, {n_excl} excluded), "
              f"{len(neurons)} neurons")
    print(f"{len(neurons)} neurons across "
          f"{len({n.indicator for n in neurons})} groups")

    if a.save_final:
        model = train_final(neurons, dst_fs=a.target_fs, epochs=max(a.epochs, 40),
                            scale_dropout=a.scale_dropout)
        os.makedirs(os.path.dirname(a.save_final) or ".", exist_ok=True)
        torch.save(model, a.save_final)
        print(f"saved final temporal model -> {a.save_final}")
        return

    report = run_loio(neurons, dst_fs=a.target_fs, epochs=a.epochs,
                      scale_dropout=a.scale_dropout)
    for r in report:
        print(f"  held-out {r['held_out']:>24}: heldout_corr {r['heldout_corr']} "
              f"| gap {r['transfer_gap']} | theta {r['learned_theta']} "
              f"| n_test {r['n_test_windows']}")
    if a.report:
        os.makedirs(os.path.dirname(a.report), exist_ok=True)
        json.dump(report, open(a.report, "w"), indent=2)


if __name__ == "__main__":
    main()
