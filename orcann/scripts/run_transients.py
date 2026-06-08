#!/usr/bin/env python
"""Temporal transient detection for ONE recording -> clean data outputs.

Runs the trained temporal model over every ROI trace in a recording and writes
a minimal, self-describing result folder for a later analysis module. Called
per recording by the transients array job.

INPUT is the per-ROI fluorescence trace file (e.g. temporal_traces.npy), shape
(n_roi, T). NOT a spike train.

OUTPUT  <out-dir>/<recording_id>/
    rates.npy     (n_roi, T) float32   per-bin event rate, one row per ROI
    events.npz    long-format event table, one entry per detected transient:
                    roi (int32), time_s, duration_s, amplitude (float32)
    meta.json     recording id, frame_rate, n_roi, n_frames, n_events,
                  detection params, source trace path, model path, timestamp
Durations are inflated multiples of the decay constant (ordering faithful,
absolute seconds not). On out-of-domain organoid data the absolute rate scale
is uncertain — prefer rate shape / event structure over absolute counts.
"""
import argparse, json, os, time

import numpy as np
import torch

from orcann.temporal_dog import detect_transients


def recording_id(traces_path: str) -> str:
    # .../<recording> - Denoised/data/temporal_traces.npy  ->  <recording>
    rec = os.path.basename(os.path.dirname(os.path.dirname(traces_path)))
    return rec.replace(" - Denoised", "").strip().replace(" ", "_")


def load_traces(path: str) -> np.ndarray:
    a = np.load(path)
    if a.ndim == 1:
        a = a[None]
    if a.shape[0] > a.shape[1]:            # orient as (n_roi, T)
        a = a.T
    return a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frame-rate", type=float, default=2.0)
    ap.add_argument("--min-prominence", type=float, default=0.5)
    ap.add_argument("--floor-pct", type=float, default=25.0)
    ap.add_argument("--min-isi-s", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True, help="parent dir; a <recording_id>/ is created under it")
    ap.add_argument("--no-figures", action="store_true", help="skip the figures/ output")
    ap.add_argument("--max-roi-figures", type=int, default=0,
                    help="cap per-ROI figures (0 = all); when capped, the most active ROIs are kept")
    a = ap.parse_args()

    rec = recording_id(a.traces)
    traces = load_traces(a.traces)
    n_roi, T = traces.shape
    model = torch.load(a.model, weights_only=False, map_location="cpu").eval()

    rates = np.zeros((n_roi, T), dtype=np.float32)
    ev_roi, ev_t, ev_d, ev_a = [], [], [], []
    for i in range(n_roi):
        det = detect_transients(model, traces[i], a.frame_rate,
                                min_prominence=a.min_prominence,
                                floor_pct=a.floor_pct, min_isi_s=a.min_isi_s)
        rates[i] = det["rate"]
        k = len(det["peaks"])
        if k:
            ev_roi.append(np.full(k, i, np.int32))
            ev_t.append(det["times_s"]); ev_d.append(det["durations_s"]); ev_a.append(det["amplitudes"])

    cat = lambda L, dt: (np.concatenate(L).astype(dt) if L else np.array([], dt))
    roi = cat(ev_roi, np.int32); t_s = cat(ev_t, np.float32)
    d_s = cat(ev_d, np.float32); amp = cat(ev_a, np.float32)

    out = os.path.join(a.out_dir, rec)
    data_dir = os.path.join(out, "data")
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "rates.npy"), rates)
    np.save(os.path.join(data_dir, "temporal_traces.npy"), traces.astype(np.float32))
    np.savez_compressed(os.path.join(data_dir, "events.npz"),
                        roi=roi, time_s=t_s, duration_s=d_s, amplitude=amp)
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump({
            "recording_id": rec,
            "frame_rate": a.frame_rate,
            "n_roi": int(n_roi),
            "n_frames": int(T),
            "n_events": int(len(roi)),
            "detection": {"min_prominence": a.min_prominence,
                          "floor_pct": a.floor_pct, "min_isi_s": a.min_isi_s},
            "source_traces": os.path.abspath(a.traces),
            "model": os.path.abspath(a.model),
            "stage": "temporal_only (spatial detection pending)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)

    n_fig = 0
    if not a.no_figures:
        from orcann.figures import roi_figure
        fig_dir = os.path.join(out, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        order = list(range(n_roi))
        if a.max_roi_figures and n_roi > a.max_roi_figures:
            order = list(np.argsort(rates.sum(axis=1))[::-1][:a.max_roi_figures])
        det_kw = dict(min_prominence=a.min_prominence, floor_pct=a.floor_pct, min_isi_s=a.min_isi_s)
        for i in order:
            roi_figure(os.path.join(fig_dir, f"roi_{i:03d}.png"), model,
                       traces[i], a.frame_rate, roi_label=f"ROI {i}", **det_kw)
        n_fig = len(order)
    print(f"{rec}: {n_roi} ROIs, {len(roi)} transients, {n_fig} ROI figures -> {out}")


if __name__ == "__main__":
    main()
