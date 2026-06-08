#!/usr/bin/env python
"""Full per-recording pipeline: ONE recording -> spatial detection, trace
extraction, transient extraction, written as a complete result folder.

This is the per-task work of a batch job. Everything for one recording happens
here in a single process — there is NO separate trace/transient job. A later
GROUP analysis module reads these per-recording folders across recordings.

    movie (.nd2/.tif/.npy)
      -> spatial scattering detector        -> cellness -> centroids, footprints
      -> trace extraction (per footprint)   -> temporal_traces (ΔF/F)
      -> temporal model + transient gating   -> rates, discrete transients, durations

OUTPUT  <out-dir>/<recording_id>/
    spatial_footprints.npz   footprints (n_roi, H, W) float32, compressed
    centroids.npy            (n_roi, 2) row,col
    temporal_traces.npy      (n_roi, T) extracted ΔF/F per ROI
    rates.npy                (n_roi, T) per-bin event rate
    events.npz               long-format transients: roi, time_s, duration_s, amplitude
    meta.json                ids, frame_rate, shapes, detection params, models, timestamp
"""
import argparse, json, os, time

import numpy as np
import torch

from orcann.extract import _load_movie, soft_footprints, extract_traces
from orcann.spatial_log import extract_instances
from orcann.temporal_dog import detect_transients
from orcann.figures import roi_figure, max_projection_figure


def recording_id(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    # also handle a results layout where traces live under "<rec> - Denoised/data/"
    if base in ("temporal_traces", "traces_denoised"):
        base = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return base.replace(" - Denoised", "").strip().replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    ap.add_argument("--spatial-model", required=True)
    ap.add_argument("--temporal-model", required=True)
    ap.add_argument("--frame-rate", type=float, default=2.0)
    ap.add_argument("--det-threshold", type=float, default=0.5)
    ap.add_argument("--min-distance", type=int, default=5)
    ap.add_argument("--min-prominence", type=float, default=0.5)
    ap.add_argument("--floor-pct", type=float, default=25.0)
    ap.add_argument("--min-isi-s", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True, help="parent; a <recording_id>/ is created under it")
    ap.add_argument("--no-figures", action="store_true", help="skip the figures/ output")
    ap.add_argument("--max-roi-figures", type=int, default=0,
                    help="cap per-ROI figures (0 = all); when capped, the most active ROIs are kept")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec = recording_id(a.movie)
    movie = _load_movie(a.movie)                                  # (T, H, W)
    T = movie.shape[0]
    spatial = torch.load(a.spatial_model, map_location=device, weights_only=False).eval().to(device)
    temporal = torch.load(a.temporal_model, map_location=device, weights_only=False).eval().to(device)

    # ── spatial ──
    with torch.no_grad():
        xt = torch.from_numpy(movie.astype(np.float32))[None].to(device)
        cellness = torch.sigmoid(spatial(xt))[0, 0].cpu().numpy()
    centroids, _ = extract_instances(cellness, a.min_distance, a.det_threshold)
    footprints = soft_footprints(cellness, centroids, movie) if len(centroids) else np.zeros((0,) + movie.shape[1:], np.float32)

    # ── traces ──
    traces = extract_traces(movie, footprints) if len(footprints) else np.zeros((0, T), np.float32)
    n_roi = len(traces)

    # ── transients (in-job, shared detector) ──
    rates = np.zeros((n_roi, T), np.float32)
    ev_roi, ev_t, ev_d, ev_a = [], [], [], []
    for i in range(n_roi):
        det = detect_transients(temporal, traces[i], a.frame_rate,
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

    # ── write: data/ (arrays) + figures/ (per-ROI panels + max-proj overlay) ──
    out = os.path.join(a.out_dir, rec)
    data_dir = os.path.join(out, "data")
    os.makedirs(data_dir, exist_ok=True)
    max_proj = movie.max(axis=0).astype(np.float32)
    np.savez_compressed(os.path.join(data_dir, "spatial_footprints.npz"),
                        footprints=footprints.astype(np.float32))
    np.save(os.path.join(data_dir, "centroids.npy"), np.asarray(centroids, np.float32))
    np.save(os.path.join(data_dir, "temporal_traces.npy"), traces.astype(np.float32))
    np.save(os.path.join(data_dir, "rates.npy"), rates)
    np.save(os.path.join(data_dir, "max_projection.npy"), max_proj)
    np.savez_compressed(os.path.join(data_dir, "events.npz"),
                        roi=roi, time_s=t_s, duration_s=d_s, amplitude=amp)
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump({
            "recording_id": rec,
            "frame_rate": a.frame_rate,
            "n_roi": int(n_roi),
            "n_frames": int(T),
            "image_shape": list(movie.shape[1:]),
            "n_events": int(len(roi)),
            "detection": {"det_threshold": a.det_threshold, "min_distance": a.min_distance,
                          "min_prominence": a.min_prominence, "floor_pct": a.floor_pct,
                          "min_isi_s": a.min_isi_s},
            "source_movie": os.path.abspath(a.movie),
            "spatial_model": os.path.abspath(a.spatial_model),
            "temporal_model": os.path.abspath(a.temporal_model),
            "device": str(device),
            "stage": "spatial+trace+transient (full per-recording pipeline)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)

    n_fig = 0
    if not a.no_figures:
        fig_dir = os.path.join(out, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        max_projection_figure(os.path.join(fig_dir, "max_projection_detections.png"),
                              max_proj, centroids, footprints)
        order = list(range(n_roi))
        if a.max_roi_figures and n_roi > a.max_roi_figures:           # keep most active
            activity = rates.sum(axis=1)
            order = list(np.argsort(activity)[::-1][:a.max_roi_figures])
        det_kw = dict(min_prominence=a.min_prominence, floor_pct=a.floor_pct, min_isi_s=a.min_isi_s)
        for i in order:
            roi_figure(os.path.join(fig_dir, f"roi_{i:03d}.png"), temporal,
                       traces[i], a.frame_rate, roi_label=f"ROI {i}", **det_kw)
        n_fig = len(order)
    print(f"{rec}: {n_roi} ROIs, {len(roi)} transients, {n_fig} ROI figures -> {out}  [{device}]")


if __name__ == "__main__":
    main()
