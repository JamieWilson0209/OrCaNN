"""Analysis stage: cross-recording group analysis over the activity outputs.

A thin config->call wrapper around the calcium pipeline's group analysis
(`orcann.analysis.orchestrate.run_analysis`). It reads every calcium-format
recording folder under results/activity/ (the ones the activity stage wrote),
scores neuron quality, deduplicates ROIs, computes per-recording functional
summaries (spike rate, amplitude, pairwise correlation, synchrony, network
bursts, active fraction), and compares them across genotype and developmental
day, writing figures + tables to results/analysis/.

Unlike the per-recording stages this is a single aggregate run over all activity
outputs, so it is submitted directly (not as an array). Genotype and day are
parsed from each recording folder name by the analysis package's own name
parsers (D-line convention: line token starting with '3' is Control).
"""
import logging
import os

logger = logging.getLogger(__name__)


def run(cfg, force=False):
    from orcann.analysis.orchestrate import run_analysis

    src, out = cfg.paths.activity, cfg.paths.analysis
    ap = cfg.analysis
    if not os.path.isdir(src):
        print(f"analysis: no activity outputs in {src} (run activity first)")
        return
    have = [d for d in os.listdir(src)
            if os.path.isfile(os.path.join(src, d, "data", "temporal_traces.npy"))]
    if not have:
        print(f"analysis: no recordings with temporal_traces.npy in {src} "
              f"(run activity first)")
        return

    print(f"analysis: {len(have)} recording(s)  {src} -> {out}")
    results = run_analysis(
        results_dir=src,
        output_dir=out,
        frame_rate_override=cfg.imaging.frame_rate,
        motion_max_threshold=ap.motion_max_threshold,
        motion_residual_threshold=ap.motion_residual_threshold,
        drift_threshold=ap.drift_threshold,
        inactive_file=ap.inactive_file,
        min_roi_distance=ap.min_roi_distance,
        roi_peak_figures=ap.roi_peak_figures,
        mutant_label=ap.mutant_label,
    )
    if results:
        print(f"analysis outputs -> {out}/  ({results.get('n_datasets', '?')} datasets)")
    else:
        print("analysis: no datasets loaded - check that activity ran and folder "
              "names carry genotype/day (see dataset_features.csv)")
