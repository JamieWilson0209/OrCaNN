"""
Top-level orchestration for the group analysis pipeline.

The ``run_analysis`` entry point is the only public symbol — it loads
per-recording outputs, applies quality gating, runs all compute and figure
generation, and writes a single ``analysis_results.json`` at the end.

External callers (run.sh, group_analysis.py shim) import ``run_analysis``
from this module.
"""

import os
import json
import logging
import csv
from pathlib import Path
from typing import Optional, Dict, Any

# Redirect matplotlib cache to scratch (HPC home quota is often limited).
if 'MPLCONFIGDIR' not in os.environ:
    _scratch = os.environ.get('SCRATCH_DIR', os.getcwd())
    _mpl_cache = os.path.join(_scratch, '.cache', 'matplotlib')
    os.makedirs(_mpl_cache, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = _mpl_cache

import numpy as np

from .loading import (
    DatasetMetrics, FEATURE_NAMES,
    load_dataset_metrics, _extract_genotype, _extract_organoid_id,
)
from .metrics import build_feature_matrix
from .stats import (
    run_statistical_tests,
    run_dataset_overview,
    run_genotype_comparison,
    run_between_organoid_tests,
    run_activity_analysis,
    generate_roi_peak_figures,
)
from ..figures.overview import (
    generate_figures,
    fig_neuron_selection,
    fig_n_selected_distribution,
    fig_quality_gating,
    fig_selected_traces,
)

logger = logging.getLogger(__name__)


def run_analysis(
    results_dir: str,
    output_dir: str,
    frame_rate_override: Optional[float] = None,
    motion_max_threshold: float = 15.0,
    motion_residual_threshold: float = 2.0,
    drift_threshold: float = 1.0,
    inactive_file: Optional[str] = None,
    min_roi_distance: float = 15.0,
    roi_peak_figures: bool = False,
    mutant_label: str = 'CEP41 R242H',
) -> Dict[str, Any]:
    """
    Run full analysis on batch results.

    Parameters
    ----------
    motion_max_threshold : float
        Datasets with max shift above this (in pixels) are excluded.
    motion_residual_threshold : float
        Datasets with residual jitter std above this are excluded.
    drift_threshold : float
        Datasets with population-median baseline drift ratio above this
        are excluded.  Drift ratio = |mean(Q4) - mean(Q1)| / std(trace),
        measured on raw fluorescence of selected neurons.  Default 1.0.
    inactive_file : str, optional
        Path to a text file listing dataset names (one per line) that were
        visually confirmed to have no activity.  These datasets are kept
        in the results but marked as inactive — all spikes are zeroed,
        active fraction set to 0, and they appear in figures with a
        distinct annotation.  Lines starting with # are ignored.

    Returns dict with datasets, features, and analysis results.
    """

    
    os.makedirs(output_dir, exist_ok=True)
    results_path = Path(results_dir)

    # ── Load inactive dataset list ───────────────────────────────────────
    inactive_names = set()
    if inactive_file and os.path.isfile(inactive_file):
        with open(inactive_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    inactive_names.add(line)
        logger.info(f"Loaded {len(inactive_names)} inactive datasets from {inactive_file}")
    elif inactive_file:
        logger.warning(f"Inactive file not found: {inactive_file}")

    # ── Load all datasets ────────────────────────────────────────────────
    logger.info(f"Loading datasets from {results_dir}")
    all_datasets = []
    for subdir in sorted(results_path.iterdir()):
        if not subdir.is_dir():
            continue
        if not (subdir / 'data' / 'temporal_traces.npy').exists():
            continue
        ds = load_dataset_metrics(
            str(subdir), subdir.name,
            frame_rate_override=frame_rate_override,
            min_roi_distance=min_roi_distance,
        )
        if ds is not None:
            all_datasets.append(ds)

    if len(all_datasets) == 0:
        logger.error("No datasets loaded")
        return {}

    logger.info(f"\nLoaded {len(all_datasets)} datasets")

    # ── Mark manually inactive datasets ──────────────────────────────────
    # Datasets visually confirmed to have no activity: zero out spikes,
    # set active fraction to 0, but keep them in the analysis for
    # active fraction and demographic reporting.
    n_marked_inactive = 0
    for ds in all_datasets:
        # Match by exact name, or bidirectional substring (handles cases where
        # folder name is shorter or longer than the inactive list entry)
        is_inactive = (ds.name in inactive_names or
                       any(iname in ds.name or ds.name in iname 
                           for iname in inactive_names))
        if is_inactive:
            ds.n_active = 0
            ds.active_fraction = 0.0
            if ds.neuron_is_active is not None:
                ds.neuron_is_active[:] = False
            if ds.neuron_spike_rates is not None:
                ds.neuron_spike_rates[:] = 0.0
            if ds.neuron_spike_amplitudes is not None:
                ds.neuron_spike_amplitudes[:] = 0.0
            ds.mean_spike_rate = 0.0
            ds.median_spike_rate = 0.0
            ds.mean_spike_amplitude = 0.0
            ds.manually_inactive = True
            n_marked_inactive += 1
            logger.info(f"  Marked as inactive (no visible activity): {ds.name}")
        else:
            ds.manually_inactive = False

    if n_marked_inactive > 0:
        logger.info(f"  Total manually inactive: {n_marked_inactive}/{len(all_datasets)}")

    # ── Quality gating (motion + baseline drift) ──────────────────────────
    logger.info(f"\nQuality gating (max_shift<={motion_max_threshold}px, "
                f"residual_std<={motion_residual_threshold}px, "
                f"baseline_drift<={drift_threshold}):")

    datasets = []
    excluded = []
    for ds in all_datasets:
        reasons = []
        if ds.motion_max_shift > motion_max_threshold:
            reasons.append(f"max_shift={ds.motion_max_shift:.1f}px")
        if ds.motion_residual_std > motion_residual_threshold:
            reasons.append(f"residual_std={ds.motion_residual_std:.2f}px")
        if ds.baseline_drift > drift_threshold:
            reasons.append(f"baseline_drift={ds.baseline_drift:.2f}")
            ds.baseline_drift_excluded = True

        if reasons:
            ds.motion_excluded = True
            excluded.append(ds)
            logger.warning(f"  EXCLUDED {ds.name}: {', '.join(reasons)}")
        else:
            datasets.append(ds)
            logger.info(f"  OK {ds.name}: shift={ds.motion_max_shift:.1f}px, "
                        f"residual={ds.motion_residual_std:.2f}px, "
                        f"drift={ds.baseline_drift:.2f}")

    logger.info(f"\n{len(datasets)} included, {len(excluded)} excluded by quality gating")

    # Save exclusion report
    quality_report = {
        'threshold_max_shift': motion_max_threshold,
        'threshold_residual_std': motion_residual_threshold,
        'threshold_baseline_drift': drift_threshold,
        'included': [d.name for d in datasets],
        'excluded': [
            {'name': d.name, 'max_shift': d.motion_max_shift,
             'residual_std': d.motion_residual_std,
             'baseline_drift': d.baseline_drift}
            for d in excluded
        ],
    }
    
    # ── Create organized output directory structure ────────────────────────
    # analysis/
    # ├── figures/
    # │   ├── 1 - Main Results/        UMAP, heatmap, between-organoid
    # │   ├── 1b - Metrics/            Individual metric plots
    # │   ├── Correlation Graphs/      Correlation matrices  
    # │   └── Full Overview/           Population activity, quality, neuron selection
    # └── data/
    #     ├── analysis_results.json
    #     ├── dataset_features.csv
    #     └── quality_gating.json
    
    fig_dir = os.path.join(output_dir, 'figures')
    data_dir = os.path.join(output_dir, 'data')
    
    # New directory structure (v2.0: added genotype comparison)
    main_results_dir = os.path.join(fig_dir, '1 - Main Results')
    metrics_dir = os.path.join(fig_dir, '1b - Metrics')
    genotype_dir = os.path.join(fig_dir, '2 - Genotype Comparison')
    correlation_dir = os.path.join(fig_dir, 'Correlation Graphs')
    overview_dir = os.path.join(fig_dir, 'Full Overview')
    
    for d in [fig_dir, data_dir, main_results_dir, metrics_dir, genotype_dir,
              correlation_dir, overview_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Save quality gating JSON to data/
    with open(os.path.join(data_dir, 'quality_gating.json'), 'w') as f:
        json.dump(quality_report, f, indent=2)

    # ── Generate motion quality figure ───────────────────────────────────
    fig_quality_gating(all_datasets, motion_max_threshold,
                        motion_residual_threshold, drift_threshold, overview_dir)

    if len(datasets) < 3:
        logger.error(f"Need at least 3 datasets after motion exclusion, "
                     f"got {len(datasets)}")
        return {}



    # ── Selected trace figures ───────────────────────────────────────────
    fig_selected_traces(datasets, output_dir)

    # ── ROI peak frame figures (optional, slow) ──────────────────────────
    if roi_peak_figures:
        try:
            generate_roi_peak_figures(datasets, output_dir)
        except Exception as e:
            logger.warning(f"ROI peak figures failed: {e}")
    else:
        logger.info("ROI peak frame figures disabled")

    # ── Neuron selection transparency ────────────────────────────────────
    fig_neuron_selection(datasets, fig_dir)

    # ── n_selected distribution ──────────────────────────────────────────
    try:
        fig_n_selected_distribution(datasets, output_dir, mutant_label=mutant_label)
    except Exception as e:
        logger.warning(f"n_selected distribution figure failed: {e}")

    # ── Build feature matrix ─────────────────────────────────────────────
    X, names = build_feature_matrix(datasets)
    feat_labels = [fl for _, fl in FEATURE_NAMES]

    # ── Save feature CSV ─────────────────────────────────────────────────
    csv_path = os.path.join(data_dir, 'dataset_features.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset'] + feat_labels)
        for i, name in enumerate(names):
            writer.writerow([name] + [f'{X[i, j]:.4f}' for j in range(X.shape[1])])
    logger.info(f"Saved feature matrix: {csv_path}")

    # ── Save per-ROI listing ─────────────────────────────────────────────
    roi_csv_path = os.path.join(data_dir, 'selected_rois.csv')
    with open(roi_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset', 'roi_index', 'quality_score', 'is_active',
            'n_spikes', 'spike_rate_per_10s', 'mean_amplitude',
            'genotype', 'organoid_day', 'manually_inactive',
        ])
        for ds in datasets:
            geno = _extract_genotype(ds.name)
            day = _extract_organoid_id(ds.name)
            n_sel = ds.n_selected
            for j in range(n_sel):
                roi_idx = int(ds.selected_roi_indices[j]) if ds.selected_roi_indices is not None and j < len(ds.selected_roi_indices) else j
                q = float(ds.selected_quality[j]) if ds.selected_quality is not None and j < len(ds.selected_quality) else 0.0
                is_active = bool(ds.neuron_is_active[j]) if ds.neuron_is_active is not None and j < len(ds.neuron_is_active) else False
                rate = float(ds.neuron_spike_rates[j]) if ds.neuron_spike_rates is not None and j < len(ds.neuron_spike_rates) else 0.0
                amp = float(ds.neuron_spike_amplitudes[j]) if ds.neuron_spike_amplitudes is not None and j < len(ds.neuron_spike_amplitudes) else 0.0
                n_spk = int(round(rate * ds.duration_seconds / 10.0)) if ds.duration_seconds > 0 else 0
                writer.writerow([
                    ds.name, roi_idx, f'{q:.3f}', is_active,
                    n_spk, f'{rate:.2f}', f'{amp:.4f}',
                    geno, day, ds.manually_inactive,
                ])
    logger.info(f"Saved selected ROI listing: {roi_csv_path}")

    # ── Save results JSON ────────────────────────────────────────────────
    results = {
        'n_datasets': len(datasets),
        'n_features': len(feat_labels),
        'motion_excluded': [d.name for d in excluded],
        'n_excluded': len(excluded),
        'neuron_selection': {
            'mode': 'deconv_pass + distance_dedup',
            'per_dataset': {
                d.name: {
                    'n_total': d.n_neurons,
                    'n_deconv_pass': d.n_confident,
                    'n_distance_removed': d.n_distance_removed,
                    'n_selected': d.n_selected,
                    'mean_snr': float(d.mean_quality_score),
                }
                for d in datasets
            },
        },
        'dataset_metrics': {
            d.name: {
                'mean_spike_rate': d.mean_spike_rate,
                'mean_spike_amplitude': d.mean_spike_amplitude,
                'pairwise_correlation': d.pairwise_correlation_mean,
                'synchrony_index': d.synchrony_index,
                'n_network_bursts': d.n_network_bursts,
                'burst_rate': d.burst_rate,
            }
            for d in datasets
        },
    }

    json_path = os.path.join(data_dir, 'analysis_results.json')

    # ── Statistical tests ────────────────────────────────────────────────
    try:
        stat_results = run_statistical_tests(datasets, output_dir)
        results['statistical_tests'] = stat_results['tests']
    except Exception as e:
        logger.error(f"Statistical tests failed: {e}")
        import traceback; logger.error(traceback.format_exc())

    # ── Dataset overview ─────────────────────────────────────────────────
    try:
        overview_results = run_dataset_overview(datasets, output_dir)
        results['dataset_overview'] = overview_results
    except Exception as e:
        logger.error(f"Dataset overview failed: {e}")
        import traceback; logger.error(traceback.format_exc())

    # ── Between-organoid comparison ──────────────────────────────────────
    try:
        between_results = run_between_organoid_tests(datasets, output_dir)
        results['between_organoid'] = between_results
    except Exception as e:
        logger.error(f"Between-organoid tests failed: {e}")
        import traceback; logger.error(traceback.format_exc())

    # ── Genotype comparison ──────────────────────────────────────────────
    try:
        genotype_results = run_genotype_comparison(datasets, output_dir,
                                                    mutant_label=mutant_label)
        results['genotype_comparison'] = genotype_results
    except Exception as e:
        logger.error(f"Genotype comparison failed: {e}")
        import traceback; logger.error(traceback.format_exc())

    # ── Activity analysis ────────────────────────────────────────────────
    try:
        activity_results = run_activity_analysis(datasets, output_dir,
                                                  mutant_label=mutant_label)
        results['activity_analysis'] = activity_results
    except Exception as e:
        logger.error(f"Activity analysis failed: {e}")
        import traceback; logger.error(traceback.format_exc())

    # ── Core overview figures ────────────────────────────────────────────
    generate_figures(datasets, X, feat_labels, names, fig_dir)

    # ── Single JSON dump at the end ──────────────────────────────────────
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Wrote {json_path}")

    return results
