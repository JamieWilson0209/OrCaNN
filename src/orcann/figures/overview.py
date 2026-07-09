"""
Overview figures: rasters, correlation matrices, neuron selection,
quality gating, population activity, bar charts.

These are the non-genotype-comparison figures — the panels that show
each recording on its own terms.
"""
import os
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.preprocessing import StandardScaler

from ..analysis.loading import (
    DatasetMetrics, FEATURE_NAMES, _abbrev,
    _extract_organoid_id, _extract_genotype,
    _trace_snr, _load_valid_mask,
)
from ..analysis.metrics import (
    _get_neuron_rates, _get_neuron_amplitudes, _recording_metric,
    build_feature_matrix,
)
from ._style import _fmt_p, _sig_stars, _draw_sig_bracket

logger = logging.getLogger(__name__)


# =============================================================================
# FIGURE FUNCTIONS
# =============================================================================

def generate_figures(
    datasets: List[DatasetMetrics],
    X: np.ndarray,
    feat_labels: List[str],
    names: List[str],
    output_dir: str,
) -> List[str]:
    """Generate all analysis figures with organised directory structure.

    Output layout:
        figures/
        ├── 1 - Main Results/       Headline figures
        ├── 1b - Metrics/           Individual bar chart metrics
        ├── Correlation Graphs/     Correlation matrices (one per dataset)
        └── Full Overview/          Combined multi-dataset views
    """

    # Create new directory structure matching user preferences
    base_dir = output_dir
    dirs = {
        'main_results': os.path.join(base_dir, '1 - Main Results'),
        'metrics': os.path.join(base_dir, '1b - Metrics'),
        'correlations': os.path.join(base_dir, 'Correlation Graphs'),
        'overview': os.path.join(base_dir, 'Full Overview'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    paths = []
    n_ds = len(datasets)
    default_color = '#5B8DBE'
    labels = np.ones(n_ds, dtype=int)
    colors = {1: default_color}

    # ── Main Results figures ───────────────────────────────────────────────
    X_std = StandardScaler().fit_transform(X)
    paths.append(_fig_feature_heatmap(X_std, names, feat_labels, labels, dirs['main_results']))
    paths.append(_fig_neuron_distributions(datasets, labels, colors, dirs['overview']))
    
    # ── Per-recording correlation matrices ────────────────────────────────
    paths.extend(_fig_correlation_matrices_split(datasets, labels, colors, dirs['correlations']))
    paths.append(_fig_population_activity(datasets, labels, colors, dirs['overview']))
    
    # ── Per-metric figures ────────────────────────────────────────────────
    paths.extend(_fig_bar_charts(datasets, labels, colors, dirs['overview'], dirs['metrics']))

    logger.info(f"Generated {len(paths)} figures in {output_dir}")
    return paths


def _fig_feature_heatmap(X_std, names, feat_labels, labels, output_dir):

    abbrevs = [_abbrev(n) for n in names]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.5), 8))

    # Reorder by cluster then by linkage within cluster
    order = np.argsort(labels)
    X_ordered = X_std[order]
    names_ordered = [abbrevs[i] for i in order]
    labels_ordered = labels[order]

    im = ax.imshow(X_ordered.T, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(names_ordered)))
    ax.set_xticklabels(names_ordered, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.set_title('Standardised Feature Heatmap', fontweight='bold')

    # Dataset separators (cosmetic)
    prev = labels_ordered[0]
    for i in range(1, len(labels_ordered)):
        if labels_ordered[i] != prev:
            ax.axvline(i - 0.5, color='white', linewidth=2)
            prev = labels_ordered[i]

    plt.colorbar(im, ax=ax, label='Z-score', shrink=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, 'feature_heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def _fig_neuron_distributions(datasets, labels, colors, output_dir):
    """Per-neuron distributions from selected neurons, pooled across all datasets."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ('spike_rate', 'Event rate (events/10s)'),
        ('quality', 'Quality Score'),
        ('spike_amplitude', 'Mean Transient amplitude (dF/F)'),
    ]

    for ax, (metric_key, title) in zip(axes, metrics):
        pooled = []
        for d in datasets:
            if metric_key == 'spike_rate':
                pooled.extend(_get_neuron_rates(d))
            elif metric_key == 'quality':
                if d.selected_quality is not None:
                    pooled.extend(d.selected_quality)
            elif metric_key == 'spike_amplitude':
                pooled.extend(_get_neuron_amplitudes(d))

        if len(pooled) == 0:
            continue
        sorted_vals = np.sort(pooled)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color='#5B8DBE', linewidth=2,
                label=f'All datasets (n={len(pooled)})')

        ax.set_xlabel(title)
        ax.set_ylabel('Cumulative Fraction')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Per-Neuron Distributions (Selected Neurons)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'neuron_distributions.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def _fig_correlation_matrices_split(datasets, labels, colors, corr_dir):
    """
    Generate individual correlation matrix figures for each dataset.
    
    Outputs:
    - correlations/<dataset_name>.png  — individual correlation matrices
    """

    os.makedirs(corr_dir, exist_ok=True)
    paths = []
    
    for i, ds in enumerate(datasets):
        result_path = Path(ds.filepath)
        
        # Prefer denoised
        denoised_path = result_path / 'data' / 'traces_denoised.npy'
        raw_path = result_path / 'data' / 'temporal_traces.npy'
        
        if denoised_path.exists():
            C_all = np.load(denoised_path)
        elif raw_path.exists():
            C_all = np.load(raw_path)
        else:
            continue
        
        # Exclude edge ROIs
        valid = _load_valid_mask(result_path)
        if valid is not None and len(valid) == C_all.shape[0]:
            valid_idx = np.where(valid)[0]
            C = C_all[valid_idx]
        else:
            valid_idx = np.arange(C_all.shape[0])
            C = C_all
        N = C.shape[0]
        
        if N < 3:
            continue
        
        # Limit to top 100 neurons by robust quality
        if N > 100:
            R = np.load(raw_path)[valid_idx] if raw_path.exists() else None
            quality = np.array([_trace_snr(C[j]) for j in range(C.shape[0])])
            top = np.argsort(quality)[::-1][:100]
            C = C[top]
            N = 100
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(C)
        corr = np.nan_to_num(corr, nan=0.0)
        
        # Sort by hierarchical clustering
        try:
            from scipy.spatial.distance import squareform; dist = np.clip(1 - corr, 0, 2); np.fill_diagonal(dist, 0); Z = linkage(squareform(dist), method='ward')
            order = leaves_list(Z)
        except Exception:
            order = np.arange(N)
        
        corr_sorted = corr[np.ix_(order, order)]
        
        # Create individual figure
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor('white')
        
        im = ax.imshow(corr_sorted, cmap='RdBu_r', vmin=-0.3, vmax=0.8,
                       aspect='equal', interpolation='none')
        ax.set_title(f'{_abbrev(ds.name)}\nPairwise Correlation ({N} neurons)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Neuron #', fontsize=10)
        ax.set_ylabel('Neuron #', fontsize=10)
        ax.tick_params(labelsize=8)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Pearson r', fontsize=10)
        
        # Add mean correlation annotation
        mask = np.triu(np.ones_like(corr), k=1).astype(bool)
        mean_corr = corr[mask].mean()
        ax.text(0.02, 0.98, f'Mean r = {mean_corr:.3f}', 
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save to correlations directory
        safe_name = ds.name.replace('/', '_').replace(' ', '_')[:50]
        corr_path = os.path.join(corr_dir, f'{safe_name}.png')
        plt.savefig(corr_path, dpi=200, bbox_inches='tight', facecolor='white')
        paths.append(corr_path)
        
        plt.close()
    
    logger.info(f"Generated {len(paths)} individual correlation matrices in {corr_dir}")
    return paths


def _fig_population_activity(datasets, labels, colors, output_dir):
    """Dataset-level metrics as labelled scatter + bar summary with organoid colors."""

    metrics = [
        ('mean_spike_rate',          'Spike Rate\n(events/10s)'),
        ('mean_spike_amplitude',     'Transient amplitude\n(dF/F)'),
        ('pairwise_correlation_mean','Pairwise\nCorrelation (r)'),
        ('synchrony_index',          'Synchrony\nIndex'),
        ('burst_rate',               'Burst Rate\n(bursts/10s)'),
        ('mean_iei',                 'Inter-Event\nInterval (s)'),
    ]

    # Extract organoid IDs and create color map
    organoid_ids = [_extract_organoid_id(ds.name) for ds in datasets]
    unique_organoids = list(OrderedDict.fromkeys(organoid_ids))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    org_colors = {oid: palette[i % len(palette)] for i, oid in enumerate(unique_organoids)}
    ds_colors = [org_colors[oid] for oid in organoid_ids]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 2.8, 5))
    fig.patch.set_facecolor('white')

    rng = np.random.default_rng(42)

    for ax, (attr, title) in zip(axes, metrics):
        ax.set_facecolor('white')
        vals = [getattr(ds, attr, 0) for ds in datasets]

        mean_val = np.mean(vals) if vals else 0
        ax.bar(0, mean_val, width=0.5, color='#5B8DBE', alpha=0.25,
               edgecolor='#5B8DBE', linewidth=1.5)

        jitter = rng.uniform(-0.15, 0.15, len(vals))
        x_pts = np.zeros(len(vals)) + jitter
        
        # Scatter points colored by organoid
        for i, (x, y, col) in enumerate(zip(x_pts, vals, ds_colors)):
            ax.scatter(x, y, c=col, s=35, edgecolor='white', linewidth=0.5, zorder=5)

        ax.hlines(mean_val, -0.25, 0.25, color='#333333', linewidth=2, zorder=6)

        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add legend for organoids
    legend_elements = [Patch(facecolor=org_colors[oid], edgecolor='white', label=oid) 
                      for oid in unique_organoids]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.95),
              fontsize=7, title='Organoid', title_fontsize=8, framealpha=0.9,
              ncol=min(3, len(unique_organoids)))

    fig.suptitle('Population Activity Summary (selected neurons)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(output_dir, 'population_activity.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def _draw_bar_panel(ax, key, ylabel, has_neurons, per_ds, ds_names, n_ds,
                    bar_width=0.65):
    """Draw a single bar chart panel. Shared by combined and individual figures."""

    BASE_COLOR = '#5B8DBE'

    ax.set_facecolor('white')
    x = np.arange(n_ds)
    means, sems = [], []

    for d in per_ds:
        if has_neurons and len(d[key]) > 0:
            vals = d[key]
            means.append(float(np.mean(vals)))
            sems.append(float(sp_stats.sem(vals)) if len(vals) > 1 else 0)
        else:
            means.append(float(d[key]))
            sems.append(0)

    bars = ax.bar(x, means, yerr=sems, width=bar_width,
                  color=BASE_COLOR, edgecolor='#3D6D8E', linewidth=0.8,
                  capsize=3, error_kw={'linewidth': 1.0, 'color': '#555',
                                       'capthick': 1.0},
                  zorder=3)

    # Individual neuron data points
    if has_neurons:
        rng = np.random.default_rng(42)
        for i, d in enumerate(per_ds):
            if len(d[key]) > 0:
                jitter = rng.uniform(-0.18, 0.18, len(d[key]))
                ax.scatter(i + jitter, d[key], color='#333333',
                           s=12, alpha=0.4, zorder=5,
                           linewidths=0, edgecolors='none')

    # Grand mean line
    if len(means) > 0:
        grand_mean = np.mean(means)
        ax.axhline(grand_mean, color='#888888', linewidth=0.8,
                   linestyle='--', alpha=0.5, zorder=2)
        ax.text(n_ds - 0.3, grand_mean, f'μ = {grand_mean:.2f}',
                fontsize=7, color='#666666', va='bottom', ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=45, ha='right', fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=9.5, fontweight='medium')
    ax.grid(axis='y', alpha=0.15, color='#999999', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['left'].set_color('#AAAAAA')
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['bottom'].set_color('#AAAAAA')
    ax.set_ylim(bottom=0)

    return means, sems

def _fig_bar_charts(datasets, labels, colors, output_dir, per_metric_dir=None):
    """
    Publication-quality bar charts of per-dataset spike rate, amplitude,
    pairwise correlation and synchrony.

    Produces a combined 4-panel figure AND individual per-metric figures.
    """

    n_ds = len(datasets)
    if n_ds == 0:
        return os.path.join(output_dir, 'bar_charts.png')

    ds_names = [_abbrev(d.name) for d in datasets]

    # ── Collect per-dataset metrics ──────────────────────────────────────
    per_ds = []
    for ds in datasets:
        rates = _get_neuron_rates(ds)
        amps = _get_neuron_amplitudes(ds)
        # Ensure amps matches rates length (some neurons may have no spikes)
        if len(rates) > 0 and len(amps) != len(rates):
            aligned = np.zeros(len(rates))
            aligned[:len(amps)] = amps[:len(rates)]
            amps = aligned
        per_ds.append({
            'rates': rates,
            'amps': amps,
            'corr': ds.pairwise_correlation_mean,
            'sync': ds.synchrony_index,
        })

    panels = [
        ('rates', 'Spike Rate (events / 10 s)',                      True,  'spike_rate'),
        ('amps',  'Transient amplitude (ΔF/F₀)',                         True,  'spike_amplitude'),
        ('corr',  'Pairwise Correlation (calcium traces, r)',         False, 'pairwise_correlation'),
        ('sync',  'Synchrony Index (0 = independent, 1 = synchronised)', False, 'synchrony_index'),
    ]

    # Legend elements (shared)
    typical_n = int(np.median([ds.n_selected for ds in datasets]))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
               markersize=6, label=f'Neuron (n={typical_n} per dataset)'),
        Line2D([0], [0], linestyle='--', color='#888888', linewidth=0.8,
               label='Grand mean'),
    ]

    fig_width = max(14, n_ds * 0.85)

    # ── Individual panel figures ─────────────────────────────────────────
    metric_dir = per_metric_dir or os.path.join(output_dir, 'per_metric')
    os.makedirs(metric_dir, exist_ok=True)

    for key, ylabel, has_neurons, fname in panels:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(fig_width, 3.8))
        fig_single.patch.set_facecolor('white')
        _draw_bar_panel(ax_single, key, ylabel, has_neurons,
                        per_ds, ds_names, n_ds)
        ax_single.set_title(ylabel, fontsize=11, fontweight='bold',
                            color='#333333', pad=10)
        fig_single.legend(handles=legend_elements, loc='upper right',
                          fontsize=7, framealpha=0.9, edgecolor='#CCCCCC',
                          bbox_to_anchor=(0.98, 0.98))
        plt.tight_layout()
        plt.savefig(os.path.join(metric_dir, f'{fname}.png'),
                    dpi=250, bbox_inches='tight', facecolor='white')
        plt.close(fig_single)

    # ── Combined 4-panel figure ──────────────────────────────────────────
    fig, axes = plt.subplots(len(panels), 1, figsize=(fig_width, len(panels) * 3.5))
    fig.patch.set_facecolor('white')

    for ax, (key, ylabel, has_neurons, _) in zip(axes, panels):
        _draw_bar_panel(ax, key, ylabel, has_neurons,
                        per_ds, ds_names, n_ds)

    fig.legend(handles=legend_elements, loc='upper right', fontsize=7.5,
               framealpha=0.9, edgecolor='#CCCCCC', bbox_to_anchor=(0.98, 0.99),
               ncol=1)
    fig.suptitle('Dataset Comparison',
                 fontsize=13, fontweight='bold', color='#333333', y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(output_dir, 'bar_charts.png')
    plt.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def _draw_neuron_selection_row(ax_hist, ax_traces, ds):
    """Draw a single dataset's neuron selection panels (histogram + traces)."""

    # ── SNR distribution ──────────────────────────────────────────────
    if ds.selected_quality is not None and len(ds.selected_quality) > 0:
        snr_sel = ds.selected_quality  # now stores SNR values
        ax_hist.hist(snr_sel, bins=20, color='#00e676', alpha=0.8,
                     edgecolor='none', label=f'Selected ({len(snr_sel)})')
        ax_hist.set_xlabel('Trace SNR', fontsize=7)
        ax_hist.set_ylabel('Count', fontsize=7)
        ax_hist.legend(fontsize=6)
        ax_hist.set_title(f'{_abbrev(ds.name)}\n{ds.n_selected}/{ds.n_neurons} selected',
                          fontsize=8, fontweight='bold')
        ax_hist.tick_params(labelsize=6)

    # ── Selected traces ─────────────────────────────────────────────
    if ds.selected_traces is not None and ds.selected_spikes is not None:
        n_sel = ds.selected_traces.shape[0]
        T = ds.selected_traces.shape[1]
        t_ax = np.arange(T) / ds.frame_rate

        for j in range(n_sel):
            trace = ds.selected_traces[j]
            offset = j * 1.2
            t_min, t_max = trace.min(), trace.max()
            t_range = t_max - t_min if t_max > t_min else 1
            trace_norm = (trace - t_min) / t_range + offset

            ax_traces.plot(t_ax, trace_norm, color='#00e676',
                           linewidth=0.6, alpha=0.8)

            spike_frames = np.where(ds.selected_spikes[j] > 0)[0]
            if len(spike_frames) > 0:
                ax_traces.scatter(
                    spike_frames / ds.frame_rate,
                    trace_norm[spike_frames],
                    color='red', s=4, zorder=5,
                )

            q = ds.selected_quality[j] if ds.selected_quality is not None else 0
            roi_idx = ds.selected_indices[j] if ds.selected_indices is not None else j
            ax_traces.text(t_ax[-1] * 1.01, offset + 0.5,
                           f'ROI {roi_idx} (q={q:.2f})',
                           fontsize=5, va='center')

        ax_traces.set_xlim(0, t_ax[-1])
        ax_traces.set_xlabel('Time (s)', fontsize=7)
        ax_traces.set_yticks([])
        ax_traces.set_title('Selected Neuron Traces (ranked by quality)',
                            fontsize=8)
        ax_traces.tick_params(labelsize=6)


def fig_neuron_selection(datasets, output_dir):
    """
    Neuron selection transparency figure.

    Produces:
    - overview/neuron_selection.png            combined overview
    """

    n_ds = len(datasets)

    # ── Combined overview ────────────────────────────────────────────────
    overview_dir = os.path.join(output_dir, 'Full Overview')
    os.makedirs(overview_dir, exist_ok=True)

    fig, axes = plt.subplots(n_ds, 2, figsize=(18, n_ds * 2.5),
                             gridspec_kw={'width_ratios': [1, 3]})
    if n_ds == 1:
        axes = axes.reshape(1, -1)

    for i, ds in enumerate(datasets):
        _draw_neuron_selection_row(axes[i, 0], axes[i, 1], ds)

    fig.suptitle('Neuron Selection — Quality Scoring & Selected Traces',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(overview_dir, 'neuron_selection.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def fig_quality_gating(datasets, max_thresh, res_thresh, drift_thresh, output_dir):
    """Show all quality metrics for all datasets with exclusion thresholds."""

    names = [_abbrev(d.name) for d in datasets]
    max_shifts = [d.motion_max_shift for d in datasets]
    residuals = [d.motion_residual_std for d in datasets]
    drifts = [d.baseline_drift for d in datasets]
    excluded = [d.motion_excluded for d in datasets]

    n_ds = len(datasets)
    fig_h = max(5, n_ds * 0.4)
    fig, axes = plt.subplots(1, 3, figsize=(20, fig_h))
    fig.patch.set_facecolor('white')

    y_pos = range(n_ds)

    # Per-dataset: determine exclusion reason for colour coding
    colors_shift = []
    colors_resid = []
    colors_drift = []
    for d in datasets:
        if d.motion_max_shift > max_thresh:
            colors_shift.append('#FF5252')
        elif d.motion_excluded:
            colors_shift.append('#FFAB91')  # excluded by other criterion
        else:
            colors_shift.append('#5B8DBE')

        if d.motion_residual_std > res_thresh:
            colors_resid.append('#FF5252')
        elif d.motion_excluded:
            colors_resid.append('#FFAB91')
        else:
            colors_resid.append('#5B8DBE')

        if d.baseline_drift > drift_thresh:
            colors_drift.append('#FF5252')
        elif d.motion_excluded:
            colors_drift.append('#FFAB91')
        else:
            colors_drift.append('#5B8DBE')

    # Panel 1: Max shift
    ax = axes[0]
    ax.set_facecolor('white')
    ax.barh(y_pos, max_shifts, color=colors_shift, edgecolor='#333333',
            linewidth=0.4, height=0.7)
    ax.axvline(max_thresh, color='#CC3333', linestyle='--', linewidth=1.5,
               label=f'Threshold ({max_thresh} px)', zorder=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Max Shift (px)', fontsize=9)
    ax.set_title('Maximum Motion Shift', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: Residual jitter
    ax = axes[1]
    ax.set_facecolor('white')
    ax.barh(y_pos, residuals, color=colors_resid, edgecolor='#333333',
            linewidth=0.4, height=0.7)
    ax.axvline(res_thresh, color='#CC3333', linestyle='--', linewidth=1.5,
               label=f'Threshold ({res_thresh} px)', zorder=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Residual Jitter Std (px)', fontsize=9)
    ax.set_title('Residual Motion After Correction', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 3: Baseline drift
    ax = axes[2]
    ax.set_facecolor('white')
    ax.barh(y_pos, drifts, color=colors_drift, edgecolor='#333333',
            linewidth=0.4, height=0.7)
    ax.axvline(drift_thresh, color='#CC3333', linestyle='--', linewidth=1.5,
               label=f'Threshold ({drift_thresh})', zorder=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Drift Ratio (|Q4-Q1| / std)', fontsize=9)
    ax.set_title('Baseline Drift', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    n_ex = sum(excluded)
    fig.suptitle(f'Dataset Quality Gating \u2014 {n_ex} excluded / {n_ds} total',
                 fontsize=13, fontweight='bold', y=1.02)

    # Legend for colours
    legend_elements = [
        Patch(facecolor='#5B8DBE', edgecolor='#333', label='Included'),
        Patch(facecolor='#FF5252', edgecolor='#333', label='Excluded (this criterion)'),
        Patch(facecolor='#FFAB91', edgecolor='#333', label='Excluded (other criterion)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = os.path.join(output_dir, 'quality_gating.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def fig_selected_traces(datasets: List[DatasetMetrics], output_dir: str) -> List[str]:
    """
    Generate trace figures for the SELECTED neurons used in statistical
    comparisons.

    For each recording, produces a figure with one row per selected neuron.
    Each row shows two overlaid traces:
      - Raw ΔF/F trace (before OASIS deconvolution) in grey
      - Denoised trace (after OASIS) in green/colour
      - Detected spike events marked as red dots

    This allows direct visual comparison of the deconvolution quality
    for every neuron entering the statistical analysis.

    Saves all figures to:
        {output_dir}/figures/Selected Traces/{recording_name}.png
    """


    traces_dir = os.path.join(output_dir, 'figures', 'Selected Traces')
    os.makedirs(traces_dir, exist_ok=True)
    paths = []

    for ds in datasets:
        result_path = Path(ds.filepath)
        denoised_path = result_path / 'data' / 'traces_denoised.npy'
        spikes_path = result_path / 'data' / 'spike_trains.npy'
        raw_path = result_path / 'data' / 'temporal_traces.npy'

        if not spikes_path.exists():
            logger.warning(f"  {ds.name}: no spike_trains.npy, skipping trace figure")
            continue

        # Load full arrays
        S_all = np.load(spikes_path)
        C_all = np.load(denoised_path) if denoised_path.exists() else None
        R_all = np.load(raw_path) if raw_path.exists() else None

        if C_all is None and R_all is None:
            logger.warning(f"  {ds.name}: no trace data, skipping")
            continue

        # Get the selected neuron indices (into full array)
        roi_idx = ds.selected_roi_indices
        if roi_idx is None or len(roi_idx) == 0:
            logger.warning(f"  {ds.name}: no selected_roi_indices, skipping")
            continue

        # Bounds-check indices against loaded arrays
        max_idx = S_all.shape[0]
        valid_mask = roi_idx < max_idx
        if C_all is not None:
            valid_mask &= roi_idx < C_all.shape[0]
        if R_all is not None:
            valid_mask &= roi_idx < R_all.shape[0]
        roi_idx_valid = roi_idx[valid_mask]
        if len(roi_idx_valid) == 0:
            continue

        S_sel = S_all[roi_idx_valid]
        C_sel = C_all[roi_idx_valid] if C_all is not None else None
        R_sel = R_all[roi_idx_valid] if R_all is not None else None
        N = len(roi_idx_valid)
        T = S_sel.shape[1]
        t_ax = np.arange(T) / ds.frame_rate
        duration_s = t_ax[-1] if len(t_ax) > 0 else 0

        # Active / inactive mask
        is_active = ds.neuron_is_active
        if is_active is not None and len(is_active) == len(roi_idx):
            is_active = is_active[valid_mask]
        else:
            is_active = np.array([np.sum(S_sel[j] > 0) > 0 for j in range(N)])

        # Per-neuron spike rates for labelling
        rates = np.array([np.sum(S_sel[j] > 0) / duration_s * 10.0
                         if duration_s > 0 else 0.0 for j in range(N)])

        # Sort by spike rate (most active at top)
        order = np.argsort(rates)[::-1]

        # ── Figure: one row per neuron ───────────────────────────────────
        row_height = 1.8
        fig_height = max(6, N * row_height + 2)
        fig, axes = plt.subplots(N, 1, figsize=(16, fig_height), sharex=True)
        fig.patch.set_facecolor('white')
        if N == 1:
            axes = [axes]

        for plot_i, neuron_i in enumerate(order):
            ax = axes[plot_i]
            ax.set_facecolor('white')

            orig_roi = int(roi_idx_valid[neuron_i])
            active = is_active[neuron_i]
            n_spikes = int(np.sum(S_sel[neuron_i] > 0))
            rate = rates[neuron_i]

            # ── Raw trace (before OASIS) ─────────────────────────────
            if R_sel is not None:
                raw_trace = R_sel[neuron_i]
                ax.plot(t_ax[:len(raw_trace)], raw_trace,
                        color='#B0BEC5', linewidth=0.5, alpha=0.7,
                        label='Raw ΔF/F', zorder=2)

            # ── Denoised trace (after OASIS) ─────────────────────────
            if C_sel is not None:
                den_trace = C_sel[neuron_i]
                trace_color = '#00e676' if active else '#78909C'
                ax.plot(t_ax[:len(den_trace)], den_trace,
                        color=trace_color, linewidth=0.8,
                        alpha=0.95 if active else 0.6,
                        label='Denoised (OASIS)', zorder=3)

            # ── Spike event markers ──────────────────────────────────
            spk_frames = np.where(S_sel[neuron_i] > 0)[0]
            if len(spk_frames) > 0:
                # Place markers on the denoised trace if available, else raw
                if C_sel is not None:
                    spk_y = C_sel[neuron_i][spk_frames]
                elif R_sel is not None:
                    spk_y = R_sel[neuron_i][spk_frames]
                else:
                    spk_y = np.zeros(len(spk_frames))
                ax.scatter(spk_frames / ds.frame_rate, spk_y,
                           color='#D32F2F', s=12, zorder=5, alpha=0.8,
                           marker='v', linewidths=0, label='Calcium events')

            # ── ROI label ────────────────────────────────────────────
            status = '●' if active else '○'
            label_color = '#333' if active else '#999'
            ax.set_ylabel(f'{status} ROI {orig_roi}\n{rate:.1f}/10s\n({n_spikes} spk)',
                         fontsize=7, fontweight='bold', color=label_color,
                         rotation=0, labelpad=55, va='center')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.grid(axis='x', alpha=0.1)

            # Legend on first row only
            if plot_i == 0:
                ax.legend(fontsize=6, loc='upper right', framealpha=0.8,
                         ncol=3, handlelength=1.5)

        # X-axis label on bottom row
        axes[-1].set_xlabel('Time (s)', fontsize=10)
        axes[-1].set_xlim(0, duration_s)

        # ── Title and caption ────────────────────────────────────────
        genotype = _extract_genotype(ds.name)
        geno_str = f'  [{genotype}]' if genotype != 'Unknown' else ''
        n_active = int(np.sum(is_active))

        fig.suptitle(
            f'{ds.name}{geno_str} — {N} selected neurons '
            f'({n_active} active, {N - n_active} inactive)',
            fontsize=12, fontweight='bold', y=1.0)

        fig.text(0.5, -0.005,
                 f'Grey = raw ΔF/F  ·  Green = OASIS denoised  ·  '
                 f'Red ▼ = detected spike events  ·  '
                 f'Duration: {duration_s:.0f}s  ·  {ds.frame_rate:.1f} Hz',
                 ha='center', fontsize=7, color='#777', style='italic')

        plt.tight_layout(rect=[0.08, 0.01, 1, 0.98])

        # Save with recording name
        safe_name = ds.name.replace('/', '_').replace(' ', '_').replace('\\', '_')
        fig_path = os.path.join(traces_dir, f'{safe_name}.png')
        plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        paths.append(fig_path)

    logger.info(f"Generated {len(paths)} selected-neuron trace figures in {traces_dir}")
    return paths


# =============================================================================
# CORE ACTIVITY ANALYSIS (v1.6)
# =============================================================================

def fig_n_selected_distribution(datasets: List, output_dir: str,
                                 mutant_label: str = 'CEP41 R242H') -> None:
    """Stacked bar chart: selected vs unselected neurons per recording by day.

    Each recording is one stacked bar. Grey segment = unselected detections
    (n_neurons - n_selected). Coloured segment on top = selected neurons
    (blue = Control, orange = Mutant). Bars are grouped by organoid day,
    sorted oldest to newest, with recordings within each day sorted by
    total detection count descending.

    Saved to figures/Full Overview/n_selected_distribution.png at 300 DPI.
    """


    overview_dir = os.path.join(output_dir, 'figures', 'Full Overview')
    os.makedirs(overview_dir, exist_ok=True)

    CTRL_COLOR   = '#4472C4'
    MUT_COLOR    = '#ED7D31'
    UNSEL_COLOR  = '#CCCCCC'
    BG_COLOR     = 'white'
    TEXT_COLOR   = '#333333'
    GRID_COLOR   = '#DDDDDD'
    SEP_COLOR    = '#CCCCCC'
    MIN_N        = 5

    def _day_num(d):
        m = re.search(r'\d+', d)
        return int(m.group()) if m else 0

    # Group recordings by organoid day
    day_map: dict = {}
    for ds in datasets:
        day = _extract_organoid_id(ds.name)
        day_map.setdefault(day, []).append(ds)

    sorted_days = sorted(day_map.keys(), key=_day_num)

    # Figure width scales with number of recordings
    n_recs = len(datasets)
    fig_w  = max(12, n_recs * 0.35 + len(sorted_days) * 0.25)
    fig_h = 10
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')
    ax.set_facecolor(BG_COLOR)

    bar_w   = 0.10
    gap_rec = 0.02    # gap between recordings within a day
    gap_day = 0.10    # extra gap between days
    x       = 0.0
    x_mids  = []      # centre of each day group for x-tick
    x_labs  = []

    for day in sorted_days:
        recs = sorted(day_map[day], key=lambda d: d.n_neurons, reverse=True)
        group_xs = []

        for ds in recs:
            geno  = _extract_genotype(ds.name)
            color = CTRL_COLOR if geno == 'Control' else MUT_COLOR
            n_sel = ds.n_selected
            n_uns = max(0, ds.n_neurons - n_sel)

            # Grey unselected base
            ax.bar(x, n_uns, width=bar_w, bottom=0,
                   color=UNSEL_COLOR, edgecolor=BG_COLOR, linewidth=0.3, zorder=2)
            # Coloured selected on top
            ax.bar(x, n_sel, width=bar_w, bottom=n_uns,
                   color=color, edgecolor=BG_COLOR, linewidth=0.3, zorder=3)

            # Label selected count above the coloured segment
            if n_sel >= 2:
                ax.text(x, n_uns + n_sel + 4, str(n_sel),
                        ha='center', va='bottom', fontsize=14,
                        color=color, fontweight='bold', zorder=4, rotation=45)

            group_xs.append(x)
            x += bar_w + gap_rec

        # Day mid-point for x-tick label
        if group_xs:
            day_start = group_xs[0] - bar_w / 2
            day_end   = group_xs[-1] + bar_w / 2
            x_mids.append((day_start + day_end) / 2)
            x_labs.append(day)

        # Separator between days
        sep_x = x - gap_rec / 2 + gap_day / 2
        if day != sorted_days[-1]:
            ax.axvline(sep_x, color=SEP_COLOR, linewidth=0.7,
                       linestyle='--', alpha=0.8, zorder=1)
        x += gap_day

    ax.set_xticks(x_mids)
    ax.set_xticklabels(x_labs, fontsize=24, fontweight='bold', color=TEXT_COLOR,
                       rotation=45, ha='center')
    ax.set_xlim(-bar_w * 0.8, x - gap_day)
    ax.set_ylabel('Number of detections (ROIs)', fontsize=24, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=20)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    # Maximum height of stacked bars
    max_bar = 500  # max bar height 
    space_above = 150  # space to leave for legend
    ax.set_ylim(0, max_bar + space_above)

    # Legend
    n_ctrl  = sum(1 for ds in datasets if _extract_genotype(ds.name) == 'Control')
    n_mut   = sum(1 for ds in datasets if _extract_genotype(ds.name) == 'Mutant')
    handles = [
        mpatches.Patch(facecolor=CTRL_COLOR,  label=f'Control — active  ({n_ctrl} recordings)'),
        mpatches.Patch(facecolor=MUT_COLOR,   label=f'{mutant_label} — active   ({n_mut} recordings)'),
        mpatches.Patch(facecolor=UNSEL_COLOR, label='No detectable activity'),
    ]
    legend = ax.legend(handles=handles, fontsize=20,
                   loc='upper right',
                   bbox_to_anchor=(1, 1),
                   framealpha=0.25, edgecolor=SEP_COLOR,
                   labelcolor=TEXT_COLOR,
                   facecolor='white')

    n_sel_vals = [ds.n_selected for ds in datasets]
    median_n   = int(np.median(n_sel_vals))
    total_sel  = sum(n_sel_vals)
    total_det  = sum(ds.n_neurons for ds in datasets)
    n_below    = sum(1 for n in n_sel_vals if n < MIN_N)

    ax.set_title(
        'Active Detections by Day Age\n'
        f'median active = {median_n}  ·  total active = {total_sel}  ·  '
        f'total detections = {total_det}',
        fontsize=28, fontweight='bold', pad=16, color=TEXT_COLOR,
    )

    fig.tight_layout()
    path = os.path.join(overview_dir, 'n_selected_distribution.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved n_selected distribution figure: {path}")
    logger.info(f"  median n_selected={median_n}, total selected={total_sel}, "
                f"total detections={total_det}, "
                f"{n_below} recordings below n={MIN_N}")

# =============================================================================
# MAIN
# =============================================================================

