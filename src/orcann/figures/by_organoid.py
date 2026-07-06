"""
"By-organoid" dot plot figures.

A single factory replacing the three near-duplicate functions
``_fig_spike_rate_by_organoid``, ``_fig_correlation_by_organoid``, and
``_fig_synchrony_by_organoid``.  Each call produces one PNG plotting a
metric on the y-axis, with x-positions grouped by organoid and then by
recording day.

Also exposes the wrapper ``plot_by_organoid_panels`` that the stats
module calls to emit the three default panels in one go.
"""

import os
import re
import logging
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..analysis.loading import DatasetMetrics
from ..analysis.metrics import _get_neuron_rates
from ._style import _save

logger = logging.getLogger(__name__)


def _extract_day_rec(name: str):
    """Pull a 6-digit date and an R# recording number out of a dataset name."""
    date_match = re.search(r'(\d{6})', name)
    rec_match = re.search(r'(R\d+)', name)
    day = date_match.group(1) if date_match else 'unknown'
    rec = rec_match.group(1) if rec_match else ''
    return day, rec


def fig_metric_by_organoid(
    datasets: List[DatasetMetrics],
    organoid_ids: List[str],
    unique_organoids: List[str],
    org_colors: dict,
    output_dir: str,
    *,
    filename: str,
    value_fn: Callable[[DatasetMetrics], Optional[float]],
    ylabel: str,
    title: str,
    fmt: str = '.3f',
    ylim: Optional[tuple] = None,
    dpi: int = 200,
    figsize_scale: float = 1.0,
) -> Optional[str]:
    """Plot one metric across datasets, grouped by organoid then by day.

    Parameters
    ----------
    value_fn : callable
        Maps a DatasetMetrics → float or None.  Returning None or a non-finite
        value skips that recording.
    filename : str
        Output filename under ``<output_dir>/figures/1 - Main Results/``.
    fmt : str
        Float format for the global-mean annotation, e.g. '.2f' or '.3f'.
    ylim : tuple, optional
        Hard y-axis limits.  If None, the y-axis is auto-fit to the data with
        a small margin.
    figsize_scale : float
        Multiplier for the auto-computed figure width and height.  The default
        size matches the smaller correlation/synchrony figures; pass ~1.6 for
        the headline event-rate figure.
    """

    rng = np.random.default_rng(42)
    org_day_vals = OrderedDict((oid, OrderedDict()) for oid in unique_organoids)
    all_vals: List[float] = []

    for ds, oid in zip(datasets, organoid_ids):
        v = value_fn(ds)
        if v is None or not np.isfinite(v):
            continue
        all_vals.append(float(v))
        day, rec = _extract_day_rec(ds.name)
        org_day_vals[oid].setdefault(day, []).append((rec, float(v)))

    total_days = sum(len(days) for days in org_day_vals.values())
    if total_days == 0:
        logger.warning(f"  No valid data for {filename} — skipping")
        return None

    n_org = len(unique_organoids)
    base_w = max(10, total_days * 0.6 + n_org * 0.8)
    fig, ax = plt.subplots(figsize=(base_w * figsize_scale, 6 * figsize_scale))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x_pos = 0
    x_ticks: List[float] = []
    x_labels: List[str] = []

    for oid in unique_organoids:
        for day, recordings in sorted(org_day_vals[oid].items()):
            vals = [r[1] for r in recordings]
            if not vals:
                continue
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(x_pos + jitter, vals, c=org_colors[oid], s=60, alpha=0.75,
                       edgecolor='white', linewidth=0.5, zorder=5)
            ax.scatter(x_pos, np.mean(vals), c=org_colors[oid], s=140,
                       marker='_', linewidths=3, zorder=6)
            day_formatted = f'{day[2:4]}/{day[0:2]}' if len(day) == 6 else day
            x_ticks.append(x_pos)
            x_labels.append(day_formatted)
            x_pos += 1
        x_pos += 0.3

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha='right')
    ax.set_xlim(-0.8, x_pos - 0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif all_vals:
        lo = max(0.0, float(np.percentile(all_vals, 1)) - 0.02)
        hi = float(np.percentile(all_vals, 99)) * 1.15
        ax.set_ylim(bottom=lo, top=max(hi, 0.1))

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Recording Date (DD/MM)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    if all_vals:
        mn = float(np.mean(all_vals))
        ax.axhline(mn, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(ax.get_xlim()[1], mn, f' mean={mn:{fmt}}',
                va='center', ha='left', fontsize=9, color='#666666')

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=org_colors[oid],
                              markersize=10, alpha=0.75, label=oid)
                       for oid in unique_organoids]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              title='Organoid', title_fontsize=11, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'figures', '1 - Main Results', filename)
    _save(fig, path, dpi=dpi)
    logger.info(f"  {filename} saved: {path}")
    return path


def _mean_neuron_rate(ds: DatasetMetrics) -> Optional[float]:
    rates = _get_neuron_rates(ds)
    if len(rates) == 0:
        return None
    return float(np.mean(rates))


def plot_by_organoid_panels(datasets, organoid_ids, unique_organoids,
                            org_colors, output_dir) -> List[str]:
    """Emit the three default by-organoid panels (rate / correlation / synchrony)."""
    paths: List[str] = []

    p1 = fig_metric_by_organoid(
        datasets, organoid_ids, unique_organoids, org_colors, output_dir,
        filename='spike_rate_by_organoid.png',
        value_fn=_mean_neuron_rate,
        ylabel='Event rate (events/10s)',
        title='Event rate by organoid  (per-recording averages)',
        fmt='.2f',
        dpi=300,
        figsize_scale=1.35,
    )
    if p1: paths.append(p1)

    p2 = fig_metric_by_organoid(
        datasets, organoid_ids, unique_organoids, org_colors, output_dir,
        filename='correlation_by_organoid.png',
        value_fn=lambda ds: ds.pairwise_correlation_mean,
        ylabel='Pairwise Correlation (r)',
        title='Pairwise Correlation by Organoid',
        fmt='.3f',
        dpi=200,
    )
    if p2: paths.append(p2)

    p3 = fig_metric_by_organoid(
        datasets, organoid_ids, unique_organoids, org_colors, output_dir,
        filename='synchrony_by_organoid.png',
        value_fn=lambda ds: ds.synchrony_index,
        ylabel='Synchrony Index',
        title='Synchrony Index by Organoid',
        fmt='.3f',
        ylim=(0, 1.05),
        dpi=200,
    )
    if p3: paths.append(p3)

    return paths
