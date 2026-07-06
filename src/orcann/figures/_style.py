"""
Shared figure styling and statistical-annotation helpers.

All figure modules import from here for consistent formatting.
"""

import os
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# Colourblind-friendly palette used across the analysis figures.
ORG_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Genotype colour map (used by genotype.py)
GENOTYPE_COLOURS = {
    'Control': '#5B8DBE',
    'Mutant':  '#D17A22',
    'Unknown': '#999999',
}


def _fmt_p(p):
    """Format p-value for display: avoid showing p=0.000."""
    if p < 0.0001:
        return f'p={p:.1e}'
    elif p < 0.001:
        return f'p={p:.4f}'
    else:
        return f'p={p:.3f}'


def _sig_stars(p):
    """Return significance stars for a p-value (***, **, *, ns)."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def _draw_sig_bracket(ax, x1, x2, y, h, text, fontsize=7, color='#333333'):
    """Draw a significance bracket between two x positions."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color,
            linewidth=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom',
            fontsize=fontsize, color=color, fontweight='bold')


def _save(fig_or_path, path=None, dpi=200):
    """Consolidated savefig wrapper with white facecolor and tight bbox.

    Usage:
        _save(path)              # uses current plt figure
        _save(fig, path)         # uses given figure
    """
    if path is None:
        path, fig_obj = fig_or_path, plt
    else:
        fig_obj = fig_or_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig_obj.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    if hasattr(fig_obj, 'clf'):
        plt.close(fig_obj)
    else:
        plt.close()
