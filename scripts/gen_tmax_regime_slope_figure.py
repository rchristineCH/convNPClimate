#!/usr/bin/env python3
"""CV -> 2024 slope chart for the seven tmax models — the four headline metrics.

Each model is one line between its CV-holdout score (left) and its 2024-holdout
score (right), so the sign of the change is visible directly: the atmospheric
models tilt down (they improve on the unseen year), the surface models tilt up.

Every error panel carries its bilinear ERA5-Land reference as a note rather
than a line — the reference sits at 2.5 to 3.4 °C against model scores of 0.7
to 2.0, so plotting it would flatten the model lines into a band. MAE and RMSE
notes come from the ``ref_mae_degC`` / ``ref_rmse_degC`` columns of the
comparison CSV; the CRPS reference IS the reference MAE (a deterministic
forecast's CRPS reduces to its absolute error). CRPSS is itself relative to
that reference, so its panel needs none.

Data: CLEAN_trained_models/tmax_model_comparison/metrics_comparison.csv in the
repo this script sits in. Writes LATEX_REPORT/images/tmax_regime_slope.png.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / 'CLEAN_trained_models/tmax_model_comparison/metrics_comparison.csv'
OUT = REPO / 'LATEX_REPORT/images/tmax_regime_slope.png'

# colour by input family, line style by training budget
STYLE = {
    'atm':             ('#2a78d6', '-'),
    'atm-clip60':      ('#2a78d6', '--'),
    'atm+wind':        ('#1baf7a', '-'),
    'atm+wind-clip60': ('#1baf7a', '--'),
    'atm+sfcanchors':  ('#e87ba4', '-'),
    'sfc':             ('#eb6834', '-'),
    # sfc-nogeo trains the standard 30-epoch budget, so it must stay solid:
    # the dash means 'extended budget' in the legend. Separated by shade.
    'sfc-nogeo':       ('#f6a97f', '-'),
}
INK, MUTED = '#1a1a19', '#5c5b55'

# (CSV column, panel title, decimals, reference column or None)
PANELS = [
    ('mae',   'MAE (°C) — lower is better',   2, 'ref_mae_degC'),
    ('rmse',  'RMSE (°C) — lower is better',  2, 'ref_rmse_degC'),
    ('crps',  'CRPS (°C) — lower is better',  2, 'ref_mae_degC'),  # deterministic ref: CRPS = MAE
    ('skill', 'CRPSS (–) — higher is better', 3, None),  # already relative to the reference
]

rows = list(csv.DictReader(open(CSV)))
vals = {(r['model'], r['regime']): r for r in rows}
models = list(STYLE)

def spread(labels, min_gap):
    """Nudge overlapping right-hand label positions apart, top-down."""
    order = sorted(range(len(labels)), key=lambda i: -labels[i])
    for a, b in zip(order, order[1:]):
        if labels[a] - labels[b] < min_gap:
            labels[b] = labels[a] - min_gap
    return labels


fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6))
for ax, (metric, title, dec, ref_col) in zip(axes.ravel(), PANELS):
    y0s = [float(vals[(m, 'cv')][metric]) for m in models]
    y1s = [float(vals[(m, '2024')][metric]) for m in models]
    gap = 0.035 * (max(y0s + y1s) - min(y0s + y1s))
    y1_labels = spread(list(y1s), gap)
    for m, y0, y1, y1lab in zip(models, y0s, y1s, y1_labels):
        c, ls = STYLE[m]
        ax.plot([0, 1], [y0, y1], ls, color=c, lw=2, marker='o', ms=5,
                markerfacecolor=c, markeredgecolor='white', zorder=3)
        ax.annotate(f'{m}  {y1:.{dec}f}', xy=(1, y1), xytext=(1.04, y1lab),
                    va='center', fontsize=8.5, color=c,
                    annotation_clip=False)
        ax.annotate(f'{y0:.{dec}f}', xy=(0, y0), xytext=(-0.04, y0),
                    ha='right', va='center', fontsize=8.5, color=MUTED,
                    annotation_clip=False)
    if ref_col is not None:
        r0 = float(vals[('atm', 'cv')][ref_col])
        r1 = float(vals[('atm', '2024')][ref_col])
        ax.set_xlabel(f'bilinear ERA5-Land reference: {r0:.2f} / {r1:.2f} °C',
                      fontsize=8, color=MUTED, labelpad=8)
    ax.set_xlim(-0.28, 1.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['CV holdout\n2020–2023', '2024\nholdout'],
                       fontsize=9.5, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', labelsize=8.5, colors=MUTED, length=0)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8)
    ax.set_axisbelow(True)

from matplotlib.lines import Line2D

handles = [
    Line2D([], [], color='#eb6834', lw=2, label='surface input'),
    Line2D([], [], color='#f6a97f', lw=2, label='surface, no elevation channel'),
    Line2D([], [], color='#2a78d6', lw=2, label='atmospheric'),
    Line2D([], [], color='#1baf7a', lw=2, label='atmospheric + wind'),
    Line2D([], [], color='#e87ba4', lw=2, label='atmospheric + surface anchors'),
    Line2D([], [], color=MUTED, lw=2, ls='-',  label='standard budget (30 epochs)'),
    Line2D([], [], color=MUTED, lw=2, ls='--', label='extended budget (60 epochs)'),
]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
           fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.5, 0.035),
           handlelength=2.4, columnspacing=2.2)
fig.text(0.5, 0.012,
         'Axes are scaled to the models; the ERA5-Land reference lies far above them and is '
         'quoted under each panel instead of drawn.',
         ha='center', fontsize=7.5, color=MUTED)

fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.17,
                    wspace=0.55, hspace=0.48)
fig.savefig(OUT, dpi=200, facecolor='white')
print('wrote', OUT)
