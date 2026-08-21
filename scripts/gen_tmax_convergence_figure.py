#!/usr/bin/env python3
"""Convergence evidence for the wind ablation: atm vs atm+wind, 30 vs 60 epochs.

Backs the claim in the temperature results that the wind channels only pay once
the optimisation has time to use them: at the 30-epoch budget ``atm+wind`` sits
above ``atm``, and under the extended budget the order reverses.

Left panel: the held-out score per epoch written by ``training_elev`` (the
``Mean absolute error`` column of ``stats.csv``, i.e. ``eval_epoch_elev``'s
median over the days of the held-out block), scaled to degrees Celsius with each
run's own ``data_std``, taken as the median over folds and smoothed with a
five-epoch rolling median because the epoch-to-epoch scatter is larger than the
difference between the runs.

Right panel: the epoch at which each fold's best checkpoint was taken. Selection
is on the held-out NLL, the training objective, so this is the quantity that
actually decided the weights carried into the results chapter. It is the sharper
evidence: both extended runs place every best checkpoint in the last quarter of
the 60-epoch budget, so neither had finished improving when it was stopped.

The curve is a median over days whereas the headline MAE of the results chapter
is a mean over all point-days, so the two are the same quantity measured
differently; the levels here should not be read against the table.

Writes LATEX_REPORT/images/tmax_convergence_wind.png.
"""
import csv
import json
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLEAN = ROOT / 'CLEAN_trained_models'
OUT = ROOT / 'LATEX_REPORT/images/tmax_convergence_wind.png'

RUNS = [
    ('atm',             'clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8',                '#2a78d6', '-'),
    ('atm+wind',        'clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8',  '#1baf7a', '-'),
    ('atm-clip60',      'lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8',                '#2a78d6', '--'),
    ('atm+wind-clip60', 'lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8',      '#1baf7a', '--'),
]
INK, MUTED = '#1a1a19', '#5c5b55'


def load(run_dir):
    """-> {fold: [(epoch, test_nll, mae_degC), ...]} sorted by epoch."""
    base = CLEAN / run_dir / 'tmax'
    std = json.load(open(base / 'metadata.json'))['data_std']
    out = defaultdict(list)
    for r in csv.DictReader(open(base / 'stats.csv')):
        out[int(r['Fold'])].append((int(r['Epoch']), float(r['test NLL']),
                                    float(r['Mean absolute error']) * std))
    return {f: sorted(v) for f, v in out.items()}


def rolling_median(xs, ys, w=5):
    half = w // 2
    return xs, [float(np.median(ys[max(0, i - half):i + half + 1])) for i in range(len(ys))]


fig, (axc, axb) = plt.subplots(1, 2, figsize=(11.0, 4.8),
                               gridspec_kw={'width_ratios': [1.65, 1]})

best_rows = []
ends = []
for label, run_dir, colour, ls in RUNS:
    folds = load(run_dir)
    epochs = sorted({e for v in folds.values() for e, _, _ in v})
    med_e, med_v = [], []
    for e in epochs:
        vals = [m for v in folds.values() for ep, _, m in v if ep == e]
        if len(vals) >= 3:
            med_e.append(e)
            med_v.append(float(np.median(vals)))
    sx, sy = rolling_median(med_e, med_v)
    axc.plot(sx, sy, ls, color=colour, lw=2.1, zorder=3)
    ends.append((sx[-1], sy[-1], label, colour))

    bests = [min(v, key=lambda t: t[1]) for v in folds.values()]
    best_rows.append((label, colour, [b[0] for b in bests],
                      float(np.median([b[2] for b in bests]))))

# labels at the curve ends, pushed apart where two runs finish at the same epoch
for x in sorted({e[0] for e in ends}):
    grp = sorted([e for e in ends if e[0] == x], key=lambda e: e[1])
    ys = [e[1] for e in grp]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.075)
    for (_x, yv, label, colour), yl in zip(grp, ys):
        axc.annotate(f'  {label}', xy=(x, yv), xytext=(x + 0.9, yl), va='center',
                     fontsize=8.5, color=colour, annotation_clip=False,
                     arrowprops=dict(arrowstyle='-', color=colour, lw=0.6,
                                     shrinkA=0, shrinkB=2, alpha=0.5))

axc.axvline(29, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
axc.text(28.2, 2.28, 'standard budget ends ', ha='right', va='top',
         fontsize=8.5, color=MUTED)
axc.set_xlabel('Epoch', fontsize=10, color=INK)
axc.set_ylabel('Held-out MAE (°C)', fontsize=10, color=INK)
axc.set_title('Convergence (fold median, 5-epoch rolling median)',
              fontsize=10.5, color=INK)
axc.set_xlim(0, 76)
axc.set_ylim(0.95, 2.35)
axc.spines[['top', 'right']].set_visible(False)
axc.tick_params(labelsize=9, colors=MUTED)
axc.grid(axis='y', color='#e6e5e0', lw=0.8)
axc.set_axisbelow(True)

for i, (label, colour, eps, mae) in enumerate(best_rows):
    y = len(best_rows) - 1 - i
    axb.scatter(eps, [y] * len(eps), s=42, color=colour, alpha=0.75,
                edgecolor='white', linewidth=0.8, zorder=3)
    axb.annotate(f'{mae:.2f} °C', xy=(62.5, y), fontsize=8.5, va='center',
                 color=colour)
axb.axvline(29, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
axb.set_yticks(range(len(best_rows)))
axb.set_yticklabels([r[0] for r in reversed(best_rows)], fontsize=9)
axb.set_xlabel('Epoch of the best checkpoint (one dot per fold)', fontsize=10, color=INK)
axb.set_title('Checkpoint selection, and its MAE', fontsize=10.5, color=INK)
axb.set_xlim(0, 76)
axb.set_ylim(-0.6, len(best_rows) - 0.4)
axb.spines[['top', 'right', 'left']].set_visible(False)
axb.tick_params(labelsize=9, colors=MUTED, length=0)
axb.grid(axis='x', color='#e6e5e0', lw=0.8)
axb.set_axisbelow(True)

fig.text(0.5, 0.015,
         'Checkpoints are selected on the held-out NLL, the training objective. Both extended runs '
         'take every best checkpoint in the last quarter of the budget, so neither had stopped improving.',
         ha='center', fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.055, 1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f'wrote {OUT}')
