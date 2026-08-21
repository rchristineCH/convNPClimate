#!/usr/bin/env python3
"""Overlay of the terrain-conditioned error curves: atm vs sfc, CV holdout.

Reproduces the per-grid-point MAE against |elevation mismatch| and against
altitude exactly as error_analysis.py computes them (per-point MAE over all CV
days, percentile-binned mean +- SE, bins with fewer than 10 points dropped),
but drawn for the two input families in one axes so the curves can be compared.

Data: pred_cache/cv.npz of the clean_solo (atm) and baseline (sfc) runs.
Writes LATEX_REPORT/images/tmax_mae_vs_elevmismatch_overlay_cv.png and
LATEX_REPORT/images/tmax_mae_vs_altitude_overlay_cv.png.
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/marc/convNPClimate/CLEAN_trained_models'
RUNS = [
    ('atm', '#2a78d6', f'{BASE}/clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8/tmax/pred_cache/cv.npz'),
    ('sfc', '#eb6834', f'{BASE}/baseline__tmax_sfc_flat_y2020-2023_e30f5_b8/tmax/pred_cache/cv.npz'),
]
OUT_DIR = '/home/marc/convNPClimate-doc/LATEX_REPORT/images'
INK, MUTED = '#1a1a19', '#5c5b55'


def binned(x, y, nbins=20):
    """Percentile-binned mean +- SE, matching error_analysis._binned."""
    edges = np.unique(np.nanpercentile(x, np.linspace(0, 100, nbins + 1)))
    centers, means, ses = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 10:
            centers.append(0.5 * (lo + hi))
            means.append(np.nanmean(y[m]))
            ses.append(np.nanstd(y[m]) / np.sqrt(m.sum()))
    return np.array(centers), np.array(means), np.array(ses)


def load(path):
    z = np.load(path, allow_pickle=True)
    mae = np.nanmean(np.abs(z['preds_degC'] - z['truths_degC']), axis=0)
    topo = z['target_topo'][z['valid_mask'].astype(bool).ravel()]
    alt, elevdiff = topo[:, 0], topo[:, 1]
    ok = np.isfinite(mae) & np.isfinite(alt) & np.isfinite(elevdiff)
    return mae[ok], alt[ok], np.abs(elevdiff)[ok]


data = {label: (load(path), color) for label, color, path in RUNS}

for feat, xlabel, fname, logx in [
        (2, '|elev mismatch| (m)', 'tmax_mae_vs_elevmismatch_overlay_cv.png', True),
        (1, 'altitude (m)', 'tmax_mae_vs_altitude_overlay_cv.png', False)]:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for label, ((mae, alt, absdiff), color) in data.items():
        x = (absdiff + 1) if logx else alt
        ax.scatter(x, mae, s=2.5, alpha=0.05, color=color, rasterized=True)
        cx, cy, ce = binned(x, mae)
        ax.errorbar(cx, cy, yerr=ce, fmt='-o', color=color, lw=2.2, ms=4,
                    capsize=2, zorder=5)
        ax.annotate(label, xy=(cx[-1], cy[-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=11,
                    color=color, fontweight='bold', annotation_clip=False)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel('per-grid-point MAE (°C)', fontsize=10, color=INK)
    ax.set_ylim(0.5, 3.2)
    ax.tick_params(labelsize=9, colors=MUTED)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.09, right=0.93, top=0.97, bottom=0.13)
    fig.savefig(f'{OUT_DIR}/{fname}', dpi=200, facecolor='white')
    plt.close(fig)
    print('wrote', fname)
