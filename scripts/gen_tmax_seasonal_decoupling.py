#!/usr/bin/env python3
"""Monthly error of ref, sfc and atm over lowlands and the high Alps.

Mean MAE per calendar month for the bilinear-ERA5 reference and the sfc and
atm predictions, over the lowland grid points (< 600 m, left) and the alpine
grid points (> 2000 m, right). Thick lines pool the four CV-holdout years
2020--2023 (each day predicted out-of-sample by the fold that never saw it);
thin lines are the 2024 holdout (five-fold ensemble).

The two panels carry the two halves of the argument: over the relief the
reference error is large and systematic, and both models correct most of it in
every season; over the lowlands the error the reference makes is situational,
and sfc converges towards the reference in the cold months while atm keeps its
distance.

Data: pred_cache/{cv,holdout_2024}.npz of the clean_solo (atm) and baseline
(sfc) runs. Writes LATEX_REPORT/images/tmax_seasonal_decoupling.png.
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/marc/convNPClimate/CLEAN_trained_models'
RUNS = {
    'atm': f'{BASE}/clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8/tmax/pred_cache',
    'sfc': f'{BASE}/baseline__tmax_sfc_flat_y2020-2023_e30f5_b8/tmax/pred_cache',
}
OUT = '/home/marc/convNPClimate-doc/LATEX_REPORT/images/tmax_seasonal_decoupling.png'
COLOR = {'atm': '#2a78d6', 'sfc': '#eb6834', 'ERA5 reference': '#8c8c8c'}
INK, MUTED = '#1a1a19', '#5c5b55'
BANDS = {'lowlands (below 600 m)': lambda a: a < 600,
         'high Alps (above 2000 m)': lambda a: a > 2000}


def monthly_curves(fname):
    """{band: {series: 12 monthly MAE values}} for one cached regime."""
    bundles = {k: np.load(f'{p}/{fname}', allow_pickle=True)
               for k, p in RUNS.items()}
    z = bundles['atm']
    dates = pd.to_datetime(z['dates'])
    truth = z['truths_degC']
    alt = z['target_topo'][z['valid_mask'].astype(bool).ravel()][:, 0]
    series = {'ERA5 reference': z['era5_ref_degC']}
    series.update({k: b['preds_degC'] for k, b in bundles.items()})
    out = {}
    for band, sel in BANDS.items():
        m = sel(alt)
        out[band] = {
            name: pd.Series(np.nanmean(np.abs(p - truth)[:, m], axis=1),
                            index=dates).groupby(dates.month).mean()
            for name, p in series.items()}
    return out


cv = monthly_curves('cv.npz')
ho = monthly_curves('holdout_2024.npz')

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
months = np.arange(1, 13)
for ax, band in zip(axes, BANDS):
    ax.fill_between(months, cv[band]['atm'], cv[band]['sfc'],
                    color='#eb6834', alpha=0.15, lw=0)
    for name in ('ERA5 reference', 'sfc', 'atm'):
        ls = (0, (5, 3)) if name == 'ERA5 reference' else '-'
        ax.plot(months, ho[band][name].values, color=COLOR[name], lw=1.1,
                ls=ls, alpha=0.45, zorder=2)
        ax.plot(months, cv[band][name].values, color=COLOR[name], lw=2.4,
                ls=ls, marker='o', ms=4, markerfacecolor=COLOR[name],
                markeredgecolor='white', zorder=3)
        if ax is axes[1]:
            ax.annotate(name, xy=(12, cv[band][name].values[-1]),
                        xytext=(7, 0), textcoords='offset points',
                        va='center', fontsize=10, color=COLOR[name],
                        fontweight='bold', annotation_clip=False)
    ax.set_title(band, fontsize=11, color=INK)
    ax.set_xticks(months)
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O',
                        'N', 'D'], fontsize=9.5, color=INK)
    ax.tick_params(labelsize=9, colors=MUTED)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8)
    ax.set_axisbelow(True)

axes[0].set_ylabel('MAE (°C)', fontsize=10, color=INK)
axes[0].set_ylim(0, 5.2)
axes[0].text(0.02, 0.955, 'thick: CV holdout 2020–2023\nthin: 2024 holdout',
             transform=axes[0].transAxes, ha='left', va='top', fontsize=8.5,
             color=MUTED)
fig.subplots_adjust(left=0.055, right=0.85, top=0.92, bottom=0.09, wspace=0.08)
fig.savefig(OUT, dpi=200, facecolor='white')
print('wrote', OUT)
for reg, cur in (('cv', cv), ('2024', ho)):
    for band in BANDS:
        for name, m in cur[band].items():
            print(reg, band.split()[0], name, np.round(m.values, 2))
