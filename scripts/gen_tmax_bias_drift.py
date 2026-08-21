#!/usr/bin/env python3
"""Monthly domain-mean bias of the selected tmax model, CV against 2024.

Visualises the warm drift of atm+wind-clip60 on the transfer to the unseen
year: monthly mean of (prediction - truth) over all valid grid points, for the
CV holdout 2020--2023 (each month backed by four years, single-fold
out-of-sample predictions) and for the 2024 holdout (five-fold ensemble).

Writes LATEX_REPORT/images/tmax_bias_drift.png.
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = ('/home/marc/convNPClimate/CLEAN_trained_models/'
        'lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8/tmax/pred_cache')
OUT = '/home/marc/convNPClimate-doc/LATEX_REPORT/images/tmax_bias_drift.png'
INK, MUTED = '#1a1a19', '#5c5b55'
CV_C, HO_C = '#2a78d6', '#eb6834'


def monthly_bias(fname):
    z = np.load(f'{BASE}/{fname}', allow_pickle=True)
    dates = pd.to_datetime(z['dates'])
    daily = np.nanmean(z['preds_degC'] - z['truths_degC'], axis=1)
    return pd.Series(daily, index=dates).groupby(dates.month).mean(), np.nanmean(daily)


cv, cv_mean = monthly_bias('cv.npz')
ho, ho_mean = monthly_bias('holdout_2024.npz')

fig, ax = plt.subplots(figsize=(8.6, 4.2))
months = np.arange(1, 13)
ax.axhline(0, color='#b9b8b2', lw=1)
for series, mean, color, label in ((cv, cv_mean, CV_C, 'CV holdout 2020–2023'),
                                   (ho, ho_mean, HO_C, '2024 holdout')):
    ax.plot(months, series.values, '-o', color=color, lw=2.2, ms=4.5,
            markerfacecolor=color, markeredgecolor='white', zorder=3)
    ax.axhline(mean, color=color, lw=1.1, ls=(0, (4, 3)), alpha=0.7)
    ax.annotate(f'{label}\nmean {mean:+.2f} °C', xy=(12, series.values[-1]),
                xytext=(8, 0), textcoords='offset points', va='center',
                fontsize=9.5, color=color, fontweight='bold',
                annotation_clip=False)

ax.set_xticks(months)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   fontsize=9.5, color=INK)
ax.set_ylabel('domain-mean bias (°C)', fontsize=10, color=INK)
ax.tick_params(labelsize=9, colors=MUTED)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#e6e5e0', lw=0.8)
ax.set_axisbelow(True)
fig.subplots_adjust(left=0.08, right=0.80, top=0.96, bottom=0.10)
fig.savefig(OUT, dpi=200, facecolor='white')
print('wrote', OUT)
print('cv monthly', np.round(cv.values, 2))
print('2024 monthly', np.round(ho.values, 2))
