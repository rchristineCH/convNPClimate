#!/usr/bin/env python3
"""Delivered spread against actual error, month by month, for the selected model.

The 2024 ensemble spread is the single-fold spread plus the disagreement of the
five fold means (moment matching). This figure compares, per calendar month of
2024, the quadratic mean of the delivered sigma, of its within-fold part alone,
and the RMSE of the actual errors. Annually the errors are the size of the
within-fold spread; the between-fold addition (shaded) is the surplus that
makes the delivered intervals too wide.

Data: pred_cache/holdout_2024.npz of the lbclip_wind run.
Writes LATEX_REPORT/images/tmax_sigma_vs_error.png.
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NPZ = ('/home/marc/convNPClimate/CLEAN_trained_models/'
       'lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8/'
       'tmax/pred_cache/holdout_2024.npz')
OUT = '/home/marc/convNPClimate-doc/LATEX_REPORT/images/tmax_sigma_vs_error.png'
INK, MUTED = '#1a1a19', '#5c5b55'
TOT_C, WIN_C = '#2a78d6', '#2a78d6'

z = np.load(NPZ, allow_pickle=True)
dates = pd.to_datetime(z['dates'])
err2 = (z['preds_degC'] - z['truths_degC']) ** 2
tot2 = z['sigmas_degC'] ** 2
win2 = z['sigma_within_degC'] ** 2


def qmean(a):
    s = pd.Series(np.nanmean(a, axis=1), index=dates)
    return np.sqrt(s.groupby(dates.month).mean())


rmse, s_tot, s_win = qmean(err2), qmean(tot2), qmean(win2)

fig, ax = plt.subplots(figsize=(8.6, 4.4))
months = np.arange(1, 13)
ax.fill_between(months, s_win.values, s_tot.values, color=TOT_C, alpha=0.15,
                lw=0)
ax.plot(months, s_tot.values, '-o', color=TOT_C, lw=2.2, ms=4.5,
        markerfacecolor=TOT_C, markeredgecolor='white', zorder=3)
ax.plot(months, s_win.values, color=WIN_C, lw=2.0, ls=(0, (5, 3)), zorder=3)
ax.plot(months, rmse.values, '-o', color=INK, lw=2.2, ms=4.5,
        markerfacecolor=INK, markeredgecolor='white', zorder=4)
for yv, label, color in ((s_tot.values[-1], 'delivered $\\sigma$', TOT_C),
                         (rmse.values[-1], 'actual error (RMSE)', INK),
                         (s_win.values[-1] - 0.05, 'within-fold $\\sigma$', TOT_C)):
    ax.annotate(label, xy=(12, yv), xytext=(8, 0), textcoords='offset points',
                va='center', fontsize=9.5, color=color, fontweight='bold',
                annotation_clip=False)
ax.text(6.5, (s_win.values[5] + s_tot.values[5]) / 2 + 0.09,
        'between-fold addition', ha='center', fontsize=8.5, color=TOT_C)

ax.set_xticks(months)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   fontsize=9.5, color=INK)
ax.set_ylabel('quadratic monthly mean (°C)', fontsize=10, color=INK)
ax.set_ylim(0.9, 2.0)
ax.tick_params(labelsize=9, colors=MUTED)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#e6e5e0', lw=0.8)
ax.set_axisbelow(True)
fig.subplots_adjust(left=0.08, right=0.80, top=0.96, bottom=0.10)
fig.savefig(OUT, dpi=200, facecolor='white')
print('wrote', OUT)
print('annual: rmse %.3f total %.3f within %.3f' % (
    np.sqrt(np.nanmean(err2)), np.sqrt(np.nanmean(tot2)),
    np.sqrt(np.nanmean(win2))))
