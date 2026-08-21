#!/usr/bin/env python3
"""Ensemble sigma decomposition on the 2024 holdout, all seven tmax models.

On the unseen year the five folds are combined by Gaussian moment matching,
sigma^2 = mean(sigma_k^2) + var(mu_k), so the delivered spread has a within-fold
part (what each fold predicts) and a between-fold part (how much the folds
disagree). This figure shows both parts per model, with the between share of the
total variance annotated.

Data: eval_2024/report_metrics.json -> sigma_decomposition of each run in
CLEAN_trained_models. Writes LATEX_REPORT/images/tmax_sigma_decomposition.png.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/marc/convNPClimate/CLEAN_trained_models'
OUT = '/home/marc/convNPClimate-doc/LATEX_REPORT/images/tmax_sigma_decomposition.png'

RUNS = [
    ('atm',             'clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8'),
    ('atm-clip60',      'lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8'),
    ('atm+wind',        'clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8'),
    ('atm+wind-clip60', 'lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8'),
    ('atm+sfcanchors',  'sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8'),
    ('sfc',             'baseline__tmax_sfc_flat_y2020-2023_e30f5_b8'),
    ('sfc-nogeo',       'baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo'),
]
WITHIN_FC, BETWEEN_FC = '#2a78d6', '#eb6834'
INK, MUTED = '#1a1a19', '#5c5b55'

labels, within, between, share = [], [], [], []
for label, run in RUNS:
    d = json.load(open(f'{BASE}/{run}/tmax/eval_2024/report_metrics.json'))
    s = d['sigma_decomposition']
    w, b = s['mean_sigma_within_degC'], s['mean_sigma_between_degC']
    labels.append(label)
    within.append(w)
    between.append(b)
    share.append(b ** 2 / (w ** 2 + b ** 2))

fig, ax = plt.subplots(figsize=(8.8, 4.2))
ypos = range(len(labels))[::-1]
h = 0.36
wmax = max(within)
for y, w, b, sh in zip(ypos, within, between, share):
    ax.barh(y + h / 2, w, height=h, color=WITHIN_FC)
    ax.barh(y - h / 2, b, height=h, color=BETWEEN_FC)
    ax.text(w + 0.02, y + h / 2, f'{w:.2f}', va='center', fontsize=8.5,
            color=WITHIN_FC)
    ax.text(b + 0.02, y - h / 2, f'{b:.2f}', va='center', fontsize=8.5,
            color=BETWEEN_FC)
    ax.text(wmax * 1.55, y, f'{100 * sh:.0f} % of variance between folds',
            va='center', ha='right', fontsize=8.5, color=MUTED)

ax.set_yticks(list(ypos))
ax.set_yticklabels(labels, fontsize=9.5, color=INK)
ax.set_xlim(0, wmax * 1.58)
ax.set_xticks([0, 0.5, 1.0, 1.5])
ax.set_xlabel('mean predictive spread on 2024 (°C)', fontsize=9.5, color=INK)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(axis='x', labelsize=8.5, colors=MUTED)
ax.tick_params(axis='y', length=0)
handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in (WITHIN_FC, BETWEEN_FC)]
fig.legend(handles, ['within folds  (mean of the five $\\sigma_k$)',
                     'between folds  (spread of the five $\\mu_k$)'],
           loc='lower center', bbox_to_anchor=(0.56, 0.02), ncol=2,
           frameon=False, fontsize=9)
fig.subplots_adjust(left=0.17, right=0.99, top=0.97, bottom=0.20)
fig.savefig(OUT, dpi=200, facecolor='white')
print('wrote', OUT)
