#!/usr/bin/env python3
"""Objective-function training curves for the five precip runs.

The appendix figure that replaces the five-panel metric overlay of
``compare_precip_models.training_comparison``. It shows the objective and
nothing else, on two panels:

  left   each run's OWN held-out objective (``test NLL``, which for the runs
         trained on the CRPS loss is -CRPS, not a likelihood). The curves are
         therefore NOT cross-comparable between objectives; the panel shows
         convergence, not ranking, and each run is on its own scale.
  right  the held-out Bernoulli-Gamma CRPS, the one score every run shares and
         the criterion the fine-tuning stops on. This is the panel to compare
         on, and the marker on each curve is its best epoch.

MAE, Pearson and Spearman are deliberately absent: they are reported as results
in section 5.2, and repeating them as training curves invites reading a training
diagnostic as a score.

`precip-FT-CRPS` contributes only its ten fine-tune epochs, which start from the
converged `precip-NLL` model, so its epoch axis counts epochs after the warm
start and is drawn dashed.

Data: <run>/precip/stats.csv, fold-averaged per epoch (tracked in this repo).
Writes LATEX_REPORT/images/precip_training_curves.png.
"""
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / 'CLEAN_trained_models'
OUT = REPO / 'LATEX_REPORT/images/precip_training_curves.png'

RUNS = {
    'precip-NLL': 'clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8',
    'precip-CRPS': 'clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8',
    'precip-FT-CRPS': 'clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps',
    'precip-WIND': 'clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8',
    'precip-SFC-TP': 'clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8',
}
# colour by input family, line style by training objective — the conventions of
# figures 5.12 and 5.13, so the three precipitation figures read as one set.
STYLE = {
    'precip-NLL':     ('#2a78d6', '-'),
    'precip-CRPS':    ('#2a78d6', ':'),
    'precip-FT-CRPS': ('#2a78d6', '--'),
    'precip-WIND':    ('#1baf7a', '-'),
    'precip-SFC-TP':  ('#eb6834', '-'),
}
# The CRPS-trained runs log -CRPS in the same column the likelihood runs log NLL.
OWN_OBJECTIVE = {
    'precip-NLL': 'Bernoulli–Gamma NLL',
    'precip-CRPS': 'sampled CRPS',
    'precip-FT-CRPS': 'sampled CRPS (fine-tune)',
    'precip-WIND': 'Bernoulli–Gamma NLL',
    'precip-SFC-TP': 'Bernoulli–Gamma NLL',
}
INK, MUTED = '#1a1a19', '#5c5b55'

curves = {name: pd.read_csv(BASE / d / 'precip' / 'stats.csv').groupby('Epoch').mean(numeric_only=True)
          for name, d in RUNS.items()}

fig, (ax_obj, ax_crps) = plt.subplots(1, 2, figsize=(11.5, 4.6))

for name, df in curves.items():
    c, ls = STYLE[name]
    ax_obj.plot(df.index, df['test NLL'], ls, color=c, lw=1.8)
    crps = df['test CRPS']
    ax_crps.plot(df.index, crps, ls, color=c, lw=1.8)
    best = crps.idxmin()
    ax_crps.plot([best], [crps[best]], 'o', ms=6, color=c, markeredgecolor='white',
                 zorder=3)

# The best-epoch values go in one ranked block rather than five labels strewn
# along the curves, which collided at the right edge where four of them sit.
ranked = sorted(curves.items(), key=lambda kv: kv[1]['test CRPS'].min())
ax_crps.text(0.985, 0.97, 'best held-out CRPS', transform=ax_crps.transAxes,
             ha='right', va='top', fontsize=8.5, color=MUTED)
for i, (name, df) in enumerate(ranked):
    crps = df['test CRPS']
    ax_crps.text(0.985, 0.90 - 0.075 * i,
                 f'{name}  {crps.min():.3f}  (epoch {crps.idxmin()})',
                 transform=ax_crps.transAxes, ha='right', va='top',
                 fontsize=8.5, color=STYLE[name][0])

ax_obj.set_title("Each run's own held-out objective — not cross-comparable",
                 fontsize=10.5, color=INK)
ax_crps.set_title('Held-out Bernoulli–Gamma CRPS — the shared criterion',
                  fontsize=10.5, color=INK)
for ax in (ax_obj, ax_crps):
    ax.set_xlabel('epoch  (fine-tune epochs for precip-FT-CRPS)', fontsize=9,
                  color=MUTED)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    ax.grid(color='#e6e5e0', lw=0.8)
    ax.set_axisbelow(True)

handles = [plt.Line2D([], [], color=STYLE[n][0], ls=STYLE[n][1], lw=1.8,
                      label=f'{n} — {OWN_OBJECTIVE[n]}') for n in RUNS]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
           fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.5, 0.005),
           handlelength=2.6, columnspacing=2.0)
fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.30, wspace=0.22)
fig.savefig(OUT, dpi=200, facecolor='white')
print('wrote', OUT)
for name, df in curves.items():
    crps = df['test CRPS']
    print(f'  {name:16s} best CRPS {crps.min():.3f} at epoch {crps.idxmin()}')
