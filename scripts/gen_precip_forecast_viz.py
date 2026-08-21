#!/usr/bin/env python3
"""Wet/dry-day Bernoulli-Gamma forecast as an area-true visualisation.

Two forecasts from the same head as 100% probability bars: the block at exactly
zero is the dry atom, the blue bands are the shared Gamma body cut at 1, 10 and
40 mm. Only rho differs between the bars, which is the mechanism of the head.

The Gamma body is fitted to the observed wet-day mean (SDII) and 98th
percentile of the clean precipitation run's CV evaluation
(CLEAN_trained_models/clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8,
eval_precip_metrics.json); the two rho values are illustrative endpoints, not
measured. Run from the repository root; writes
LATEX_REPORT/images/precip_forecast_viz.png.
"""
import numpy as np
from scipy import optimize, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MEAN, Q98 = 9.894001960754395, 43.10695266723633
shape = optimize.brentq(lambda a: stats.gamma.ppf(0.98, a, scale=MEAN/a)-Q98, 0.2, 50.0)
scale = MEAN/shape
F = lambda x: stats.gamma.cdf(x, shape, scale=scale)

BANDS = [(0,1),(1,10),(10,40),(40,np.inf)]
BAND_LABELS = ['0–1 mm','1–10 mm','10–40 mm','$\\geq$40 mm']
DRY_FC = '#c9c8c0'
WET_FC = ['#cde2fb','#86b6ef','#3987e5','#1c5cab']
INK, MUTED = '#1a1a19', '#5c5b55'

def masses(rho):
    out=[1-rho]
    for a,b in BANDS:
        hi = 1.0 if np.isinf(b) else F(b)
        out.append(rho*(hi-F(a)))
    return out

CASES=[(0.10,'Dry forecast'),(0.90,'Wet forecast')]
fig,ax=plt.subplots(figsize=(9.2,3.1))
for row,(rho,name) in enumerate(CASES):
    y=1-row
    x=0.0
    for i,m in enumerate(masses(rho)):
        fc = DRY_FC if i==0 else WET_FC[i-1]
        ax.barh(y,m,left=x,height=0.52,color=fc,edgecolor='white',lw=2)
        if m>0.035:
            dark = i>=3 or i==0 and False
            ax.text(x+m/2,y,f'{100*m:.0f}%',ha='center',va='center',
                    color='white' if i>=3 else INK,fontsize=10,fontweight='bold')
        x+=m
    ax.text(-0.015,y,f'{name}\n$\\rho={rho:.2f}$',ha='right',va='center',fontsize=10.5,color=INK)
# category labels below the wet bar, whose segments are wide enough to name
x=0.0
for i,m in enumerate(masses(0.90)):
    lbl = 'exactly dry' if i==0 else BAND_LABELS[i-1]
    if m>0.05:
        ax.annotate(lbl,(x+m/2,-0.36),ha='center',fontsize=9,color=MUTED)
    elif i==len(BANDS):
        ax.annotate(lbl,(x+m/2,-0.36),ha='left',fontsize=9,color=MUTED,
                    xytext=(x+m/2+0.012,-0.36),textcoords='data')
    x+=m
# arrow between the bars: what moves
ax.annotate('', xy=(0.30,0.60), xytext=(0.30,0.40),
            arrowprops=dict(arrowstyle='<->',color=MUTED,lw=1.2))
ax.text(0.315,0.50,'only $\\rho$ moves: probability shifts between the dry block\n'
                  'and the same Gamma-shaped wet bands',
        va='center',fontsize=9,color=MUTED)
ax.set(xlim=(-0.16,1.02),ylim=(-0.55,1.45))
ax.axis('off')
ax.set_title('Two forecasts from the Bernoulli–Gamma head — area = probability',
             fontsize=12,color=INK,pad=10)
fig.tight_layout()
fig.savefig('LATEX_REPORT/images/precip_forecast_viz.png',dpi=200,facecolor='white')
print({n:[round(m,3) for m in masses(r)] for r,n in CASES})
