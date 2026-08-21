#!/usr/bin/env python3
"""Temporal k-fold scheme of the clean runs, as an area-true timeline.

Four training years (2020--2023) are split into five contiguous blocks; each
fold trains on four blocks and holds out the fifth, giving five models that act
as an ensemble at evaluation. The year 2024 is never touched during training
and serves as the unseen final evaluation.

Run from the repository root; writes LATEX_REPORT/images/cv_scheme.png.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_FC, HOLD_FC, UNSEEN_FC = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED = '#1a1a19', '#5c5b55'
N_FOLDS = 5
YEARS = ['2020', '2021', '2022', '2023']

fig, ax = plt.subplots(figsize=(8.4, 2.9))
for fold in range(N_FOLDS):
    y = N_FOLDS - 1 - fold
    for b in range(N_FOLDS):
        fc = HOLD_FC if b == fold else TRAIN_FC
        ax.barh(y, 1 / N_FOLDS, left=b / N_FOLDS, height=0.62,
                color=fc, edgecolor='white', lw=1.6)
    ax.barh(y, 0.22, left=1.06, height=0.62, color=UNSEEN_FC,
            edgecolor='white', lw=1.6, alpha=0.35)
    ax.text(-0.015, y, f'fold {fold}', ha='right', va='center',
            fontsize=10, color=INK)

for i, yr in enumerate(YEARS):
    ax.text((i + 0.5) / len(YEARS), N_FOLDS - 0.28, yr, ha='center',
            fontsize=9, color=MUTED)
ax.text(1.17, N_FOLDS - 0.28, '2024', ha='center', fontsize=9, color=MUTED)

ax.text(0.5, -0.62, 'training years, five contiguous blocks', ha='center',
        fontsize=9.5, color=MUTED)
ax.text(1.17, -0.62, 'unseen year', ha='center', fontsize=9.5, color=MUTED)

handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=a)
           for c, a in ((TRAIN_FC, 1), (HOLD_FC, 1), (UNSEEN_FC, 0.35))]
ax.legend(handles, ['training days', 'held-out block (early stopping)',
                    'final evaluation only'],
          loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
          frameon=False, fontsize=9)
ax.set(xlim=(-0.12, 1.30), ylim=(-0.9, N_FOLDS + 0.15))
ax.axis('off')
fig.subplots_adjust(top=0.98, bottom=0.24, left=0.02, right=0.99)
fig.savefig('LATEX_REPORT/images/cv_scheme.png', dpi=200, facecolor='white')
print('wrote LATEX_REPORT/images/cv_scheme.png')
