#!/usr/bin/env python3
"""CV -> 2024 slope chart for the five precip models — the headline metrics.

The precipitation counterpart of ``gen_tmax_regime_slope_figure.py``, in the same
visual language: each model is one line between its CV-holdout score (left) and
its 2024-holdout score (right), colour encodes the input family and line style
the training objective.

One thing differs from the temperature figure, and it is the reason the layout
gains a panel. The tmax reference sits at 2.5 to 3.4 °C against model scores of
0.7 to 2.0, so it is quoted under each panel rather than drawn; the precip
reference is *inside* the model range — bilinear ERA5-Land scores an MAE of
2.29 mm against 2.36 to 2.82 mm for the five runs — so here it is drawn as its
own grey line and the panels stay readable. The observed target, where a panel
has one, is the black rule.

Panels, all under both regimes:

  MAE (mm)      point accuracy, with the bilinear ERA5-Land reference drawn
  Bias (mm)     signed error, reference drawn, observed ideal at 0
  Spearman      rank correlation, reference drawn
  MAE skill     1 - MAE_model / MAE_ref, so the reference IS the zero line
  RPSS          five-category ranked probability skill, reference at zero

Data, all tracked in this repo — no prediction bundle and no GPU needed:
  <run>/precip/eval_precip_metrics[_2024].json        overall + baseline block
  <run>/precip/eval_figures[_2024]/precip_category_pooled.json   RPSS
  precip_processing_comparison/baseline_correlations.json  the reference's own
      Spearman/Pearson, R10 and P98 (not in the per-run baseline block)

Writes LATEX_REPORT/images/precip_regime_slope.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / 'CLEAN_trained_models'
OUT_DIR = REPO / 'LATEX_REPORT/images'

RUNS = {
    'precip-NLL': 'clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8',
    'precip-CRPS': 'clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8',
    'precip-FT-CRPS': 'clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps',
    'precip-WIND': 'clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8',
    'precip-SFC-TP': 'clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8',
}
# colour by input family, line style by training objective
STYLE = {
    'precip-NLL':     ('#2a78d6', '-'),
    'precip-CRPS':    ('#2a78d6', ':'),
    'precip-FT-CRPS': ('#2a78d6', '--'),
    'precip-WIND':    ('#1baf7a', '-'),
    'precip-SFC-TP':  ('#eb6834', '-'),
}
INK, MUTED, REF, OBS = '#1a1a19', '#5c5b55', '#8a8880', '#1a1a19'
REGIMES = ['cv', '2024']

# The per-run metric JSONs carry only four baseline quantities (MAE, bias, R01,
# SDII). The reference's own rank correlation, heavy-day frequency and tail come
# from this file, written by scripts/gen_precip_baseline_corr.py against the same
# w0606 series; load() asserts the MAE matches before using any of it.
BASELINE_CORR = json.loads(
    (BASE / 'precip_processing_comparison/baseline_correlations.json').read_text())


def load(run_dir, regime):
    """Every headline number of one run under one regime."""
    d = BASE / run_dir / 'precip'
    suffix = '' if regime == 'cv' else '_2024'
    m = json.loads((d / f'eval_precip_metrics{suffix}.json').read_text())
    o, b = m['overall'], m.get('baseline', {})
    fig_dir = d / ('eval_figures' if regime == 'cv' else 'eval_figures_2024')
    pooled = fig_dir / 'precip_category_pooled.json'
    # The baseline block stores occurrence as R01, i.e. already divided by the
    # observed wet-day frequency, so the reference frequency has to be put back
    # on the observed scale before it can share an axis with the runs.
    ref_r01 = b.get('baseline_R01_rel_wetday_freq')
    bc = BASELINE_CORR[regime]
    # Same reference series, so the MAEs must agree — but not bit-exactly:
    # precip-SFC-TP scores 1460 CV days against the atmospheric runs' 1461 (its
    # tp series ends 2023-12-30), which moves the reference MAE by ~1e-4 relative.
    assert abs(bc['baseline_mae_mm'] - b['baseline_mae_mm']) / b['baseline_mae_mm'] < 1e-3, (
        'baseline_correlations.json scores a different reference than this run')
    return {
        'mae': o['mae_mm'],
        'bias': o['bias_mm'],
        'spearman': o['spearman_pooled'],
        'wetfreq': o['wetday_freq_pred'],
        'sdii': o['SDII_pred_mm'],
        'skill': b.get('skill_mae'),
        'rpss': json.loads(pooled.read_text())['rpss'] if pooled.exists() else None,
        'ref_mae': b.get('baseline_mae_mm'),
        'ref_bias': b.get('baseline_bias_mm'),
        'ref_wetfreq': ref_r01 * o['wetday_freq_obs'] if ref_r01 is not None else None,
        'ref_sdii': b.get('baseline_SDII_mm'),
        'obs_wetfreq': o['wetday_freq_obs'],
        'obs_sdii': o['SDII_obs_mm'],
        # The four Vaughan indices, each as a predicted-over-observed ratio so
        # the ideal is 1 on every panel. R01 already is one; R10 and P98 take the
        # reference from BASELINE_CORR, the baseline block carrying neither.
        'r01': o['R01_rel_wetday_freq'],
        'sdii_ratio': o['SDII_pred_mm'] / o['SDII_obs_mm'],
        'r10_ratio': o['R10_freq_pred'] / o['R10_freq_obs'],
        'p98_ratio': o['P98_pred_mm'] / o['P98_obs_mm'],
        'ref_r01': ref_r01,
        'ref_sdii_ratio': (b['baseline_SDII_mm'] / o['SDII_obs_mm']
                           if 'baseline_SDII_mm' in b else None),
        'ref_spearman': bc['baseline_spearman_pooled'],
        'ref_r10_ratio': bc['baseline_R10_freq'] / o['R10_freq_obs'],
        'ref_p98_ratio': bc['baseline_P98_mm'] / o['P98_obs_mm'],
    }


# Two figures: the measured scores, then the two skill scores that are defined
# against the reference. Occurrence and intensity have an observed target that
# moves between the regimes, so it is drawn as its own black line rather than a
# fixed rule.
# (key, panel title, decimals, reference key, zero-line label, observed key)
ACCURACY = [
    ('mae',      'MAE (mm) — lower is better',      2, 'ref_mae',     None,       None),
    ('bias',     'Bias (mm) — 0 is ideal',          2, 'ref_bias',    'observed', None),
    ('spearman', 'Spearman (–) — higher is better', 3, 'ref_spearman', None,      None),
    ('wetfreq',  'Wet-day frequency (≥ 1 mm)',      3, 'ref_wetfreq', None,       'obs_wetfreq'),
    ('sdii',     'SDII (mm) — wet-day intensity',   2, 'ref_sdii',    None,       'obs_sdii'),
]
SKILL = [
    ('skill',    'MAE skill (–) — higher is better', 3, None, 'bilinear ERA5-Land reference', None),
    ('rpss',     'RPSS (–) — higher is better',      3, None, 'bilinear ERA5-Land reference', None),
]
INDICES = [
    ('r01',        'R01 — wet-day occurrence',   3, 'ref_r01',        'observed', None, 1.0),
    ('sdii_ratio', 'SDII — wet-day intensity',   3, 'ref_sdii_ratio', 'observed', None, 1.0),
    ('r10_ratio',  'R10 — heavy-day frequency',  3, 'ref_r10_ratio',  'observed', None, 1.0),
    ('p98_ratio',  'P98 — the tail',             3, 'ref_p98_ratio',  'observed', None, 1.0),
]
FIGURES = [
    (ACCURACY, 'precip_regime_slope.png', (14.0, 9.0), (2, 3)),
    (SKILL,    'precip_regime_slope_skill.png', (9.6, 5.0), (1, 2)),
    (INDICES,  'precip_regime_slope_indices.png', (10.5, 8.6), (2, 2)),
]

data = {(m, r): load(run, r) for m, run in RUNS.items() for r in REGIMES}
models = list(RUNS)


def spread(labels, min_gap):
    """Nudge overlapping right-hand label positions apart, top-down."""
    order = sorted(range(len(labels)), key=lambda i: -labels[i])
    for a, b in zip(order, order[1:]):
        if labels[a] - labels[b] < min_gap:
            labels[b] = labels[a] - min_gap
    return labels


HANDLES = [
    Line2D([], [], color='#2a78d6', lw=2, label='atmospheric $z$, $t$, $q$'),
    Line2D([], [], color='#1baf7a', lw=2, label='atmospheric + wind'),
    Line2D([], [], color='#eb6834', lw=2, label='surface (ERA5-Land tp + $t_{2m}$)'),
    Line2D([], [], color=MUTED, lw=2, ls='-',  label='mixture NLL objective'),
    Line2D([], [], color=MUTED, lw=2, ls=':',  label='CRPS from scratch'),
    Line2D([], [], color=MUTED, lw=2, ls='--', label='NLL, then CRPS fine-tune'),
]
REF_HANDLE = Line2D([], [], color=REF, lw=2, ls='-.',
                    label='bilinear ERA5-Land reference')
OBS_HANDLE = Line2D([], [], color=OBS, lw=2, ls='-',
                    label='observed (MeteoSwiss RhiresD)')
missing = []


def handles_for(panels):
    """Only legend what the figure actually draws as a series.

    The skill figure draws neither: its reference is the zero rule, labelled in
    the panel itself, and it has no observed target at all.
    """
    h = list(HANDLES)
    if any(p[3] for p in panels):
        h.append(REF_HANDLE)
    if any(p[5] for p in panels):
        h.append(OBS_HANDLE)
    return h


def render(panels, out_name, figsize, grid):
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.ravel()
    for ax, panel in zip(flat, panels):
        draw_panel(ax, *panel)
    for ax in flat[len(panels):]:               # unused cells carry nothing
        ax.axis('off')
    bottom = 0.28 if nrows == 1 else 0.15
    fig.legend(handles=handles_for(panels), loc="lower center", ncol=4, frameon=False,
               fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.5, 0.01),
               handlelength=2.4, columnspacing=2.2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=bottom,
                        wspace=0.62, hspace=0.45)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=200, facecolor='white')
    plt.close(fig)
    print('wrote', out)


def draw_panel(ax, metric, title, dec, ref_key, zero_label, obs_key=None, ideal=0.0):
    series = []                                  # (label, colour, style, y0, y1)
    for m in models:
        y0, y1 = data[(m, 'cv')][metric], data[(m, '2024')][metric]
        c, ls = STYLE[m]
        series.append((m, c, ls, y0, y1))
        if y1 is None:
            missing.append(f'{m} has no {metric.upper()} on 2024')
    if ref_key is not None:
        series.append(('bilinear ERA5-Land', REF, '-.',
                       data[(models[0], 'cv')][ref_key],
                       data[(models[0], '2024')][ref_key]))
    if obs_key is not None:
        series.append(('observed', OBS, '-',
                       data[(models[0], 'cv')][obs_key],
                       data[(models[0], '2024')][obs_key]))

    finite = [v for _, _, _, y0, y1 in series for v in (y0, y1) if v is not None]
    if zero_label is not None:
        finite.append(ideal)    # the ideal rule stretches the axis, so labels need more room
    # The panels are wide and short, so labels need a bigger slice of the axis
    # than the four-panel temperature figure gives them.
    gap = 0.075 * (max(finite) - min(finite))
    right_lab = spread([y1 if y1 is not None else y0 for _, _, _, y0, y1 in series], gap)
    left_lab = spread([y0 for _, _, _, y0, _ in series], gap)

    for (label, c, ls, y0, y1), ylab, y0lab in zip(series, right_lab, left_lab):
        if y1 is None:                            # CV-only: no line to draw
            ax.plot([0], [y0], 'o', ms=5, color=c, markeredgecolor='white', zorder=3)
            ax.annotate(f'{label}  2024 n/a', xy=(1, y0), xytext=(1.06, ylab),
                        va='center', fontsize=8.5, color=c, annotation_clip=False)
        else:
            ax.plot([0, 1], [y0, y1], ls, color=c, lw=2, marker='o', ms=5,
                    markerfacecolor=c, markeredgecolor='white', zorder=3)
            ax.annotate(f'{label}  {y1:.{dec}f}', xy=(1, y1), xytext=(1.06, ylab),
                        va='center', fontsize=8.5, color=c, annotation_clip=False)
        ax.annotate(f'{y0:.{dec}f}', xy=(0, y0), xytext=(-0.05, y0lab),
                    ha='right', va='center', fontsize=8.5, color=MUTED,
                    annotation_clip=False)

    if zero_label is not None:
        rule_color = REF if 'reference' in zero_label else OBS
        ax.axhline(ideal, color=rule_color, lw=1.2, ls='-.' if rule_color is REF else '-',
                   zorder=1)
        ax.annotate(zero_label, xy=(0, ideal), xytext=(0.02, ideal), va='bottom',
                    fontsize=7.5, color=rule_color)
    # spread() only pushes labels downwards, so a tight cluster (the wet-day
    # frequencies sit within 0.02 of each other) can end up under the axis.
    # Grow the axis to whatever the labels need instead of clipping them.
    span = finite + right_lab + left_lab
    pad = 0.05 * (max(span) - min(span))
    ax.set_ylim(min(span) - pad, max(span) + pad)
    ax.set_xlim(-0.32, 2.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['CV holdout\n2020–2023', '2024\nholdout'],
                       fontsize=9.5, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', labelsize=8.5, colors=MUTED, length=0)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8)
    ax.set_axisbelow(True)

for panels, out_name, figsize, grid in FIGURES:
    render(panels, out_name, figsize, grid)
for gap_note in sorted(set(missing)):
    print('  note:', gap_note)
for m in models:
    row = ' '.join(
        f'{k}={data[(m, r)][k]:+.3f}' if data[(m, r)][k] is not None else f'{k}=n/a'
        for r in REGIMES for k in ('mae', 'bias', 'spearman', 'skill', 'rpss'))
    print(f'  {m:16s} {row}')
