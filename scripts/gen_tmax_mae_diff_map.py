#!/usr/bin/env python3
"""Map of the per-point MAE difference between two tmax runs.

Both runs' cached prediction bundles score the same 46,718 valid points of the
240x370 MeteoSwiss tmax grid, so the difference of their per-point MAE shows
where one input configuration pays off over the other. The map shows the *gain*
of model A over model B (MAE_B - MAE_A), so positive = A is better, binned into
the discrete improvement classes of the colour bar.

    python scripts/gen_tmax_mae_diff_map.py                    # atm-clip60 vs sfc, 2024
    python scripts/gen_tmax_mae_diff_map.py \
        --model-a atm+wind-clip60 --model-b atm-clip60 --regime cv \
        --out LATEX_REPORT/images/tmax_wind_gain_map.png

Data: pred_cache/{cv,holdout_2024}.npz of the chosen runs in CLEAN_trained_models.
"""
import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

BASE = Path('/home/marc/convNPClimate/CLEAN_trained_models')
DOC = Path('/home/marc/convNPClimate-doc')
RUNS = {
    'sfc': 'baseline__tmax_sfc_flat_y2020-2023_e30f5_b8',
    'atm': 'clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8',
    'atm-clip60': 'lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8',
    'atm+wind-clip60': 'lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8',
}
CACHE = {'cv': 'cv.npz', '2024': 'holdout_2024.npz'}
INK, MUTED = '#1a1a19', '#5c5b55'

# Discrete ΔMAE classes, in °C of MAE difference (model B − model A). The first
# band collects every point where A is worse by more than 0.05 °C; the colour
# bar is extended below it, so LOW only fixes where that band's tick sits.
LOW = -0.35
# Rescaled after the first run on the real bundles: with a median gain of +0.03 °C
# the ladder above 0.25 held 222 then 9 points (0.5 % and 0.0 % of 46 718 valid),
# so everything beyond 0.25 is one open-ended class and the 0.35...1.0 steps are
# gone. The decline side mirrors the improvement side in 0.1 °C steps, so a red
# region can be read at the same granularity as a blue one.
BOUNDS = [LOW, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 1.0]
BAND_NAMES = ['very strong decline', 'strong decline', 'decline', 'neutral',
              'improvement', 'strong improvement', 'very strong improvement']
NEUTRAL_COLOR = '#ecebe6'


def gain_scale():
    """Discrete colour map, norm and tick labels for the gain classes."""
    blues = plt.get_cmap('Blues')
    reds = plt.get_cmap('Reds')
    neutral_i = BOUNDS.index(-0.05)      # bands below this one are declines
    n_decline = neutral_i
    n_improve = len(BOUNDS) - 2 - neutral_i
    colors = ([reds(0.85 - 0.55 * i / (n_decline - 1)) for i in range(n_decline)]
              + [NEUTRAL_COLOR]
              + [blues(0.30 + 0.70 * i / (n_improve - 1)) for i in range(n_improve)])
    cmap = ListedColormap(colors)
    cmap.set_under(colors[0])
    cmap.set_bad('#f4f3ef')
    norm = BoundaryNorm(BOUNDS, cmap.N)

    def fmt(v):
        return f'{v:g}'.replace('-', '−')

    ticks, labels = [], []
    for i, name in enumerate(BAND_NAMES):
        lo, hi = BOUNDS[i], BOUNDS[i + 1]
        ticks.append(0.5 * (lo + hi))
        if i == 0:
            span = f'< {fmt(hi)}'
        elif i == len(BAND_NAMES) - 1:      # open-ended top class
            span = f'> {fmt(lo)}'
        else:
            span = f'{fmt(lo)} to {fmt(hi)}'
        labels.append(f'{span}   {name}'.rstrip())
    return cmap, norm, ticks, labels


def point_mae(label, regime):
    path = BASE / RUNS[label] / 'tmax' / 'pred_cache' / CACHE[regime]
    z = np.load(path, allow_pickle=True)
    mae = np.nanmean(np.abs(z['preds_degC'] - z['truths_degC']), axis=0)
    return mae, z['valid_mask'], z['lat'], z['lon']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-a', default='atm-clip60', choices=RUNS)
    ap.add_argument('--model-b', default='sfc', choices=RUNS)
    ap.add_argument('--regime', default='2024', choices=CACHE)
    ap.add_argument('--out', default='LATEX_REPORT/images/tmax_mae_diff_map.png')
    args = ap.parse_args()

    mae_a, mask_a, lat, lon = point_mae(args.model_a, args.regime)
    mae_b, mask_b, _, _ = point_mae(args.model_b, args.regime)
    assert np.array_equal(mask_a, mask_b), 'runs score different point sets'
    gain = mae_b - mae_a                 # positive = model A is better

    grid = np.full(mask_a.shape, np.nan)
    grid[mask_a.astype(bool)] = gain
    lat2d = lat.reshape(mask_a.shape)
    lon2d = lon.reshape(mask_a.shape)

    cmap, norm, ticks, labels = gain_scale()
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    pc = ax.pcolormesh(lon2d, lat2d, np.ma.masked_invalid(grid), cmap=cmap,
                       norm=norm, shading='nearest', rasterized=True)
    ax.set_aspect(1 / np.cos(np.deg2rad(float(np.nanmean(lat)))))
    ax.set_xlabel('longitude (°E)', fontsize=9.5, color=INK)
    ax.set_ylabel('latitude (°N)', fontsize=9.5, color=INK)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    for s in ax.spines.values():
        s.set_color('#c9c8c2')
    # The class names make for long tick labels, so the colour bar gets its own
    # axes and the map keeps the rest of the canvas.
    fig.subplots_adjust(left=0.07, right=0.70, top=0.93, bottom=0.09)
    cax = fig.add_axes([0.725, 0.12, 0.022, 0.76])
    cb = fig.colorbar(pc, cax=cax, extend='min', spacing='uniform', ticks=ticks)
    cax.set_title(f'ΔMAE (°C)\n{args.model_b} − {args.model_a}',
                  fontsize=9, color=INK, pad=8, loc='left')
    cb.ax.set_yticklabels(labels)
    cb.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    cb.outline.set_edgecolor('#c9c8c2')
    share = (gain > 0).mean()
    regime_name = 'CV holdout 2020–2023' if args.regime == 'cv' else '2024 holdout'
    ax.set_title(f'{regime_name} — {args.model_a} better (blue) at '
                 f'{100 * share:.0f} % of the {gain.size:,} points',
                 fontsize=10.5, color=INK)
    out = Path(args.out)
    if not out.is_absolute():
        out = DOC / out
    fig.savefig(out, dpi=200, facecolor='white')
    counts = np.histogram(gain, bins=[-np.inf] + BOUNDS[1:-1] + [np.inf])[0]
    print(f'wrote {out} | mean gain {gain.mean():+.3f} °C, median '
          f'{np.median(gain):+.3f}, {args.model_a} better at {100 * share:.1f} %')
    for label, n in zip(labels, counts):
        print(f'  {label:<28} {n:7,d}  ({100 * n / gain.size:5.1f} %)')


if __name__ == '__main__':
    main()
