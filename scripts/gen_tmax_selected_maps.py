#!/usr/bin/env python3
"""Error and skill maps of the selected tmax model on the 2024 holdout.

Three panels from the cached prediction bundle of atm+wind-clip60: per-grid-point
MAE, per-grid-point CRPSS against the bilinear-ERA5 reference (Gaussian CRPS in
closed form, reference CRPS equal to its absolute error), and the grid points
above the 90th error percentile.

Writes LATEX_REPORT/images/tmax_wind_2024_maps.png.
"""
import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NPZ = ('/home/marc/convNPClimate/CLEAN_trained_models/'
       'lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8/'
       'tmax/pred_cache/holdout_2024.npz')
OUT = '/home/marc/convNPClimate-doc/LATEX_REPORT/images/tmax_wind_2024_maps.png'
INK, MUTED = '#1a1a19', '#5c5b55'

z = np.load(NPZ, allow_pickle=True)
mask = z['valid_mask'].astype(bool)
mu, sig = z['preds_degC'], z['sigmas_degC']
y, ref = z['truths_degC'], z['era5_ref_degC']
lat = z['lat'].reshape(mask.shape)
lon = z['lon'].reshape(mask.shape)

zz = (y - mu) / sig
crps = sig * (zz * (2 * norm.cdf(zz) - 1) + 2 * norm.pdf(zz) - 1 / np.sqrt(np.pi))
pix_crps = np.nanmean(crps, axis=0)
pix_ref = np.nanmean(np.abs(ref - y), axis=0)
pix_skill = 1 - pix_crps / pix_ref
pix_mae = np.nanmean(np.abs(mu - y), axis=0)
hi = pix_mae >= np.nanpercentile(pix_mae, 90)

glob = 1 - np.nanmean(crps) / np.nanmean(np.abs(ref - y))
print(f'global skill {glob:.3f} (report: 0.708) | median pixel skill '
      f'{np.nanmedian(pix_skill):.3f} | pixel skill range '
      f'{np.nanmin(pix_skill):.2f}..{np.nanmax(pix_skill):.2f}')


def to_grid(v):
    g = np.full(mask.size, np.nan)
    g[mask.ravel()] = v
    return g.reshape(mask.shape)


aspect = 1 / np.cos(np.deg2rad(float(np.nanmean(lat))))
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
panels = [
    (to_grid(pix_mae), 'YlOrRd', 'per-grid-point MAE (°C)',
     dict(vmax=np.nanpercentile(pix_mae, 98))),
    (to_grid(pix_skill), 'RdYlGn', 'per-grid-point skill (CRPSS)',
     dict(vmin=0, vmax=1)),
]
glob_mae = np.nanmean(np.abs(mu - y))
insets = [f'mean {glob_mae:.2f} °C', f'global {glob:.2f}']
for ax, (grid, cmap, title, kw), note in zip(axes, panels, insets):
    pc = ax.pcolormesh(lon, lat, grid, cmap=cmap, shading='nearest',
                       rasterized=True, **kw)
    cb = fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.text(0.03, 0.96, note, transform=ax.transAxes, ha='left', va='top',
            fontsize=9, color=INK,
            bbox=dict(facecolor='white', edgecolor='#c9c8c2',
                      boxstyle='round,pad=0.3'))

ax = axes[2]
gl = to_grid(np.ones(mask.sum()))
ax.pcolormesh(lon, lat, gl, cmap='Greys', vmin=0, vmax=4,
              shading='nearest', rasterized=True)
lat_v, lon_v = z['lat'][mask.ravel()], z['lon'][mask.ravel()]
ax.scatter(lon_v[hi], lat_v[hi], s=2.5, color='#c22e2e', rasterized=True)
ax.set_title('high-error grid points (P90)', fontsize=10.5, color=INK)

for ax in axes:
    ax.set_aspect(aspect)
    ax.set_xlabel('longitude (°E)', fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color('#c9c8c2')
axes[0].set_ylabel('latitude (°N)', fontsize=9, color=INK)

fig.subplots_adjust(left=0.045, right=0.99, top=0.93, bottom=0.11, wspace=0.16)
fig.savefig(OUT, dpi=200, facecolor='white')
print('wrote', OUT)
