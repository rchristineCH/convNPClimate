#!/usr/bin/env python3
"""Diagnostic evaluation figures for a trained precipitation (Bernoulli-Gamma) model.

``eval_precip.py`` writes the numeric metrics; this script renders the figures.
It obtains predictions via ``eval_precip.load_or_predict`` (persisted inference
state under ``<model_dir>/pred_cache/``), so inference matches the metrics
exactly and runs at most once per (model, regime). It writes the complete
diagnostic set under ``<model_dir>/eval_figures[_<year>]/``:

  0. grid_considerations.png         - configs + all grids (--grids-only, no inference)
  1. spatial_mean_obs_pred_bias.png  - observed / predicted / bias mean maps
  2. spatial_mae_spearman.png        - per-point MAE + Spearman skill maps
  3. spatial_wetday_freq.png         - observed / predicted wet-day freq + R01
  4. obs_vs_pred_density.png         - pooled daily obs-vs-pred density (hexbin)
  5. amount_distribution.png         - wet-day amount PDF: observed vs sampled
  6. qq_wetday_amounts.png           - Q-Q of wet-day amounts (obs vs sampled)
  7. pit_histogram.png               - randomized PIT histogram (Vaughan Fig. 10)
  8. metric_distributions_boxplots.png - per-point metric box plots (+ baseline)
  9. perfold_skill_bars.png          - per-fold NLL/MAE/Pearson/Spearman
 10. vaughan_comparison.png          - headline metrics vs Vaughan et al. reference
 11. spatial_wetday_freq_thresholds.png - P(Y>=t) maps + Brier skill vs ERA5-Land
 12. wetfreq_threshold_histograms.png   - pooled exceedance histograms per threshold
 13. precip_category_spatial.png        - intensity-category maps + per-category BSS
 14. precip_category_distribution.png   - pooled category distribution + RPSS
     (+ precip_category_pooled.json with the pooled numbers)
 15. sweep_vs_full_physical.png      - only with --sweep-glob: run vs OFAT sweep

Figures 11-14 also persist as ``<name>.fig.pkl`` + ``<name>.npz`` (editable saves).

Example
-------
    python eval_precip_figures.py --trial-dir trained_models/<run> \
        --sweep-glob 'trained_models/sweep_bg_*'
"""

import argparse
import glob as _glob
import json
import logging
import pickle
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import xarray as xr
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import BoundaryNorm, ListedColormap

import params as params_mod
import model_factory
import eval_precip as ep
from visualization import _save_fig

logger = logging.getLogger("eval_precip_figures")

# Subsample caps so the pooled-cloud figures stay light.
DENSITY_MAX = 300_000
SAMPLE_POINT_CAP = 400_000

# Daily accumulation thresholds (mm) for the multi-threshold spatial wet-day
# frequency figure, and the internal CATEGORY edges for the intensity-category
# evaluation (bins [0,0.1), [0.1,10), [10,40), [40,80), [80,inf)). Rendered by
# default; --wetfreq-thresholds overrides the edges.
WETFREQ_THRESHOLDS_MM = [0.1, 10.0, 40.0, 80.0]

# Vaughan et al. 2022 precipitation results (medians across the 86 VALUE stations,
# from docs/Vaughan-2022.pdf, Sec. 3 / Table 2). R01 and SDII bias are reported only
# as "comparable to the best VALUE baselines" (no number) -> treated as ~ideal with a
# note; 98P and R10 biases have explicit values. ideal = perfectly-unbiased target.
VAUGHAN_PRECIP = {
    "R01":       {"value": None,   "ideal": 1.0, "note": "~comparable to best VALUE"},
    "SDII_bias": {"value": None,   "ideal": 0.0, "note": "~comparable to best VALUE"},
    "P98_bias":  {"value": -2.04,  "ideal": 0.0, "note": "median across 86 stations"},
    "R10_bias":  {"value": -0.003, "ideal": 0.0, "note": "median across stations"},
}
_VAUGHAN_PANELS = [("R01", "R01 (wet-day freq ratio)"), ("SDII_bias", "SDII bias (mm)"),
                   ("P98_bias", "98th-pct bias (mm)"), ("R10_bias", "R10 freq bias")]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
# Swiss-border lat/lon box, matching the tmax TmaxD ``ch01r`` valid extent. The
# precip RhiresD ``ch01h`` product covers a buffer beyond the border (~0.2 deg
# further N into DE, ~0.1 deg further S into IT); clipping the *maps* to this box
# makes the precip silhouette comparable to the tmax maps. NOTE: this is display
# only — all reported metrics are still computed over the full RhiresD domain.
SWISS_LAT = (45.80, 47.82)
SWISS_LON = (5.93, 10.51)

_UNSET = object()  # sentinel for the per-B Swiss-mask cache (avoids array truthiness)


def _clip_to_swiss(vals, B):
    """Mask per-point values outside the Swiss-border lat/lon box (display only).

    Returns a copy with out-of-box points set to NaN, or ``vals`` unchanged if
    per-point coordinates are unavailable.
    """
    lat = getattr(B, "target_lat", None)
    lon = getattr(B, "target_lon", None)
    if lat is None or lon is None or len(lat) != vals.size:
        return vals
    inside = ((lat >= SWISS_LAT[0]) & (lat <= SWISS_LAT[1]) &
              (lon >= SWISS_LON[0]) & (lon <= SWISS_LON[1]))
    out = vals.astype(float, copy=True)
    out[~inside] = np.nan
    return out


def _index_map(parent, child, tol=1.0):
    """For each value in ``child``, the index of the matching ``parent`` cell.

    Returns an int array (len ``child``); entries with no match within ``tol``
    (LV95 metres) are -1. Used to align the TmaxD grid onto the RhiresD grid by
    coordinate value rather than assuming a fixed row offset.
    """
    parent = np.asarray(parent, dtype=float)
    order = np.argsort(parent)
    sp = parent[order]
    pos = np.clip(np.searchsorted(sp, child), 0, sp.size - 1)
    out = np.full(np.size(child), -1, dtype=int)
    for k, c in enumerate(np.atleast_1d(child)):
        j = pos[k]
        cand = [j] + ([j - 1] if j > 0 else [])
        best = min(cand, key=lambda i: abs(sp[i] - c))
        if abs(sp[best] - c) <= tol:
            out[k] = order[best]
    return out


def _swiss_domain_mask(B):
    """Flat (n_points,) bool mask of the TmaxD Swiss domain on the precip grid.

    The precip RhiresD ``ch01h`` field covers a buffer beyond the Swiss border;
    the tmax TmaxD ``ch01r`` field is masked to Switzerland. Both grids share the
    LV95 1 km lattice (identical E axis, N offset by a few cells), so the TmaxD
    valid mask aligns onto the precip grid by matching N/E coordinate values —
    giving the *exact* Swiss silhouette the tmax maps use. Cached on ``B``.
    Returns ``None`` (callers fall back to the lat/lon box) if it can't be built.
    """
    cached = getattr(B, "_swiss_mask_cache", _UNSET)
    if cached is not _UNSET:
        return cached
    mask = None
    try:
        gs = getattr(B, "grid_shape", None)
        p = getattr(B, "params", None)
        precip_glob = getattr(p, "METEO_SWISS_PRECIP_GLOB", None)
        tmax_glob = getattr(p, "METEO_SWISS_MAX_TEMP_GLOB", None)
        if gs is not None and precip_glob and tmax_glob:
            pds = xr.open_dataset(sorted(_glob.glob(str(precip_glob)))[0])
            tds = xr.open_dataset(sorted(_glob.glob(str(tmax_glob)))[0])
            pN, pE = np.asarray(pds["N"].values), np.asarray(pds["E"].values)
            tN, tE = np.asarray(tds["N"].values), np.asarray(tds["E"].values)
            tvar = next(v for v in tds.data_vars if "time" in tds[v].dims)
            tvalid = np.isfinite(np.asarray(tds[tvar].isel(time=0).values))  # (tN, tE)
            ri, ci = _index_map(pN, tN), _index_map(pE, tE)
            ok_r, ok_c = ri >= 0, ci >= 0
            full = np.zeros((pN.size, pE.size), dtype=bool)
            full[np.ix_(ri[ok_r], ci[ok_c])] = tvalid[np.ix_(ok_r, ok_c)]
            if full.shape == tuple(gs):
                mask = full.reshape(-1)
    except Exception as exc:  # pragma: no cover - fall back to lat/lon box
        logger.warning("could not build TmaxD Swiss mask (%r); using lat/lon box", exc)
        mask = None
    B._swiss_mask_cache = mask
    return mask


def _crop_to_valid(grid, pad=3):
    """Crop a (N, E) masked grid to its valid-data bounding box (+pad cells).

    Returns the cropped grid and the ``imshow`` extent (e0, e1, n0, n1) in
    original-index coordinates, so axis ticks still read true N/E indices.
    Falls back to the full grid if nothing (or everything) is valid.
    """
    valid = ~np.ma.getmaskarray(grid)
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return grid, (0, grid.shape[1], 0, grid.shape[0])
    n0 = max(int(rows[0]) - pad, 0); n1 = min(int(rows[-1]) + 1 + pad, grid.shape[0])
    e0 = max(int(cols[0]) - pad, 0); e1 = min(int(cols[-1]) + 1 + pad, grid.shape[1])
    return grid[n0:n1, e0:e1], (e0, e1, n0, n1)


def _grid_map(ax, vals, B, title, cmap, vmin=None, vmax=None, label="",
              norm=None, boundaries=None, extend="neither", ticks=None,
              bad_color="#f2f2f2", tick_labels=None):
    """Render a per-point field as an undistorted Swiss map.

    Reshapes the (n_points,) array to the MeteoSwiss (N, E) LV95 grid and
    ``imshow``s it (square 1 km cells, ``origin='lower'`` since the N axis runs
    south->north) — the same approach as the tmax evaluation, which avoids the
    lon/lat aspect distortion of a degree-space scatter. Points outside the Swiss
    domain are masked (using the TmaxD ``ch01r`` valid mask aligned onto the
    precip grid, or a lat/lon box if that's unavailable) and the grid cropped to
    the valid bbox, so the RhiresD ``ch01h`` map frames Switzerland like the
    tmax maps (display only; metrics use the full domain).
    Falls back to a cos(lat)-corrected scatter if the grid shape is unknown.

    Pass ``norm`` (e.g. a ``BoundaryNorm``) + ``boundaries`` for a DISCRETE
    (stepped) colour scale + colorbar; otherwise the continuous ``vmin/vmax``
    scale is used (default, so all existing callers are unaffected).
    """
    discrete = norm is not None
    cbar_kw = dict(shrink=0.8)
    if discrete:
        cbar_kw.update(boundaries=boundaries,
                       ticks=(ticks if ticks is not None else boundaries),
                       spacing="uniform", extend=extend)
    gs = getattr(B, "grid_shape", None)
    if gs is not None and vals.size == gs[0] * gs[1]:
        cmap_obj = (cmap.copy() if hasattr(cmap, "copy") else plt.get_cmap(cmap).copy())
        cmap_obj.set_bad(bad_color)  # masked / NaN (domain mask or undefined value)
        m = _swiss_domain_mask(B)
        if m is not None and m.size == vals.size:
            vals = vals.astype(float, copy=True)
            vals[~m] = np.nan
        else:
            vals = _clip_to_swiss(vals, B)
        grid = np.ma.masked_invalid(vals.reshape(gs))  # (N, E)
        grid, (e0, e1, n0, n1) = _crop_to_valid(grid, pad=3)
        imshow_kw = dict(origin="lower", cmap=cmap_obj, aspect="equal",
                         interpolation="nearest", extent=(e0, e1, n0, n1))
        if discrete:
            imshow_kw["norm"] = norm
        else:
            imshow_kw.update(vmin=vmin, vmax=vmax)
        im = ax.imshow(grid, **imshow_kw)
        ax.set_xlabel("E (easting index)"); ax.set_ylabel("N (northing index)")
        cb = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    else:
        valid = np.isfinite(vals)
        lon, lat = B.target_lon, B.target_lat
        sc_kw = dict(s=2, cmap=cmap, linewidths=0, rasterized=True)
        if discrete:
            sc_kw["norm"] = norm
        else:
            sc_kw.update(vmin=vmin, vmax=vmax)
        sc = ax.scatter(lon[valid], lat[valid], c=vals[valid], **sc_kw)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.nanmean(lat)))))
        cb = ax.figure.colorbar(sc, ax=ax, **cbar_kw)
    if tick_labels is not None:
        cb.set_ticklabels(tick_labels)
        cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)
    if label:
        cb.set_label(label, fontsize=8)


def _precip_valid_mask(B):
    """Flat (n_points,) bool: where the RhiresD field has data (CH + buffer)."""
    pds = xr.open_dataset(sorted(_glob.glob(str(B.params.METEO_SWISS_PRECIP_GLOB)))[0])
    pvar = next(v for v in pds.data_vars if "time" in pds[v].dims)
    return np.isfinite(np.asarray(pds[pvar].isel(time=0).values)).reshape(-1)


def _box_mask(B):
    """Flat (n_points,) bool for the old lat/lon Swiss box (for comparison)."""
    lat, lon = np.asarray(B.target_lat), np.asarray(B.target_lon)
    return ((lat >= SWISS_LAT[0]) & (lat <= SWISS_LAT[1]) &
            (lon >= SWISS_LON[0]) & (lon <= SWISS_LON[1]))


def _read_grid_extent(pattern):
    """Read dims + lat/lon bounding box from the first file matching ``pattern``.

    Handles both the MeteoSwiss 2-D (N, E) curvilinear lat/lon and the ERA5 1-D
    latitude/longitude, and repairs the geopotential file's swapped lat/lon
    variable names (it stores latitudes under ``longitude`` and vice-versa).
    """
    fs = sorted(_glob.glob(str(pattern)))
    if not fs:
        return None
    ds = xr.open_dataset(fs[0])

    def pick(*names):
        for n in names:
            if n in ds.variables:
                return np.asarray(ds[n].values, dtype=float)
        return None

    lat, lon = pick("lat", "latitude"), pick("lon", "longitude")
    if lat is None or lon is None:
        return None
    swapped = np.nanmean(lat) < 30 and np.nanmean(lon) > 30  # CH lat~47, lon~8
    if swapped:
        lat, lon = lon, lat
    dims = lat.shape if lat.ndim == 2 else (lat.size, lon.size)
    return {
        "files": len(fs), "dims": dims, "swapped": bool(swapped),
        "lat_min": float(np.nanmin(lat)), "lat_max": float(np.nanmax(lat)),
        "lon_min": float(np.nanmin(lon)), "lon_max": float(np.nanmax(lon)),
    }


def fig_grid_considerations(B, out_dir):
    """Standalone diagnostic summarising every grid/config the pipeline uses.

    Needs only ``B.params``, ``B.grid_shape`` and ``B.target_lat/lon`` — no model
    predictions — so it can be produced by the ``--grids-only`` fast path.
    """
    p = B.params
    pl_pat = str(Path(p.ERA5_PRESSURE_LEVEL_DIR) / "z" / "z_pl-*-06.nc")
    specs = [
        ("RhiresD ch01h (precip target)", p.METEO_SWISS_PRECIP_GLOB, "1 km LV95", "tab:blue"),
        ("TmaxD ch01r (tmax target)",      p.METEO_SWISS_MAX_TEMP_GLOB, "1 km LV95", "tab:green"),
        ("ERA5-Land t2m_max (surface/norm)", p.ERA5_MAX_TEMP_GLOB, "0.1 deg", "tab:orange"),
        ("ERA5-Land z (geopotential/elev)",  p.ERA5_GEOPOTENTIAL_GLOB, "0.1 deg", "tab:red"),
        ("ERA5 pressure levels (atmos ctx)", pl_pat, "0.25 deg", "tab:purple"),
    ]
    grids = []
    for name, pat, res, color in specs:
        info = _read_grid_extent(pat)
        if info:
            info.update(name=name, res=res, color=color)
            grids.append(info)

    gs = tuple(B.grid_shape)
    precip_valid = _precip_valid_mask(B).reshape(gs)
    tmask = _swiss_domain_mask(B)
    tmax_on_p = tmask.reshape(gs) if tmask is not None else np.zeros(gs, bool)
    box_on_p = _box_mask(B).reshape(gs) & precip_valid

    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # --- A. geographic extents -------------------------------------------------
    axA = axes[0, 0]
    for g in grids:
        axA.add_patch(Rectangle(
            (g["lon_min"], g["lat_min"]), g["lon_max"] - g["lon_min"],
            g["lat_max"] - g["lat_min"], fill=False, lw=2, edgecolor=g["color"],
            label=f"{g['name']}  [{g['dims'][0]}x{g['dims'][1]}, {g['res']}]"))
    axA.set_xlim(4.4, 11.6); axA.set_ylim(45.0, 48.6)
    axA.set_aspect(1.0 / np.cos(np.deg2rad(46.8)))
    axA.set_xlabel("lon"); axA.set_ylabel("lat")
    axA.set_title("A. Grid geographic extents (WGS84 bounding boxes)", fontsize=11)
    axA.legend(fontsize=7, loc="lower center", framealpha=0.9)
    axA.grid(alpha=0.3)

    # --- B. target domains on the RhiresD LV95 grid ----------------------------
    axB = axes[0, 1]
    cat = np.zeros(gs, dtype=int)
    cat[precip_valid & ~tmax_on_p] = 1   # precip-only buffer (dropped from maps)
    cat[precip_valid & tmax_on_p] = 2    # shared Swiss domain (TmaxD)
    axB.imshow(cat, origin="lower", aspect="equal", interpolation="nearest",
               cmap=ListedColormap(["#f2f2f2", "tab:orange", "tab:blue"]),
               vmin=0, vmax=2)
    n_ch = int((precip_valid & tmax_on_p).sum())
    n_buf = int((precip_valid & ~tmax_on_p).sum())
    axB.set_title(f"B. Target domains on RhiresD grid {gs[0]}x{gs[1]}\n"
                  f"blue = Swiss (TmaxD) {n_ch:,} cells · "
                  f"orange = precip-only buffer {n_buf:,} cells", fontsize=11)
    axB.set_xlabel("E index"); axB.set_ylabel("N index")

    # --- C. clip comparison: lat/lon box vs TmaxD mask -------------------------
    axC = axes[1, 0]
    catC = np.zeros(gs, dtype=int)
    catC[box_on_p & tmax_on_p] = 1            # kept by both
    catC[box_on_p & ~tmax_on_p] = 2           # box-only (wrongly kept buffer)
    catC[tmax_on_p & ~box_on_p] = 3           # mask-only (box wrongly dropped)
    axC.imshow(catC, origin="lower", aspect="equal", interpolation="nearest",
               cmap=ListedColormap(["#f2f2f2", "#9ecae1", "tab:red", "tab:green"]),
               vmin=0, vmax=3)
    axC.set_title("C. Clip comparison\nblue = both · red = box-only (over-includes)"
                  " · green = TmaxD-only (box drops)", fontsize=11)
    axC.set_xlabel("E index"); axC.set_ylabel("N index")

    # --- D. config + grid summary text -----------------------------------------
    axD = axes[1, 1]; axD.axis("off")
    cfg = [
        "MODEL CONFIG",
        f"  variable / dist : {p.VARIABLE} / {p.DISTRIBUTION}",
        f"  years           : {p.DATA_YEAR_START}-{p.DATA_YEAR_END}   folds: {p.N_FOLDS}",
        f"  encoder         : {p.ENCODER}   in_channels: {p.IN_CHANNELS}",
        f"  conv            : {p.N_CHANNELS}ch x {p.N_BLOCKS}blk k{p.KERNEL_SIZE}"
        f"   length_scale: {p.LENGTH_SCALE}",
        f"  atmos vars      : {','.join(p.ATMOS_VARIABLES)}  native_grid={p.ATMOS_NATIVE_GRID}",
        f"  atmos levels    : {p.ATMOS_LEVELS}",
        f"  atmos hours     : {p.ATMOS_HOURS}",
        f"  elevation/mTPI  : {p.USE_ELEVATION}/{p.USE_MTPI}   surface: {p.USE_SURFACE}",
        "",
        "GRIDS",
    ]
    for g in grids:
        note = "  (lat/lon names SWAPPED in file)" if g["swapped"] else ""
        cfg.append(f"  {g['name']}")
        cfg.append(f"      {g['dims'][0]}x{g['dims'][1]} @ {g['res']}  "
                   f"lat[{g['lat_min']:.2f},{g['lat_max']:.2f}] "
                   f"lon[{g['lon_min']:.2f},{g['lon_max']:.2f}]{note}")
    cfg += [
        "",
        "CONSIDERATIONS",
        "  - RhiresD (precip) & TmaxD (tmax) share the LV95 1km lattice:",
        "    identical E axis, N offset by a few cells -> exact mask reuse.",
        "  - Maps masked to the TmaxD Swiss domain (display only);",
        "    metrics still cover the full RhiresD domain incl. buffer.",
    ]
    axD.text(0.0, 1.0, "\n".join(cfg), va="top", ha="left", family="monospace",
             fontsize=8.5, transform=axD.transAxes)

    fig.suptitle("Grid considerations — configs & grids used by this run", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save_fig(fig, out_dir / "grid_considerations.png")


def _pooled_obs_pred(B):
    """Pooled (obs, predicted-mean) over evaluated days/points, valid only."""
    eval_params = B.params_full[B.day_mask]
    pred = B.get_value_fn(eval_params).numpy().reshape(-1)
    obs = B.truth_full[B.day_mask].reshape(-1)
    valid = np.isfinite(obs)
    return obs[valid], pred[valid], eval_params, valid


def _wetday_samples(B, valid):
    """Sampled accumulations (mm) over the valid pooled set (capped)."""
    eval_params = B.params_full[B.day_mask]
    flat = eval_params.reshape(-1, eval_params.shape[-1])[torch.from_numpy(valid)]
    n = flat.shape[0]
    if n > SAMPLE_POINT_CAP:
        g = torch.Generator().manual_seed(ep.PIT_SEED)
        idx = torch.randperm(n, generator=g)[:SAMPLE_POINT_CAP]
        flat = flat[idx]
    sampler = ep._SAMPLERS[B.distribution]
    return sampler(flat, ep.N_SAMPLES).numpy().reshape(-1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_spatial_means(pp, B, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax = np.nanpercentile(pp["mean_obs"], 99)
    _grid_map(axes[0], pp["mean_obs"], B, "Mean observed (mm/day)", "Blues", 0, vmax)
    _grid_map(axes[1], pp["mean_pred"], B, "Mean predicted (mm/day)", "Blues", 0, vmax)
    blim = np.nanpercentile(np.abs(pp["bias"]), 98)
    _grid_map(axes[2], pp["bias"], B, "Mean bias (pred - obs, mm/day)", "RdBu_r", -blim, blim)
    fig.suptitle("Time-mean precipitation: observed / predicted / bias", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, out_dir / "spatial_mean_obs_pred_bias.png")


def fig_spatial_mae_spearman(pp, B, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _grid_map(axes[0], pp["mae"], B, "Per-point MAE (mm/day)", "viridis",
              0, np.nanpercentile(pp["mae"], 99))
    _grid_map(axes[1], pp["spearman"], B, "Per-point Spearman (obs vs pred)", "RdYlGn", -1, 1)
    fig.suptitle("Per-point error and rank-correlation skill", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, out_dir / "spatial_mae_spearman.png")


def fig_spatial_wetfreq(pp, B, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _grid_map(axes[0], pp["wetfreq_obs"], B, "Wet-day freq (obs)", "YlGnBu", 0, 1)
    _grid_map(axes[1], pp["wetfreq_pred"], B, "Wet-day freq (pred, E[P(Y>=1mm)])", "YlGnBu", 0, 1)
    _grid_map(axes[2], pp["r01"], B, "R01 (pred:obs wet-day freq)", "RdBu_r", 0, 2)
    fig.suptitle("Wet-day frequency and R01 ratio", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, out_dir / "spatial_wetday_freq.png")


def _wetfreq_at_threshold(B, t, base=None):
    """Per-point (obs, ERA5, model, BSS) for daily accumulation >= ``t`` mm.

    MeteoSwiss (``B.truth_full``) is the ground truth throughout. Returns:
      ``obs_f``  : observed exceedance frequency (fraction of days obs >= t);
      ``era_f``  : ERA5-Land bilinear-baseline exceedance frequency
                   (fraction of covered days with ERA5 >= t), or NaN if no
                   baseline field;
      ``pred_f`` : model mean exceedance probability P(Y>=t) =
                   rho*(1-GammaCDF(t; alpha, scale=1/beta));
      ``bss``    : Brier skill of the model's daily P(Y>=t) vs the ERA5-Land
                   reference forecast 1{ERA5>=t} (or the point climatology when
                   ``base`` is None), scored on the binary MeteoSwiss event;
                   BSS = 1 - BS_model/BS_ref (1 perfect, 0 ties the reference,
                   <0 worse; NaN where the reference makes no error).
    ``base`` is the ERA5-Land daily field (D, P) aligned to the evaluated days.
    """
    eval_params = B.params_full[B.day_mask]          # torch (D, P, K)
    obs = B.truth_full[B.day_mask]                   # np   (D, P)
    P = obs.shape[1]
    point_valid = np.isfinite(obs).all(axis=0)
    obs_f = np.full(P, np.nan)
    era_f = np.full(P, np.nan)
    pred_f = np.full(P, np.nan)
    bss = np.full(P, np.nan)
    if not point_valid.any():
        return obs_f, era_f, pred_f, bss, point_valid

    o = obs[:, point_valid]                                       # (D, Pv)
    pv = eval_params[:, torch.from_numpy(point_valid), :].numpy()  # (D, Pv, K)
    rho, alpha, beta = pv[..., 0], pv[..., 1], pv[..., 2]
    surv = np.ones_like(rho) if t <= 0 else (1.0 - scipy_stats.gamma.cdf(t, a=alpha, scale=1.0 / beta))
    pexc = rho * surv                                            # P(Y>=t) per (day, point)
    o_evt = (o >= t).astype(np.float64)                          # binary obs event (D, Pv)

    obs_f[point_valid] = o_evt.mean(axis=0)
    pred_f[point_valid] = pexc.mean(axis=0)

    be = None
    if base is not None:
        b = base[:, point_valid]                                 # (D, Pv) ERA5 mm, may be NaN
        with np.errstate(invalid="ignore"):
            be = np.where(np.isfinite(b), (b >= t).astype(np.float64), np.nan)  # 1{ERA5>=t}
        w = np.isfinite(b).astype(float)
        denom = w.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            era_f[point_valid] = np.where(denom > 0, (w * np.nan_to_num(be)).sum(axis=0) / denom, np.nan)

    bss[point_valid] = _brier_skill(pexc, o_evt, be)
    return obs_f, era_f, pred_f, bss, point_valid


# Ratio (pred:obs) discrete diverging scheme. The centre bin [0.95, 1.05]
# ISOLATES the perfect score (ratio = 1) in a distinct colour; drier-than-obs
# (<1) run blue, wetter-than-obs (>1) run red, and ratio > 2 gets the over arrow.
_RATIO_BOUNDS = [0.0, 0.5, 0.8, 0.95, 1.05, 1.25, 1.5, 2.0]
_RATIO_COLORS = [
    "#08306b",  # [0.00,0.50) strong dry
    "#4292c6",  # [0.50,0.80) dry
    "#c6dbef",  # [0.80,0.95) slightly dry
    "#1a9850",  # [0.95,1.05) PERFECT -> distinct green
    "#fdae61",  # [1.05,1.25) slightly wet
    "#f46d43",  # [1.25,1.50) wet
    "#a50026",  # [1.50,2.00) strong wet
]


_ZERO_GREY = "#9e9e9e"   # exact-zero frequency (distinct from the light domain mask)
_ZERO_EPS = 1e-9


def _freq_discrete(fmax, step=0.1):
    """Fixed step-``step`` discrete colour scale + norm for the frequency maps.

    Shared 0 -> fmax (rounded up to a whole ``step``) across every threshold row,
    so the maps are directly comparable and literally use a 0.1 step. High
    thresholds (40-80 mm), where the wet-day frequency is far below 0.1, collapse
    into the lowest bin by construction.

    A TRUE-ZERO frequency (value exactly 0 -- e.g. a point that never exceeded t)
    gets its own distinct grey bin ``[0, eps)`` so it is visually separated from
    small-but-nonzero frequencies (and from the lighter out-of-domain grey).
    Returns (cmap, norm, boundaries, ticks): ``boundaries`` includes the tiny eps
    edge; ``ticks`` are the clean step boundaries for the colorbar.
    """
    top = max(step, np.ceil(max(fmax, 1e-9) / step) * step)
    ticks = np.round(np.arange(0.0, top + 0.5 * step, step), 4)
    if len(ticks) < 2:
        ticks = np.array([0.0, step])
    bounds = np.concatenate([[0.0, _ZERO_EPS], ticks[1:]])   # [0, eps, step, 2step, ...]
    colors = [_ZERO_GREY] + list(
        plt.get_cmap("YlGnBu")(np.linspace(0.12, 1.0, len(bounds) - 2)))
    cmap = ListedColormap(colors)
    cmap.set_bad("#f2f2f2")
    return cmap, BoundaryNorm(bounds, cmap.N), bounds, ticks


def _nice_step(vmax, target_bins=6):
    """A 1/2/2.5/5 x10^k step giving ~target_bins bins up to vmax."""
    if vmax <= 0:
        return 0.1
    raw = vmax / target_bins
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= raw:
            return round(m * mag, 10)
    return 10.0 * mag


def _ratio_discrete():
    cmap = ListedColormap(_RATIO_COLORS)
    cmap.set_over("#4d0013")   # ratio > 2 (severe over-prediction)
    cmap.set_bad("#f2f2f2")
    bounds = np.asarray(_RATIO_BOUNDS)
    return cmap, BoundaryNorm(bounds, cmap.N), bounds


# Brier-skill-score discrete diverging scale (skill vs the point's climatology).
# 1 = perfect, 0 = no better than the local base rate, negative = worse; the
# under-arrow catches BSS < -1, undefined points (no obs events) render grey.
# Discrete BSS scale. A bottom bin [-2,-1) absorbs "far worse than the reference"
# (real BSS is clipped into it), so the below-range (set_under) slot is free to
# mark UNDEFINED points in dark grey; out-of-domain stays light grey (set_bad).
_BSS_BOUNDS = np.array([-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
_BSS_COLORS = ["#a50026", "#d73027", "#f46d43", "#fee08b",
               "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850"]   # red -> green through 0
_BSS_UNDEF = -3.0      # sentinel (< -2) for undefined-in-domain -> set_under dark grey
_BSS_LABEL = "dark grey = undefined (reference makes no error) · light grey = outside domain"


def _bss_ticklabels(ref="ERA5-Land"):
    """Comprehensive interpretation scale printed next to a Brier-skill colorbar."""
    return ["≤−2\nfar worse", "−1", "−0.5\nworse", "−0.25", f"0\n= {ref}",
            "+0.25", "+0.5\nbetter", "+0.75", "+1\nperfect"]


def _skill_range(bss):
    """Title suffix with the UNCLIPPED min/median/max of a per-point skill field.

    The map itself clips into the discrete bins; this annotation preserves the
    true extremes (a min far below -2 signals how badly the worst point loses to
    the reference). Empty string if no finite value.
    """
    fin = bss[np.isfinite(bss)]
    if fin.size == 0:
        return ""
    return (f"\nskill min {fin.min():+.2f} · median {np.median(fin):+.2f}"
            f" · max {fin.max():+.2f}")


def _bss_for_plot(bss, point_valid):
    """Prepare a BSS field for display: clip into range; undefined-in-domain -> sentinel.

    Finite BSS is clipped to (-2, 1] (values below -2 fall into the 'far worse'
    bin). Points that are valid (in-domain, have obs) but whose BSS is undefined
    (the reference forecast makes no error, so there is nothing to beat) are set to
    a below-range sentinel that renders dark grey; genuinely out-of-domain points
    stay NaN and render light grey.
    """
    plot = np.full(bss.shape, np.nan)
    fin = np.isfinite(bss)
    plot[fin] = np.clip(bss[fin], -1.999, 1.0)
    plot[point_valid & ~fin] = _BSS_UNDEF
    return plot


_AUTO_BASE = object()   # sentinel: auto-load the ERA5-Land reference from disk

# ERA5-Land reference behind every skill panel in these figures.
#
# Defaults to the corrected, RhiresD-aligned rebuild (06-06 UTC) rather than
# datasets/ERA5_Land/precipitation/, where tp-2024.nc is ~1.53x inflated and
# day-shifted. The glob deliberately spans all years: bilinear_era5_precip
# reindexes onto the model's own day axis, so the extra years cost a little
# interpolation and select nothing. Override with PRECIP_BASELINE_GLOB if you
# need to reproduce an older figure against the legacy field.
ERA5_BASELINE_GLOB = os.environ.get("PRECIP_BASELINE_GLOB", "tp_hourly/w0606/tp-*.nc")


def _era5_baseline(B):
    """Bilinear-ERA5-Land daily precip (mm) aligned to the evaluated days, or None.

    The reference forecast for the Brier / ranked skill scores: the coarse
    ERA5-Land ``tp`` field interpolated to the target points. Returns
    ``(D_eval, P)`` on the evaluated day slice, or ``None`` if no ERA5 field is
    present (skill then falls back to climatology).
    """
    year = int(getattr(B, "eval_year", None) or B.params.DATA_YEAR_START)
    base, _src = ep.precip_baseline.bilinear_era5_precip(
        B.target_lat, B.target_lon, B.time_dates, year=year,
        glob=ERA5_BASELINE_GLOB)
    return None if base is None else base[B.day_mask]


def _brier_skill(pexc, o_evt, be):
    """Per-point Brier skill of prob. forecast ``pexc`` vs a reference, on obs ``o_evt``.

    All inputs are (D, Pv). If ``be`` (the ERA5-Land reference exceedance flag,
    same shape, may contain NaN for uncovered days/points) is given, the reference
    is the ERA5-Land deterministic forecast scored on the days it covers; else the
    reference is the point's climatology F_obs(1-F_obs). Returns the (Pv,) BSS
    (NaN where the reference makes no error, i.e. nothing to improve on).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if be is not None:
            w = np.isfinite(be).astype(float)                 # covered day mask
            denom = w.sum(axis=0)
            f_ref = np.where(np.isfinite(be), be, 0.0)         # 0/1 ERA5 forecast
            bs_model = (w * (pexc - o_evt) ** 2).sum(axis=0) / denom
            bs_ref = (w * (f_ref - o_evt) ** 2).sum(axis=0) / denom
            return np.where((denom > 0) & (bs_ref > 0), 1.0 - bs_model / bs_ref, np.nan)
        f = o_evt.mean(axis=0)
        bs_model = ((pexc - o_evt) ** 2).mean(axis=0)
        bs_ref = f * (1.0 - f)
        return np.where(bs_ref > 0, 1.0 - bs_model / bs_ref, np.nan)


def _bss_discrete():
    cmap = ListedColormap(_BSS_COLORS)
    cmap.set_under("#4d4d4d")   # sentinel < -2 : undefined (no reference error) -> dark grey
    cmap.set_bad("#f2f2f2")     # NaN : outside domain -> light grey
    return cmap, BoundaryNorm(_BSS_BOUNDS, cmap.N), _BSS_BOUNDS


def fig_spatial_wetfreq_thresholds(B, out_dir, thresholds=None, base=_AUTO_BASE):
    """Spatial wet-day frequency + Brier skill across accumulation thresholds.

    A leading "no rain" row (Y < lowest threshold; complement frequencies, skill
    identical to the lowest-threshold row by Brier-score complement invariance),
    then one row per threshold t (mm/day); columns = observed frequency, predicted
    frequency (analytic Bernoulli-Gamma exceedance P(Y>=t)), and the per-point
    BRIER SKILL SCORE of the model's daily exceedance probability against the
    binary MeteoSwiss event, referenced to the point's climatology (1 = perfect,
    0 = no better than the local base rate, <0 = worse; grey = no events). BSS
    replaces the old pred:obs ratio, which explodes as the observed frequency
    approaches zero at high thresholds.

    DISCRETE (stepped) legends: the frequency columns use a fixed 0.1-step scale
    shared across thresholds, EXCEPT a row whose values all fall within the first
    2 bands (row max <= 0.2) is rescaled to a finer per-row step so the
    high-threshold maps keep spatial detail (obs and pred still share the row
    scale). Each row is annotated with the number of observed exceedance events
    (pooled day x point) in the evaluated period -- a direct read on how
    sampling-limited the high thresholds are.
    """
    ts = [float(t) for t in (thresholds if thresholds else WETFREQ_THRESHOLDS_MM)]
    if base is _AUTO_BASE:
        base = _era5_baseline(B)                      # ERA5-Land reference field (or None)
    ref = "ERA5-Land" if base is not None else "climatology"
    rows = [_wetfreq_at_threshold(B, t, base=base) for t in ts]
    # Observed event counts (pooled over finite day x point) per threshold.
    obs_all = B.truth_full[B.day_mask]
    finite = np.isfinite(obs_all)
    n_pd = int(finite.sum())                       # total valid point-days
    counts = [int(np.sum(obs_all[finite] >= t)) for t in ts]

    # Leading "no rain" row: the complement of the lowest threshold. Frequencies
    # are 1 - exceedance; the Brier skill of the complementary event is IDENTICAL
    # (the Brier score is invariant under event complementation), so the skill
    # map reuses row ts[0]'s and says so in its title.
    of0, ef0, pf0, bss0, pv0 = rows[0]
    dry_row = (1.0 - of0, 1.0 - ef0, 1.0 - pf0, bss0, pv0)
    t0 = ts[0]
    panel_rows = [("dry", f"Y < {t0:g} mm", dry_row, n_pd - counts[0],
                   f"Brier skill vs {ref}  (Y < {t0:g} mm ≡ Y ≥ {t0:g} mm)")]
    panel_rows += [(f"{t:g}mm", f"Y ≥ {t:g} mm", r, c,
                    f"Brier skill vs {ref}  (Y ≥ {t:g} mm){_skill_range(r[3])}")
                   for t, r, c in zip(ts, rows, counts)]
    nrow = len(panel_rows)

    fmax = max([np.nanmax(a) for _, _, (of, ef, pf, _, _), _, _ in panel_rows
                for a in (of, ef, pf) if np.isfinite(a).any()] + [0.1])
    SHARED_STEP = 0.1
    shared = _freq_discrete(fmax, step=SHARED_STEP)   # (cmap, norm, bounds, ticks)
    bcmap, bnorm, bbounds = _bss_discrete()
    bss_ticklabels = _bss_ticklabels(ref)

    fig, axes = plt.subplots(nrow, 4, figsize=(24, 5 * nrow), squeeze=False)
    for i, (tag, evt_lab, (of, ef, pf, bss, pvalid), n_evt, skill_title) in enumerate(panel_rows):
        row_max = max([np.nanmax(a) for a in (of, ef, pf) if np.isfinite(a).any()] + [0.0])
        # Rescale this row's frequency legend when every value falls within the
        # first 2 bands of the shared 0.1-step scale (row_max <= 0.2), so the
        # otherwise-uniform high-threshold maps regain spatial detail (columns
        # share the row scale; cross-row comparability is traded away).
        if 0 < row_max <= 2 * SHARED_STEP:
            fcmap, fnorm, fbounds, fticks = _freq_discrete(row_max, step=_nice_step(row_max))
        else:
            fcmap, fnorm, fbounds, fticks = shared
        evt = f"\nN={n_evt:,} obs events ({n_evt / max(n_pd, 1):.2%} of point-days)"
        _grid_map(axes[i][0], of, B, f"MeteoSwiss obs freq  ({evt_lab}){evt}",
                  fcmap, label="", norm=fnorm, boundaries=fbounds, extend="neither", ticks=fticks)
        _grid_map(axes[i][1], ef, B, f"ERA5-Land freq  ({evt_lab})",
                  fcmap, label="", norm=fnorm, boundaries=fbounds, extend="neither", ticks=fticks)
        _grid_map(axes[i][2], pf, B, f"Model pred  (P({evt_lab}))",
                  fcmap, label="", norm=fnorm, boundaries=fbounds, extend="neither", ticks=fticks)
        _grid_map(axes[i][3], _bss_for_plot(bss, pvalid), B, skill_title,
                  bcmap, label=_BSS_LABEL, norm=bnorm, boundaries=bbounds, extend="min",
                  tick_labels=bss_ticklabels)
    fig.tight_layout()
    _save_fig(fig, out_dir / "spatial_wetday_freq_thresholds.png")
    data = {"thresholds": np.asarray(ts, np.float32),
            "counts": np.asarray(counts, np.int64),
            "n_point_days": np.asarray(n_pd, np.int64),
            "reference": np.asarray(ref),
            "grid_shape": np.asarray(getattr(B, "grid_shape", None) or (-1, -1))}
    for tag, _, (of, ef, pf, bss, _), _, _ in panel_rows:
        data[f"obs_{tag}"] = of.astype(np.float32)
        data[f"era_{tag}"] = ef.astype(np.float32)
        data[f"model_{tag}"] = pf.astype(np.float32)
        data[f"bss_{tag}"] = bss.astype(np.float32)
    _save_editable(fig, out_dir / "spatial_wetday_freq_thresholds.png", data)


# ---------------------------------------------------------------------------
# Precipitation-intensity CATEGORY evaluation (0-0.1, 0.1-10, ... mm) + RPSS
# ---------------------------------------------------------------------------
def _save_editable(fig, png_path, data=None):
    """Persist an editable copy of the figure next to its PNG.

    ``<name>.fig.pkl``: the live matplotlib Figure, pickled -- reload with
    ``fig = pickle.load(open(p, 'rb'))``, tweak titles/labels/layout, then
    ``fig.savefig(...)`` -- no recomputation (matplotlib-version-sensitive).
    ``<name>.npz``    : the plotted arrays (float32), for full rebuilds with a
    different layout even across matplotlib versions.
    """
    p = Path(png_path)
    try:
        with open(p.with_suffix(".fig.pkl"), "wb") as fh:
            pickle.dump(fig, fh)
    except Exception as exc:  # never let persistence break the render
        logger.warning("could not pickle figure %s: %r", p.name, exc)
    if data:
        try:
            np.savez_compressed(p.with_suffix(".npz"), **data)
        except Exception as exc:
            logger.warning("could not save %s.npz: %r", p.stem, exc)


def _cat_labels(cat_edges):
    labs = []
    for a, b in zip(cat_edges[:-1], cat_edges[1:]):
        labs.append(f"{a:g}-{b:g}" if not np.isinf(b) else f">={a:g}")
    return labs


def _precip_category_eval(B, edges, base=_AUTO_BASE):
    """Per-point intensity-category evaluation of the model vs MeteoSwiss + RPSS.

    Categories are the disjoint bins [0, e1), ..., [eK, inf) from the internal
    edges ``edges`` (mm) -- the "precipitation behaviours". MeteoSwiss is ground
    truth; the ERA5-Land bilinear baseline is the reference forecast. Returns, per
    target point: observed / ERA5 / model category frequency (P, C); per-category
    Brier skill of the model's daily category probability vs the ERA5-Land
    reference forecast 1{ERA5 in cat} (or climatology when no ERA5 field); and the
    per-point RANKED PROBABILITY SKILL SCORE, RPSS = 1 - RPS_model/RPS_ref over the
    ordered category boundaries. Also returns pooled RPSS + domain-mean category
    frequencies (obs / ERA5 / model) for the pooled distribution figure.
    """
    eval_params = B.params_full[B.day_mask]
    obs = B.truth_full[B.day_mask]
    P = obs.shape[1]
    pv = np.isfinite(obs).all(axis=0)
    if base is _AUTO_BASE:
        base = _era5_baseline(B)
    edges = [float(e) for e in edges]
    cat_edges = [0.0] + edges + [np.inf]
    C = len(cat_edges) - 1
    labels = _cat_labels(cat_edges)
    out = dict(obs_cat=np.full((P, C), np.nan), era_cat=np.full((P, C), np.nan),
               model_cat=np.full((P, C), np.nan), bss_cat=np.full((P, C), np.nan),
               rpss=np.full(P, np.nan), point_valid=pv, cat_labels=labels,
               reference=("ERA5-Land" if base is not None else "climatology"), pooled={})
    if not pv.any():
        return out

    o = obs[:, pv]                                              # (D, Pv)
    prm = eval_params[:, torch.from_numpy(pv), :].numpy()       # (D, Pv, K)
    rho, alpha, beta = prm[..., 0], prm[..., 1], prm[..., 2]
    n_pd = int(np.isfinite(obs).sum())
    be = base[:, pv] if base is not None else None              # (D, Pv) ERA5 mm or None
    w = np.isfinite(be).astype(float) if be is not None else None
    denom = w.sum(axis=0) if w is not None else None

    def Pge(e):
        if e <= 0:
            return np.ones_like(rho)
        return rho * (1.0 - scipy_stats.gamma.cdf(e, a=alpha, scale=1.0 / beta))

    npv = int(pv.sum())
    obs_cat_v = np.full((npv, C), np.nan)
    era_cat_v = np.full((npv, C), np.nan)
    model_cat_v = np.full((npv, C), np.nan)
    bss_cat_v = np.full((npv, C), np.nan)
    rps_model = np.zeros(npv)
    rps_ref = np.zeros(npv)
    counts = []
    prev = Pge(0.0)                                            # P(Y>=0)=1
    for c in range(C):
        upper = cat_edges[c + 1]
        Pge_up = np.zeros_like(rho) if np.isinf(upper) else Pge(upper)
        Pc = prev - Pge_up                                    # (D, Pv) model prob cat c
        lo = cat_edges[c]
        obs_in = ((o >= lo) & (o < upper)).astype(np.float64)  # (D, Pv)
        model_cat_v[:, c] = Pc.mean(axis=0)
        f_c = obs_in.mean(axis=0)
        obs_cat_v[:, c] = f_c
        counts.append(int(obs_in.sum()))
        with np.errstate(divide="ignore", invalid="ignore"):
            if be is not None:
                era_in = np.where(np.isfinite(be), ((be >= lo) & (be < upper)).astype(np.float64), np.nan)
                era_cat_v[:, c] = np.where(denom > 0, (w * np.nan_to_num(era_in)).sum(axis=0) / denom, np.nan)
                bss_cat_v[:, c] = _brier_skill(Pc, obs_in, era_in)
            else:
                bs_m = ((Pc - obs_in) ** 2).mean(axis=0)
                bs_r = f_c * (1.0 - f_c)
                bss_cat_v[:, c] = np.where(bs_r > 0, 1.0 - bs_m / bs_r, np.nan)
        if not np.isinf(upper):                               # internal boundary -> RPS term
            Fk = 1.0 - Pge_up                                 # model forecast CDF at edge
            Ok = (o < upper).astype(np.float64)               # obs CDF indicator
            if be is not None:
                era_cdf = np.where(np.isfinite(be), (be < upper).astype(np.float64), 0.0)
                rps_model += (w * (Fk - Ok) ** 2).sum(axis=0)
                rps_ref += (w * (era_cdf - Ok) ** 2).sum(axis=0)
            else:
                rps_model += ((Fk - Ok) ** 2).mean(axis=0)
                ocdf = Ok.mean(axis=0)
                rps_ref += ocdf * (1.0 - ocdf)
        prev = Pge_up

    with np.errstate(divide="ignore", invalid="ignore"):
        if be is not None:
            rpss_v = np.where((denom > 0) & (rps_ref > 0), 1.0 - rps_model / rps_ref, np.nan)
            pooled_rpss = float(1.0 - rps_model.sum() / rps_ref.sum()) if rps_ref.sum() > 0 else float("nan")
        else:
            rpss_v = np.where(rps_ref > 0, 1.0 - rps_model / rps_ref, np.nan)
            pooled_rpss = float(1.0 - rps_model.sum() / rps_ref.sum()) if rps_ref.sum() > 0 else float("nan")

    out["obs_cat"][pv] = obs_cat_v
    out["era_cat"][pv] = era_cat_v
    out["model_cat"][pv] = model_cat_v
    out["bss_cat"][pv] = bss_cat_v
    out["rpss"][pv] = rpss_v
    out["pooled"] = {
        "rpss": pooled_rpss,
        "reference": out["reference"],
        "obs_freq": np.nanmean(obs_cat_v, axis=0).tolist(),
        "era_freq": np.nanmean(era_cat_v, axis=0).tolist() if be is not None else None,
        "model_freq": np.nanmean(model_cat_v, axis=0).tolist(),
        "obs_event_counts": counts,
        "n_point_days": n_pd,
        "cat_labels": labels,
    }
    return out


def fig_precip_category_distribution(cat, out_dir, name=""):
    """Pooled intensity-category distribution: observed vs model, + RPSS.

    Domain-mean fraction of days in each precip category (0-0.1, 0.1-1, ... mm),
    observed vs the model's mean predicted category probability -- the pooled
    "how often does each precipitation behaviour occur" check -- with the ranked
    probability skill score (ordered multi-category skill vs climatology).
    """
    labels = cat["cat_labels"]
    of = np.asarray(cat["pooled"]["obs_freq"])
    mf = np.asarray(cat["pooled"]["model_freq"])
    ef = cat["pooled"].get("era_freq")
    ref = cat["pooled"].get("reference", "climatology")
    counts = cat["pooled"]["obs_event_counts"]
    x = np.arange(len(labels))
    have_era = ef is not None
    fig, ax = plt.subplots(figsize=(12, 5.5))
    if have_era:
        ax.bar(x - 0.28, of, width=0.28, color="k", label="MeteoSwiss (obs)")
        ax.bar(x, np.asarray(ef), width=0.28, color="#d95f02", label="ERA5-Land")
        ax.bar(x + 0.28, mf, width=0.28, color="C0", label="model")
    else:
        ax.bar(x - 0.2, of, width=0.4, color="k", label="MeteoSwiss (obs)")
        ax.bar(x + 0.2, mf, width=0.4, color="C0", label="model")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nN={c:,}" for l, c in zip(labels, counts)], fontsize=8)
    ax.set_xlabel("daily precipitation category (mm)")
    ax.set_ylabel("domain-mean frequency (fraction of days, log)")
    ax.set_title(f"Pooled precipitation-intensity distribution: MeteoSwiss / ERA5-Land / model"
                 f"{(' — ' + name) if name else ''}\n"
                 f"ranked probability skill score RPSS = {cat['pooled']['rpss']:+.3f} "
                 f"(1 perfect, 0 = ties {ref})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save_fig(fig, out_dir / "precip_category_distribution.png")
    _save_editable(fig, out_dir / "precip_category_distribution.png", {
        "cat_labels": np.asarray(labels),
        "obs_freq": np.asarray(of, np.float32),
        "era_freq": np.asarray(ef if ef is not None else [], np.float32),
        "model_freq": np.asarray(mf, np.float32),
        "counts": np.asarray(counts, np.int64),
        "rpss": np.asarray(cat["pooled"]["rpss"], np.float32),
        "reference": np.asarray(ref)})


def fig_precip_category_spatial(cat, B, out_dir):
    """Spatial maps of the precip-intensity categories: obs / ERA5 / model / Brier skill.

    One row per category (0-0.1, 0.1-1, ... mm); columns = MeteoSwiss observed
    frequency, ERA5-Land frequency, model mean category probability, and the
    per-category Brier skill of the model vs the ERA5-Land reference forecast.
    Frequency columns share the fixed 0.1-step discrete scale (low rows rescaled,
    true-zero grey); the skill column uses the diverging BSS scale (light grey =
    outside domain, dark grey = undefined).
    """
    labels = cat["cat_labels"]
    C = len(labels)
    obs_cat, era_cat, model_cat = cat["obs_cat"], cat["era_cat"], cat["model_cat"]
    bss_cat, pv = cat["bss_cat"], cat["point_valid"]
    ref = cat.get("reference", "climatology")
    have_era = np.isfinite(era_cat).any()
    counts = cat["pooled"]["obs_event_counts"]
    n_pd = cat["pooled"]["n_point_days"]

    cols = 4 if have_era else 3
    fmax = max([np.nanmax(a[:, c]) for a in (obs_cat, era_cat, model_cat)
                for c in range(C) if np.isfinite(a[:, c]).any()] + [0.1])
    SHARED_STEP = 0.1
    shared = _freq_discrete(fmax, step=SHARED_STEP)
    bcmap, bnorm, bbounds = _bss_discrete()
    bss_ticklabels = _bss_ticklabels(ref)

    fig, axes = plt.subplots(C, cols, figsize=(6 * cols, 5 * C), squeeze=False)
    for c in range(C):
        of, ef, mf, bs = obs_cat[:, c], era_cat[:, c], model_cat[:, c], bss_cat[:, c]
        row_max = max([np.nanmax(a) for a in (of, ef, mf) if np.isfinite(a).any()] + [0.0])
        if 0 < row_max <= 2 * SHARED_STEP:
            fcmap, fnorm, fbounds, fticks = _freq_discrete(row_max, step=_nice_step(row_max))
        else:
            fcmap, fnorm, fbounds, fticks = shared
        n_evt = counts[c]
        evt = f"\nN={n_evt:,} obs events ({n_evt / max(n_pd, 1):.2%} of point-days)"
        j = 0
        _grid_map(axes[c][j], of, B, f"MeteoSwiss obs freq  (Y in {labels[c]} mm){evt}",
                  fcmap, label="", norm=fnorm, boundaries=fbounds, extend="neither", ticks=fticks); j += 1
        if have_era:
            _grid_map(axes[c][j], ef, B, f"ERA5-Land freq  (Y in {labels[c]} mm)",
                      fcmap, label="", norm=fnorm, boundaries=fbounds, extend="neither", ticks=fticks); j += 1
        _grid_map(axes[c][j], mf, B, f"Model pred  (P(Y in {labels[c]} mm))",
                  fcmap, label="", norm=fnorm, boundaries=fbounds, extend="neither", ticks=fticks); j += 1
        _grid_map(axes[c][j], _bss_for_plot(bs, pv), B,
                  f"Brier skill vs {ref}  (Y in {labels[c]} mm){_skill_range(bs)}",
                  bcmap, label=_BSS_LABEL, norm=bnorm, boundaries=bbounds, extend="min",
                  tick_labels=bss_ticklabels)
    fig.suptitle(f"Spatial precipitation-intensity categories: MeteoSwiss / ERA5-Land / model / "
                 f"Brier skill vs {ref}  (RPSS = {cat['pooled']['rpss']:+.3f})", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out_dir / "precip_category_spatial.png")
    _save_editable(fig, out_dir / "precip_category_spatial.png", {
        "cat_labels": np.asarray(labels),
        "obs_cat": obs_cat.astype(np.float32),
        "era_cat": era_cat.astype(np.float32),
        "model_cat": model_cat.astype(np.float32),
        "bss_cat": bss_cat.astype(np.float32),
        "rpss": cat["rpss"].astype(np.float32),
        "counts": np.asarray(counts, np.int64),
        "reference": np.asarray(ref),
        "grid_shape": np.asarray(getattr(B, "grid_shape", None) or (-1, -1))})


def _pooled_wetfreq_samples(B):
    """Matched pooled subsample for the histogram-based exceedance figure.

    Returns (obs_sub, flat, samples): the observed daily accumulations, the
    predicted Bernoulli-Gamma params, and Monte-Carlo accumulation draws
    (``N_SAMPLES`` per point, including the dry atom -> zeros) -- all on the
    *same* random subsample of valid pooled point-days (cap ``SAMPLE_POINT_CAP``,
    seed ``ep.PIT_SEED``) so the empirical/analytic/sampled exceedance curves are
    directly comparable. Same sampler the amount/QQ figures use (``ep._SAMPLERS``).
    """
    obs_all, _, eval_params, valid = _pooled_obs_pred(B)
    flat_all = eval_params.reshape(-1, eval_params.shape[-1])[torch.from_numpy(valid)]
    n = flat_all.shape[0]
    if n > SAMPLE_POINT_CAP:
        g = torch.Generator().manual_seed(ep.PIT_SEED)
        idx = torch.randperm(n, generator=g)[:SAMPLE_POINT_CAP]
        flat = flat_all[idx]
        obs_sub = obs_all[idx.numpy()]
    else:
        flat, obs_sub = flat_all, obs_all
    sampler = ep._SAMPLERS[B.distribution]
    samples = sampler(flat, ep.N_SAMPLES).numpy().reshape(-1)
    return obs_sub, flat.numpy(), samples


def fig_wetfreq_threshold_histograms(B, out_dir, thresholds=None):
    """Wet-day frequency across thresholds, computed from histograms (sampling).

    Complements ``spatial_wetday_freq_thresholds.png`` with a domain-aggregate,
    histogram-based view and a method cross-check. Left: histograms (log density)
    of daily accumulation over the pooled valid domain -- observed vs predicted
    Monte-Carlo samples from the Bernoulli-Gamma head (dry atom included). Right:
    the exceedance frequency P(accum >= t) vs threshold t (the CCDF read off the
    histograms), comparing observed (empirical), predicted (sampled/histogram),
    and predicted (analytic rho*(1-GammaCDF)) -- the analytic curve is the same
    quantity mapped in the spatial figure, so its agreement with the sampled
    curve validates that figure. Discrete thresholds are marked and printed.
    """
    ts = sorted(float(t) for t in (thresholds if thresholds else WETFREQ_THRESHOLDS_MM))
    obs, flat, samples = _pooled_wetfreq_samples(B)
    rho, alpha, beta = flat[:, 0], flat[:, 1], flat[:, 2]

    hi = float(max(max(ts) * 1.5, np.percentile(obs[obs > 0], 99.9) if (obs > 0).any() else 30.0))
    grid = np.linspace(0.0, hi, 160)

    def _emp(arr):
        return np.array([(arr >= x).mean() for x in grid])

    def _analytic(x):
        if x <= 0:
            return float(rho.mean())
        return float((rho * (1.0 - scipy_stats.gamma.cdf(x, a=alpha, scale=1.0 / beta))).mean())

    obs_exc = _emp(obs)
    samp_exc = _emp(samples)
    ana_exc = np.array([_analytic(x) for x in grid])

    # Per-threshold table (also printed to stdout for the record).
    logger.info("wet-day exceedance frequency P(accum >= t)  [histogram vs analytic]:")
    logger.info("  %6s | %8s %8s %8s", "t(mm)", "obs", "pred_hist", "pred_ana")
    rows = []
    for t in ts:
        oe = float((obs >= t).mean())
        se = float((samples >= t).mean())
        ae = _analytic(t)
        rows.append((t, oe, se, ae))
        logger.info("  %6g | %8.4f %8.4f %8.4f", t, oe, se, ae)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) accumulation histograms (obs vs sampled), log-y, threshold guides.
    bins = np.linspace(0, hi, 80)
    ax0.hist(obs, bins=bins, density=True, histtype="step", color="k", label="observed")
    ax0.hist(samples, bins=bins, density=True, histtype="step", color="C0",
             label="predicted (sampled)")
    for t in ts:
        ax0.axvline(t, color="grey", ls=":", lw=0.8)
    ax0.set_yscale("log")
    ax0.set_xlim(0, hi)
    ax0.set_xlabel("daily accumulation (mm)")
    ax0.set_ylabel("density (log)")
    ax0.set_title("Daily accumulation histograms (all pooled days)")
    ax0.legend(fontsize=8)

    # (b) exceedance frequency vs threshold (CCDF from the histograms).
    ax1.plot(grid, obs_exc, color="k", lw=1.5, label="obs (empirical)")
    ax1.plot(grid, samp_exc, color="C0", lw=1.5, label="pred (sampled / histogram)")
    ax1.plot(grid, ana_exc, color="C3", lw=1.2, ls="--", label="pred (analytic BG)")
    for t, oe, se, _ in rows:
        ax1.scatter([t], [oe], color="k", s=22, zorder=5)
        ax1.scatter([t], [se], color="C0", s=22, zorder=5)
    ax1.set_yscale("log")
    ax1.set_xlim(0, hi)
    ax1.set_xlabel("threshold t (mm)")
    ax1.set_ylabel("exceedance freq  P(accum >= t)")
    ax1.set_title("Wet-day frequency vs threshold (histogram-derived)")
    ax1.legend(fontsize=8)

    fig.suptitle("Histogram-based wet-day frequency across accumulation thresholds", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out_dir / "wetfreq_threshold_histograms.png")
    obs_hist, _ = np.histogram(obs, bins=bins, density=True)
    samp_hist, _ = np.histogram(samples, bins=bins, density=True)
    _save_editable(fig, out_dir / "wetfreq_threshold_histograms.png", {
        "thresholds": np.asarray(ts, np.float32),
        "threshold_table_obs_hist_ana": np.asarray(rows, np.float32),
        "grid": grid.astype(np.float32),
        "obs_exc": obs_exc.astype(np.float32),
        "samp_exc": samp_exc.astype(np.float32),
        "ana_exc": ana_exc.astype(np.float32),
        "hist_bins": bins.astype(np.float32),
        "obs_hist_density": obs_hist.astype(np.float32),
        "samp_hist_density": samp_hist.astype(np.float32)})


def fig_density(B, out_dir):
    obs, pred, _, _ = _pooled_obs_pred(B)
    if obs.size > DENSITY_MAX:
        rng = np.random.default_rng(ep.PIT_SEED)
        idx = rng.choice(obs.size, DENSITY_MAX, replace=False)
        obs_s, pred_s = obs[idx], pred[idx]
    else:
        obs_s, pred_s = obs, pred
    sp, pr = ep._pooled_correlations(obs, pred)
    hi = float(np.percentile(obs[obs > 0], 99)) if (obs > 0).any() else 50.0
    fig, ax = plt.subplots(figsize=(6.5, 6))
    hb = ax.hexbin(obs_s, pred_s, gridsize=60, bins="log", cmap="magma",
                   extent=(0, hi, 0, hi))
    ax.plot([0, hi], [0, hi], "w--", lw=1)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("observed (mm/day)"); ax.set_ylabel("predicted mean (mm/day)")
    ax.set_title(f"Pooled daily obs vs predicted mean\nSpearman={sp:.3f}  Pearson={pr:.3f}")
    fig.colorbar(hb, ax=ax, label="log10(count)")
    fig.tight_layout()
    _save_fig(fig, out_dir / "obs_vs_pred_density.png")


def fig_amount_distribution(B, out_dir):
    obs, _, _, valid = _pooled_obs_pred(B)
    samples = _wetday_samples(B, valid)
    obs_wet = obs[obs >= ep.WET_THRESHOLD_MM]
    samp_wet = samples[samples >= ep.WET_THRESHOLD_MM]
    bins = np.linspace(0, max(np.percentile(obs_wet, 99.5), np.percentile(samp_wet, 99.5)), 80)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(obs_wet, bins=bins, density=True, histtype="step", color="k", label="observed")
    ax.hist(samp_wet, bins=bins, density=True, histtype="step", color="C0", label="sampled")
    ax.axvline(ep.R10_THRESHOLD_MM, color="grey", ls=":", lw=1, label="R10 (10 mm)")
    p98 = np.percentile(obs_wet, 98)
    ax.axvline(p98, color="C3", ls="--", lw=1, label=f"obs P98={p98:.1f} mm")
    ax.set_yscale("log")
    ax.set_xlabel("wet-day accumulation (mm)"); ax.set_ylabel("density (log)")
    ax.set_title("Wet-day amount distribution: observed vs sampled")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_dir / "amount_distribution.png")


def fig_qq(B, out_dir):
    obs, _, _, valid = _pooled_obs_pred(B)
    samples = _wetday_samples(B, valid)
    obs_wet = obs[obs >= ep.WET_THRESHOLD_MM]
    samp_wet = samples[samples >= ep.WET_THRESHOLD_MM]
    q = np.linspace(1, 99.5, 99)
    oq = np.percentile(obs_wet, q)
    sq = np.percentile(samp_wet, q)
    hi = max(oq.max(), sq.max())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(oq, sq, "o-", ms=3, color="C0")
    ax.plot([0, hi], [0, hi], "k--", lw=1)
    p98o, p98s = np.percentile(obs_wet, 98), np.percentile(samp_wet, 98)
    ax.scatter([p98o], [p98s], color="C3", zorder=5, label=f"P98: obs={p98o:.1f}, pred={p98s:.1f}")
    ax.set_xlabel("observed wet-day quantile (mm)")
    ax.set_ylabel("sampled wet-day quantile (mm)")
    ax.set_title("Q-Q of wet-day accumulations (obs vs sampled)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_dir / "qq_wetday_amounts.png")


def fig_pit(B, out_dir):
    pit = ep.compute_pit(B)
    if not pit.get("available"):
        logger.warning("PIT unavailable (%s); skipping pit_histogram", pit.get("note"))
        return
    edges = np.asarray(pit["bin_edges"]); dens = np.asarray(pit["density"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(centers, dens, width=(edges[1] - edges[0]) * 0.95, color="C0", alpha=0.8)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform (calibrated)")
    ax.set_xlabel("PIT value"); ax.set_ylabel("density")
    ax.set_title(f"Randomized PIT histogram (Vaughan Fig. 10 analog)\n"
                 f"mean={pit['mean_pit']:.3f}  frac<0.1={pit['frac_pit_below_0.1']:.3f}  "
                 f"frac>0.9={pit['frac_pit_above_0.9']:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_dir / "pit_histogram.png")


def fig_boxplots(pp, baseline, out_dir):
    specs = [("mae", "MAE (mm)"), ("bias", "Bias (mm)"), ("spearman", "Spearman"),
             ("r01", "R01"), ("sdii_bias", "SDII bias (mm)")]
    fig, axes = plt.subplots(1, len(specs), figsize=(3 * len(specs), 4.5))
    for ax, (key, label) in zip(axes, specs):
        vals = pp[key][np.isfinite(pp[key])]
        ax.boxplot(vals, vert=True, showfliers=False, widths=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        if key in ("bias", "sdii_bias"):
            ax.axhline(0, color="grey", ls=":", lw=1)
        if key == "r01":
            ax.axhline(1.0, color="grey", ls=":", lw=1)
        if baseline and baseline.get("available"):
            if key == "mae":
                ax.axhline(baseline["baseline_mae_mm"], color="C3", ls="--", lw=1,
                           label="baseline")
                ax.legend(fontsize=7)
            if key == "bias":
                ax.axhline(baseline["baseline_bias_mm"], color="C3", ls="--", lw=1)
    fig.suptitle("Per-point metric distributions across target points (median/IQR)", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, out_dir / "metric_distributions_boxplots.png")


def fig_perfold(model_dir, out_dir):
    import pandas as pd
    stats = model_dir / "stats.csv"
    if not stats.exists():
        logger.warning("no stats.csv; skipping perfold_skill_bars")
        return
    df = pd.read_csv(stats)
    best = df.loc[df.groupby("Fold")["test NLL"].idxmin()].sort_values("Fold")
    folds = best["Fold"].astype(int).tolist()
    metrics = [("test NLL", "test NLL"), ("Mean absolute error", "MAE (mm)"),
               ("Pearson correlation", "Pearson"), ("Spearman correlation", "Spearman")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (col, label) in zip(axes, metrics):
        ax.bar([str(f) for f in folds], best[col].values, color="C0")
        ax.set_title(label, fontsize=10); ax.set_xlabel("fold")
    fig.suptitle("Per-fold held-out skill at best epoch (by test NLL)", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, out_dir / "perfold_skill_bars.png")


def fig_vaughan_comparison(ours: dict, out_dir):
    """Compare full_bg precip biases to Vaughan et al. 2022's reported values.

    ``ours`` holds our values for R01 / SDII_bias / P98_bias / R10_bias. One panel
    per metric: our bar vs Vaughan's (grey; hatched + "~" where the paper only
    states "comparable to best baselines"), with the unbiased ideal as a dashed line.
    """
    fig, axes = plt.subplots(1, len(_VAUGHAN_PANELS), figsize=(3.4 * len(_VAUGHAN_PANELS), 4.6))
    for ax, (key, label) in zip(axes, _VAUGHAN_PANELS):
        ref = VAUGHAN_PRECIP[key]
        v_val = ref["ideal"] if ref["value"] is None else ref["value"]
        bars = ax.bar([0, 1], [ours[key], v_val], color=["C3", "0.6"], width=0.6)
        if ref["value"] is None:
            bars[1].set_hatch("//"); bars[1].set_alpha(0.5)
        ax.axhline(ref["ideal"], color="grey", ls="--", lw=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["full_bg", "Vaughan22"], fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.annotate(f"{ours[key]:.2f}", (0, ours[key]), ha="center",
                    va="bottom" if ours[key] >= 0 else "top", fontsize=8)
        vtxt = "~ideal" if ref["value"] is None else f"{ref['value']:.3g}"
        ax.annotate(vtxt, (1, v_val), ha="center",
                    va="bottom" if v_val >= 0 else "top", fontsize=8)
        ax.annotate(ref["note"], (0.5, 0.02), xycoords="axes fraction",
                    ha="center", fontsize=6, color="0.4")
    fig.suptitle("full_bg vs Vaughan et al. 2022 — precipitation bias metrics\n"
                 "(ours: Switzerland gridded, 1-yr 2023; Vaughan: 86 European VALUE "
                 "stations, multi-decade — see caveats)", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_dir / "vaughan_comparison.png")


def fig_sweep(model_dir, sweep_glob, out_dir):
    import glob as globmod
    runs = []
    full_json = model_dir / "eval_precip_metrics.json"
    if full_json.exists():
        runs.append(("full_bg", json.loads(full_json.read_text())))
    for d in sorted(globmod.glob(sweep_glob)):
        for j in Path(d).rglob("eval_precip_metrics.json"):
            label = Path(d).name.replace("sweep_bg_", "").split("__")[0]
            runs.append((label, json.loads(j.read_text())))
            break
    if len(runs) < 2:
        logger.warning("only %d runs with metrics; skipping sweep figure", len(runs))
        return
    metrics = [("R01_rel_wetday_freq", "R01"), ("SDII_bias_mm", "SDII bias (mm)"),
               ("P98_abs_bias_mm", "P98 abs bias (mm)"), ("mae_mm", "MAE (mm)"),
               ("spearman_pooled", "Spearman (pooled)")]
    labels = [r[0] for r in runs]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4.5))
    for ax, (key, title) in zip(axes, metrics):
        vals = [r[1]["overall"].get(key, np.nan) for r in runs]
        colors = ["C3" if l == "full_bg" else "C0" for l in labels]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        if key == "R01_rel_wetday_freq":
            ax.axhline(1.0, color="grey", ls=":", lw=1)
    fig.suptitle("Full run vs bg OFAT sweep variants (held-out physical metrics)", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, out_dir / "sweep_vs_full_physical.png")


# ---------------------------------------------------------------------------
def _lightweight_bundle(model_dir):
    """Minimal ``B`` for the grids-only diagnostic: params + RhiresD grid coords.

    No model, no DataBundle, no inference — just what ``fig_grid_considerations``
    needs (``params``, ``grid_shape``, ``target_lat``/``target_lon``).
    """
    p = params_mod.Params.load_json(model_dir / "params.json")
    pds = xr.open_dataset(sorted(_glob.glob(str(p.METEO_SWISS_PRECIP_GLOB)))[0])
    N, E = int(pds.sizes["N"]), int(pds.sizes["E"])
    return SimpleNamespace(
        params=p, grid_shape=(N, E), n_target_points=N * E,
        target_lat=np.asarray(pds["lat"].values).reshape(-1),
        target_lon=np.asarray(pds["lon"].values).reshape(-1),
    )


def render_category_figures(B, out_dir: Path, base, edges, name: str = "") -> float:
    """Intensity-category evaluation: both figures + precip_category_pooled.json.

    Returns the pooled RPSS (vs the ERA5-Land reference in ``base``).
    """
    cat = _precip_category_eval(B, edges, base=base)
    fig_precip_category_spatial(cat, B, out_dir)
    fig_precip_category_distribution(cat, out_dir, name=name)
    (Path(out_dir) / "precip_category_pooled.json").write_text(json.dumps({
        "rpss": cat["pooled"]["rpss"],
        "reference": cat["pooled"].get("reference"),
        "cat_labels": cat["pooled"]["cat_labels"],
        "obs_freq": cat["pooled"]["obs_freq"],
        "era_freq": cat["pooled"].get("era_freq"),
        "model_freq": cat["pooled"]["model_freq"],
        "obs_event_counts": cat["pooled"]["obs_event_counts"],
        "n_point_days": cat["pooled"]["n_point_days"]}, indent=2))
    return cat["pooled"]["rpss"]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", help="Dir with params.json/model_fold_*.")
    src.add_argument("--trial-dir", help="Trial dir; its single variable subdir is used.")
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None, help="Figure dir (default <model_dir>/eval_figures).")
    ap.add_argument("--sweep-glob", default=None, help="Glob of sweep trial dirs for the comparison fig.")
    ap.add_argument("--baseline-glob", default=None)
    ap.add_argument("--precip-glob", default=None,
                    help="Override the ERA5-Land tp field fed to the model as an INPUT "
                         "channel (ERA5_PRECIP_GLOB), for USE_SURFACE_PRECIP runs. The "
                         "stored params.json is not modified, and the prediction cache "
                         "is bypassed so a default-input bundle is never reused.")
    ap.add_argument("--grids-only", action="store_true",
                    help="Render only grid_considerations.png (no inference; seconds).")
    ap.add_argument("--wetfreq-thresholds", nargs="+", type=float, default=None,
                    metavar="MM",
                    help="Override the daily accumulation threshold/category edges (mm) "
                         "for the threshold and intensity-category figures; default "
                         "0.1 10 40 80 (bins 0-0.1, 0.1-10, 10-40, 40-80, >=80). "
                         "These figures are always rendered.")
    ap.add_argument("--eval-year", type=int, default=None,
                    help="Render figures for a genuinely unseen holdout YEAR (e.g. 2024): "
                         "fold-ensemble with training-time normalization. Default out-dir "
                         "becomes <model_dir>/eval_figures_<year>.")
    ap.add_argument("--fig-label", default=None,
                    help="Display label for the category-distribution figure title "
                         "(default: derived from the trial dir + regime).")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Re-run inference even if a pred_cache bundle exists, then overwrite it.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Neither read nor write the pred_cache prediction bundle.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    params_mod.configure_renku_cuda()

    model_dir = Path(args.model_dir) if args.model_dir else ep._resolve_model_dir(args)
    default_figdir = ("eval_figures" if args.eval_year is None
                      else f"eval_figures_{args.eval_year}")
    out_dir = Path(args.out_dir) if args.out_dir else (model_dir / default_figdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.grids_only:
        try:
            fig_grid_considerations(_lightweight_bundle(model_dir), out_dir)
            print(f"Wrote grid_considerations.png to {out_dir}")
            return 0
        except Exception:
            logger.exception("grid_considerations failed")
            return 1

    device = torch.device(args.device) if args.device else params_mod.select_device()
    logger.info("obtaining predictions (cached inference state if available)...")
    B = ep.load_or_predict(model_dir, device, eval_year=args.eval_year, folds=args.folds,
                           refresh=args.refresh_cache, use_cache=not args.no_cache,
                           precip_glob=args.precip_glob)
    pp = ep.per_point_metrics(B)
    overall = ep.compute_precip_metrics(B.params_full[B.day_mask], B.truth_full[B.day_mask],
                                        B.distribution, B.get_value_fn)
    # Attempt the baseline for every regime; it degrades gracefully to
    # available=False when tp-<year>.nc is absent (so 2024 skill/figures light up
    # automatically once the field lands).
    baseline = ep.compute_baseline_skill(B, overall, baseline_glob=args.baseline_glob)

    figures = [
        ("grid considerations", lambda: fig_grid_considerations(B, out_dir)),
        ("spatial means", lambda: fig_spatial_means(pp, B, out_dir)),
        ("mae/spearman maps", lambda: fig_spatial_mae_spearman(pp, B, out_dir)),
        ("wet-day freq maps", lambda: fig_spatial_wetfreq(pp, B, out_dir)),
        ("obs-vs-pred density", lambda: fig_density(B, out_dir)),
        ("amount distribution", lambda: fig_amount_distribution(B, out_dir)),
        ("qq wet-day amounts", lambda: fig_qq(B, out_dir)),
        ("PIT histogram", lambda: fig_pit(B, out_dir)),
        ("metric boxplots", lambda: fig_boxplots(pp, baseline, out_dir)),
        ("per-fold skill", lambda: fig_perfold(model_dir, out_dir)),
        ("vaughan comparison", lambda: fig_vaughan_comparison({
            "R01": float(np.nanmedian(pp["r01"])),
            "SDII_bias": float(np.nanmedian(pp["sdii_bias"])),
            "P98_bias": overall["P98_pred_mm"] - overall["P98_obs_mm"],
            "R10_bias": overall["R10_freq_pred"] - overall["R10_freq_obs"],
        }, out_dir)),
    ]
    # Threshold + intensity-category figures are part of the standard set; the
    # ERA5-Land reference is computed once and shared between them.
    edges = args.wetfreq_thresholds or WETFREQ_THRESHOLDS_MM
    regime_tag = "CV" if args.eval_year is None else str(args.eval_year)
    fig_label = args.fig_label or f"{model_dir.parent.name} {regime_tag}"
    base = _era5_baseline(B)
    cat_result: dict = {}
    figures += [
        ("wet-day freq thresholds",
         lambda: fig_spatial_wetfreq_thresholds(B, out_dir, edges, base=base)),
        ("wet-day freq threshold histograms",
         lambda: fig_wetfreq_threshold_histograms(B, out_dir, edges)),
        ("intensity categories (+RPSS)",
         lambda: cat_result.update(
             rpss=render_category_figures(B, out_dir, base, edges, name=fig_label))),
    ]
    if args.sweep_glob:
        figures.append(("sweep comparison",
                        lambda: fig_sweep(model_dir, args.sweep_glob, out_dir)))

    n_ok = 0
    for name, fn in figures:
        try:
            fn(); n_ok += 1
            logger.info("  [ok] %s", name)
        except Exception as exc:  # keep rendering the rest
            logger.exception("  [FAIL] %s: %r", name, exc)
    if "rpss" in cat_result:
        print(f"### DONE {fig_label} | RPSS={cat_result['rpss']:+.4f} -> {out_dir}", flush=True)
    print(f"Wrote {n_ok}/{len(figures)} figures to {out_dir}")
    return 0 if n_ok == len(figures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
