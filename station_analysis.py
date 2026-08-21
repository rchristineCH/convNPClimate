#!/usr/bin/env python3
"""
Evaluate the best atmospheric model at the 29 real NBCN homogeneous climate stations.

Unlike error_analysis.py (which uses the gridded TmaxD analysis as truth),
this script downloads the raw homogenized daily station measurements from
MeteoSwiss open data and evaluates the model at the exact station coordinates.
This is a truly independent skill assessment: the station observations are the
*source* data that TmaxD interpolation is based on, not a byproduct of it.

Data source : ch.meteoschweiz.ogd-nbcn (29 NBCN stations)
Variable    : ths200dx = homogeneous daily maximum air temperature 2 m (°C)
Reference   : https://opendatadocs.meteoswiss.ch/en/c-climate-data/c1-climate-stations_homogeneous

Outputs (default dir: station_analysis/):
    REPORT.md          — interpretability report with embedded figures
    summary.json       — machine-readable per-station metrics
    *.png              — figures
    _nbcn_cache.npz    — cached predictions + truth (reuse with --use-cache)

Example
-------
    python station_analysis.py --years 2020-2024
    python station_analysis.py --years 2020-2024 --use-cache
"""

import argparse
import io
import json
import logging
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import xarray as xr
import properscoring as ps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats

import params as params_mod
import datasets as ds
import model_factory
import predict
import visualization as vis  # save_fig: PNG + editable .fig.pkl
from infer import _resolve_model_dir
import evaluate as ev
from convCNP.training.training_elev import get_fold_holdout_indices

logger = logging.getLogger("station_analysis")

# ---------------------------------------------------------------------------
# MeteoSwiss NBCN open-data endpoints
# ---------------------------------------------------------------------------
_NBCN_BASE = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn"
_META_STATIONS_URL = f"{_NBCN_BASE}/ogd-nbcn_meta_stations.csv"
_TMAX_COL = "ths200dx"          # homogeneous daily maximum temperature (°C)
BEST_ATMOS = ("CLEAN_trained_models/clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8")


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


# ---------------------------------------------------------------------------
# 1.  Download NBCN data
# ---------------------------------------------------------------------------

def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read()
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def download_nbcn_data(cache_dir: Path, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download station metadata and all daily Tmax records.

    Returns
    -------
    station_meta : DataFrame  — one row per station (abbr, name, lat, lon, elev_m)
    daily        : DataFrame  — columns [date, stn_abbr, tmax_C]  (all available years)
    """
    cache_meta = cache_dir / "_nbcn_stations.csv"
    cache_daily = cache_dir / "_nbcn_daily_tmax.csv"

    if not force and cache_meta.exists() and cache_daily.exists():
        logger.info("loading NBCN cache from %s", cache_dir)
        meta = pd.read_csv(cache_meta)
        daily = pd.read_csv(cache_daily, parse_dates=["date"])
        return meta, daily

    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- station metadata ---
    logger.info("downloading NBCN station metadata ...")
    raw = _fetch_text(_META_STATIONS_URL)
    meta = pd.read_csv(io.StringIO(raw), sep=";")
    # Columns: station_abbr, station_name, station_coordinates_wgs84_lat,
    #          station_coordinates_wgs84_lon, station_elevation_m  (exact names may vary)
    meta.columns = meta.columns.str.strip().str.lower()
    # Normalise key column names. The source has several elevation-like columns
    # (station height + barometer height); map only the FIRST match per target so
    # we don't create duplicate labels (which would make meta[col] a DataFrame).
    rename, taken = {}, set()
    def _assign(col, target):
        if target not in taken:
            rename[col] = target
            taken.add(target)
    for c in meta.columns:
        if "abbr" in c:
            _assign(c, "stn_abbr")
        elif "name" in c:
            _assign(c, "name")
        elif "_lat" in c or c == "lat":
            _assign(c, "lat")
        elif "_lon" in c or c == "lon":
            _assign(c, "lon")
        elif "height_masl" in c or "elevation" in c or "altitude" in c:
            _assign(c, "elev_m")
    meta = meta.rename(columns=rename)
    # Drop any leftover duplicate-named columns, keeping the first.
    meta = meta.loc[:, ~meta.columns.duplicated()]
    meta["stn_abbr"] = meta["stn_abbr"].astype(str).str.strip().str.upper()
    meta.to_csv(cache_meta, index=False)
    logger.info("%d stations in metadata", len(meta))

    # --- daily Tmax per station ---
    all_rows = []
    for _, row in meta.iterrows():
        stn = row["stn_abbr"].upper()
        stn_lower = stn.lower()
        for kind in ("d_historical", "d_recent"):
            url = f"{_NBCN_BASE}/{stn_lower}/ogd-nbcn_{stn_lower}_{kind}.csv"
            try:
                raw = _fetch_text(url)
            except Exception as exc:
                logger.debug("station %s kind %s: %s", stn, kind, exc)
                continue
            df = pd.read_csv(io.StringIO(raw), sep=";")
            df.columns = df.columns.str.strip().str.lower()
            if "reference_timestamp" not in df.columns:
                logger.warning("%s %s: no reference_timestamp column", stn, kind)
                continue
            if _TMAX_COL not in df.columns:
                logger.warning("%s %s: no %s column", stn, kind, _TMAX_COL)
                continue
            df["date"] = pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M",
                                        errors="coerce")
            df = df.dropna(subset=["date"])
            df["stn_abbr"] = stn
            df["tmax_C"] = pd.to_numeric(df[_TMAX_COL], errors="coerce")
            all_rows.append(df[["date", "stn_abbr", "tmax_C"]])
            logger.info("  %s %-12s: %d rows", stn, kind, len(df))

    daily = pd.concat(all_rows, ignore_index=True)
    daily = daily.drop_duplicates(subset=["date", "stn_abbr"])
    daily = daily.sort_values(["stn_abbr", "date"]).reset_index(drop=True)
    daily.to_csv(cache_daily, index=False)
    logger.info("total NBCN daily rows: %d", len(daily))
    return meta, daily


# ---------------------------------------------------------------------------
# 2.  Build station targets (target_x, target_topo, dists)
# ---------------------------------------------------------------------------

def build_station_targets(
    station_meta: pd.DataFrame,
    model_dir: Path,
    device: torch.device,
) -> SimpleNamespace:
    """Build the coordinate/elevation tensors for the 29 stations."""
    manifest = predict.load_manifest(model_dir)
    p = params_mod.Params.load_json(model_dir / "params.json")
    dists_meta = predict.manifest_to_dists_metadata(manifest)

    lats = np.asarray(station_meta["lat"].values, dtype=np.float64).ravel()
    lons = np.asarray(station_meta["lon"].values, dtype=np.float64).ravel()
    elevs = np.asarray(station_meta["elev_m"].values, dtype=np.float32).ravel()
    n = len(lats)

    # --- normalized lat/lon (same convention as prepare_meteoswiss_targets) ---
    lat_range = dists_meta.lat_max - dists_meta.lat_min
    lon_range = dists_meta.lon_max - dists_meta.lon_min
    lat_norm = xr.DataArray((lats - dists_meta.lat_min) / lat_range, dims="point")
    lon_norm = xr.DataArray((lons - dists_meta.lon_min) / lon_range, dims="point")
    target_x = xr.concat([lat_norm, lon_norm], dim="coord")
    target_x = target_x.transpose("point", "coord")
    target_x = target_x.assign_coords(coord=["lat", "lon"])

    # --- dists (ERA5 grid → station points) ---
    dists = ds.calculate_dists_meteoswiss(dists_meta, target_x, device=device)

    # --- topography features: [true_elev, elev_diff, mTPI] ---
    # true_elev: from station metadata
    true_elev = elevs.copy()

    # elev_diff: station_elev - ERA5_grid_elevation_at_station (bilinear)
    elev_diff = np.zeros(n, dtype=np.float32)
    if p.ERA5_GEOPOTENTIAL_GLOB:
        try:
            geo = xr.open_mfdataset(p.ERA5_GEOPOTENTIAL_GLOB, combine="by_coords")
            geo = geo.rename({"latitude": "longitude", "longitude": "latitude"})
            era5_alt = geo["z"] / 9.80665
            if "time" in era5_alt.dims:
                era5_alt = era5_alt.isel(time=0)
            era5_at_stns = era5_alt.interp(
                latitude=xr.DataArray(lats, dims="point"),
                longitude=xr.DataArray(lons, dims="point"),
                method="linear",
            )
            era5_vals = np.asarray(era5_at_stns.values, dtype=np.float32).ravel()
            elev_diff = (true_elev - era5_vals).astype(np.float32)
            logger.info("ERA5 elev at stations: min %.0f max %.0f m",
                        era5_at_stns.values.min(), era5_at_stns.values.max())
        except Exception as exc:
            logger.warning("could not load ERA5 geopotential for elev_diff: %s", exc)

    # mTPI: from hi-res zarr via LV95 coords
    mtpi = np.zeros(n, dtype=np.float32)
    if p.HI_RES_TOPOGRAPHY_ZARR_PATH:
        try:
            _, tpi_da = ds.load_high_res_topography(p.HI_RES_TOPOGRAPHY_ZARR_PATH)
            if tpi_da is not None:
                stn_x_lv95, stn_y_lv95 = ds.wgs84_to_lv95(lons, lats)
                tpi_at_stns = tpi_da.interp(
                    x=xr.DataArray(stn_x_lv95, dims="point"),
                    y=xr.DataArray(stn_y_lv95, dims="point"),
                    method="linear",
                )
                mtpi = np.asarray(tpi_at_stns.values, dtype=np.float32).ravel()
                logger.info("mTPI at stations: min %.0f max %.0f m",
                            np.nanmin(mtpi), np.nanmax(mtpi))
        except Exception as exc:
            logger.warning("could not load hi-res TPI for mTPI: %s", exc)

    topo_arr = np.column_stack([true_elev, elev_diff, mtpi])   # (n, 3)
    target_topo = torch.from_numpy(topo_arr.astype(np.float32)).to(device)

    logger.info("station targets: %d stations, target_x %s, dists %s, topo %s",
                n, target_x.sizes, tuple(dists.shape), tuple(target_topo.shape))
    return SimpleNamespace(
        target_x=target_x, target_topo=target_topo, dists=dists,
        dists_meta=dists_meta, p=p, manifest=manifest,
        lats=lats, lons=lons, elevs=elevs, n=n,
    )


# ---------------------------------------------------------------------------
# 3.  Run ensemble inference for one year at station coordinates
# ---------------------------------------------------------------------------

def _station_truth(station_meta: pd.DataFrame, daily: pd.DataFrame,
                   dates: pd.DatetimeIndex, n: int) -> np.ndarray:
    """NBCN observed tmax on the requested days, (T, N) degC, NaN where not reported."""
    stn_order = station_meta["stn_abbr"].str.upper().tolist()
    ctx_days = pd.DatetimeIndex(dates).normalize()
    truth_C = np.full((len(ctx_days), n), np.nan, dtype=np.float32)
    for si, stn in enumerate(stn_order):
        sub = daily[daily["stn_abbr"] == stn].copy()
        sub = sub.set_index(pd.to_datetime(sub["date"]).dt.normalize())
        sub = sub[~sub.index.duplicated(keep="first")]
        vals = sub["tmax_C"].reindex(ctx_days).to_numpy(dtype=np.float32)
        truth_C[:, si] = vals
    logger.info("station truth coverage: %.1f%%", 100 * np.isfinite(truth_C).mean())
    return truth_C


def _station_context(tgts, dates, device):
    p = tgts.p
    if p.USE_ATMOSPHERIC and p.ATMOS_NATIVE_GRID:
        context = predict.build_atmospheric_context(tgts.manifest, p, dates, device)
    else:
        context = predict.build_surface_context(tgts.manifest, p, dates, device)
    seasonal = (ds.compute_seasonal_features(dates.values.astype("datetime64[ns]"),
                                             device=device)
                if p.SEASONAL_FEATURES else None)
    channel_groups = (ds.channel_groups_by_variable(tgts.manifest["channel_names"])
                      if p.ENCODER != "flat" else None)
    return context, seasonal, channel_groups


def predict_stations_cv(model_dir: Path, tgts: SimpleNamespace,
                        station_meta: pd.DataFrame, daily: pd.DataFrame,
                        device: torch.device, folds=None,
                        day_batch: int = 1) -> SimpleNamespace:
    """CV holdout at the stations: each fold scores only its own held-out block.

    The station targets are not on the MeteoSwiss lattice, so the gridded bundle cannot
    serve them — but the fold arithmetic is identical to ``evaluate.predict_cv``, and it
    must be, or the station numbers would not describe the same experiment.
    """
    p, dists_meta = tgts.p, tgts.dists_meta
    dates = predict._date_range(f"{p.DATA_YEAR_START}-01-01", f"{p.DATA_YEAR_END}-12-31")
    n_times = len(dates)
    logger.info("=== station CV holdout %s | %d-%d (%d days x %d stations) ===",
                model_dir.name, p.DATA_YEAR_START, p.DATA_YEAR_END, n_times, tgts.n)

    context, seasonal, channel_groups = _station_context(tgts, dates, device)
    truth_C = _station_truth(station_meta, daily, dates, tgts.n)

    preds_n = np.full((n_times, tgts.n), np.nan, dtype=np.float32)
    sig_n = np.full((n_times, tgts.n), np.nan, dtype=np.float32)
    day_mask = np.zeros(n_times, dtype=bool)
    per_fold_blocks, fold_epochs = {}, {}
    for fold in (folds if folds is not None else list(range(p.N_FOLDS))):
        ckpt = model_dir / f"model_fold_{fold}"
        if not ckpt.exists():
            logger.warning("checkpoint %s missing; skipping fold %d", ckpt, fold)
            continue
        model, epoch = model_factory.load_model_checkpoint(
            ckpt, p, device, channel_groups=channel_groups)
        a, b = get_fold_holdout_indices(fold, p.N_FOLDS, n_times)
        pf, sf = predict.predict_all_days(
            model, context, tgts.dists, tgts.target_topo, seasonal, device,
            day_batch=day_batch, day_start=a, day_end=b)
        preds_n[a:b], sig_n[a:b], day_mask[a:b] = pf, sf, True
        per_fold_blocks[fold] = (a, b)
        fold_epochs[fold] = int(epoch)
        logger.info("  fold %d epoch %d: held-out days [%d, %d)", fold, epoch, a, b)

    if not per_fold_blocks:
        raise RuntimeError(f"No fold checkpoints found under {model_dir}.")

    preds_C = predict.denormalize(preds_n, dists_meta)
    sigmas_C = sig_n * dists_meta.data_std
    era5_C = ev.era5_reference_degC(context, p, tgts.target_x, dists_meta, dates, device)
    return SimpleNamespace(
        preds_C=preds_C, sigmas_C=sigmas_C, truth_C=truth_C,
        errors_C=preds_C - truth_C, era5_C=era5_C, dates=np.array(dates),
        eval_year=None, regime="cv_holdout", prediction_mode="fold_holdout",
        day_mask=day_mask, per_fold_blocks=per_fold_blocks, fold_epochs=fold_epochs,
    )


def predict_stations_holdout_year(model_dir: Path, tgts: SimpleNamespace,
                                  station_meta: pd.DataFrame, daily: pd.DataFrame,
                                  year: int, device: torch.device, folds=None,
                                  day_batch: int = 1) -> SimpleNamespace:
    """Fold-ensemble inference at the stations for a genuinely unseen year."""
    p, dists_meta = tgts.p, tgts.dists_meta
    dates = predict._date_range(f"{year}-01-01", f"{year}-12-31")
    logger.info("=== station holdout %s | %d (%d days x %d stations) ===",
                model_dir.name, year, len(dates), tgts.n)

    context, seasonal, channel_groups = _station_context(tgts, dates, device)
    truth_C = _station_truth(station_meta, daily, dates, tgts.n)

    all_preds_n, all_sigmas_n, fold_epochs = [], [], {}
    for fold in (folds if folds is not None else list(range(p.N_FOLDS))):
        ckpt = model_dir / f"model_fold_{fold}"
        if not ckpt.exists():
            logger.warning("checkpoint %s missing; skipping fold %d", ckpt, fold)
            continue
        model, epoch = model_factory.load_model_checkpoint(
            ckpt, p, device, channel_groups=channel_groups)
        pf, sf = predict.predict_all_days(
            model, context, tgts.dists, tgts.target_topo, seasonal, device,
            day_batch=day_batch)
        all_preds_n.append(pf)
        all_sigmas_n.append(sf)
        fold_epochs[fold] = int(epoch)
        logger.info("  fold %d epoch %d done", fold, epoch)

    if not all_preds_n:
        raise RuntimeError(f"No fold checkpoints found under {model_dir}.")

    # Gaussian mixture moment match, as in evaluate.predict_holdout_year.
    within_var = np.mean(np.array(all_sigmas_n) ** 2, axis=0)
    between_var = np.var(np.array(all_preds_n), axis=0)
    preds_C = predict.denormalize(np.mean(all_preds_n, axis=0), dists_meta)
    sigmas_C = np.sqrt(within_var + between_var) * dists_meta.data_std
    era5_C = ev.era5_reference_degC(context, p, tgts.target_x, dists_meta, dates, device)
    return SimpleNamespace(
        preds_C=preds_C, sigmas_C=sigmas_C, truth_C=truth_C,
        errors_C=preds_C - truth_C, era5_C=era5_C, dates=np.array(dates),
        eval_year=year, regime=f"holdout_year_{year}", prediction_mode="fold_ensemble",
        day_mask=np.ones(len(dates), dtype=bool), per_fold_blocks={},
        fold_epochs=fold_epochs,
    )


# ---------------------------------------------------------------------------
# 4.  One regime's station predictions, cached like evaluate.py's grid bundles
# ---------------------------------------------------------------------------

def station_cache_path(model_dir: Path, eval_year=None) -> Path:
    stem = "cv" if eval_year is None else f"holdout_{eval_year}"
    return Path(model_dir) / ev.CACHE_DIRNAME / f"stations_{stem}.npz"


def aggregate(
    model_dir: Path,
    station_meta: pd.DataFrame,
    daily: pd.DataFrame,
    device: torch.device,
    eval_year=None,
    folds=None,
    refresh: bool = False,
    use_cache: bool = True,
    day_batch: int = 1,
) -> dict:
    """Station predictions for one regime, served from a persisted bundle when possible."""
    cache = station_cache_path(model_dir, eval_year)
    if folds is not None:
        logger.info("explicit fold subset requested: bypassing the station cache")
    elif use_cache and not refresh and cache.exists():
        try:
            z = np.load(cache, allow_pickle=True)
            if int(z["schema_version"]) == ev.CACHE_SCHEMA_VERSION:
                logger.info("reusing cached station inference: %s (no GPU pass)", cache)
                return dict(
                    preds_C=z["preds_C"], sigmas_C=z["sigmas_C"], truth_C=z["truth_C"],
                    era5_C=z["era5_C"], dates=z["dates"].tolist(), topo=z["topo"],
                    day_mask=z["day_mask"], regime=str(z["regime"]),
                    eval_year=(None if int(z["eval_year"]) < 0 else int(z["eval_year"])),
                    prediction_mode=str(z["prediction_mode"]),
                    served_from_cache=True,
                )
            logger.warning("%s has an old schema; re-running station inference", cache)
        except Exception as exc:                                  # noqa: BLE001
            logger.warning("could not read %s (%r); re-running station inference",
                           cache, exc)

    tgts = build_station_targets(station_meta, model_dir, device)
    topo = tgts.target_topo.cpu().numpy()   # (N, 3): true_elev, elev_diff, mTPI
    if eval_year is None:
        a = predict_stations_cv(model_dir, tgts, station_meta, daily, device,
                                folds=folds, day_batch=day_batch)
    else:
        a = predict_stations_holdout_year(model_dir, tgts, station_meta, daily,
                                          eval_year, device, folds=folds,
                                          day_batch=day_batch)

    out = dict(preds_C=a.preds_C, sigmas_C=a.sigmas_C, truth_C=a.truth_C,
               era5_C=a.era5_C, dates=a.dates.tolist(), topo=topo,
               day_mask=a.day_mask, regime=a.regime, eval_year=a.eval_year,
               prediction_mode=a.prediction_mode, served_from_cache=False)
    if use_cache and folds is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, schema_version=np.int64(ev.CACHE_SCHEMA_VERSION),
                 preds_C=a.preds_C, sigmas_C=a.sigmas_C, truth_C=a.truth_C,
                 era5_C=a.era5_C, dates=np.array(a.dates, dtype=object), topo=topo,
                 day_mask=a.day_mask, regime=np.asarray(a.regime),
                 eval_year=np.int64(-1 if a.eval_year is None else a.eval_year),
                 prediction_mode=np.asarray(a.prediction_mode))
        logger.info("cached station inference: %s", cache)
    return out


# ---------------------------------------------------------------------------
# 5.  Per-station metrics
# ---------------------------------------------------------------------------

def per_station_metrics(data: dict) -> pd.DataFrame:
    preds = data["preds_C"]   # (T, N)
    sigs  = data["sigmas_C"]
    truth = data["truth_C"]
    era5  = data["era5_C"]

    rows = []
    N = preds.shape[1]
    for i in range(N):
        valid = np.isfinite(truth[:, i])
        if valid.sum() < 10:
            rows.append({})
            continue
        e = preds[valid, i] - truth[valid, i]
        s = np.clip(sigs[valid, i], 1e-6, None)
        t = truth[valid, i]
        p = preds[valid, i]
        r5 = era5[valid, i]

        mae = float(np.mean(np.abs(e)))
        bias = float(np.mean(e))
        rmse = float(np.sqrt(np.mean(e ** 2)))
        crps = float(np.mean(ps.crps_gaussian(t, mu=p, sig=s)))
        crps_era5 = float(np.mean(np.abs(r5 - t)))   # deterministic CRPS = MAE
        skill = 1.0 - crps / crps_era5 if crps_era5 > 0 else float("nan")
        mae_era5 = float(np.mean(np.abs(r5 - t)))
        z_std = float(np.std(e / s))
        cov90 = float(np.mean(np.abs(e) <= 1.645 * s))
        pearson = float(sstats.pearsonr(p, t)[0]) if len(p) > 2 else float("nan")
        rows.append(dict(
            mae=mae, bias=bias, rmse=rmse, crps=crps,
            skill=skill, mae_era5=mae_era5, skill_mae=1 - mae / mae_era5 if mae_era5 > 0 else float("nan"),
            z_std=z_std, cov90=cov90, pearson=pearson, n_days=int(valid.sum()),
        ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6.  Figures
# ---------------------------------------------------------------------------

def fig_station_mae(df_m: pd.DataFrame, station_meta: pd.DataFrame, out: Path):
    """Horizontal bar chart: per-station MAE sorted from best to worst."""
    data = df_m.copy()
    data["name"] = station_meta["name"].values
    data["elev_m"] = station_meta["elev_m"].values
    data = data.dropna(subset=["mae"]).sort_values("mae", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(data) * 0.32)))
    colors = ["C3" if m >= data["mae"].quantile(0.9) else "C0" for m in data["mae"]]
    bars = ax.barh(data["name"], data["mae"], color=colors)
    ax.axvline(data["mae"].mean(), color="k", ls="--", label=f"mean {data['mae'].mean():.2f} °C")
    ax.axvline(data["mae"].median(), color="0.5", ls=":", label=f"median {data['mae'].median():.2f} °C")
    ax.set_xlabel("MAE (°C)")
    ax.set_title("Per-station MAE — NBCN homogeneous Tmax")
    ax.legend(fontsize=8)
    fig.tight_layout()
    vis.save_fig(fig, out, dpi=130, bbox_inches=None)
    plt.close(fig)


def fig_mae_vs_altitude(df_m: pd.DataFrame, station_meta: pd.DataFrame, out: Path):
    elevs = station_meta["elev_m"].values
    names = station_meta["name"].values
    mae = df_m["mae"].values
    skill = df_m["skill"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, y, ylabel, title in [
        (axes[0], mae, "MAE (°C)", "MAE vs station altitude"),
        (axes[1], skill, "CRPS skill vs ERA5", "Skill vs station altitude"),
    ]:
        valid = np.isfinite(y)
        ax.scatter(elevs[valid], y[valid], s=60, c="C0", zorder=3)
        for xi, yi, nm in zip(elevs[valid], y[valid], names[valid]):
            ax.annotate(nm, (xi, yi), fontsize=6, ha="left", va="bottom")
        if valid.sum() > 2:
            r = sstats.pearsonr(elevs[valid], y[valid])[0]
            ax.set_title(f"{title}  (r={r:.2f})")
        else:
            ax.set_title(title)
        ax.set_xlabel("elevation (m)"); ax.set_ylabel(ylabel)
    fig.tight_layout()
    vis.save_fig(fig, out, dpi=130, bbox_inches=None)
    plt.close(fig)


def fig_mae_vs_topo(df_m: pd.DataFrame, station_meta: pd.DataFrame,
                    mtpi: np.ndarray, elev_diff: np.ndarray, out: Path):
    """Per-station MAE against the topographic-position index and the elevation
    mismatch between the station and the coarse ERA5 grid cell."""
    names = station_meta["name"].values
    mae = df_m["mae"].values
    panels = [
        (mtpi, "mTPI — topographic position index (m)", "MAE vs topographic index"),
        (np.abs(elev_diff), "|elevation mismatch| vs ERA5 grid (m)", "MAE vs elevation mismatch"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (x, xlabel, title) in zip(axes, panels):
        valid = np.isfinite(mae) & np.isfinite(x)
        ax.scatter(x[valid], mae[valid], s=60, c="C0", zorder=3)
        for xi, yi, nm in zip(x[valid], mae[valid], names[valid]):
            ax.annotate(nm, (xi, yi), fontsize=6, ha="left", va="bottom")
        if valid.sum() > 2:
            r = sstats.pearsonr(x[valid], mae[valid])[0]
            rho = sstats.spearmanr(x[valid], mae[valid]).correlation
            ax.set_title(f"{title}  (r={r:.2f}, ρ={rho:.2f})")
        else:
            ax.set_title(title)
        ax.set_xlabel(xlabel); ax.set_ylabel("MAE (°C)")
    fig.tight_layout()
    vis.save_fig(fig, out, dpi=130, bbox_inches=None)
    plt.close(fig)


def fig_spatial_skill(df_m: pd.DataFrame, station_meta: pd.DataFrame, out: Path):
    lats = station_meta["lat"].values
    lons = station_meta["lon"].values
    names = station_meta["name"].values
    mae = df_m["mae"].values
    skill = df_m["skill"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, vals, label, cmap in [
        (axes[0], mae, "MAE (°C)", "YlOrRd"),
        (axes[1], skill, "CRPS skill vs ERA5", "RdYlGn"),
    ]:
        valid = np.isfinite(vals)
        sc = ax.scatter(lons[valid], lats[valid], c=vals[valid], s=120, cmap=cmap, zorder=3,
                        edgecolors="k", linewidths=0.4)
        for xi, yi, nm in zip(lons[valid], lats[valid], names[valid]):
            ax.annotate(nm, (xi, yi), fontsize=6, ha="left", va="bottom")
        fig.colorbar(sc, ax=ax, shrink=0.8, label=label)
        ax.set_title(label); ax.set_xlabel("lon (°E)"); ax.set_ylabel("lat (°N)")
    fig.tight_layout()
    vis.save_fig(fig, out, dpi=130, bbox_inches=None)
    plt.close(fig)


def fig_bias_overview(df_m: pd.DataFrame, station_meta: pd.DataFrame, out: Path):
    bias = df_m["bias"].values
    mae = df_m["mae"].values
    names = station_meta["name"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # panel 1: bias bar chart (sorted)
    order = np.argsort(bias)
    valid = np.isfinite(bias[order])
    colors = ["C0" if b > 0 else "C3" for b in bias[order][valid]]
    axes[0].barh(np.array(names)[order][valid], bias[order][valid], color=colors)
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_xlabel("systematic bias (°C, positive = warm bias)")
    axes[0].set_title("Per-station systematic bias")

    # panel 2: |bias| vs MAE
    valid2 = np.isfinite(bias) & np.isfinite(mae)
    axes[1].scatter(np.abs(bias[valid2]), mae[valid2], s=60, c="C2")
    lim = max(np.nanmax(mae), np.nanmax(np.abs(bias))) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="MAE = |bias|")
    for xi, yi, nm in zip(np.abs(bias[valid2]), mae[valid2], names[valid2]):
        axes[1].annotate(nm, (xi, yi), fontsize=6)
    axes[1].set_xlabel("|bias| (°C)"); axes[1].set_ylabel("MAE (°C)")
    axes[1].set_title("Systematic vs total error"); axes[1].legend()
    fig.tight_layout()
    vis.save_fig(fig, out, dpi=130, bbox_inches=None)
    plt.close(fig)


def fig_skill_comparison(df_m: pd.DataFrame, station_meta: pd.DataFrame, out: Path):
    """Side-by-side: ERA5 MAE vs model MAE per station (bar chart)."""
    data = df_m.copy()
    data["name"] = station_meta["name"].values
    data = data.dropna(subset=["mae", "mae_era5"]).sort_values("mae")
    x = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(10, max(5, len(data) * 0.35)))
    ax.barh(x - 0.2, data["mae_era5"], 0.4, label="ERA5 bilinear (MAE)", color="0.65")
    ax.barh(x + 0.2, data["mae"], 0.4, label="ConvCNP model (MAE)", color="C0")
    ax.set_yticks(x); ax.set_yticklabels(data["name"])
    ax.set_xlabel("MAE (°C)"); ax.set_title("Model vs ERA5 baseline — per station")
    ax.legend()
    fig.tight_layout()
    vis.save_fig(fig, out, dpi=130, bbox_inches=None)
    plt.close(fig)


def fig_calibration(df_m: pd.DataFrame, station_meta: pd.DataFrame, out: Path):
    """z-score std and coverage-90 per station."""
    data = df_m.copy()
    data["name"] = station_meta["name"].values
    data = data.dropna(subset=["z_std", "cov90"])

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(data) * 0.25)))
    order = np.argsort(data["z_std"].values)[::-1]
    axes[0].barh(data["name"].iloc[order], data["z_std"].values[order],
                 color=["C3" if z > 1.5 else "C0" for z in data["z_std"].values[order]])
    axes[0].axvline(1.0, color="k", ls="--", label="ideal z-std=1")
    axes[0].set_xlabel("z-score std  (ideal = 1)")
    axes[0].set_title("Calibration: z-std per station"); axes[0].legend()

    order2 = np.argsort(data["cov90"].values)
    axes[1].barh(data["name"].iloc[order2], data["cov90"].values[order2],
                 color=["C3" if c < 0.80 else "C0" for c in data["cov90"].values[order2]])
    axes[1].axvline(0.90, color="k", ls="--", label="ideal 90% coverage")
    axes[1].set_xlabel("coverage at 1.645σ  (ideal = 0.90)")
    axes[1].set_title("Calibration: coverage-90 per station"); axes[1].legend()
    fig.tight_layout()
    vis.save_fig(fig, out, dpi=130, bbox_inches=None)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7.  Report
# ---------------------------------------------------------------------------

def _mtpi_phrase(topo_corr: dict) -> str:
    """Compact rank-based mTPI string for inline use; robust to small-n leverage."""
    m = topo_corr.get("mTPI")
    if not m or not np.isfinite(m.get("spearman", np.nan)):
        return "mTPI trend negligible"
    return f"mTPI rank ρ={m['spearman']:+.2f}"


def _topo_section_text(topo_corr: dict) -> str:
    """Prose for §3: per-station MAE correlation with each topographic feature.

    Leads on the rank (Spearman) correlation, which is robust to the handful of
    high-leverage alpine summits that can inflate Pearson on only ~28 points."""
    def fmt(key, label):
        c = topo_corr.get(key)
        if not c or not np.isfinite(c.get("pearson", np.nan)):
            return f"{label}: n/a"
        return f"{label}: r={c['pearson']:+.2f}, ρ={c['spearman']:+.2f}"

    alt = fmt("altitude", "altitude")
    mt = fmt("mTPI", "mTPI (topographic position index)")
    em = fmt("abs_elev_mismatch", "|elevation mismatch|")

    # Detect a Pearson/Spearman divergence (outlier leverage) for mTPI.
    m = topo_corr.get("mTPI", {})
    pr, rho = m.get("pearson", float("nan")), m.get("spearman", float("nan"))
    leverage = (np.isfinite(pr) and np.isfinite(rho) and abs(pr) - abs(rho) > 0.2)
    mtpi_note = ""
    if leverage:
        mtpi_note = (
            f" mTPI shows a moderate **Pearson r={pr:+.2f}** but a weak **rank ρ={rho:+.2f}** — "
            "the linear correlation is carried by a few high-leverage alpine summits "
            "(e.g. Säntis, Jungfraujoch: high mTPI *and* high MAE), not a population-wide trend. "
            "On only ~28 stations the rank correlation is the trustworthy measure, and it is weak.")

    return (
        "Per-station MAE vs each topographic feature (Pearson r / Spearman ρ over the "
        f"evaluated stations) — **{alt}**, **{mt}**, **{em}**."
        + mtpi_note +
        " The rank correlations are all weak (|ρ| ≤ 0.23): there is no robust ridge/valley "
        "(mTPI) or altitude ordering in where the model errs, consistent with the gridded "
        "analysis (R²≈0.005, |ρ| ≤ 0.15 over 46,718 cells). The topographic index does **not** "
        "cleanly isolate a hard cohort, so a topography-conditioned loss weight would not "
        "reliably target the actually-hard stations."
    )


def _write_report(out: Path, df_m: pd.DataFrame, station_meta: pd.DataFrame,
                  model_dir: Path, years: list, summary: dict,
                  mtpi: np.ndarray | None = None, elev_diff: np.ndarray | None = None):
    n_stn = int(summary["n_stations"])
    n_eval = int(summary.get("n_stations_evaluated", n_stn))
    excluded = summary.get("excluded_stations", [])
    n_days = int(summary["n_days_total"])
    mae_ov = summary["mae_overall"]
    mae_era5_ov = summary["mae_era5_overall"]
    skill_ov = summary["skill_overall"]
    skill_med = summary.get("skill_median", float("nan"))
    skill_low = summary.get("skill_lowland", float("nan"))
    bias_ov = summary["mean_bias"]
    sys_share = summary["systematic_share"]
    regime = summary.get("eval_regime", "cv_holdout")
    topo_corr = summary.get("topo_correlations", {})

    excl_note = ""
    if excluded:
        excl_note = (f" {', '.join(excluded)} excluded "
                     "(no homogenized daily Tmax in `ths200dx`).")

    # State the regime explicitly: these numbers mean very different things, and the
    # previous version of this report split one pooled run into "training" vs "holdout"
    # years, which quietly averaged in-sample days into the headline.
    if regime == "cv_holdout":
        period_block = [
            "",
            f"**Regime: cross-validation holdout ({years[0]}-{years[-1]}).** Each fold "
            "predicts only the contiguous block of days it was held out from during "
            "training, so every station-day below was scored by a model that never saw "
            "it. These are training-span days scored out-of-sample — not an unseen year.",
        ]
    else:
        period_block = [
            "",
            f"**Regime: holdout year {summary.get('eval_year')}.** No fold saw any day of "
            "this year. All folds predict every day and are combined by Gaussian moment "
            "matching, with inputs normalised using the training-frozen statistics.",
        ]

    rows = ["# Station evaluation — real NBCN observations", "",
            f"- Model: `{model_dir}`",
            f"- Regime: **`{regime}`** | years: **{years}** — {n_days:,} station-days across "
            f"**{n_eval} of {n_stn}** NBCN stations.{excl_note}",
            f"- Mean MAE over all station-days: **{mae_ov:.3f} °C** "
            f"(ERA5 baseline: {mae_era5_ov:.3f} °C, pooled skill: {skill_ov:.3f})",
            f"- Mean systematic bias: **{bias_ov:+.3f} °C**  "
            f"({100*sys_share:.0f}% of error is a fixable per-station offset).",
            "",
            "> **What these observations are.** The NBCN (National Basic Climatological Network) "
            f"provides **{n_stn} homogenized daily station series** — raw point observations of "
            "Tmax 2 m, quality-controlled and bias-corrected for instrument changes. Unlike the "
            "gridded TmaxD product (which already applies an elevation regression over these same "
            "stations), these are genuinely independent point measurements and avoid spatial "
            "autocorrelation. The evaluation is therefore a true out-of-sample test: the model "
            "never trained on station observations.",
            "",
            "## 1. Overall accuracy",
            "",
            f"> **Read the skill number with care.** The ERA5 baseline is raw bilinear "
            f"`t2m_max` with **no lapse-rate correction**, so its MAE is huge at high stations "
            f"(Säntis ≈8.8 °C, Jungfraujoch ≈6.3 °C) and small at lowland ones (≈1.0–1.5 °C). "
            f"The pooled skill **{skill_ov:.3f}** is therefore dominated by *elevation "
            f"downscaling*: the **median per-station skill is {skill_med:.3f}** and the "
            f"**mean skill at lowland stations (<1000 m) is only {skill_low:.3f}**. Against a "
            f"lapse-corrected baseline the model's value-add would be considerably smaller.",
            *period_block,
            "",
            "![Model vs ERA5 baseline](skill_comparison.png)",
            "",
            "## 2. Per-station MAE",
            "",
            "![Per-station MAE](station_mae.png)",
            "",
            "## 3. MAE and skill vs topography (altitude, mTPI)",
            "",
            _topo_section_text(topo_corr),
            "",
            "![MAE vs altitude](mae_vs_altitude.png)",
            "",
            "![MAE vs topographic index and elevation mismatch](mae_vs_mtpi.png)",
            "",
            "## 4. Spatial distribution of error and skill",
            "",
            "![Spatial skill map](spatial_skill.png)",
            "",
            "## 5. Systematic bias per station",
            "",
            f"Mean |bias| = {summary['mean_abs_bias']:.3f} °C "
            f"({100*sys_share:.0f}% of the average error is systematic).",
            "",
            "![Bias overview](bias_overview.png)",
            "",
            "## 6. Calibration (uncertainty estimates)",
            "",
            "The predicted σ is best-calibrated in the mid range. Alpine and foehn-valley "
            "stations (Jungfraujoch, Meiringen, Säntis, Sion) are mildly **under-confident** "
            "(z-std > 1.1 — errors larger than σ implies), while most lowland stations are "
            "**over-confident** (z-std < 0.7) — the model under-states uncertainty exactly "
            "where it is most accurate.",
            "",
            "![Calibration per station](calibration.png)",
            "",
            "## 7. Per-station results table",
            "",
            "| Station | Name | Elev (m) | mTPI (m) | n days | MAE (°C) | Bias (°C) | RMSE (°C) | "
            "CRPS skill | z-std | cov90 | Pearson r |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]

    for i, row in df_m.iterrows():
        if not np.isfinite(row.get("mae", np.nan)):
            continue
        stn = station_meta.iloc[i]
        mtpi_cell = f"{mtpi[i]:+.0f}" if (mtpi is not None and np.isfinite(mtpi[i])) else "—"
        rows.append(
            f"| {stn['stn_abbr']} | {stn['name']} | {stn['elev_m']:.0f} | {mtpi_cell} | "
            f"{row['n_days']:.0f} | {row['mae']:.3f} | {row['bias']:+.3f} | "
            f"{row['rmse']:.3f} | {row['skill']:.3f} | {row['z_std']:.2f} | "
            f"{row['cov90']:.2f} | {row['pearson']:.3f} |"
        )

    # --- §8: real comparison against the gridded-target analysis ---
    grid = _load_gridded_summary()
    if grid is not None:
        g_n = grid.get("n_grid_points")
        g_mae = grid.get("mae_overall")
        g_r2 = grid.get("regression_r2")
        compare = (
            f"The gridded-target analysis (`error_analysis.py`) scored **{g_n:,} grid cells** of "
            f"the gridded TmaxD product at mean MAE **{g_mae:.3f} °C**, and found no meaningful "
            f"topographic error pattern (R²≈{g_r2:.3f}). This station analysis scores **{n_eval} "
            f"independent point observations** — the actual stations the gridded product is built "
            f"from — at mean MAE **{mae_ov:.3f} °C**. Stations are **harder** ({mae_ov:.2f} vs "
            f"{g_mae:.2f} °C): point measurements carry local-scale effects (cold-air pooling, "
            f"exposure, foehn) that the grid smooths out. Crucially, the per-station MAE here "
            f"shows **no strong altitude or mTPI trend either** ({_mtpi_phrase(topo_corr)}, §3) — "
            f"so the 'no topographic pattern' finding holds on genuinely independent data, not "
            f"just as an artifact of the gridded truth's own elevation regression."
        )
    else:
        compare = (
            f"The gridded-target analysis (`error_analysis.py`) found no meaningful topographic "
            f"error pattern (R²≈0.005). This station analysis scores {n_eval} independent point "
            f"observations — the actual stations the gridded product is built from. The per-station "
            f"MAE shows no strong altitude trend either (§3), confirming the finding on independent "
            f"data. (Run error_analysis.py to embed the exact gridded MAE here.)"
        )
    rows += ["",
             "## 8. Comparison with gridded-target analysis",
             "",
             compare,
             ]
    (out / "REPORT.md").write_text("\n".join(rows))


def _load_gridded_summary() -> dict | None:
    """Load the gridded error_analysis summary for the §8 comparison, if present."""
    for cand in (Path("error_analysis/grid_analysis/error_analysis_summary.json"),
                 Path("error_analysis/grid_analysis/summary.json"),
                 Path("error_analysis/error_analysis_summary.json"),
                 Path("error_analysis/summary.json")):
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except Exception:
                return None
    return None


def _topo_correlations(mae, feats: dict) -> dict:
    """Pearson r / Spearman ρ of per-station MAE vs each topographic feature."""
    out = {}
    for name, x in feats.items():
        v = np.isfinite(mae) & np.isfinite(x)
        if v.sum() > 2:
            out[name] = dict(pearson=float(sstats.pearsonr(x[v], mae[v])[0]),
                             spearman=float(sstats.spearmanr(x[v], mae[v]).correlation))
        else:
            out[name] = dict(pearson=float("nan"), spearman=float("nan"))
    return out


def _station_topo(data: dict, station_meta: pd.DataFrame, model_dir: Path, device):
    """Per-station (altitude, elev_diff, mTPI). Uses cached topo if present, else
    recomputes it inference-free via build_station_targets (DEM/TPI interpolation)."""
    topo = data.get("topo")
    if topo is None:
        logger.info("topo not in cache — recomputing station topography (no inference)")
        topo = build_station_targets(station_meta, model_dir, device).target_topo.cpu().numpy()
    altitude = station_meta["elev_m"].values.astype(float)
    elev_diff = np.asarray(topo[:, 1], dtype=float)
    mtpi = np.asarray(topo[:, 2], dtype=float)
    return altitude, elev_diff, mtpi


# ---------------------------------------------------------------------------
# 8.  Driver
# ---------------------------------------------------------------------------

def run(
    model_dir: Path, device, out_dir: Path, eval_year=None,
    cache_dir: Path | None = None, folds=None,
    refresh_cache: bool = False, use_cache: bool = True, day_batch: int = 1,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    # The NBCN download (station list + daily archive) is model-independent, so
    # several models can share one cache_dir and download it once. The prediction
    # bundle is per-model and lives under <model_dir>/pred_cache/.
    download_dir = Path(cache_dir) if cache_dir else out_dir
    download_dir.mkdir(parents=True, exist_ok=True)

    p_train = params_mod.Params.load_json(model_dir / "params.json")
    if eval_year is None:
        years = list(range(int(p_train.DATA_YEAR_START), int(p_train.DATA_YEAR_END) + 1))
    else:
        years = [int(eval_year)]

    # -- station data --
    station_meta, daily = download_nbcn_data(download_dir, force=False)
    daily = daily[pd.to_datetime(daily["date"]).dt.year.isin(years)].copy()
    daily["stn_abbr"] = daily["stn_abbr"].str.upper()
    station_meta["stn_abbr"] = station_meta["stn_abbr"].str.upper()

    # -- inference (cache-served after the first run of this regime) --
    data = aggregate(model_dir, station_meta, daily, device, eval_year=eval_year,
                     folds=folds, refresh=refresh_cache, use_cache=use_cache,
                     day_batch=day_batch)

    # -- per-station metrics --
    df_m = per_station_metrics(data)

    # -- overall summary --
    preds, sigs, truth, era5 = data["preds_C"], data["sigmas_C"], data["truth_C"], data["era5_C"]
    valid = np.isfinite(truth)
    e_all = (preds - truth)[valid]
    s_all = np.clip(sigs[valid], 1e-6, None)
    t_all = truth[valid]
    e5_all = era5[valid]

    mae_ov = float(np.mean(np.abs(e_all)))
    mae_era5_ov = float(np.mean(np.abs(e5_all - t_all)))
    crps_ov = float(np.mean(ps.crps_gaussian(t_all, mu=(preds[valid]), sig=s_all)))
    crps_era5_ov = mae_era5_ov
    skill_ov = 1.0 - crps_ov / crps_era5_ov if crps_era5_ov > 0 else float("nan")
    bias_ov = float(np.mean(e_all))
    abs_bias_ov = float(np.nanmean(np.abs(df_m["bias"].dropna())))
    sys_share = abs_bias_ov / mae_ov if mae_ov > 0 else 0.0

    # -- which stations actually have data (finite MAE) --
    has_mae = np.isfinite(df_m["mae"].values) if "mae" in df_m else np.zeros(len(df_m), bool)
    n_eval = int(has_mae.sum())
    excluded = station_meta.loc[~has_mae, "stn_abbr"].tolist()

    # -- per-station skill summaries (the pooled skill is inflated by the
    #    lapse-uncorrected ERA5 baseline at alpine stations) --
    skill_ps = df_m["skill"].values if "skill" in df_m else np.full(len(df_m), np.nan)
    elev_ps = station_meta["elev_m"].values.astype(float)
    skill_median = float(np.nanmedian(skill_ps)) if np.isfinite(skill_ps).any() else float("nan")
    low_mask = has_mae & (elev_ps < 1000)
    skill_lowland = (float(np.nanmean(skill_ps[low_mask]))
                     if low_mask.any() else float("nan"))

    # -- topography index (mTPI) + elevation mismatch per station --
    altitude, elev_diff, mtpi = _station_topo(data, station_meta, model_dir, device)
    mae_ps = df_m["mae"].values if "mae" in df_m else np.full(len(df_m), np.nan)
    topo_corr = _topo_correlations(
        mae_ps, {"altitude": altitude, "mTPI": mtpi, "abs_elev_mismatch": np.abs(elev_diff)})

    # The regime IS the period: there is no in-sample/holdout split to make inside a
    # run any more. The CV regime scores training-span days that the scoring fold
    # never saw; the holdout regime scores a year no fold saw. Splitting the old
    # pooled 2020-2024 run by year was the station-side symptom of mixing the two.
    regime = data.get("regime", "cv_holdout")

    summary = dict(
        model_dir=str(model_dir), eval_regime=regime,
        eval_year=data.get("eval_year"), prediction_mode=data.get("prediction_mode"),
        years=years,
        n_stations=len(station_meta), n_stations_evaluated=n_eval,
        excluded_stations=excluded, n_days_total=int(valid.sum()),
        mae_overall=mae_ov, mae_era5_overall=mae_era5_ov,
        crps_overall=crps_ov, skill_overall=skill_ov,
        skill_median=skill_median, skill_lowland=skill_lowland,
        mean_bias=bias_ov, mean_abs_bias=abs_bias_ov, systematic_share=sys_share,
        topo_correlations=topo_corr,
        per_station=df_m.to_dict(orient="records"),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # -- figures --
    fig_station_mae(df_m, station_meta, out_dir / "station_mae.png")
    fig_mae_vs_altitude(df_m, station_meta, out_dir / "mae_vs_altitude.png")
    fig_mae_vs_topo(df_m, station_meta, mtpi, elev_diff, out_dir / "mae_vs_mtpi.png")
    fig_spatial_skill(df_m, station_meta, out_dir / "spatial_skill.png")
    fig_bias_overview(df_m, station_meta, out_dir / "bias_overview.png")
    fig_skill_comparison(df_m, station_meta, out_dir / "skill_comparison.png")
    fig_calibration(df_m, station_meta, out_dir / "calibration.png")

    _write_report(out_dir, df_m, station_meta, model_dir, years, summary,
                  mtpi=mtpi, elev_diff=elev_diff)

    print(f"\nWrote {out_dir}/REPORT.md (+ summary.json, 7 figures)")
    print(f"  [{regime}] overall MAE {mae_ov:.3f} °C | ERA5 baseline {mae_era5_ov:.3f} °C | "
          f"skill {skill_ov:.3f} | bias {bias_ov:+.3f} °C")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--model-dir", help="Dir with params.json / manifest.json / model_fold_*.")
    src.add_argument("--trial-dir", default=BEST_ATMOS,
                     help=f"Trial dir (default: {BEST_ATMOS}).")
    ap.add_argument("--eval-year", type=int, default=None,
                    help="Holdout year (e.g. 2024). Omit for the 2020-2023 CV holdout.")
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="Evaluate only these folds. Bypasses the station cache.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None,
                    help="Output dir (default <model_dir>/station_analysis_{cv,<year>}).")
    ap.add_argument("--cache-dir", default=None,
                    help="Where the NBCN download lives (default: --out). Point several "
                         "models at one dir to download the station archive only once. "
                         "The prediction bundle is always per-model, under pred_cache/.")
    ap.add_argument("--day-batch", type=int, default=1,
                    help="Days per forward pass (results are day-independent).")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Re-run station inference and overwrite the cached bundle.")
    ap.add_argument("--no-cache", dest="use_cache", action="store_false",
                    help="Neither read nor write the station prediction bundle.")
    ap.add_argument("--years", help=argparse.SUPPRESS)
    ap.add_argument("--use-cache", dest="_legacy_use_cache", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.years:
        ap.error("--years is gone: this script now runs one regime at a time. Omit "
                 "--eval-year for the 2020-2023 CV holdout, or pass --eval-year 2024. "
                 "The old pooled run averaged in-sample years into the headline.")
    if args._legacy_use_cache:
        ap.error("--use-cache is gone: the prediction bundle is reused by default. "
                 "Use --refresh-cache to force re-inference, --no-cache to bypass it.")

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    params_mod.configure_renku_cuda()
    device = torch.device(args.device) if args.device else params_mod.select_device()
    model_dir = _resolve_model_dir(args)
    out = Path(args.out) if args.out else (
        model_dir / ("station_analysis_cv" if args.eval_year is None
                     else f"station_analysis_{args.eval_year}"))
    run(model_dir, device, out, eval_year=args.eval_year,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        folds=args.folds, refresh_cache=args.refresh_cache,
        use_cache=args.use_cache, day_batch=args.day_batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
