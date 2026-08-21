#!/usr/bin/env python3
"""
Apply a trained convNPClimate model to new (unseen) dates.

Reads ``manifest.json`` for normalization — no training data required.
The companion script ``infer.py`` is for evaluating in-distribution holdout
folds; this script is for applying the trained model to entirely new dates.

Target points default to the MeteoSwiss TmaxD station grid.
Supply ``--targets`` to predict at arbitrary lat/lon coordinates
(CSV with columns ``lat``, ``lon``; one row per point).

Output is a compressed NumPy ``.npz`` archive containing:
  dates        (T,)         ISO-format date strings
  lat          (P,)         target latitudes  (° N)
  lon          (P,)         target longitudes (° E)
  pred_mean    (T, P)       predicted tmax mean  (°C)
  pred_sigma   (T, P)       predicted tmax sigma (°C); within-model, or ensemble
                            with --ensemble-var
  pred_sigma_within   (T,P) within-model sigma  (°C)   [ensemble runs only]
  pred_sigma_ensemble (T,P) within + between-fold sigma (°C) [ensemble runs only]
  pred_sigma_between  (T,P) between-fold-only sigma (°C)      [ensemble runs only]

Example
-------
    # Best fold, new year:
    python predict.py --model-dir trained_models/my_run/tmax \\
        --date-start 2024-01-01 --date-end 2024-12-31 \\
        --output predictions_2024.npz

    # Ensemble (average over all fold checkpoints):
    python predict.py --model-dir trained_models/my_run/tmax \\
        --all-folds --date-start 2024-01-01 --date-end 2024-12-31

    # Custom target locations:
    python predict.py --model-dir trained_models/my_run/tmax \\
        --targets my_sites.csv --date-start 2024-01-01 --date-end 2024-12-31
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr

# Headless plotting (only used by --plot); set before pyplot is imported.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import params as params_mod
import datasets as ds
import model_factory
from inference import predict_single_day, predict_day_range
from convCNP.validation.utils import get_dists

logger = logging.getLogger("predict")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(model_dir: Path) -> dict:
    path = model_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {model_dir}. "
            "Re-train with the current train.py to generate it."
        )
    return json.loads(path.read_text())


def manifest_to_dists_metadata(manifest: dict) -> ds.Era5Metadata:
    """Reconstruct the Era5Metadata needed by calculate_dists_meteoswiss."""
    dg = manifest["dists_grid"]
    norm = manifest["normalization"]
    sdata = norm.get("surface_data", {}) or {}
    selev = norm.get("surface_elevation", {}) or {}
    return ds.Era5Metadata(
        data_mean=sdata.get("mean", 0.0),
        data_std=sdata.get("std", 1.0),
        lat_min=dg["lat_bounds"][0],
        lat_max=dg["lat_bounds"][1],
        lon_min=dg["lon_bounds"][0],
        lon_max=dg["lon_bounds"][1],
        lat_coords=np.array(dg["lat_coords"]),
        lon_coords=np.array(dg["lon_coords"]),
        elev_mean=selev.get("mean"),
        elev_std=selev.get("std"),
    )


# ---------------------------------------------------------------------------
# Context construction (normalized with stored manifest stats)
# ---------------------------------------------------------------------------

def _date_range(date_start: str, date_end: str) -> pd.DatetimeIndex:
    return pd.date_range(date_start, date_end, freq="D")


def _seasonal_channels(dates: pd.DatetimeIndex, n_lat: int, n_lon: int,
                        device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos_ch, sin_ch), each (T, lat, lon)."""
    doy = dates.dayofyear.values
    rads = (doy - 1) / 365.0 * 2 * np.pi
    cos_t = torch.from_numpy(np.cos(rads).astype(np.float32)).to(device)
    sin_t = torch.from_numpy(np.sin(rads).astype(np.float32)).to(device)
    cos_ch = cos_t.view(-1, 1, 1).expand(-1, n_lat, n_lon)
    sin_ch = sin_t.view(-1, 1, 1).expand(-1, n_lat, n_lon)
    return cos_ch, sin_ch


def build_surface_context(
    manifest: dict,
    p: params_mod.Params,
    dates: pd.DatetimeIndex,
    device: torch.device,
) -> torch.Tensor:
    """
    Load ERA5 surface data for ``dates`` and normalize with stored manifest
    stats (instead of recomputing from the new data).

    Returns context tensor (T, C, lat, lon) in manifest channel order.
    """
    norm = manifest["normalization"]
    data_mean = norm["surface_data"]["mean"]
    data_std  = norm["surface_data"]["std"]
    lat_min, lat_max = norm["lat_bounds"]
    lon_min, lon_max = norm["lon_bounds"]

    logger.info("loading ERA5 surface data %s – %s", dates[0].date(), dates[-1].date())
    raw = xr.open_mfdataset(p.ERA5_MAX_TEMP_GLOB, combine="by_coords", preprocess=ds._era5_mf_preprocess)
    raw = raw.sel(time=slice(str(dates[0].date()), str(dates[-1].date())))

    # Align exactly to requested dates (avoid surprises from partial files).
    req = dates.normalize()
    avail = pd.to_datetime(raw.time.values).normalize()
    missing = req.difference(avail)
    if len(missing):
        raise ValueError(f"ERA5 surface data missing {len(missing)} requested dates "
                         f"(e.g. {missing[0].date()}).")
    raw = raw.sel(time=avail[avail.isin(req)])

    lats = raw.latitude.values
    lons = raw.longitude.values
    T, nlat, nlon = len(raw.time), len(lats), len(lons)

    lat_norm = ((lats - lat_min) / (lat_max - lat_min)).astype(np.float32)
    lon_norm = ((lons - lon_min) / (lon_max - lon_min)).astype(np.float32)
    lat_grid = np.broadcast_to(lat_norm[:, None], (nlat, nlon)).copy()
    lon_grid = np.broadcast_to(lon_norm[None, :], (nlat, nlon)).copy()
    lat_ch = torch.from_numpy(lat_grid).to(device).unsqueeze(0).expand(T, -1, -1)
    lon_ch = torch.from_numpy(lon_grid).to(device).unsqueeze(0).expand(T, -1, -1)

    name_to_tensor: dict[str, torch.Tensor] = {"lat": lat_ch, "lon": lon_ch}

    if p.USE_SURFACE:
        raw_vals = raw["t2m_max"].values.astype(np.float32)
        data_norm = (raw_vals - data_mean) / data_std
        name_to_tensor["data"] = torch.from_numpy(data_norm).to(device)

    if p.SEASONAL_FEATURES:
        cos_ch, sin_ch = _seasonal_channels(dates, nlat, nlon, device)
        name_to_tensor["cos_time"] = cos_ch
        name_to_tensor["sin_time"] = sin_ch

    # Elevation channel (normalize with stored stats). Skipped when the model was
    # trained with --no-geopotential (USE_ELEVATION_CHANNEL=False); the manifest's
    # channel_names then omits 'elevation', so _stack_channels would drop it anyway,
    # but guarding here avoids a pointless geopotential load.
    selev = norm.get("surface_elevation", {}) or {}
    if (selev.get("mean") is not None and p.ERA5_GEOPOTENTIAL_GLOB
            and getattr(p, "USE_ELEVATION_CHANNEL", True)):
        logger.info("loading geopotential for elevation channel")
        geo = xr.open_mfdataset(p.ERA5_GEOPOTENTIAL_GLOB, combine="by_coords", preprocess=ds._era5_mf_preprocess)
        # The geopotential file has swapped axis labels (known quirk); swap back.
        geo = geo.rename({"latitude": "longitude", "longitude": "latitude"})
        altitude = geo["z"] / 9.80665  # geopotential → metres
        if "time" in altitude.dims:
            altitude = altitude.isel(time=0)
        # rename() relabels axes but keeps their order, so after the swap the dims
        # are (longitude, latitude); force (latitude, longitude) to match the other
        # channels before stacking (the training path transposes likewise).
        alt_interp = altitude.interp(
            latitude=lats, longitude=lons, method="linear"
        ).transpose("latitude", "longitude").values.astype(np.float32)
        elev_norm = (alt_interp - selev["mean"]) / selev["std"]
        elev_ch = torch.from_numpy(elev_norm).to(device).unsqueeze(0).expand(T, -1, -1)
        name_to_tensor["elevation"] = elev_ch

    # ERA5-Land precipitation channel, normalized with the stored training stats.
    # This goes through the SAME loader as training (ds.load_era5_precip_aligned),
    # so the two paths cannot drift in how the field is read, converted or
    # date-aligned -- only the normalization differs (frozen here, computed there).
    per_ch = norm.get("per_channel", {}) or {}
    if getattr(p, "USE_SURFACE_PRECIP", False) and "tp" in per_ch:
        logger.info("loading ERA5-Land precipitation channel")
        tp_vals, tp_covered = ds.load_era5_precip_aligned(
            p.ERA5_PRECIP_GLOB, lats, lons, req
        )
        if not tp_covered.all():
            gaps = req[~tp_covered]
            raise ValueError(
                f"ERA5 precip does not cover {len(gaps)} of the requested dates "
                f"(e.g. {gaps[0].date()}); cannot build the 'tp' channel for "
                f"{req[0].date()}..{req[-1].date()}."
            )
        tp_norm = (tp_vals - per_ch["tp"]["mean"]) / per_ch["tp"]["std"]
        name_to_tensor["tp"] = torch.from_numpy(tp_norm.astype(np.float32)).to(device)

    return _stack_channels(manifest["channel_names"], name_to_tensor)


def precip_covered_dates(p: params_mod.Params, dates: pd.DatetimeIndex,
                         label: str = "") -> pd.DatetimeIndex:
    """Trim a requested day range to the span ERA5-Land tp actually covers.

    A no-op unless the model uses the tp channel. The ERA5-Land precip files are
    labelled by accumulation window, so a calendar year is typically missing its
    31 Dec; every consumer of the tp channel therefore has to score the days it
    has rather than fail on the last one. An INTERIOR gap still raises -- that is
    a data defect, not something to silently slice around.
    """
    if not getattr(p, "USE_SURFACE_PRECIP", False):
        return dates
    _, covered = ds.load_era5_precip_aligned(p.ERA5_PRECIP_GLOB, None, None, dates)
    if not covered.any():
        raise ValueError(f"ERA5 precip covers none of the requested days{label}.")
    idx = np.flatnonzero(covered)
    span = slice(int(idx[0]), int(idx[-1]) + 1)
    if not covered[span].all():
        gaps = dates[span][~covered[span]]
        raise ValueError(
            f"ERA5 precip has {len(gaps)} interior gap(s){label} "
            f"(e.g. {gaps[0].date()}); refusing to work around a hole.")
    if span.start != 0 or span.stop != len(dates):
        logger.warning("ERA5 precip covers %d/%d days%s (%s..%s); using that span",
                       span.stop - span.start, len(dates), label,
                       dates[span][0].date(), dates[span][-1].date())
    return dates[span]


def build_atmospheric_context(
    manifest: dict,
    p: params_mod.Params,
    dates: pd.DatetimeIndex,
    device: torch.device,
) -> torch.Tensor:
    """
    Load ERA5 pressure-level data for ``dates`` at native coarse grid,
    normalize with stored manifest per-channel stats.

    Returns context tensor (T, C, lat_coarse, lon_coarse) in manifest channel order.
    """
    norm = manifest["normalization"]
    per_ch = norm["per_channel"]
    dg = manifest["dists_grid"]
    lat_min, lat_max = norm["lat_bounds"]
    lon_min, lon_max = norm["lon_bounds"]

    req = dates.normalize()
    name_to_tensor: dict[str, torch.Tensor] = {}
    native_lat: np.ndarray | None = None
    native_lon: np.ndarray | None = None

    logger.info("loading ERA5 pressure levels %s – %s", dates[0].date(), dates[-1].date())
    for var in p.ATMOS_VARIABLES:
        for hour in p.ATMOS_HOURS:
            glob = str(Path(p.ERA5_PRESSURE_LEVEL_DIR) / var / f"{var}_pl-*-{hour}.nc")
            ds_pl = xr.open_mfdataset(glob, combine="by_coords", preprocess=ds._era5_mf_preprocess)

            # Normalise dimension names.
            rename = {}
            if "valid_time" in ds_pl.dims and "time" not in ds_pl.dims:
                rename["valid_time"] = "time"
            for lname in ("level", "plev", "isobaricInhPa"):
                if lname in ds_pl.dims and "pressure_level" not in ds_pl.dims:
                    rename[lname] = "pressure_level"
                    break
            if rename:
                ds_pl = ds_pl.rename(rename)

            ds_pl = ds_pl.assign_coords(
                time=pd.to_datetime(ds_pl.time.values).normalize()
            )
            missing = req.difference(pd.to_datetime(ds_pl.time.values))
            if len(missing):
                raise ValueError(
                    f"Pressure-level {var}@{hour}UTC missing {len(missing)} dates "
                    f"(e.g. {missing[0].date()})."
                )
            ds_pl = ds_pl.sel(time=req)

            # Pin coordinate labels to first file so all files align identically.
            if native_lat is None:
                native_lat = ds_pl.latitude.values
                native_lon = ds_pl.longitude.values
            else:
                ds_pl = ds_pl.assign_coords(latitude=native_lat, longitude=native_lon)

            for level in p.ATMOS_LEVELS:
                ch_name = f"{var}{level}_{hour}"
                if ch_name not in per_ch:
                    raise KeyError(
                        f"Channel {ch_name!r} not found in manifest.normalization.per_channel. "
                        "Ensure this model was trained with the same ATMOS_VARIABLES/LEVELS/HOURS."
                    )
                raw_vals = ds_pl[var].sel(pressure_level=level).values.astype(np.float32)
                norm_vals = (raw_vals - per_ch[ch_name]["mean"]) / per_ch[ch_name]["std"]
                name_to_tensor[ch_name] = torch.from_numpy(norm_vals).to(device)

    T   = len(dates)
    nlat = len(native_lat)
    nlon = len(native_lon)

    # Coarse scaffold uses the dists_grid bounds (same as training).
    lat_norm = ((native_lat - lat_min) / (lat_max - lat_min)).astype(np.float32)
    lon_norm = ((native_lon - lon_min) / (lon_max - lon_min)).astype(np.float32)
    lat_grid = np.broadcast_to(lat_norm[:, None], (nlat, nlon)).copy()
    lon_grid = np.broadcast_to(lon_norm[None, :], (nlat, nlon)).copy()
    lat_ch = torch.from_numpy(lat_grid).to(device).unsqueeze(0).expand(T, -1, -1)
    lon_ch = torch.from_numpy(lon_grid).to(device).unsqueeze(0).expand(T, -1, -1)
    name_to_tensor["lat"] = lat_ch
    name_to_tensor["lon"] = lon_ch

    if p.SEASONAL_FEATURES:
        cos_ch, sin_ch = _seasonal_channels(dates, nlat, nlon, device)
        name_to_tensor["cos_time"] = cos_ch
        name_to_tensor["sin_time"] = sin_ch

    if "elevation" in per_ch:
        # Coarse-grid elevation: normalize with stored stats from training.
        # We need the ERA5 surface geopotential to build it.
        if p.ERA5_GEOPOTENTIAL_GLOB:
            logger.info("interpolating coarse elevation scaffold")
            geo = xr.open_mfdataset(p.ERA5_GEOPOTENTIAL_GLOB, combine="by_coords", preprocess=ds._era5_mf_preprocess)
            geo = geo.rename({"latitude": "longitude", "longitude": "latitude"})
            altitude = geo["z"] / 9.80665
            if "time" in altitude.dims:
                altitude = altitude.isel(time=0)
            lat_c = np.clip(native_lat, float(altitude.latitude.min()),
                            float(altitude.latitude.max()))
            lon_c = np.clip(native_lon, float(altitude.longitude.min()),
                            float(altitude.longitude.max()))
            elev = altitude.interp(latitude=lat_c, longitude=lon_c, method="linear")
            elev = elev.assign_coords(latitude=native_lat, longitude=native_lon)
            elev_vals = elev.transpose("latitude", "longitude").values.astype(np.float32)
            elev_norm = (elev_vals - per_ch["elevation"]["mean"]) / per_ch["elevation"]["std"]
            elev_ch = torch.from_numpy(elev_norm).to(device).unsqueeze(0).expand(T, -1, -1)
            name_to_tensor["elevation"] = elev_ch

    # Optional surface-anchor channels.
    if p.USE_SFC_ATMOS and p.ERA5_SURFACE_DIR:
        logger.info("loading ERA5 surface anchor channels")
        sfc_stats_dummy: list = []
        sfc, sfc_names = ds.load_era5_surface_levels(
            p.ERA5_SURFACE_DIR,
            lat_coords=native_lat,
            lon_coords=native_lon,
            time_coords=dates.values.astype("datetime64[ns]"),
            variables=list(p.ATMOS_SFC_VARIABLES),
            hours=list(p.ATMOS_HOURS),
            stats_out=sfc_stats_dummy,
            device=device,
        )
        # Re-normalize with stored stats (undo on-the-fly normalization, apply stored).
        for i, name in enumerate(sfc_names):
            if name not in per_ch:
                raise KeyError(f"Surface anchor channel {name!r} missing from manifest.")
            computed = sfc_stats_dummy[i]
            # channel i of sfc is already (raw-computed_mean)/computed_std; invert then restd
            raw_ch = sfc[:, i] * computed["std"] + computed["mean"]
            norm_ch = (raw_ch - per_ch[name]["mean"]) / per_ch[name]["std"]
            name_to_tensor[name] = norm_ch

    return _stack_channels(manifest["channel_names"], name_to_tensor)


def _stack_channels(channel_names: list[str],
                    name_to_tensor: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack channels in manifest order into (T, C, lat, lon)."""
    missing = [n for n in channel_names if n not in name_to_tensor]
    if missing:
        raise RuntimeError(
            f"Could not build context: channels missing from loaded data: {missing}. "
            "Check that data paths cover the required ERA5 variables."
        )
    parts = [name_to_tensor[n] for n in channel_names]
    # Each part is (T, lat, lon); stack along new dim 1 → (T, C, lat, lon).
    return torch.stack(parts, dim=1)


# ---------------------------------------------------------------------------
# Target-point preparation
# ---------------------------------------------------------------------------

def load_meteoswiss_targets(
    manifest: dict,
    p: params_mod.Params,
    device: torch.device,
) -> tuple[xr.DataArray, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Load the MeteoSwiss station grid as target points.

    Returns (target_x, target_topo, lat_arr, lon_arr).
    target_x is normalized (point, coord) DataArray consumed by
    calculate_dists_meteoswiss.
    """
    dists_meta = manifest_to_dists_metadata(manifest)

    hi_res_elevation, hi_res_tpi = ds.load_high_res_topography(
        p.HI_RES_TOPOGRAPHY_ZARR_PATH
    )
    # Grid elevation for elev_diff feature; surface mode has it from geopotential.
    grid_elev = None
    if p.ERA5_GEOPOTENTIAL_GLOB:
        geo = xr.open_mfdataset(p.ERA5_GEOPOTENTIAL_GLOB, combine="by_coords", preprocess=ds._era5_mf_preprocess)
        geo = geo.rename({"latitude": "longitude", "longitude": "latitude"})
        alt = geo["z"] / 9.80665
        if "time" in alt.dims:
            alt = alt.isel(time=0)
        grid_elev = alt

    target_x, _, target_topo = ds.prepare_meteoswiss_targets(
        p.METEO_SWISS_MAX_TEMP_GLOB,
        normalization_stats=dists_meta,
        data_var="TmaxD",
        grid_elevation=grid_elev if p.USE_ELEVATION else None,
        hi_res_elevation=hi_res_elevation if p.USE_ELEVATION else None,
        hi_res_tpi=hi_res_tpi if p.USE_MTPI else None,
        convert_to_kelvin=True,
        normalize_targets=True,
        device=device,
    )

    lat_arr = (
        target_x.sel(coord="lat").values
        * (dists_meta.lat_max - dists_meta.lat_min)
        + dists_meta.lat_min
    )
    lon_arr = (
        target_x.sel(coord="lon").values
        * (dists_meta.lon_max - dists_meta.lon_min)
        + dists_meta.lon_min
    )
    return target_x, target_topo, lat_arr, lon_arr


def load_custom_targets(
    targets_csv: Path,
    manifest: dict,
    p: params_mod.Params,
    device: torch.device,
) -> tuple[xr.DataArray, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Load custom target points from a CSV (columns: lat, lon).

    Elevation features are interpolated from the hi-res topo if available.
    Returns (target_x, target_topo, lat_arr, lon_arr).
    """
    df = pd.read_csv(targets_csv)
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError(f"targets CSV must have columns 'lat' and 'lon'; got {list(df.columns)}")
    lat_arr = df["lat"].values.astype(np.float64)
    lon_arr = df["lon"].values.astype(np.float64)

    dists_meta = manifest_to_dists_metadata(manifest)
    lat_min, lat_max = dists_meta.lat_min, dists_meta.lat_max
    lon_min, lon_max = dists_meta.lon_min, dists_meta.lon_max

    lat_norm = (lat_arr - lat_min) / (lat_max - lat_min)
    lon_norm = (lon_arr - lon_min) / (lon_max - lon_min)

    target_x = xr.DataArray(
        np.stack([lat_norm, lon_norm], axis=1).astype(np.float32),
        dims=["point", "coord"],
        coords={"coord": ["lat", "lon"]},
    )

    # Build topo features (true_elev, elev_diff, mTPI) via hi-res DEM.
    hi_res_elevation, hi_res_tpi = ds.load_high_res_topography(
        p.HI_RES_TOPOGRAPHY_ZARR_PATH
    )
    n_pts = len(lat_arr)
    if p.USE_ELEVATION and hi_res_elevation is not None:
        x_lv95, y_lv95 = ds.wgs84_to_lv95(lon_arr, lat_arr)
        true_elev = hi_res_elevation.interp(
            x=xr.DataArray(x_lv95, dims="point"),
            y=xr.DataArray(y_lv95, dims="point"),
            method="linear",
        ).values.astype(np.float32)

        # ERA5-grid elevation at target coords (for elev_diff).
        grid_elev_at_pts = np.zeros(n_pts, dtype=np.float32)
        if p.ERA5_GEOPOTENTIAL_GLOB:
            geo = xr.open_mfdataset(p.ERA5_GEOPOTENTIAL_GLOB, combine="by_coords", preprocess=ds._era5_mf_preprocess)
            geo = geo.rename({"latitude": "longitude", "longitude": "latitude"})
            alt = geo["z"] / 9.80665
            if "time" in alt.dims:
                alt = alt.isel(time=0)
            grid_elev_at_pts = alt.interp(
                latitude=xr.DataArray(lat_arr, dims="point"),
                longitude=xr.DataArray(lon_arr, dims="point"),
                method="linear",
            ).values.astype(np.float32)

        tpi = np.zeros(n_pts, dtype=np.float32)
        if hi_res_tpi is not None and p.USE_MTPI:
            tpi = hi_res_tpi.interp(
                x=xr.DataArray(x_lv95, dims="point"),
                y=xr.DataArray(y_lv95, dims="point"),
                method="linear",
            ).values.astype(np.float32)

        topo = np.stack([true_elev, true_elev - grid_elev_at_pts, tpi], axis=1)
    else:
        topo = np.zeros((n_pts, 3), dtype=np.float32)

    target_topo = torch.from_numpy(topo).to(device)
    return target_x, target_topo, lat_arr, lon_arr


# ---------------------------------------------------------------------------
# Prediction loop
# ---------------------------------------------------------------------------

def predict_all_days(
    model: torch.nn.Module,
    context: torch.Tensor,
    dists: torch.Tensor,
    target_topo: torch.Tensor,
    seasonal: torch.Tensor | None,
    device: torch.device,
    day_batch: int = 1,
    day_start: int = 0,
    day_end: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a day slice of context. Returns (preds, sigmas) in normalized units.

    ``day_batch`` days are processed per forward pass. ``day_batch=1`` is the original
    one-day-at-a-time behaviour; larger values batch the day axis (the model's batch axis)
    so the GPU is not starved by 1-row forwards — the key speedup for GPU inference. Results
    are independent of ``day_batch`` (each day is predicted independently either way).

    ``day_start``/``day_end`` restrict the loop to ``context[day_start:day_end]``, returning
    ``(day_end - day_start, P)``. The default spans everything, so existing callers are
    unaffected. This is what lets a cross-validation fold predict *only* its own held-out
    block without materialising predictions for days it trained on.
    """
    T = context.shape[0]
    day_end = T if day_end is None else min(day_end, T)
    if not 0 <= day_start < day_end:
        raise ValueError(f"empty day slice [{day_start}, {day_end}) over {T} days")
    preds, sigmas = [], []
    for start in range(day_start, day_end, max(1, day_batch)):
        end = min(start + max(1, day_batch), day_end)
        p, s = predict_day_range(
            model, context, start, end, dists, target_topo, seasonal, device
        )
        preds.append(p.cpu().numpy())   # (B, P)
        sigmas.append(s.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(sigmas, axis=0)


def denormalize(arr: np.ndarray, dists_meta: ds.Era5Metadata) -> np.ndarray:
    """Denormalize model output from normalized units → Celsius."""
    return arr * dists_meta.data_std + dists_meta.data_mean - ds.KELVIN_OFFSET


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(
    model_dir: Path,
    date_start: str,
    date_end: str,
    targets_csv: Optional[Path],
    folds: Optional[list[int]],
    all_folds: bool,
    data_dir: Optional[Path],
    device: torch.device,
    output: Path,
    ensemble_var: bool = False,
    plot: bool = False,
):
    manifest = load_manifest(model_dir)
    p = params_mod.Params.load_json(model_dir / "params.json")
    p.DEVICE = str(device)

    # Override data paths if the model was trained on a different machine.
    if data_dir is not None:
        new_paths = ds.build_data_paths(data_dir)
        import dataclasses
        p = dataclasses.replace(
            p,
            ERA5_MAX_TEMP_GLOB=new_paths.ERA5_MAX_TEMP_GLOB,
            ERA5_PRECIP_GLOB=new_paths.ERA5_PRECIP_GLOB,
            ERA5_GEOPOTENTIAL_GLOB=new_paths.ERA5_GEOPOTENTIAL_GLOB,
            ERA5_PRESSURE_LEVEL_DIR=new_paths.ERA5_PRESSURE_LEVEL_DIR,
            ERA5_SURFACE_DIR=new_paths.ERA5_SURFACE_DIR,
            METEO_SWISS_MAX_TEMP_GLOB=new_paths.METEO_SWISS_MAX_TEMP_GLOB,
            METEO_SWISS_PRECIP_GLOB=new_paths.METEO_SWISS_PRECIP_GLOB,
            HI_RES_TOPOGRAPHY_ZARR_PATH=new_paths.HI_RES_TOPOGRAPHY_ZARR_PATH,
        )

    dates = _date_range(date_start, date_end)
    logger.info("prediction window: %s – %s (%d days)", date_start, date_end, len(dates))

    # ---- Build context ----
    if p.USE_ATMOSPHERIC and p.ATMOS_NATIVE_GRID:
        context = build_atmospheric_context(manifest, p, dates, device)
    else:
        context = build_surface_context(manifest, p, dates, device)
    logger.info("context tensor: %s", tuple(context.shape))

    # ---- Build dists + targets ----
    dists_meta = manifest_to_dists_metadata(manifest)

    if targets_csv is not None:
        logger.info("loading custom targets from %s", targets_csv)
        target_x, target_topo, lat_arr, lon_arr = load_custom_targets(
            targets_csv, manifest, p, device
        )
    else:
        logger.info("loading MeteoSwiss target grid")
        target_x, target_topo, lat_arr, lon_arr = load_meteoswiss_targets(
            manifest, p, device
        )

    dists = ds.calculate_dists_meteoswiss(dists_meta, target_x, device=device)
    logger.info("dists tensor: %s, targets: %d", tuple(dists.shape), len(lat_arr))

    # ---- Seasonal features for MLP ----
    seasonal = (
        ds.compute_seasonal_features(dates.values.astype("datetime64[ns]"), device=device)
        if p.SEASONAL_FEATURES
        else None
    )

    # ---- Determine fold checkpoints ----
    channel_groups = (
        ds.channel_groups_by_variable(manifest["channel_names"])
        if p.ENCODER != "flat"
        else None
    )
    if all_folds:
        ckpts = sorted(model_dir.glob("model_fold_*"))
        if not ckpts:
            raise FileNotFoundError(f"No model_fold_* checkpoints in {model_dir}")
        fold_ids = [int(c.name.split("_")[-1]) for c in ckpts]
    elif folds:
        fold_ids = folds
    else:
        fold_ids = [0]

    logger.info("using fold checkpoints: %s", fold_ids)

    # ---- Predict (ensemble if multiple folds) ----
    all_preds, all_sigmas = [], []
    for fold in fold_ids:
        ckpt = model_dir / f"model_fold_{fold}"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        model, epoch = model_factory.load_model_checkpoint(
            ckpt, p, device, channel_groups=channel_groups
        )
        logger.info("fold %d (epoch %d): running %d days × %d points ...",
                    fold, epoch, len(dates), len(lat_arr))
        preds_norm, sigmas_norm = predict_all_days(
            model, context, dists, target_topo, seasonal, device
        )
        all_preds.append(preds_norm)
        all_sigmas.append(sigmas_norm)

    # Ensemble: mean of means for the prediction.
    preds_mean = np.mean(all_preds, axis=0)

    # Two predictive-sigma modes, both from this single inference pass:
    #   within   sigma^2 = mean(sigma_k^2)               (average within-model var)
    #   total    sigma^2 = mean(sigma_k^2) + var(mu_k)   (+ between-fold disagreement)
    is_ensemble = len(all_preds) > 1
    within_var = np.mean(np.array(all_sigmas) ** 2, axis=0)
    between_var = (
        np.var(np.array(all_preds), axis=0) if is_ensemble
        else np.zeros_like(within_var)
    )
    sigma_within = np.sqrt(within_var)
    sigma_total = np.sqrt(within_var + between_var)
    # Primary pred_sigma honors --ensemble-var for backward compatibility.
    sigma_primary = sigma_total if (ensemble_var and is_ensemble) else sigma_within

    # Denormalize to Celsius (sigma is a spread: only the std scale applies).
    sd = dists_meta.data_std
    preds_degc         = denormalize(preds_mean, dists_meta)
    sigma_primary_degc = sigma_primary * sd
    sigma_within_degc  = sigma_within * sd
    sigma_total_degc   = sigma_total * sd
    sigma_between_degc = np.sqrt(between_var) * sd  # the added fold-disagreement term

    # ---- Save ----
    date_strings = dates.strftime("%Y-%m-%d").tolist()
    save_kwargs = dict(
        dates=np.array(date_strings),
        lat=lat_arr.astype(np.float32),
        lon=lon_arr.astype(np.float32),
        pred_mean=preds_degc.astype(np.float32),
        pred_sigma=sigma_primary_degc.astype(np.float32),
    )
    if is_ensemble:
        # Both modes are always available when ensembling, regardless of the flag.
        save_kwargs.update(
            pred_sigma_within=sigma_within_degc.astype(np.float32),
            pred_sigma_ensemble=sigma_total_degc.astype(np.float32),
            pred_sigma_between=sigma_between_degc.astype(np.float32),
        )
    np.savez_compressed(output, **save_kwargs)

    _print_summary(preds_degc, sigma_primary_degc, date_strings, lat_arr, lon_arr, output,
                   fold_ids, manifest, is_ensemble, sigma_within_degc, sigma_total_degc)

    if plot:
        if is_ensemble:
            _plot_sigma_modes(sigma_within_degc, sigma_total_degc, sigma_between_degc,
                              lat_arr, lon_arr, output)
        else:
            logger.warning("--plot ignored: sigma-mode comparison needs >1 fold "
                           "(use --all-folds).")


def _print_summary(preds, sigmas, dates, lat, lon, output, fold_ids, manifest,
                   is_ensemble=False, sigma_within=None, sigma_total=None):
    print()
    print(f"=== Predictions written: {output} ===")
    print(f"  dates  : {dates[0]} – {dates[-1]}  ({len(dates)} days)")
    print(f"  points : {len(lat):,}  (lat {lat.min():.2f}–{lat.max():.2f}, "
          f"lon {lon.min():.2f}–{lon.max():.2f})")
    print(f"  folds  : {fold_ids}")
    print(f"  grid   : {manifest['grid_mode']}")
    print(f"  pred_mean  : {preds.mean():.2f} ± {preds.std():.2f} °C  "
          f"[{preds.min():.1f}, {preds.max():.1f}]")
    print(f"  pred_sigma : {sigmas.mean():.2f} °C  (mean, the saved pred_sigma)")
    if is_ensemble and sigma_within is not None and sigma_total is not None:
        infl = (sigma_total.mean() / sigma_within.mean() - 1.0) * 100.0
        print(f"  sigma modes: within {sigma_within.mean():.3f} °C | "
              f"ensemble (within+between) {sigma_total.mean():.3f} °C  "
              f"(+{infl:.1f}% from fold disagreement)")
    print()
    print("Load with:  data = np.load('<output>.npz'); "
          "pred = data['pred_mean']  # shape (days, points)")


# ---------------------------------------------------------------------------
# Sigma-mode comparison plot
# ---------------------------------------------------------------------------
def _plot_sigma_modes(sigma_within, sigma_total, sigma_between, lat, lon, output):
    """Compare the two predictive-sigma modes (per-point, time-averaged).

    Panels: within-model sigma map, ensemble (within+between) sigma map, the
    added between-fold term, and the distribution of both modes. Saved next to
    the predictions npz as ``<stem>_sigma_modes.png``.
    """
    mean_within = sigma_within.mean(axis=0)      # (P,) time-averaged per point
    mean_total = sigma_total.mean(axis=0)
    mean_between = sigma_between.mean(axis=0)

    # Shared colour scale so the two sigma maps are directly comparable.
    vmax = float(np.nanmax(mean_total))
    vmin = float(np.nanmin(mean_within))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    def _map(ax, vals, title, vmin_, vmax_, cmap="viridis"):
        sc = ax.scatter(lon, lat, c=vals, s=4, cmap=cmap, vmin=vmin_, vmax=vmax_)
        ax.set_title(title)
        ax.set_xlabel("lon (°E)")
        ax.set_ylabel("lat (°N)")
        fig.colorbar(sc, ax=ax, shrink=0.85, label="σ (°C)")

    _map(axes[0, 0], mean_within, "Within-model σ  (mean(σ_k²))", vmin, vmax)
    _map(axes[0, 1], mean_total, "Ensemble σ  (within + between)", vmin, vmax)
    _map(axes[1, 0], mean_between, "Added: between-fold σ  √var(μ_k)",
         0.0, float(np.nanmax(mean_between)), cmap="magma")

    # Distribution of both modes over all points/days.
    ax = axes[1, 1]
    w = sigma_within.ravel()
    t = sigma_total.ravel()
    hi = float(np.nanpercentile(t, 99.5))
    bins = np.linspace(0, hi, 60)
    ax.hist(w, bins=bins, alpha=0.55, label=f"within (μ={w.mean():.3f})", color="C0")
    ax.hist(t, bins=bins, alpha=0.55, label=f"ensemble (μ={t.mean():.3f})", color="C3")
    ax.set_title("σ distribution: within vs ensemble")
    ax.set_xlabel("σ (°C)")
    ax.set_ylabel("count")
    ax.legend()

    fig.suptitle("Predictive σ modes — within-model vs ensemble (between-fold added)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_path = Path(output).with_name(Path(output).stem + "_sigma_modes.png")
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    logger.info("sigma-mode comparison plot written: %s", save_path)
    print(f"  sigma plot : {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model-dir", required=True,
                    help="Dir with params.json / manifest.json / model_fold_* checkpoints.")
    ap.add_argument("--date-start", required=True, help="First date to predict (YYYY-MM-DD).")
    ap.add_argument("--date-end",   required=True, help="Last date to predict (YYYY-MM-DD).")
    ap.add_argument("--targets", default=None,
                    help="CSV with columns lat,lon for custom target points. "
                         "Default: MeteoSwiss TmaxD station grid.")
    fld = ap.add_mutually_exclusive_group()
    fld.add_argument("--fold", type=int, default=None,
                     help="Single fold checkpoint to use (default 0).")
    fld.add_argument("--all-folds", action="store_true",
                     help="Ensemble over all available fold checkpoints.")
    ap.add_argument("--ensemble-var", action="store_true",
                    help="Make the saved pred_sigma the ensemble mode (within + "
                         "between-fold var(mu_k)) instead of within-model only. "
                         "Both modes are saved when ensembling regardless; this "
                         "only selects pred_sigma. No effect for a single fold.")
    ap.add_argument("--plot", action="store_true",
                    help="Write a sigma-mode comparison figure (within vs "
                         "ensemble) next to the predictions npz. Needs --all-folds.")
    ap.add_argument("--data-dir", default=None,
                    help="Override ERA5/MeteoSwiss data root (useful when the model "
                         "was trained on another machine with different paths).")
    ap.add_argument("--device", default=None, help="Override device (cuda/cpu).")
    ap.add_argument("--output", default=None,
                    help="Output .npz path (default: <model_dir>/predictions_<start>_<end>.npz).")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    params_mod.configure_renku_cuda()
    device = torch.device(args.device) if args.device else params_mod.select_device()

    model_dir = Path(args.model_dir)
    folds = [args.fold] if args.fold is not None else None

    out = Path(args.output) if args.output else (
        model_dir / f"predictions_{args.date_start}_{args.date_end}.npz"
    )

    run(
        model_dir=model_dir,
        date_start=args.date_start,
        date_end=args.date_end,
        targets_csv=Path(args.targets) if args.targets else None,
        folds=folds,
        all_folds=args.all_folds,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        device=device,
        output=out,
        ensemble_var=args.ensemble_var,
        plot=args.plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
