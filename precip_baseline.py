#!/usr/bin/env python3
"""Bilinear-ERA5 precipitation baseline for the precip evaluation.

A trivial reference predictor: take the coarse ERA5 daily total-precipitation
field and bilinearly interpolate it onto the MeteoSwiss target points. This is
the precipitation analogue of the bilinear-ERA5 temperature baseline used for
the tmax skill score (the report's methodology chapter) and mirrors the
baseline comparison in Vaughan et al. (2022). The downscaling model should beat
it; ``skill = 1 - MAE_model / MAE_baseline`` quantifies by how much.

The default field is ``datasets/ERA5_Land/precipitation/tp-<year>.nc`` (daily
``tp`` in metres on the ~0.1 deg ERA5-Land grid). If no field is found the
baseline is reported as unavailable (the evaluation still runs without it).
"""

from __future__ import annotations

import glob as globmod
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import datasets as ds_mod  # _era5_mf_preprocess: per-file coord snap (phantom-grid guard)

logger = logging.getLogger("precip_baseline")

DEFAULT_GLOB = "datasets/ERA5_Land/precipitation/tp-{year}.nc"
M_TO_MM = 1000.0


# One definition, shared with the training/inference precip channel loader in
# datasets.py, so the baseline and the model input can never date-align differently.
_as_day_index = ds_mod._as_day_index


def bilinear_era5_precip(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    time_dates,
    year: int,
    glob: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """Bilinearly interpolate coarse ERA5 daily precip onto the target points.

    Args:
        target_lat, target_lon: (n_points,) WGS84 degrees of each target point.
        time_dates: the model's time axis (length n_times); used to date-align.
        year: data year (selects the default ERA5 file).
        glob: optional override glob for the ERA5 ``tp`` field.

    Returns:
        (array, source) where array is (n_times, n_points) precip in mm with NaN
        for any (day, point) the baseline cannot cover, or (None, reason) if no
        field is available.
    """
    pattern = (glob or DEFAULT_GLOB).format(year=year)
    files = sorted(globmod.glob(pattern))
    if not files:
        return None, f"no ERA5 precip field matched {pattern!r}"

    try:
        # Snap each file's lat/lon before the merge: per-year ERA5 files carry
        # ~1e-13 coordinate drift (tp-2024.nc does), and combine='by_coords'
        # would OUTER-JOIN drifted grid lines into a phantom NaN border that
        # bleeds NaN through the bilinear interpolation (see datasets.py).
        dsx = (xr.open_mfdataset(files, combine="by_coords",
                                 preprocess=ds_mod._era5_mf_preprocess)
               if len(files) > 1
               else ds_mod._era5_mf_preprocess(xr.open_dataset(files[0])))
    except Exception as exc:  # pragma: no cover - data-dependent
        logger.warning("could not open baseline field %r: %r", pattern, exc)
        return None, f"failed to open {pattern!r}: {exc!r}"

    var = "tp" if "tp" in dsx.data_vars else list(dsx.data_vars)[0]
    da = dsx[var]
    # Normalise dim/coord names to latitude/longitude/time.
    rename = {}
    for cand in ("lat",):
        if cand in da.dims or cand in da.coords:
            rename[cand] = "latitude"
    for cand in ("lon",):
        if cand in da.dims or cand in da.coords:
            rename[cand] = "longitude"
    for cand in ("valid_time",):
        if cand in da.dims or cand in da.coords:
            rename[cand] = "time"
    if rename:
        da = da.rename(rename)
    # Drop stray scalar coords (e.g. ERA5 'number'/'expver') that block interp.
    for stray in ("number", "expver", "step", "surface", "heightAboveGround"):
        if stray in da.coords:
            da = da.reset_coords(stray, drop=True)

    lat = xr.DataArray(np.asarray(target_lat, dtype=float), dims="point")
    lon = xr.DataArray(np.asarray(target_lon, dtype=float), dims="point")
    interp = da.interp(latitude=lat, longitude=lon, method="linear")  # (time, point)
    interp = (interp.astype("float32") * M_TO_MM).compute()

    # Date-align the ERA5 series to the model's day axis.
    src_days = _as_day_index(interp["time"].values)
    interp = interp.assign_coords(time=src_days)
    # Collapse any duplicate days (accumulation conventions can repeat a date).
    interp = interp.groupby("time").mean()
    tgt_days = _as_day_index(time_dates)
    interp = interp.reindex(time=tgt_days)

    arr = np.asarray(interp.transpose("time", "point").values, dtype=np.float32)
    n_cover = int(np.isfinite(arr).any(axis=1).sum())
    logger.info("baseline %s -> %d/%d model days covered", pattern, n_cover, arr.shape[0])
    source = (f"{files[0]} (bilinear, mm/day)" if len(files) == 1 else
              f"{files[0]} .. {files[-1]} ({len(files)} files, bilinear, mm/day)")
    return arr, source
