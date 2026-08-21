#!/usr/bin/env python
"""Settle the day-label conventions between the SMN stations and the gridded truths.

Precip: ``rre150d0`` on day D is the total from 06 UTC of D to 06 UTC of D+1.
RhiresD uses the same 06-06 window, but which calendar day a window is filed under
is a labelling convention on both sides, and the tp-2024 audit showed what a silent
one-day shift does to every downstream number. Temperature: ``tre200dx`` against
TmaxD, same question. So the lag is measured, not assumed: each station is
correlated against its nearest grid cell at lags -1, 0 and +1, per variable, and
the winning lag is recorded for everything downstream to import.

The grids are interpolated FROM (a superset of) these very stations, so at the
correct lag the correlation should be near 1 and the verdict unambiguous — that
expectation is itself part of the check: a mushy winner would mean the station
table or the cell matching is wrong.

Writes cache/alignment.json: {"precip": {...}, "tmax": {...}}.

Usage:  python station_exploration/check_alignment.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
GRIDS = {
    "precip": (HERE.parent / "datasets/MeteoSwiss/RhiresD_v2.0_swiss.lv95",
               ("RhiresD_ch01h", "rhiresd_ch01h"), "RhiresD", "precip_mm",
               "smn_precip_daily.csv"),
    "tmax": (HERE.parent / "datasets/MeteoSwiss/TmaxD_v2.0_swiss.lv95",
             ("TmaxD_ch01r", "tmaxd_ch01r"), "TmaxD", "tmax_C",
             "smn_tmax_daily.csv"),
}


def load_grid(variable: str, years: range) -> xr.DataArray:
    """The gridded daily field for the given years, whichever file naming vintage."""
    grid_dir, stems, var_name, _vcol, _obs = GRIDS[variable]
    paths = []
    for y in years:
        hits: list[str] = []
        for stem in stems:
            hits = (glob.glob(str(grid_dir / f"{stem}.swiss.lv95_{y}0101*.nc"))
                    or glob.glob(str(grid_dir / f"*{stem}.swiss.lv95_{y}0101*.nc")))
            if hits:
                break
        if not hits:
            raise FileNotFoundError(f"no {var_name} file for {y} under {grid_dir}")
        paths.extend(hits)
    ds = xr.open_mfdataset(paths, combine="by_coords")
    var = var_name if var_name in ds else [v for v in ds.data_vars
                                           if getattr(ds[v], "ndim", 0) == 3][0]
    return ds[var]


def nearest_cell(da: xr.DataArray, lat: float, lon: float) -> tuple[int, int]:
    """Index of the closest valid grid cell by great-circle-ish squared degrees."""
    # The 2024+ OGD vintage broadcasts the static lat/lon over time on concat;
    # collapse back before using them as 2-D coordinate fields.
    glat_da, glon_da = da["lat"], da["lon"]
    if "time" in glat_da.dims:
        glat_da, glon_da = glat_da.isel(time=0), glon_da.isel(time=0)
    glat, glon = glat_da.values, glon_da.values
    d2 = (glat - lat) ** 2 + ((glon - lon) * np.cos(np.deg2rad(lat))) ** 2
    return np.unravel_index(np.nanargmin(d2), d2.shape)


def check_variable(variable: str, meta: pd.DataFrame) -> dict:
    _dir, _stems, grid_name, value_col, obs_name = GRIDS[variable]
    obs = pd.read_csv(CACHE / obs_name, parse_dates=["date"])
    grid = load_grid(variable, range(2020, 2025)).load()
    times = pd.DatetimeIndex(grid["time"].values).normalize()
    stns = set(obs["stn"])

    rows = []
    for _, st in meta.iterrows():
        if st["stn"] not in stns:
            continue
        series = obs[obs["stn"] == st["stn"]].set_index("date")[value_col]
        i, j = nearest_cell(grid, st["lat"], st["lon"])
        cell = pd.Series(grid.values[:, i, j], index=times)
        if np.isnan(cell).all():          # station just outside the analysis domain
            rows.append({"stn": st["stn"], "corr_m1": np.nan, "corr_0": np.nan,
                         "corr_p1": np.nan, "best_lag": None, "outside_domain": True})
            continue
        both = pd.concat([cell.rename("grid"), series.rename("station")],
                         axis=1, join="inner").dropna()
        corr = {lag: both["grid"].corr(both["station"].shift(lag))
                for lag in (-1, 0, 1)}
        best = max(corr, key=lambda k: corr[k] if np.isfinite(corr[k]) else -9)
        rows.append({"stn": st["stn"], "corr_m1": corr[-1], "corr_0": corr[0],
                     "corr_p1": corr[1], "best_lag": best, "outside_domain": False,
                     "n_days": len(both)})
    df = pd.DataFrame(rows)
    in_dom = df[~df["outside_domain"]]
    counts = in_dom["best_lag"].value_counts().to_dict()
    med = {k: float(in_dom[f"corr_{s}"].median())
           for k, s in ((-1, "m1"), (0, "0"), (1, "p1"))}
    verdict = max(counts, key=counts.get)
    out = {"verdict_lag": int(verdict),
           "meaning": f"station series must be shifted by this many days to match "
                      f"{grid_name}'s day labels (0 = same-day labels agree)",
           "stations_by_best_lag": {str(k): int(v) for k, v in counts.items()},
           "median_corr_by_lag": {str(k): v for k, v in med.items()},
           "outside_domain": df[df["outside_domain"]]["stn"].tolist(),
           "per_station": df.to_dict(orient="records")}
    print(f"{variable}: verdict lag {verdict} "
          f"(stations by best lag {counts}; median corr by lag {med})")
    print(f"  outside domain: {out['outside_domain']}")
    return out


def main() -> int:
    meta = pd.read_csv(CACHE / "smn_stations.csv")
    out = {variable: check_variable(variable, meta) for variable in GRIDS}
    json.dump(out, open(CACHE / "alignment.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
