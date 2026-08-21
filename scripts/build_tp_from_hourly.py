#!/usr/bin/env python3
"""Build daily ERA5-Land total precipitation from the HOURLY dataset.

Why not the derived daily set: ``derived-era5-land-daily-statistics`` rejects
``total_precipitation`` outright -- "Daily statistics of accumulated variables
are not supported for this dataset" (job 00b3d563, 2026-07-31; same failure on
2026-07-23). No time_zone setting changes that. So the daily field has to be
built from ``reanalysis-era5-land`` hourly data, which does work on this
account, and aggregated here.

ERA5-Land ``tp`` accumulates from 00 UTC and resets each day, so the total for
day D is the accumulation at 00:00 of day D+1 -- NOT the max over D's own
00:00..23:00, which returns max(total(D-1), partial(D)): both inflated and
biased to the previous day. That is the defect in the current tp-2024.nc
(6.889 mm/day against an expected 3.3-4.5).

The aggregation rule is not assumed -- ``--validate`` scores several candidate
rules against the known-good legacy files and reports which one reproduces
them.

    # download (resume-safe; one request per year-month)
    python scripts/build_tp_from_hourly.py --years 2020-2024 --download

    # decide the rule empirically against legacy tp-2020..2023
    python scripts/build_tp_from_hourly.py --years 2020 --validate

    # write tp-<year>.nc
    python scripts/build_tp_from_hourly.py --years 2020-2024 --aggregate
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import xarray as xr

AREA = [48.2, 5.0, 45.4, 11.0]        # N, W, S, E -> 29 x 61 on the 0.1 deg grid
DATASET = "reanalysis-era5-land"
RAW = pathlib.Path("tp_hourly/raw")
OUT = pathlib.Path("tp_hourly")
LEGACY = pathlib.Path("datasets/ERA5_Land/precipitation")
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def month_file(year: int, month: int) -> pathlib.Path:
    return RAW / f"tp_hourly-{year}-{month:02d}.nc"


def download(years: list[int]) -> None:
    """One request per year-month, all days, all 24 hours.

    Closing day 31 Dec of the last year needs 00:00 of 1 Jan of the year after,
    so a single extra day is fetched for that boundary.
    """
    import cdsapi

    RAW.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    jobs = [(y, m, ALL_DAYS) for y in years for m in range(1, 13)]
    jobs.append((max(years) + 1, 1, ["01"]))          # boundary day only

    for year, month, days in jobs:
        out = month_file(year, month)
        if out.exists():
            print(f"[skip] {out}", flush=True)
            continue
        print(f"[download] tp hourly {year}-{month:02d} ({len(days)} days x 24 h)", flush=True)
        c.retrieve(DATASET, {
            "variable": ["total_precipitation"],
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": ALL_HOURS,
            "area": AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }, str(out))
        print(f"[done] {out}", flush=True)


def load_hourly(years: list[int]) -> xr.DataArray:
    """Open every monthly file covering `years` plus the boundary day."""
    paths = [month_file(y, m) for y in years for m in range(1, 13)]
    paths.append(month_file(max(years) + 1, 1))
    have = [p for p in paths if p.exists()]
    if not have:
        sys.exit(f"no hourly files under {RAW}/ -- run --download first")
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"[warn] {len(missing)} month(s) missing: {', '.join(missing[:6])}"
              f"{' ...' if len(missing) > 6 else ''}")
    ds = xr.open_mfdataset([str(p) for p in have], combine="by_coords")
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    for stray in ("number", "expver", "step", "surface"):
        if stray in ds.coords:
            ds = ds.reset_coords(stray, drop=True)
    return ds["tp"].sortby("time")


def _at(tp: xr.DataArray, stamps: pd.DatetimeIndex, days: pd.DatetimeIndex) -> xr.DataArray:
    """Accumulation sampled at `stamps`, re-labelled onto `days`."""
    return tp.reindex(time=stamps).assign_coords(time=days)


def candidates(tp: xr.DataArray) -> dict[str, xr.DataArray]:
    """Candidate daily-total rules, all returning (time=day, lat, lon) in metres.

    next00   accumulation at 00:00 of D+1        00-00 UTC; matches the legacy files
    w0606    06 UTC D -> 06 UTC D+1              MeteoSwiss climatological day (RhiresD)
    max24    max over D 00:00..23:00             the defect in tp-2024
    at23     accumulation at D 23:00             drops the final hour

    ``tp`` accumulates from 00 UTC and resets, so acc(00:00 of D+1) is day D's
    total and a 06-06 window has to be assembled from two partial cycles:
    (acc(00:00 D+1) - acc(06:00 D)) + acc(06:00 D+1).

    Empirically (2020, 120 days, domain-mean daily vs RhiresD):
    next00 corr 0.910, w0606 corr 0.958 -- RhiresD's day is the 06-06 window
    labelled by its START date.
    """
    out = {}
    t = pd.to_datetime(tp.time.values)
    days = pd.date_range(t.min().normalize(), t.max().normalize(), freq="D")

    out["next00"] = _at(tp, days + pd.Timedelta(days=1), days)

    out["w0606"] = ((_at(tp, days + pd.Timedelta(days=1), days)
                     - _at(tp, days + pd.Timedelta(hours=6), days))
                    + _at(tp, days + pd.Timedelta(days=1, hours=6), days))

    day_of = pd.to_datetime(tp.time.values).normalize()
    out["max24"] = tp.assign_coords(day=("time", day_of)).groupby("day").max().rename(day="time")

    a23 = tp.isel(time=(tp.time.dt.hour == 23))
    out["at23"] = a23.assign_coords(time=pd.to_datetime(a23.time.values).normalize())

    return {k: v.sortby("time") for k, v in out.items()}


def legacy_for(year: int) -> xr.DataArray | None:
    p = LEGACY / f"tp-{year}.nc"
    if not p.exists():
        return None
    d = xr.open_dataset(p, engine="h5netcdf")
    if "valid_time" in d.dims and "time" not in d.dims:
        d = d.rename({"valid_time": "time"})
    return d["tp"].assign_coords(time=pd.to_datetime(d.time.values).normalize())


def validate(years: list[int]) -> None:
    """Score each candidate rule against the known-good legacy files."""
    tp = load_hourly(years)
    cand = candidates(tp)
    print(f"\nhourly loaded: {tp.sizes['time']} steps  "
          f"{str(tp.time.values[0])[:13]} .. {str(tp.time.values[-1])[:13]}\n")
    print(f"{'rule':<9} {'year':<6} {'n_cmp':>6} {'mean mm/d':>10} {'legacy':>8} "
          f"{'ratio':>7} {'maxdiff':>9} {'corr':>8}")
    for year in years:
        leg = legacy_for(year)
        if leg is None:
            print(f"   (no legacy tp-{year}.nc to compare)")
            continue
        for name, da in cand.items():
            common = np.intersect1d(da.time.values, leg.time.values)
            if len(common) == 0:
                continue
            a = np.asarray(da.sel(time=common).values, float) * 1000.0
            b = np.asarray(leg.sel(time=common).values, float) * 1000.0
            fin = np.isfinite(a) & np.isfinite(b)
            corr = np.corrcoef(a[fin], b[fin])[0, 1]
            print(f"{name:<9} {year:<6} {len(common):>6} {a[fin].mean():>10.4f} "
                  f"{b[fin].mean():>8.4f} {a[fin].mean()/b[fin].mean():>7.3f} "
                  f"{np.abs(a[fin]-b[fin]).max():>9.5f} {corr:>8.5f}")
    print("\nThe correct rule reproduces legacy: ratio ~1.000, maxdiff ~0, corr ~1.")


def aggregate(years: list[int], rule: str) -> None:
    """Write one calendar-year file per year under the chosen window rule.

    Rules other than the default land in a `<rule>/` subdir, so a 00-00 set and
    a 06-06 set can coexist: train on one, score the baseline against the other
    via eval_precip's --baseline-glob.
    """
    tp = load_hourly(years)
    daily = candidates(tp)[rule]
    out_dir = OUT if rule == "next00" else OUT / rule
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        sel = daily.sel(time=str(year))
        if sel.sizes["time"] == 0:
            print(f"[skip] {year}: no days produced")
            continue
        ds = sel.to_dataset(name="tp")
        out = out_dir / f"tp-{year}.nc"
        ds.to_netcdf(out, engine="h5netcdf")
        mm = float(sel.mean()) * 1000
        exp = 366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365
        flag = "OK" if 3.3 <= mm <= 4.6 else "*** SUSPECT ***"
        print(f"[done] {out}  n={sel.sizes['time']}/{exp}  "
              f"grid={sel.sizes['latitude']}x{sel.sizes['longitude']}  "
              f"{str(sel.time.values[0])[:10]}..{str(sel.time.values[-1])[:10]}  "
              f"mean={mm:.3f} mm/day  {flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2020-2024", help="e.g. 2020-2024 or 2020,2024")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--rule", default="next00", choices=["next00", "w0606", "max24", "at23"])
    a = ap.parse_args()
    years = parse_years(a.years)
    if a.download:
        download(years)
    if a.validate:
        validate(years)
    if a.aggregate:
        aggregate(years, a.rule)
    if not (a.download or a.validate or a.aggregate):
        ap.error("pick at least one of --download / --validate / --aggregate")


if __name__ == "__main__":
    main()
