#!/usr/bin/env python
"""Download daily precipitation observations for the SwissMetNet (SMN) gauges.

The NBCN homogeneous daily set the tmax station analysis is built on carries no
precipitation at daily granularity, so precip verification needs the SMN network
(``ch.meteoschweiz.ogd-smn``): ~160 automatic stations, per-station daily CSVs.

The parameter is ``rre150d0`` — "Precipitation; daily total 6 UTC - 6 UTC following
day" — which is the accumulation window RhiresD itself uses. Its sibling ``rka150d0``
(0 UTC - 0 UTC) is deliberately NOT read: a 0-0 total against a 06-06 analysis would
re-create exactly the day-straddling defect the tp-2024 audit documented. Which
calendar day a 06-06 total belongs to is a labelling convention, not a fact; it is
settled empirically by check_alignment.py before anything downstream reads this cache.

The same per-station files carry ``tre200dx`` — the OPERATIONAL daily maximum 2 m
air temperature (the homogenised counterpart, NBCN's ``ths200dx``, exists for the
28 NBCN sites only) — so one pass also builds the temperature-verification cohort.

Output (in station_exploration/cache/, gitignored):
  smn_stations.csv    station meta: abbr, name, lat, lon, elev_m, is_nbcn
                      (every station with EITHER variable; per-variable membership
                      is whichever daily table the station appears in)
  smn_precip_daily.csv  long table: date, stn, precip_mm  (2020-2024 only)
  smn_tmax_daily.csv    long table: date, stn, tmax_C     (2020-2024 only)
  smn_coverage.json   per-station day counts and the applied coverage cut, per variable

Usage:  python station_exploration/download_smn_precip.py [--min-coverage 0.95]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
CACHE = Path(__file__).resolve().parent / "cache"
YEARS = (2020, 2024)          # inclusive span the study needs
PRECIP_COL = "rre150d0"       # 06-06 UTC daily total; NEVER rka150d0 (0-0 UTC)
TMAX_COL = "tre200dx"         # operational daily max 2 m T (NOT homogenised)
# The NBCN cohort of the tmax station chapter, for the is_nbcn flag.
NBCN_ABBRS = {
    "ALT", "ANT", "BAS", "BER", "CDF", "CHD", "CHM", "DAV", "ELM", "ENG",
    "GRC", "GRH", "GSB", "GVE", "JUN", "LUG", "LUZ", "MER", "NEU", "OTL",
    "PAY", "RAG", "SAE", "SAM", "SBE", "SIA", "SIO", "SMA", "STG",
}


def _get(url: str, retries: int = 3) -> bytes | None:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - a 404 just means no such file
            if "404" in str(e):
                return None
            if attempt == retries - 1:
                print(f"  giving up on {url}: {e}")
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def load_station_meta() -> pd.DataFrame:
    raw = _get(f"{BASE}/ogd-smn_meta_stations.csv")
    if raw is None:
        sys.exit("station metadata download failed")
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="cp1252")
    out = pd.DataFrame({
        "stn": df["station_abbr"].str.upper(),
        "name": df["station_name"],
        "lat": df["station_coordinates_wgs84_lat"],
        "lon": df["station_coordinates_wgs84_lon"],
        "elev_m": df["station_height_masl"],
    })
    out["is_nbcn"] = out["stn"].isin(NBCN_ABBRS)
    return out


def load_station_daily(stn: str) -> dict[str, pd.DataFrame]:
    """2020-2024 daily rre150d0 / tre200dx for one station; only variables present."""
    raw = _get(f"{BASE}/{stn.lower()}/ogd-smn_{stn.lower()}_d_historical.csv")
    if raw is None:
        return {}
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="cp1252",
                     usecols=lambda c: c in ("station_abbr", "reference_timestamp",
                                             PRECIP_COL, TMAX_COL))
    df["date"] = pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M")
    df = df[(df["date"].dt.year >= YEARS[0]) & (df["date"].dt.year <= YEARS[1])]
    out = {}
    for col, value_name in ((PRECIP_COL, "precip_mm"), (TMAX_COL, "tmax_C")):
        if col not in df.columns:
            continue
        d = df.dropna(subset=[col])
        if d.empty:
            continue
        out[value_name] = pd.DataFrame({"date": d["date"].dt.normalize(), "stn": stn,
                                        value_name: d[col].astype(float)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="minimum fraction of 2020-2024 days a gauge must report")
    ap.add_argument("--force", action="store_true", help="re-download everything")
    args = ap.parse_args(argv)

    CACHE.mkdir(parents=True, exist_ok=True)
    obs_path = CACHE / "smn_precip_daily.csv"
    meta_path = CACHE / "smn_stations.csv"
    if obs_path.exists() and meta_path.exists() and not args.force:
        print(f"cache present ({obs_path}); use --force to re-download")
        return 0

    meta = load_station_meta()
    print(f"{len(meta)} SMN stations in metadata ({meta['is_nbcn'].sum()} NBCN)")

    frames: dict[str, list] = {"precip_mm": [], "tmax_C": []}
    missing = []
    for i, stn in enumerate(meta["stn"], 1):
        got = load_station_daily(stn)
        if not got:
            missing.append(stn)
        for value_name, frame in got.items():
            frames[value_name].append(frame)
        if i % 25 == 0:
            print(f"  {i}/{len(meta)} stations fetched")

    n_days_total = (pd.Timestamp(f"{YEARS[1]}-12-31")
                    - pd.Timestamp(f"{YEARS[0]}-01-01")).days + 1
    coverage: dict = {"n_days_span": n_days_total, "min_coverage": args.min_coverage,
                      "no_data_at_all": sorted(missing)}
    kept_any: set[str] = set()
    for value_name, out_name in (("precip_mm", "smn_precip_daily.csv"),
                                 ("tmax_C", "smn_tmax_daily.csv")):
        obs = pd.concat(frames[value_name], ignore_index=True)
        counts = obs.groupby("stn").size()
        keep = counts[counts >= args.min_coverage * n_days_total].index
        dropped = sorted(set(counts.index) - set(keep))
        obs = obs[obs["stn"].isin(keep)]
        obs.to_csv(CACHE / out_name, index=False)
        kept_any |= set(keep)
        coverage[value_name] = {"kept": sorted(keep),
                                "dropped_low_coverage": dropped,
                                "per_station_days": counts.to_dict()}
        print(f"{value_name}: kept {len(keep)} stations "
              f"(>= {args.min_coverage:.0%} of {n_days_total} days); "
              f"dropped {len(dropped)} low-coverage -> {out_name} ({len(obs)} rows)")

    # Meta keeps every station that survives in EITHER variable; per-variable
    # membership is the corresponding daily table, not this file.
    meta = meta[meta["stn"].isin(kept_any)]
    meta.to_csv(meta_path, index=False)
    json.dump(coverage, open(CACHE / "smn_coverage.json", "w"), indent=2)
    print(f"wrote {meta_path} ({len(meta)} stations in >=1 cohort)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
