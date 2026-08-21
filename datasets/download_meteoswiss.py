#!/usr/bin/env python3
"""
Download MeteoSwiss gridded daily climate reference data (the model targets) by
year, from the MeteoSwiss Open Government Data (OGD) STAC archive on
data.geo.admin.ch.

These are the same "v2.0_swiss.lv95" products already under
``datasets/MeteoSwiss/`` (TmaxD/TminD/TabsD/RhiresD), redistributed as OGD. The
full-year historical files live in the ``archive-ch`` item of the collection
``ch.meteoschweiz.ogd-surface-derived-grid`` (tmax/tmin from 1971, tabs/precip
from 1961, through the last completed year).

The downloaded ``.nc`` files drop straight into the existing per-variable dirs;
the loader globs ``*.nc`` and combines by coordinates, so no rename is needed.

NOTE: evaluating the models on a new year also needs the matching ERA5 *inputs*
(``datasets/download_era5.py --years <year>``); MeteoSwiss only supplies the
reference truth.

Examples
--------
    # tmax for 2024 (default variable)
    python download_meteoswiss.py --years 2024

    # all four variables, a range of years
    python download_meteoswiss.py --years 2020-2024 --variables all

    # see what would be fetched, download nothing
    python download_meteoswiss.py --years 2022,2024 --variables tmax,precip --dry-run
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# --- OGD STAC source -------------------------------------------------------
STAC_ITEM_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/"
    "ch.meteoschweiz.ogd-surface-derived-grid/items/archive-ch"
)

# variable -> (product token in the asset name, destination subdir under MeteoSwiss/)
VARIABLES = {
    "tmax":   ("tmaxd_",   "TmaxD_v2.0_swiss.lv95"),
    "tmin":   ("tmind_",   "TminD_v2.0_swiss.lv95"),
    "tabs":   ("tabsd_",   "TabsD_v2.0_swiss.lv95"),
    "precip": ("rhiresd_", "RhiresD_v2.0_swiss.lv95"),
}


# --- CLI parsing helpers (same syntax as download_era5.py) -----------------
def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y) for y in spec.split(",")]


def parse_variables(spec: str) -> list[str]:
    if spec == "all":
        return list(VARIABLES)
    names = [v.strip() for v in spec.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIABLES]
    if unknown:
        raise SystemExit(
            f"Unknown variable(s): {unknown}. "
            f"Choose from {list(VARIABLES)} or 'all'."
        )
    return names


# --- STAC discovery --------------------------------------------------------
def load_archive_assets() -> dict:
    """Fetch the archive STAC item and return its {asset_key: href} map."""
    with urllib.request.urlopen(STAC_ITEM_URL, timeout=60) as resp:
        item = json.load(resp)
    return {k: a["href"] for k, a in item.get("assets", {}).items()}


def find_asset(assets: dict, product_token: str, year: int) -> tuple[str, str] | None:
    """Return (asset_key, href) for one product/year, or None if not present.

    Matches the product token (e.g. ``tmaxd_``) and the file's start date
    ``_{year}0101`` so we pick exactly that calendar year's annual file.
    """
    needle_start = f"_{year}0101"
    for key, href in assets.items():
        if product_token in key and needle_start in key:
            return key, href
    return None


def available_years(assets: dict, product_token: str) -> list[int]:
    """Sorted list of years available for a product (for error messages)."""
    years = []
    for key in assets:
        if product_token not in key:
            continue
        # asset names contain ..._{YYYY}0101000000_{YYYY}1231000000.nc
        for part in key.split("_"):
            if len(part) >= 8 and part[:8].isdigit() and part[4:8] == "0101":
                years.append(int(part[:4]))
                break
    return sorted(set(years))


# --- download --------------------------------------------------------------
def download(href: str, dest: Path) -> None:
    """Stream ``href`` to ``dest`` atomically (via a .part temp file)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(href, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if total:
                    pct = 100 * read / total
                    print(f"\r    {read/1048576:7.1f} / {total/1048576:.1f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
        print()
    os.replace(tmp, dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--years", default="2024",
                        help="Year(s): '2024', '2020-2024', or '2022,2024'.")
    parser.add_argument("--variables", default="tmax",
                        help=f"Subset of {list(VARIABLES)} (comma-separated) or "
                             "'all'. Default: tmax.")
    parser.add_argument("--dest", default=None,
                        help="MeteoSwiss base dir (default: datasets/MeteoSwiss "
                             "next to this script).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download even if the target file exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve and print URLs/paths; download nothing.")
    args = parser.parse_args(argv)

    years = parse_years(args.years)
    variables = parse_variables(args.variables)
    base = Path(args.dest) if args.dest else (Path(__file__).resolve().parent / "MeteoSwiss")

    print(f"Resolving MeteoSwiss archive assets ({STAC_ITEM_URL.split('/')[-1]}) ...")
    assets = load_archive_assets()

    n_ok = n_skip = n_miss = 0
    for var in variables:
        product_token, subdir = VARIABLES[var]
        out_dir = base / subdir
        for year in years:
            match = find_asset(assets, product_token, year)
            if match is None:
                avail = available_years(assets, product_token)
                rng = f"{avail[0]}-{avail[-1]}" if avail else "none"
                print(f"[{var} {year}] NOT in archive (available: {rng})")
                n_miss += 1
                continue
            asset_key, href = match
            dest = out_dir / asset_key
            if dest.exists() and not args.overwrite:
                print(f"[{var} {year}] exists, skipping: {dest}")
                n_skip += 1
                continue
            if args.dry_run:
                print(f"[{var} {year}] would download:\n    {href}\n    -> {dest}")
                n_ok += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{var} {year}] downloading -> {dest}")
            download(href, dest)
            n_ok += 1

    print(f"\nDone: {n_ok} {'resolved' if args.dry_run else 'downloaded'}, "
          f"{n_skip} skipped, {n_miss} unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
