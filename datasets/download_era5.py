"""Download ERA5-Land daily statistics (temperature mean/min/max, precipitation)
and the static geopotential field from the Copernicus CDS into the datasets/
layout expected by datasets.py.

It also downloads ERA5 pressure-level fields (geopotential, temperature, and
specific or relative humidity) that capture the vertical structure of the
atmosphere. ERA5-Land has no pressure levels, so these come from the hourly
`reanalysis-era5-pressure-levels` set, sampled at the synoptic hours plus the
summer convective peak (00/06/12/15/18 UTC) so the diurnal cycle is retained.
One request/file per hour (e.g. z_pl-2023-12.nc) to stay under the CDS cost
limit, each with dims (time, pressure_level, lat, lon).

Only ERA5-Land is available on the CDS. MeteoSwiss grid products (TabsD/TminD/
TmaxD/RhiresD) and the swisstopo topo_subset.zarr must be obtained separately.

Usage:
    python download_era5.py --years 2023
    python download_era5.py --years 2014-2023
    python download_era5.py --years 2023 --fields max_temperature,precipitation

    # Pressure-level fields (vertical structure, 00/06/12/15/18 UTC by default):
    python download_era5.py --years 2023 --pl-fields all
    python download_era5.py --years 2023 --pl-fields z,t,q --levels 850,700,500
    python download_era5.py --years 2023 --fields none --pl-fields all  # PL only
    python download_era5.py --years 2023 --pl-fields all --pl-hours 06,12

    # Pressure-level wind (u/v components) to complement the z/t/q atmosphere:
    python download_era5.py --years 2020-2024 --fields none --pl-fields u,v

    python datasets/download_era5.py --fields none --pl-fields all --years 2023

    # ERA5 single-level surface anchor fields at the same 0.25° grid/hours as the
    # pressure levels (surface anchor for atmospheric mode; 'all' = t2m,tp):
    python download_era5.py --years 2023 --fields none --pl-fields all --sfc-fields all
    python download_era5.py --years 2023 --sfc-fields t2m,tp --pl-hours 06,12
"""

import argparse
from pathlib import Path

import cdsapi
import xarray as xr

# Switzerland bounding box: [North, West, South, East] in degrees.
# Used for the 0.25 deg pressure-level (and surface-anchor) requests; matches the
# existing PressureLevels grid (lat 45.5-48.0, lon 5.5-11.0 -> 11 x 23).
SWISS_AREA = [48.0, 5.5, 45.5, 11.0]

# The daily ERA5-Land (0.1 deg) surface fields were originally fetched on a
# slightly larger box than SWISS_AREA; the trained models' grid is lat 45.4-48.2,
# lon 5.0-11.0 -> 29 x 61. The daily download MUST use this so new years combine
# (open_mfdataset by_coords) with the existing files and match the model grid.
ERA5_LAND_AREA = [48.2, 5.0, 45.4, 11.0]

ERA5_LAND_DIR = Path("datasets/ERA5_Land")
GEOPOTENTIAL_DIR = ERA5_LAND_DIR / "geopotential"

# Pressure-level fields live alongside ERA5-Land but in their own tree because
# they carry an extra `pressure_level` dimension.
ERA5_PL_DIR = Path("datasets/ERA5_PressureLevels")

# Vertical resolution: ERA5 offers 37 pressure levels, but for a near-surface
# downscaling task over the Alps the lower/mid troposphere carries the relevant
# signal. The default set spans the planetary boundary layer up to the upper
# troposphere while keeping file sizes modest:
#   1000 hPa ~   100 m   925 hPa ~   800 m   850 hPa ~  1500 m
#    700 hPa ~  3000 m   500 hPa ~  5500 m   300 hPa ~  9000 m
# (Over high Alpine terrain the 1000/925 hPa surfaces are partly "underground"
# and ERA5 extrapolates them; keep or drop them via --levels as needed.)
DEFAULT_PRESSURE_LEVELS = ["1000", "925", "850", "700", "500", "300"]

# All 37 ERA5 pressure levels, smallest-to-largest height, for --levels all.
ALL_PRESSURE_LEVELS = [
    "1000", "975", "950", "925", "900", "875", "850", "825", "800", "775",
    "750", "700", "650", "600", "550", "500", "450", "400", "350", "300",
    "250", "225", "200", "175", "150", "125", "100", "70", "50", "30",
    "20", "10", "7", "5", "3", "2", "1",
]

# Sub-daily sampling: unlike the surface fields (daily statistics), the
# pressure-level fields are pulled from the *hourly* dataset and sampled at the
# standard synoptic hours plus 15 UTC so the diurnal cycle is retained. In Swiss
# local time (UTC+1/+2) these bracket the daily extremes and boundary-layer
# states:
#   00 UTC ~ night/inversion    06 UTC ~ near daily min temp
#   12 UTC ~ peak heating/max    15 UTC ~ afternoon convection peak (summer)
#   18 UTC ~ evening
DEFAULT_PL_HOURS = ["00", "06", "12", "15", "18"]

ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

# Daily-field specs keyed by the folder name in datasets/ERA5_Land/.
# - cds_variable:   variable name in the derived-era5-land-daily-statistics set
# - daily_statistic: aggregation applied over each day
# - out_var:        variable name datasets.load_era5_data expects (file content)
# - prefix:         output filename prefix (e.g. t2m_max-2023.nc)
# For accumulated precipitation, ERA5-Land resets the accumulation at 00 UTC, so
# the daily maximum equals the daily total.
DAILY_FIELDS = {
    "max_temperature": {
        "cds_variable": "2m_temperature",
        "daily_statistic": "daily_maximum",
        "out_var": "t2m_max",
        "prefix": "t2m_max",
    },
    "min_temperature": {
        "cds_variable": "2m_temperature",
        "daily_statistic": "daily_minimum",
        "out_var": "t2m_min",
        "prefix": "t2m_min",
    },
    "temperature": {
        "cds_variable": "2m_temperature",
        "daily_statistic": "daily_mean",
        "out_var": "t2m",
        "prefix": "t2m",
    },
    "precipitation": {
        "cds_variable": "total_precipitation",
        "daily_statistic": "daily_maximum",
        "out_var": "tp",
        "prefix": "tp",
    },
}


# Pressure-level field specs keyed by the folder/short name used on disk.
# - cds_variable:    variable name in derived-era5-pressure-levels-daily-statistics
# - out_var:         variable name in the resulting NetCDF (ERA5 short name)
# - prefix:          output filename prefix (e.g. t_pl-2023.nc)
# Geopotential `z` here is the geopotential of each pressure surface; divide by g
# (as datasets.load_era5_data already does) to get geopotential height in metres.
# Specific humidity `q` is the default moisture variable; relative humidity `r`
# is offered as an alternative ("Specific Humidity or Relative Humidity").
PRESSURE_LEVEL_FIELDS = {
    "z": {
        "cds_variable": "geopotential",
        "out_var": "z",
        "prefix": "z_pl",
    },
    "t": {
        "cds_variable": "temperature",
        "out_var": "t",
        "prefix": "t_pl",
    },
    "q": {
        "cds_variable": "specific_humidity",
        "out_var": "q",
        "prefix": "q_pl",
    },
    "r": {
        "cds_variable": "relative_humidity",
        "out_var": "r",
        "prefix": "r_pl",
    },
    # Horizontal wind components on each pressure surface (m/s). u is the
    # west->east (zonal) component, v the south->north (meridional) component.
    # Together they capture advection and the vertical wind profile that shapes
    # Alpine downscaling (foehn, channelling), complementing the z/t/q fields.
    "u": {
        "cds_variable": "u_component_of_wind",
        "out_var": "u",
        "prefix": "u_pl",
    },
    "v": {
        "cds_variable": "v_component_of_wind",
        "out_var": "v",
        "prefix": "v_pl",
    },
}

# Default pressure-level fields: geopotential, temperature, specific humidity.
# Relative humidity (`r`) and wind (`u`/`v`) are excluded from "all" so they are
# not fetched unintentionally (r would duplicate the q moisture channel, and wind
# doubles the request cost); request them explicitly, e.g. --pl-fields u,v.
DEFAULT_PL_FIELDS = ["z", "t", "q"]

# ERA5 single-level surface fields from reanalysis-era5-single-levels (0.25° grid,
# same resolution as the pressure-level dataset). Used as a surface anchor when
# running the model in atmospheric-only mode (USE_SFC_ATMOS=True).
ERA5_SFC_DIR = Path("datasets/ERA5_Surface")

ERA5_SURFACE_FIELDS = {
    "t2m": {
        "cds_variable": "2m_temperature",
        "out_var": "t2m",
        "prefix": "t2m_sfc",
    },
    "d2m": {
        "cds_variable": "2m_dewpoint_temperature",
        "out_var": "d2m",
        "prefix": "d2m_sfc",
    },
    "sp": {
        "cds_variable": "surface_pressure",
        "out_var": "sp",
        "prefix": "sp_sfc",
    },
    "u10": {
        "cds_variable": "10m_u_component_of_wind",
        "out_var": "u10",
        "prefix": "u10_sfc",
    },
    "v10": {
        "cds_variable": "10m_v_component_of_wind",
        "out_var": "v10",
        "prefix": "v10_sfc",
    },
    # Total precipitation (accumulated; the hourly value ending at the sampled
    # synoptic hour). A surface-anchor moisture/precip signal for atmospheric mode.
    "tp": {
        "cds_variable": "total_precipitation",
        "out_var": "tp",
        "prefix": "tp_sfc",
    },
}

DEFAULT_SFC_FIELDS = ["t2m", "tp"]


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y) for y in spec.split(",")]


def parse_fields(spec: str) -> list[str]:
    if spec in ("none", ""):
        return []
    if spec == "all":
        return list(DAILY_FIELDS)
    fields = [f.strip() for f in spec.split(",") if f.strip()]
    unknown = [f for f in fields if f not in DAILY_FIELDS]
    if unknown:
        raise SystemExit(
            f"Unknown field(s): {unknown}. "
            f"Choose from {list(DAILY_FIELDS)}, 'all', or 'none'."
        )
    return fields


def parse_pl_fields(spec: str) -> list[str]:
    if spec in ("none", ""):
        return []
    if spec == "all":
        return list(DEFAULT_PL_FIELDS)
    fields = [f.strip() for f in spec.split(",") if f.strip()]
    unknown = [f for f in fields if f not in PRESSURE_LEVEL_FIELDS]
    if unknown:
        raise SystemExit(
            f"Unknown pressure-level field(s): {unknown}. "
            f"Choose from {list(PRESSURE_LEVEL_FIELDS)}, 'all', or 'none'."
        )
    return fields


def parse_sfc_fields(spec: str) -> list[str]:
    if spec in ("none", ""):
        return []
    if spec == "all":
        return list(DEFAULT_SFC_FIELDS)
    fields = [f.strip() for f in spec.split(",") if f.strip()]
    unknown = [f for f in fields if f not in ERA5_SURFACE_FIELDS]
    if unknown:
        raise SystemExit(
            f"Unknown surface field(s): {unknown}. "
            f"Choose from {list(ERA5_SURFACE_FIELDS)}, 'all', or 'none'."
        )
    return fields


def parse_levels(spec: str) -> list[str]:
    if spec == "all":
        return list(ALL_PRESSURE_LEVELS)
    levels = [lvl.strip() for lvl in spec.split(",") if lvl.strip()]
    unknown = [lvl for lvl in levels if lvl not in ALL_PRESSURE_LEVELS]
    if unknown:
        raise SystemExit(
            f"Unknown pressure level(s): {unknown}. "
            f"Valid ERA5 levels (hPa): {ALL_PRESSURE_LEVELS}"
        )
    return levels


def parse_hours(spec: str) -> list[str]:
    if spec == "all":
        return [f"{h:02d}" for h in range(24)]
    hours = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        # Accept '6', '06', or '06:00'.
        h = int(tok.split(":")[0])
        if not 0 <= h <= 23:
            raise SystemExit(f"Hour out of range: {tok}. Use 0-23.")
        hours.append(f"{h:02d}")
    return hours


def download_daily_field(client: cdsapi.Client, field: str, year: int) -> None:
    """Download one year of a daily ERA5-Land statistic and normalize the
    variable name to what datasets.load_era5_data expects."""
    spec = DAILY_FIELDS[field]
    out_dir = ERA5_LAND_DIR / field
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{spec['prefix']}-{year}.nc"
    if final_path.exists():
        print(f"[skip] {final_path} already exists")
        return

    raw_path = out_dir / f"_raw_{spec['prefix']}-{year}.nc"
    print(f"[download] {field} ({spec['daily_statistic']}) {year} -> {raw_path}")
    client.retrieve(
        "derived-era5-land-daily-statistics",
        {
            "variable": [spec["cds_variable"]],
            "year": str(year),
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            "daily_statistic": spec["daily_statistic"],
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": ERA5_LAND_AREA,
            "data_format": "netcdf",
        },
        str(raw_path),
    )

    # The derived dataset names the temperature variable `t2m` and precip `tp`;
    # normalize to the variable name the loader expects.
    ds = xr.open_dataset(raw_path, engine="h5netcdf")
    rename = {}
    out_var = spec["out_var"]
    if "t2m" in ds and out_var not in ds:
        rename["t2m"] = out_var
    # Some CDS responses use `valid_time` instead of `time`.
    if "valid_time" in ds.dims and "time" not in ds.dims:
        rename["valid_time"] = "time"
    if rename:
        ds = ds.rename(rename)
    ds.to_netcdf(final_path, engine="h5netcdf")
    ds.close()
    raw_path.unlink()
    print(f"[done] {final_path}  vars={list(ds.data_vars)}  dims={dict(ds.sizes)}")


def download_pressure_level_field(
    client: cdsapi.Client, field: str, year: int, levels: list[str],
    hours: list[str],
) -> None:
    """Download one year of an ERA5 pressure-level field, one synoptic hour
    (UTC) per request and per file. The hourly set bills every level x hour x
    day as a field, so a full year of all hours in one request exceeds the CDS
    cost limit; a single hour (all levels, all days) stays well under it.

    Writes one file per hour: <prefix>-<year>-<HH>.nc, each with dims
    (time, pressure_level, lat, lon) and ~365 daily time steps."""
    spec = PRESSURE_LEVEL_FIELDS[field]
    out_dir = ERA5_PL_DIR / field
    out_dir.mkdir(parents=True, exist_ok=True)

    for h in hours:
        out_path = out_dir / f"{spec['prefix']}-{year}-{h}.nc"
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue
        print(f"[download] PL {field} ({spec['cds_variable']}) {year} {h}:00 UTC "
              f"levels={levels} -> {out_path}")
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": ["reanalysis"],
                "variable": [spec["cds_variable"]],
                "pressure_level": levels,
                "year": str(year),
                "month": ALL_MONTHS,
                "day": ALL_DAYS,
                "time": [f"{h}:00"],
                "area": SWISS_AREA,
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            str(out_path),
        )
        print(f"[done] {out_path}")


def download_surface_field(
    client: cdsapi.Client, field: str, year: int, hours: list[str],
) -> None:
    """Download one year of an ERA5 single-level surface field, one synoptic
    hour (UTC) per request and per file, on the same 0.25 deg grid and hours as
    the pressure-level data. Used as a near-surface anchor channel in
    atmospheric mode.

    Writes one file per hour: <prefix>-<year>-<HH>.nc, each with dims
    (time, lat, lon) and ~365 daily time steps."""
    spec = ERA5_SURFACE_FIELDS[field]
    out_dir = ERA5_SFC_DIR / field
    out_dir.mkdir(parents=True, exist_ok=True)

    for h in hours:
        out_path = out_dir / f"{spec['prefix']}-{year}-{h}.nc"
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue
        print(f"[download] SFC {field} ({spec['cds_variable']}) {year} {h}:00 UTC "
              f"-> {out_path}")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": [spec["cds_variable"]],
                "year": str(year),
                "month": ALL_MONTHS,
                "day": ALL_DAYS,
                "time": [f"{h}:00"],
                "area": SWISS_AREA,
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            str(out_path),
        )
        print(f"[done] {out_path}")


def download_geopotential(client: cdsapi.Client) -> None:
    """Download the static ERA5-Land geopotential field (variable `z`)."""
    GEOPOTENTIAL_DIR.mkdir(parents=True, exist_ok=True)
    final_path = GEOPOTENTIAL_DIR / "era5_land_geopotential.nc"
    if final_path.exists():
        print(f"[skip] {final_path} already exists")
        return

    print(f"[download] geopotential (static) -> {final_path}")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["geopotential"],
            "year": "2023",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "area": SWISS_AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(final_path),
    )
    print(f"[done] {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="2023",
                        help="e.g. '2023', '2014-2023', or '2014,2015'")
    parser.add_argument("--fields", default="all",
                        help=f"comma-separated subset of {list(DAILY_FIELDS)}, "
                             "'all', or 'none' to skip surface fields")
    parser.add_argument("--pl-fields", default="none",
                        help=f"pressure-level fields: subset of "
                             f"{list(PRESSURE_LEVEL_FIELDS)} (z=geopotential, "
                             "t=temperature, q=specific humidity, r=relative "
                             "humidity, u/v=wind components), 'all' (=z,t,q), or "
                             "'none' (default)")
    parser.add_argument("--levels", default=",".join(DEFAULT_PRESSURE_LEVELS),
                        help="comma-separated pressure levels in hPa, or 'all' "
                             f"(default: {','.join(DEFAULT_PRESSURE_LEVELS)})")
    parser.add_argument("--pl-hours", default=",".join(DEFAULT_PL_HOURS),
                        help="synoptic hours (UTC) to sample pressure levels "
                             "(and surface anchor fields) at, comma-separated, "
                             f"or 'all' for hourly (default: {','.join(DEFAULT_PL_HOURS)})")
    parser.add_argument("--sfc-fields", default="none",
                        help=f"ERA5 single-level surface anchor fields (0.25 deg, "
                             f"same grid/hours as pressure levels): subset of "
                             f"{list(ERA5_SURFACE_FIELDS)}, 'all' "
                             f"(={','.join(DEFAULT_SFC_FIELDS)}), or 'none' "
                             "(default). Sampled at --pl-hours.")
    parser.add_argument("--skip-geopotential", action="store_true")
    args = parser.parse_args()

    fields = parse_fields(args.fields)
    pl_fields = parse_pl_fields(args.pl_fields)
    sfc_fields = parse_sfc_fields(args.sfc_fields)
    levels = parse_levels(args.levels)
    hours = parse_hours(args.pl_hours)
    client = cdsapi.Client()
    for year in parse_years(args.years):
        for field in fields:
            download_daily_field(client, field, year)
        for field in pl_fields:
            download_pressure_level_field(
                client, field, year, levels, hours
            )
        for field in sfc_fields:
            download_surface_field(client, field, year, hours)
    if not args.skip_geopotential:
        download_geopotential(client)


if __name__ == "__main__":
    main()