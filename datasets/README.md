# Input data

Everything the pipeline reads lives under this directory, resolved by
`datasets.build_data_paths()`. The data itself is not in git; fetch it with the
two scripts here. Both anchor their output to this directory, so they can be run
from anywhere.

## Required layout

```
datasets/
├── ERA5_Land/
│   ├── max_temperature/   t2m_max-<year>.nc      (daily t2m maximum, 0.1°)
│   ├── precipitation/     tp-<year>.nc           (daily tp, 0.1°; see note below)
│   └── geopotential/      *.nc                   (static elevation field)
├── ERA5_PressureLevels/
│   └── {z,t,q,u,v}/       <var>_pl-<year>-<HH>.nc  (0.25°, synoptic hours)
├── ERA5_Surface/
│   └── {t2m,tp}/          <var>_sfc-<year>-<HH>.nc (0.25°, same hours)
├── MeteoSwiss/
│   ├── TmaxD_v2.0_swiss.lv95/    *.nc            (tmax targets, ~1 km)
│   └── RhiresD_v2.0_swiss.lv95/  *.nc            (precip targets, ~1 km)
└── topo_subset.zarr/                              (DEM/TPI fields on the target grid)
```

The loaders glob `*.nc` per directory and combine by coordinates, so exact file
names don't matter. u/v pressure-level winds are only needed for the wind-input
runs; `ERA5_Surface` only for the surface-anchor and surface-tp runs.

## ERA5 (Copernicus CDS)

Needs a CDS account and `~/.cdsapirc` (see
https://cds.climate.copernicus.eu/how-to-api). Then, e.g. for the training span:

```bash
python datasets/download_era5.py --years 2020-2023 --fields max_temperature,precipitation
python datasets/download_era5.py --years 2020-2023 --fields none --pl-fields all      # z,t,q
python datasets/download_era5.py --years 2020-2023 --fields none --pl-fields u,v      # wind runs
python datasets/download_era5.py --years 2020-2023 --fields none --sfc-fields all     # t2m,tp anchors
```

**Precipitation note:** CDS-delivered daily `tp` files are labelled by
accumulation window and recent downloads can be day-shifted; the corrected
daily files are rebuilt from hourly data by
`python scripts/build_tp_from_hourly.py --years 2020-2024 --download`.
See `tp_hourly/README.md` before downloading or replacing any tp year.

## MeteoSwiss targets (Open Government Data, no account needed)

```bash
python datasets/download_meteoswiss.py --years 2020-2024 --variables tmax,precip
```

Annual files land directly in the two product directories above.

## Topography (`topo_subset.zarr`)

Derived from swisstopo elevation data on the MeteoSwiss target grid; it is not
downloadable from a public endpoint. Take it from the project's data archive
(the local-only `data-archive` branch) or from a previous checkout. If a
`zarr.json` file appears inside it, rename it away — the v3 metadata breaks the
v2 reader.

## Not required

MeteoSwiss TabsD/TminD, ERA5-Land mean/min temperature and E-OBS files may sit
alongside the above but are read by nothing in the pipeline.
