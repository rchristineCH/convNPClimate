# Datasets used

## Sources

* ERA5-Land: 
  https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation

* ERA5 hourly data on pressure levels (daily statistics derived set):
  https://cds.climate.copernicus.eu/datasets/derived-era5-pressure-levels-daily-statistics

* MeteoSwiss grid-data products:
  https://www.meteoswiss.admin.ch/dam/jcr:818a4d17-cb0c-4e8b-92c6-1a1bdf5348b7/ProdDoc_TabsD.pdf
  https://www.meteoswiss.admin.ch/dam/jcr:4f51f0f1-0fe3-48b5-9de0-15666327e63c/ProdDoc_RhiresD.pdf


## Local storage

Download and store them in the following folder structure:

- datasets
  - ERA5_Land
    - max_temperature
      - t2m_max-1960.nc
      - t2m_max-1961.nc
      - ...
      - t2m_max-2023.nc
    - min_temperature
      - t2m_min-1960.nc
      - t2m_min-1961.nc
      - ...
      - t2m_min-2023.nc
    - precipitation
      - tp-1960.nc
      - tp-1961.nc
      - ...
      - tp-2023.nc
    - temperature
      - t2m-1960.nc
      - t2m-1961.nc
      - ...
      - t2m-2023.nc
    - geopotential
      - era5_land_geopotential.nc
  - ERA5_PressureLevels
    (vertical structure of the atmosphere; dims time x pressure_level x lat x lon.
     Default levels: 1000/925/850/700/500/300 hPa, sampled at 00/06/12/15/18 UTC
     to retain the diurnal cycle. One file per variable per year per hour, e.g.
     z_pl-2023-12.nc holds all days at 12 UTC.)
    - z          (geopotential per level; divide by g for geopotential height)
      - z_pl-2023-00.nc
      - z_pl-2023-06.nc
      - ...
      - z_pl-2023-18.nc
    - t          (temperature)
      - t_pl-2023-00.nc
      - ...
      - t_pl-2023-18.nc
    - q          (specific humidity; use r/ instead for relative humidity)
      - q_pl-2023-00.nc
      - ...
      - q_pl-2023-18.nc
    - u          (zonal wind component, m/s; west->east)
      - u_pl-2023-00.nc
      - ...
      - u_pl-2023-18.nc
    - v          (meridional wind component, m/s; south->north)
      - v_pl-2023-00.nc
      - ...
      - v_pl-2023-18.nc
  - MeteoSwiss
    - RhiresD_v2.0_swiss.lv95
      - RhiresD_ch01h.swiss.lv95_196101010000_196112310000.nc
      - RhiresD_ch01h.swiss.lv95_196201010000_196212310000.nc
      - ...
      - RhiresD_ch01h.swiss.lv95_202301010000_202312310000.nc
    - TabsD_v2.0_swiss.lv95
      - TabsD_ch01r.swiss.lv95_196101010000_196112310000.nc
      - TabsD_ch01r.swiss.lv95_196201010000_196212310000.nc
      - ...
      - TabsD_ch01r.swiss.lv95_202301010000_202312310000.nc
    - TmaxD_v2.0_swiss.lv95
      - TmaxD_ch01r.swiss.lv95_197101010000_197112310000.nc
      - TmaxD_ch01r.swiss.lv95_197201010000_197212310000.nc
      - ...
      - TmaxD_ch01r.swiss.lv95_202301010000_202312310000.nc
    - TminD_v2.0_swiss.lv95
      - TminD_ch01r.swiss.lv95_197101010000_197112310000.nc
      - TminD_ch01r.swiss.lv95_197201010000_197212310000.nc
      - ...
      - TminD_ch01r.swiss.lv95_202301010000_202312310000.nc
  - topo_subset.zarr (swisstopo DHM25, all subfolders with content in zarray format 2)
    - DEM
    - EASTING
    - latitude
    - longitude
    - NORTHING
    - SN_DERIVATIVE_2000M_SIGRATIO1
    - SN_DERIVATIVE_500M_SIGRATIO1
    - TPI_2000M
    - TPI_500M
    - VALLEY_NORM_2000M_SMTHFACT0.5
    - WE_DERIVATIVE_2000M_SIGRATIO1
    - WE_DERIVATIVE_500M_SIGRATIO1
    - x
    - y
    - zarr.json
    (Note: this dataset mixes Zarr format 3 and Zarr format 2 and that causes trouble. Renaming zarr.json to zarr.json.bak makes the dataset fully loadable in format 2.)


## Dependencies

You will need `xarray` and `h5netcdf`.

```
pip install xarray h5netcdf
```


## Loading

Directly load the data like this:

```
import xarray as xr

era5_temp2m = xr.open_dataset("datasets/ERA5_Land/temperature/t2m-1960.nc")
mch_temp2m = xr.open_dataset("datasets/MeteoSwiss/TabsD_v2.0_swiss.lv95/TabsD_ch01r.swiss.lv95_196101010000_196112310000.nc")
topo_ds = xr.open_zarr("datasets/topo_subset.zarr")
```

Pressure-level fields are split into one file per synoptic hour. Load a single
hour directly, or combine all hours of a year into one time series:

```
import xarray as xr

# One hour (e.g. 12 UTC): dims (time, pressure_level, latitude, longitude)
era5_t_12utc = xr.open_dataset("datasets/ERA5_PressureLevels/t/t_pl-2023-12.nc")

# All synoptic hours of a year, merged along time
era5_t_2023 = xr.open_mfdataset(
    "datasets/ERA5_PressureLevels/t/t_pl-2023-*.nc", combine="by_coords"
)

# Geopotential height (m) per level = geopotential z / g
g = 9.80665
z = xr.open_dataset("datasets/ERA5_PressureLevels/z/z_pl-2023-12.nc")["z"]
geopotential_height = z / g
```

datasets.py contains tooling for loading and processing the datasets as well.