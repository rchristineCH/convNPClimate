"""
Dataset loading and metadata utilities for ERA5 and MeteoSwiss data.
"""

import dataclasses
import re
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pyproj
import torch
import torch.nn.functional as F
import xarray as xr

from json_utils import save_dataclass_json, load_dataclass_json
from params import set_seed
from convCNP.validation.utils import get_dists


KELVIN_OFFSET = 273.15

# Scalar/ensemble coordinates that recent CDS ERA5 downloads attach (e.g.
# 'number' = ensemble member, 'expver' = experiment version). They carry no
# information for this pipeline but break xr.concat when only *some* files in a
# multi-year glob have them (concat with coords='different' then complains the
# coord is "not present in all datasets"). Drop them on load so the channel
# concat in load_era5_data is robust to a mixed-schema dataset.
_STRAY_ERA5_COORDS = ('number', 'expver')


def _drop_stray_era5_coords(ds):
    """Drop CDS ensemble/version coords (``number``/``expver``) if present."""
    return ds.drop_vars([c for c in _STRAY_ERA5_COORDS if c in ds.coords],
                        errors='ignore')


# Precision to snap ERA5 lat/lon coords to. The ERA5(-Land) grids are >=0.1 deg,
# so 5 decimal places (1e-5 deg ~ 1 m) is far finer than any real grid spacing
# yet far coarser than the ~1e-13 float64 drift we are trying to remove.
_ERA5_COORD_DECIMALS = 5


def _snap_era5_coords(ds, decimals=_ERA5_COORD_DECIMALS):
    """Round latitude/longitude coordinates to a fixed precision.

    ERA5 NetCDF files written per-year can store the *same* nominal grid line
    (e.g. 45.4 N, 11.0 E) with float64 drift of ~1e-13 between files.
    ``xr.open_mfdataset(combine='by_coords')`` aligns by the *exact* coordinate
    value, so that drift makes it treat a drifted boundary as a distinct grid
    line and OUTER-JOINs the files into a grid one row/column larger, NaN-filling
    the phantom border for every timestep. That NaN border then flows through the
    surface ``data`` channel into the model and produces an all-NaN output on the
    first forward pass. Snapping every file's coords to a fixed precision makes
    them share identical coordinates so they concatenate onto one clean grid.
    """
    new = {name: np.round(ds[name].values.astype('float64'), decimals)
           for name in ('latitude', 'longitude') if name in ds.coords}
    return ds.assign_coords(new) if new else ds


def _era5_mf_preprocess(ds):
    """Per-file preprocess for ``open_mfdataset``: drop stray coords + snap grid.

    Applied to each dataset *before* ``combine='by_coords'`` merges them, which is
    the only point at which coordinate snapping can prevent a phantom NaN border
    (see ``_snap_era5_coords``).
    """
    return _snap_era5_coords(_drop_stray_era5_coords(ds))

# Coordinate transformers for Swiss LV95 (EPSG:2056) <-> WGS84 (EPSG:4326)
_WGS84_TO_LV95 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
_LV95_TO_WGS84 = pyproj.Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)


def wgs84_to_lv95(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert WGS84 (lon, lat) to Swiss LV95 (x, y) coordinates."""
    x, y = _WGS84_TO_LV95.transform(lon, lat)
    return x, y


def lv95_to_wgs84(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Swiss LV95 (x, y) to WGS84 (lon, lat) coordinates."""
    lon, lat = _LV95_TO_WGS84.transform(x, y)
    return lon, lat


@dataclasses.dataclass
class Era5Metadata:
    """
    Holds metadata and statistics for input data.

    These are required:
     - to normalize target coordinates into the same reference
       frame as the input context.
     - to calculate distances of to points in the target set
    """
    data_mean: float
    data_std: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    # Original coordinate arrays from the input grid (for distance calculations)
    lat_coords: np.ndarray  # Shape: (n_lat,)
    lon_coords: np.ndarray  # Shape: (n_lon,)
    # Normalization stats for the geopotential-derived elevation channel, so the
    # exact same transform can be reapplied to unseen data (None if no
    # geopotential was loaded). Optional/defaulted for backward compatibility
    # with metadata.json files written before these fields existed.
    elev_mean: float | None = None
    elev_std: float | None = None

    def denormalize(self, normalized_data: float | np.ndarray) -> float | np.ndarray:
        """
        Converts normalized data back to original units (Kelvin).
        Formula: Original = (Normalized * Std) + Mean
        """
        return (normalized_data * self.data_std) + self.data_mean


# Fields containing numpy arrays that need special JSON handling
ERA5_METADATA_NUMPY_FIELDS = ['lat_coords', 'lon_coords']


def save_metadata_json(metadata: Era5Metadata, output_path: Path) -> None:
    """Save Era5Metadata dataclass as JSON to specified path."""
    save_dataclass_json(metadata, output_path, numpy_fields=ERA5_METADATA_NUMPY_FIELDS)


def load_metadata_json(metadata_json: Path) -> Era5Metadata:
    """Load Era5Metadata dataclass from JSON file at specified path."""
    return load_dataclass_json(Era5Metadata, metadata_json, numpy_fields=ERA5_METADATA_NUMPY_FIELDS)


@dataclasses.dataclass
class DataPaths:
    """Glob patterns and paths for all datasets."""
    ERA5_MAX_TEMP_GLOB: str
    ERA5_PRECIP_GLOB: str
    ERA5_GEOPOTENTIAL_GLOB: str
    # Base directory for the ERA5 pressure-level (atmospheric) fields. Per-field
    # files live in <dir>/<var>/<var>_pl-<year>-<hour>.nc; the loader builds the
    # per-(variable, hour) globs from this base.
    ERA5_PRESSURE_LEVEL_DIR: str
    # Base directory for the ERA5 single-level surface fields (0.25 deg, same
    # grid as the pressure levels). Per-field files live in
    # <dir>/<var>/<var>_sfc-<year>-<hour>.nc; used as a surface anchor in the
    # atmospheric configuration.
    ERA5_SURFACE_DIR: str
    EOBS_MAX_TEMP_GLOB: str
    EOBS_PRECIP_GLOB: str
    METEO_SWISS_MAX_TEMP_GLOB: str
    METEO_SWISS_PRECIP_GLOB: str
    HI_RES_TOPOGRAPHY_ZARR_PATH: str


def build_data_paths(base_dataset_dir: Path | str) -> DataPaths:
    """Construct all dataset glob patterns from a base directory.

    Args:
        base_dataset_dir: Root directory containing the dataset subdirectories.

    Returns:
        DataPaths with all glob patterns resolved.
    """
    base = Path(base_dataset_dir)
    return DataPaths(
        ERA5_MAX_TEMP_GLOB=str(base / 'ERA5_Land/max_temperature') + '/*.nc',
        ERA5_PRECIP_GLOB=str(base / 'ERA5_Land/precipitation') + '/*.nc',
        ERA5_GEOPOTENTIAL_GLOB=str(base / 'ERA5_Land/geopotential') + '/*.nc',
        ERA5_PRESSURE_LEVEL_DIR=str(base / 'ERA5_PressureLevels'),
        ERA5_SURFACE_DIR=str(base / 'ERA5_Surface'),
        EOBS_MAX_TEMP_GLOB=str(base / 'EOBS/max_temperature') + '/*.nc',
        EOBS_PRECIP_GLOB=str(base / 'EOBS/precipitation') + '/*.nc',
        METEO_SWISS_MAX_TEMP_GLOB=str(base / 'MeteoSwiss/TmaxD_v2.0_swiss.lv95') + '/*.nc',
        METEO_SWISS_PRECIP_GLOB=str(base / 'MeteoSwiss/RhiresD_v2.0_swiss.lv95') + '/*.nc',
        HI_RES_TOPOGRAPHY_ZARR_PATH=str(base / 'topo_subset.zarr'),
    )


def load_era5_data(
    var_glob: Path | str,
    var_name: str = 't2m_max',
    year_start: int | None = None,
    year_end: int | None = None,
    geopotential_glob: Path | str | None = None,
    geopotential_var_name: str = 'z',
    include_seasonal_embeddings = True,
    include_data_channel: bool = True,
    include_elevation_channel: bool = True,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, Era5Metadata, torch.Tensor | None, np.ndarray]:
    """
    Loads ERA5 data from NetCDF files matching the given glob pattern.
    Optionally filters data to a specific year range.
    Optionally loads geopotential data and converts to altitude in meters.

    Args:
        var_glob: Glob pattern for NetCDF files
        var_name: Name of the variable in the dataset
        year_start: Optional first year to include (inclusive)
        year_end: Optional last year to include (inclusive)
        geopotential_glob: Optional glob pattern for geopotential NetCDF files
        geopotential_var_name: Name of geopotential variable (default 'z')
        include_data_channel: If False, omit the surface variable itself as a
            context channel while still keeping the lat/lon/seasonal/elevation
            scaffold and computing normalization stats. Used for the
            atmospheric-only configuration (USE_SURFACE=False) where the
            pressure-level channels replace the surface field.
        include_elevation_channel: If False, omit the geopotential-derived
            coarse elevation channel from the context, while STILL loading
            geopotential and returning ``altitude`` (so the elevation MLP's
            elev_diff feature keeps working). Used for the "no-geopotential"
            surface ablation (drops the input channel only, not the MLP).
        device: Torch device to load tensor to (defaults to CPU)

    Returns:
        tensor: Processed input tensor with shape (time, channels, lat, lon)
        stats: Era5Metadata containing normalization parameters and grid coordinates
        altitude: Tensor with altitude in meters, shape (lat, lon), or None if geopotential_glob not provided
        time_coords: Array of datetime64 values corresponding to each time index
    """
    if device is None:
        device = torch.device('cpu')

    var_glob = str(var_glob)  # Ensure it's a string for xarray
    # Data load
    print(f"Loading ERA5 data from: {var_glob}")
    ds = xr.open_mfdataset(var_glob, combine='by_coords', preprocess=_era5_mf_preprocess)
    print(f"Dataset loaded with dimensions: {ds.dims}")
    print(f"Original dataset time range: {ds.time.min().values} to {ds.time.max().values}")

    # Data filtering
    t_start = f'{year_start}-01-01' if year_start is not None else None
    t_end = f'{year_end}-12-31' if year_end is not None else None
    if t_start is not None or t_end is not None:
        ds = ds.sel(time=slice(t_start, t_end))
        print(f"Filtered dataset time range [{year_start}, {year_end}]: {ds.time.min().values} to {ds.time.max().values}")
    print(f"Filtered dataset dimensions: {ds.dims}")

    # Normalization
    # Note: we explicitly call .compute() here because dask lazy arrays cannot
    # be converted to python scalars via .item() directly.
    print(f"Calculating stats for {var_name} (Kelvin)...")
    mean_val = ds[var_name].mean().compute().item()
    std_val = ds[var_name].std().compute().item()
    lat_min = ds.latitude.min().compute().item()
    lat_max = ds.latitude.max().compute().item()
    lon_min = ds.longitude.min().compute().item()
    lon_max = ds.longitude.max().compute().item()

    # Extract coordinate arrays for distance calculations
    lat_coords = ds.latitude.values
    lon_coords = ds.longitude.values

    stats = Era5Metadata(
        data_mean=mean_val,
        data_std=std_val,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_coords=lat_coords,
        lon_coords=lon_coords,
    )
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    if std_val == 0:
        raise ValueError(f"ERA5 variable '{var_name}' has zero standard deviation (constant field); cannot normalize.")
    if lat_range == 0 or lon_range == 0:
        raise ValueError(f"ERA5 coordinate range is zero (lat_range={lat_range}, lon_range={lon_range}); cannot normalize.")
    norm_data = (ds[var_name] - mean_val) / std_val

    # Normalize coordinates
    lat_norm = (ds.latitude - lat_min) / lat_range
    lon_norm = (ds.longitude - lon_min) / lon_range
    # Xarray broadcast automatically expands (lat) and (lon) to (time, lat, lon)
    # dependent on the shape of `norm_data`.
    lat_channel, lon_channel, _ = xr.broadcast(lat_norm, lon_norm, norm_data)

    # Time Embeddings
    day_of_year = ds.time.dt.dayofyear
    time_rads = (day_of_year - 1) / 365.0 * 2 * np.pi
    # Use numpy functions; if inputs are dask/xarray, result is lazy
    cos_time = np.cos(time_rads)
    sin_time = np.sin(time_rads)
    # Broadcast time to spatial dimensions
    cos_channel, sin_channel, _ = xr.broadcast(cos_time, sin_time, norm_data)

    # Load geopotential and convert to altitude if provided.
    # This must run before channel stacking so the (normalized) altitude can be
    # appended as a static input channel alongside the time-varying channels.
    altitude, altitude_norm = None, None
    if geopotential_glob is not None:
        geopotential_glob = str(geopotential_glob)
        print(f"\nLoading geopotential data from: {geopotential_glob}")
        geo_ds = xr.open_mfdataset(geopotential_glob, combine='by_coords', preprocess=_era5_mf_preprocess)
        print(f"Geopotential dataset dimensions: {geo_ds.dims}")

        # The geopotential file labels its axes opposite to ERA5: its 'latitude' axis
        # actually holds longitude (~5-11 E) values and its 'longitude' axis holds
        # latitude (~45-49 N) values. Swap the axis names so the rest of the pipeline
        # (grid matching, sel/interp on lat_coords/lon_coords) operates on the real axes.
        geo_ds = geo_ds.rename({'latitude': 'longitude', 'longitude': 'latitude'})

        # Verify grids match
        geo_lat = geo_ds.latitude.values
        geo_lon = geo_ds.longitude.values
        print(f"Geopotential lat range: [{geo_lat.min():.4f}, {geo_lat.max():.4f}], shape: {geo_lat.shape}")
        print(f"Geopotential lon range: [{geo_lon.min():.4f}, {geo_lon.max():.4f}], shape: {geo_lon.shape}")
        print(f"ERA5 data lat range: [{lat_coords.min():.4f}, {lat_coords.max():.4f}], shape: {lat_coords.shape}")
        print(f"ERA5 data lon range: [{lon_coords.min():.4f}, {lon_coords.max():.4f}], shape: {lon_coords.shape}")

        # Check if grids match exactly
        grids_match_exactly = (
            geo_lat.shape == lat_coords.shape and
            geo_lon.shape == lon_coords.shape and
            np.allclose(geo_lat, lat_coords, atol=1e-6) and
            np.allclose(geo_lon, lon_coords, atol=1e-6)
        )

        if grids_match_exactly:
            print("✓ Geopotential and ERA5 data grids match exactly")
        else:
            # Check if ERA5 coords are a subset of geopotential coords (within tolerance)
            # This handles cases where geopotential covers a larger area but same resolution
            lat_subset = all(np.min(np.abs(geo_lat - t)) < 1e-4 for t in lat_coords)
            lon_subset = all(np.min(np.abs(geo_lon - t)) < 1e-4 for t in lon_coords)

            if lat_subset and lon_subset:
                print("✓ ERA5 grid points are subset of geopotential grid (using nearest-neighbor selection)")
                # Use sel with nearest method - this is like a left join on coordinates
                geo_ds = geo_ds.sel(latitude=lat_coords, longitude=lon_coords, method='nearest')
                print(f"  Selected geopotential dimensions: {geo_ds.dims}")
            else:
                print("⚠ WARNING: Geopotential grid does not contain all ERA5 points!")
                print("  Falling back to linear interpolation...")
                geo_ds = geo_ds.interp(latitude=lat_coords, longitude=lon_coords, method='linear')
                print(f"  Interpolated geopotential dimensions: {geo_ds.dims}")

        # Convert geopotential to altitude: altitude = geopotential / g
        # Geopotential is in m²/s², g = 9.80665 m/s²
        GRAVITY = 9.80665
        geopotential = geo_ds[geopotential_var_name]

        # If there's a time dimension, take first timestep (geopotential is static)
        if 'time' in geopotential.dims:
            geopotential = geopotential.isel(time=0)
            print("  Selected first time step from geopotential (static field)")

        altitude = geopotential / GRAVITY
        print(f"Converted geopotential to altitude (m). Range: [{float(altitude.min()):.1f}, {float(altitude.max()):.1f}] m")

        # Re-label altitude's grid with the exact ERA5 coordinate arrays. After a
        # nearest-neighbour .sel the coordinate *labels* are the geopotential grid's
        # own values, which can differ from the ERA5 grid by ~1e-5. xr.broadcast /
        # xr.concat align by label and OUTER-JOIN mismatched coords, which would
        # silently union the two near-identical grids and double the lat/lon size.
        # The sizes already match (verified/selected/interpolated above), so this is
        # a pure relabel that forces an exact identity alignment.
        altitude = altitude.assign_coords(latitude=lat_coords, longitude=lon_coords)

        # Z-score normalize the altitude (same style as the temperature channel)
        # so it sits on a comparable scale to the other input channels.
        alt_mean = float(altitude.mean())
        alt_std = float(altitude.std())
        if alt_std == 0:
            raise ValueError("Geopotential-derived altitude has zero standard deviation (constant field); cannot normalize.")
        altitude_norm = (altitude - alt_mean) / alt_std
        # Record so the exact elevation normalization can be reapplied to unseen data.
        stats.elev_mean = alt_mean
        stats.elev_std = alt_std

    # Channel stacking
    # Create the list of channels. The surface variable itself (norm_data) is
    # optional (include_data_channel); the lat/lon scaffold is always present so
    # the grid is well-defined even in the atmospheric-only configuration.
    if include_data_channel:
        channel_list = [norm_data, lat_channel, lon_channel]
        channel_names = ['data', 'lat', 'lon']
    else:
        channel_list = [lat_channel, lon_channel]
        channel_names = ['lat', 'lon']

    if include_seasonal_embeddings:
        channel_list.extend([cos_channel, sin_channel])
        channel_names.extend(['cos_time', 'sin_time'])

    if geopotential_glob is not None and include_elevation_channel:
        # Altitude is static (lat, lon). Broadcast it against the full time-varying
        # field so it gains a properly-indexed `time` dimension matching the other
        # channels; this lets xr.concat align all channels (it cannot mix a channel
        # whose `time` is a scalar coordinate with channels that have a `time` index).
        # Same pattern as the lat/lon/seasonal channels above.
        alt_channel, _ = xr.broadcast(altitude_norm, norm_data)
        channel_list.append(alt_channel)
        channel_names.append('elevation')

    # xr.concat preserves dask arrays.
    tensor_Z = xr.concat(channel_list, dim="channel")

    # Assign labels to the channel dimension so we know what is what
    tensor_Z = tensor_Z.assign_coords(channel=channel_names)

    # Order matches: [Data, Lat, Lon, CosTime, SinTime, (Elevation)]
    tensor_Z = tensor_Z.transpose('time', 'channel', 'latitude', 'longitude')

    # Extract time coordinates before converting to tensor
    time_coords = ds.time.values

    # Use float32 consistently - the model and mask generation use float32
    tensor_Z = torch.from_numpy(tensor_Z.values.astype(np.float32)).to(device)
    print(f"Final tensor_Z torch tensor with shape (time, channel, lat, lon): {tensor_Z.shape}, dtype: {tensor_Z.dtype}")

    return tensor_Z, stats, altitude, time_coords


def load_era5_pressure_levels(
    pressure_level_dir: Path | str,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    time_coords: np.ndarray,
    variables: list[str] = ('z', 't', 'q'),
    levels: list[int] = (1000, 925, 850, 700, 500, 300),
    hours: list[str] = ('00', '06', '12', '15', '18'),
    native_grid: bool = False,
    stats_out: list | None = None,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, list[str]]:
    """Load ERA5 pressure-level (atmospheric) fields as additional context
    channels, aligned to the surface grid and daily time axis.

    If ``stats_out`` is a list, the per-channel normalization stats are appended
    to it as ``{'channel', 'mean', 'std'}`` dicts so the exact transform can be
    reapplied to unseen data.

    The pressure-level files are stored one per (variable, year, hour) on the
    coarser 0.25 deg ERA5 grid (see download_era5.py). Each (variable, level,
    hour) combination becomes its own normalized channel, so the diurnal cycle
    is retained (e.g. t @ 850 hPa @ 12 UTC is a distinct channel from t @ 850
    hPa @ 06 UTC). The output is stacked onto the surface tensor produced by
    load_era5_data along the channel dimension.

    Args:
        pressure_level_dir: Base dir holding <var>/<var>_pl-<year>-<hour>.nc.
        lat_coords, lon_coords: Surface grid coordinate arrays (from the
            Era5Metadata returned by load_era5_data) to interpolate onto.
        time_coords: Surface daily datetime64 array to align to (by calendar
            day); the pressure-level fields are selected to match these dates.
        variables: ERA5 short names to include (e.g. 'z', 't', 'q', 'r').
        levels: Pressure levels in hPa to include.
        hours: Synoptic hours (UTC, 'HH') to include, one channel-set per hour.
        native_grid: If True, keep the field on its native coarse (0.25 deg)
            pressure-level grid instead of interpolating it up to the fine
            surface grid (lat_coords/lon_coords are then used only for the
            target-date alignment, not for regridding). The RBF final layer maps
            any grid -> target points, so the encoder can consume the coarse
            field directly. In this mode the function also returns the native
            coordinate arrays as a third element (lat_native, lon_native).
        device: Torch device for the returned tensor (defaults to CPU).

    Returns:
        tensor: shape (time, channel, lat, lon), float32, normalized per channel.
        channel_names: e.g. ['t850_12', 'q700_06', ...], one per channel.
        If native_grid=True, additionally returns (lat_native, lon_native): the
        coarse grid coordinate arrays the tensor lives on.
    """
    if device is None:
        device = torch.device('cpu')

    pressure_level_dir = Path(pressure_level_dir)
    # Daily target dates (midnight-normalized) the channels must line up with.
    target_dates = pd.to_datetime(time_coords).normalize()

    channel_list: list[xr.DataArray] = []
    channel_names: list[str] = []
    # When native_grid, all (var, hour) files share the same coarse product grid;
    # capture the first one's coords as the reference and force every later file
    # onto identical labels so xr.concat aligns by identity (it would otherwise
    # outer-join near-identical float labels and silently union the grids).
    native_lat: np.ndarray | None = None
    native_lon: np.ndarray | None = None

    for var in variables:
        for hour in hours:
            glob = str(pressure_level_dir / var / f"{var}_pl-*-{hour}.nc")
            print(f"\nLoading pressure-level {var} @ {hour} UTC from: {glob}")
            ds = xr.open_mfdataset(glob, combine='by_coords', preprocess=_era5_mf_preprocess)

            # Normalize coordinate names to (time, pressure_level, latitude, longitude).
            rename = {}
            if 'valid_time' in ds.dims and 'time' not in ds.dims:
                rename['valid_time'] = 'time'
            for level_name in ('level', 'plev', 'isobaricInhPa'):
                if level_name in ds.dims and 'pressure_level' not in ds.dims:
                    rename[level_name] = 'pressure_level'
                    break
            if rename:
                ds = ds.rename(rename)

            # Align to the daily target axis by calendar day. The hourly stamps
            # (e.g. 2023-06-01T12:00) are floored to the day so they match the
            # surface daily dates exactly.
            ds = ds.assign_coords(time=pd.to_datetime(ds.time.values).normalize())
            missing = target_dates.difference(pd.to_datetime(ds.time.values))
            if len(missing) > 0:
                raise ValueError(
                    f"Pressure-level {var} @ {hour} UTC is missing {len(missing)} "
                    f"of the surface dates (e.g. {missing[0].date()}). Ensure the "
                    "pressure-level download covers the same years as the surface data."
                )
            ds = ds.sel(time=target_dates)

            if native_grid:
                # Keep the field on its own coarse grid. Pin every file to the
                # first file's coordinate labels to force identity alignment.
                if native_lat is None:
                    native_lat = ds.latitude.values
                    native_lon = ds.longitude.values
                elif (ds.latitude.shape == native_lat.shape
                        and ds.longitude.shape == native_lon.shape):
                    ds = ds.assign_coords(latitude=native_lat, longitude=native_lon)
                else:
                    raise ValueError(
                        f"Pressure-level {var} @ {hour} UTC has a grid "
                        f"{ds.latitude.shape}x{ds.longitude.shape} that differs "
                        f"from the reference {native_lat.shape}x{native_lon.shape}; "
                        "native-grid mode requires a single shared coarse grid."
                    )
            else:
                # Regrid the coarse (0.25 deg) field onto the fine surface grid. The
                # fine surface grid extends ~0.2-0.3 deg beyond the coarse
                # pressure-level grid (CDS snaps the two products to different grid
                # spacings), which would leave a thin NaN border. Clamping the target
                # coordinates into the pressure-level range before interpolating
                # yields the nearest valid edge value there; we then relabel to the
                # true surface coordinates so the grid lines up exactly.
                lat_clamped = np.clip(lat_coords, float(ds.latitude.min()), float(ds.latitude.max()))
                lon_clamped = np.clip(lon_coords, float(ds.longitude.min()), float(ds.longitude.max()))
                ds = ds.interp(latitude=lat_clamped, longitude=lon_clamped, method='linear')
                ds = ds.assign_coords(latitude=lat_coords, longitude=lon_coords)

            for level in levels:
                da = ds[var].sel(pressure_level=level, method='nearest')
                # Z-score normalize per channel, matching the surface/elevation
                # channels' normalization style.
                mean_val = float(da.mean())
                std_val = float(da.std())
                if std_val == 0:
                    raise ValueError(
                        f"Pressure-level channel {var}{level}_{hour} has zero "
                        "standard deviation (constant field); cannot normalize."
                    )
                channel_list.append((da - mean_val) / std_val)
                channel_names.append(f"{var}{level}_{hour}")
                if stats_out is not None:
                    stats_out.append({'channel': f"{var}{level}_{hour}",
                                      'mean': mean_val, 'std': std_val})

    tensor = xr.concat(channel_list, dim='channel')
    tensor = tensor.assign_coords(channel=channel_names)
    tensor = tensor.transpose('time', 'channel', 'latitude', 'longitude')

    out = torch.from_numpy(tensor.values.astype(np.float32)).to(device)
    print(f"\nFinal pressure-level tensor (time, channel, lat, lon): {out.shape}, "
          f"dtype: {out.dtype}, channels: {len(channel_names)}, "
          f"grid: {'native coarse' if native_grid else 'regridded to surface'}")
    if native_grid:
        return out, channel_names, native_lat, native_lon
    return out, channel_names


def load_era5_surface_levels(
    surface_dir: Path | str,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    time_coords: np.ndarray,
    variables: list[str] = ('t2m', 'tp'),
    hours: list[str] = ('00', '06', '12', '15', '18'),
    stats_out: list | None = None,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, list[str]]:
    """Load ERA5 single-level surface fields as additional context channels,
    aligned to the surface grid and daily time axis.

    If ``stats_out`` is a list, the per-channel normalization stats are appended
    to it as ``{'channel', 'mean', 'std'}`` dicts.

    These are the reanalysis-era5-single-levels fields (0.25 deg, same grid and
    synoptic-hour sampling as the pressure levels) downloaded by
    download_era5.py as a near-surface anchor for the atmospheric (pressure-
    level) configuration. Unlike load_era5_pressure_levels there is no vertical
    dimension, so each (variable, hour) becomes its own normalized channel (e.g.
    t2m @ 12 UTC, tp @ 06 UTC). The output is stacked onto the atmospheric
    tensor along the channel dimension.

    Args:
        surface_dir: Base dir holding <var>/<var>_sfc-<year>-<hour>.nc.
        lat_coords, lon_coords: Surface grid coordinate arrays (from the
            Era5Metadata returned by load_era5_data) to interpolate onto.
        time_coords: Surface daily datetime64 array to align to (by calendar
            day); the surface fields are selected to match these dates.
        variables: ERA5 single-level short names to include (e.g. 't2m', 'tp').
        hours: Synoptic hours (UTC, 'HH') to include, one channel per hour.
        device: Torch device for the returned tensor (defaults to CPU).

    Returns:
        tensor: shape (time, channel, lat, lon), float32, normalized per channel.
        channel_names: e.g. ['t2m_12', 'tp_06', ...], one per channel.
    """
    if device is None:
        device = torch.device('cpu')

    surface_dir = Path(surface_dir)
    # Daily target dates (midnight-normalized) the channels must line up with.
    target_dates = pd.to_datetime(time_coords).normalize()

    channel_list: list[xr.DataArray] = []
    channel_names: list[str] = []

    for var in variables:
        for hour in hours:
            glob = str(surface_dir / var / f"{var}_sfc-*-{hour}.nc")
            print(f"\nLoading surface {var} @ {hour} UTC from: {glob}")
            ds = xr.open_mfdataset(glob, combine='by_coords', preprocess=_era5_mf_preprocess)

            # Normalize the time coordinate name to 'time' (CDS sometimes uses
            # 'valid_time'). Single-level fields have no pressure_level dim.
            if 'valid_time' in ds.dims and 'time' not in ds.dims:
                ds = ds.rename({'valid_time': 'time'})

            # Align to the daily target axis by calendar day. The hourly stamps
            # (e.g. 2023-06-01T12:00) are floored to the day so they match the
            # surface daily dates exactly.
            ds = ds.assign_coords(time=pd.to_datetime(ds.time.values).normalize())
            missing = target_dates.difference(pd.to_datetime(ds.time.values))
            if len(missing) > 0:
                raise ValueError(
                    f"Surface {var} @ {hour} UTC is missing {len(missing)} "
                    f"of the surface dates (e.g. {missing[0].date()}). Ensure the "
                    "surface download covers the same years as the input data."
                )
            ds = ds.sel(time=target_dates)

            # Regrid the coarse (0.25 deg) field onto the fine surface grid,
            # clamping target coords into the coarse extent first to avoid a thin
            # NaN border, then relabel to the true surface coords (see
            # load_era5_pressure_levels for the full rationale).
            lat_clamped = np.clip(lat_coords, float(ds.latitude.min()), float(ds.latitude.max()))
            lon_clamped = np.clip(lon_coords, float(ds.longitude.min()), float(ds.longitude.max()))
            ds = ds.interp(latitude=lat_clamped, longitude=lon_clamped, method='linear')
            ds = ds.assign_coords(latitude=lat_coords, longitude=lon_coords)

            # Z-score normalize per channel, matching the surface/pressure-level
            # channels' normalization style.
            da = ds[var]
            mean_val = float(da.mean())
            std_val = float(da.std())
            if std_val == 0:
                raise ValueError(
                    f"Surface channel {var}_{hour} has zero standard deviation "
                    "(constant field); cannot normalize."
                )
            channel_list.append((da - mean_val) / std_val)
            channel_names.append(f"{var}_{hour}")
            if stats_out is not None:
                stats_out.append({'channel': f"{var}_{hour}",
                                  'mean': mean_val, 'std': std_val})

    tensor = xr.concat(channel_list, dim='channel')
    tensor = tensor.assign_coords(channel=channel_names)
    tensor = tensor.transpose('time', 'channel', 'latitude', 'longitude')

    out = torch.from_numpy(tensor.values.astype(np.float32)).to(device)
    print(f"\nFinal surface tensor (time, channel, lat, lon): {out.shape}, "
          f"dtype: {out.dtype}, channels: {len(channel_names)}")
    return out, channel_names


# ERA5 stores total precipitation as an accumulation in metres.
M_TO_MM = 1000.0


def _as_day_index(values) -> pd.DatetimeIndex:
    """Coerce assorted date representations to day-resolution timestamps."""
    return pd.to_datetime(np.asarray(values)).floor("D")


def load_era5_precip_aligned(
    precip_glob: Path | str,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    time_coords,
    var_name: str = 'tp',
) -> Tuple[np.ndarray, np.ndarray]:
    """ERA5-Land daily total precipitation in mm, aligned onto a day axis.

    This is the SINGLE alignment implementation: training (via
    ``load_era5_precip_channel``) and inference (``predict.build_surface_context``)
    both go through it, so the two can never drift apart in how they read the
    field. They differ only in normalization -- training computes and freezes the
    stats, inference reapplies them from the manifest.

    The day convention is the one already established in ``precip_baseline``:
    floor the source stamps to a day, collapse duplicates, reindex onto the
    requested days. There is deliberately NO +/-1 shift -- the ERA5-Land files
    are labelled by accumulation window (tp-2020.nc spans 2019-12-31..2020-12-30)
    but each record IS labelled with the day it belongs to, verified against
    MeteoSwiss at lag 0 for every year 2020-2023.

    Pass ``lat_coords=None``/``lon_coords=None`` to skip the grid check and keep
    the file's own grid -- useful when only day COVERAGE is wanted and the caller
    has no surface grid to hand (see ``predict.precip_covered_dates``).

    Returns:
        values: (time, lat, lon) float32 in mm, NaN on days the source does not
            cover (the ERA5-Land series can stop short of a calendar year end).
        covered: (time,) bool, True where the day carries a complete field.
    """
    dsx = xr.open_mfdataset(str(precip_glob), combine='by_coords',
                            preprocess=_era5_mf_preprocess)
    var = var_name if var_name in dsx.data_vars else list(dsx.data_vars)[0]
    da = dsx[var]

    rename = {}
    if 'lat' in da.dims or 'lat' in da.coords:
        rename['lat'] = 'latitude'
    if 'lon' in da.dims or 'lon' in da.coords:
        rename['lon'] = 'longitude'
    if 'valid_time' in da.dims or ('valid_time' in da.coords and 'time' not in da.dims):
        rename['valid_time'] = 'time'
    if rename:
        da = da.rename(rename)
    for stray in ('number', 'expver', 'step', 'surface', 'heightAboveGround'):
        if stray in da.coords:
            da = da.reset_coords(stray, drop=True)

    da = da.astype('float32') * M_TO_MM

    # The ERA5-Land precip grid is bit-identical to the surface (t2m_max) grid, so
    # this is a relabel, not a regrid. Refuse rather than silently interpolate: a
    # mismatch means the wrong file set, not a resampling job.
    if lat_coords is not None and lon_coords is not None:
        if (da.latitude.shape != np.shape(lat_coords)
                or da.longitude.shape != np.shape(lon_coords)
                or not np.allclose(da.latitude.values, lat_coords, atol=1e-6)
                or not np.allclose(da.longitude.values, lon_coords, atol=1e-6)):
            raise ValueError(
                f"ERA5 precip grid {da.latitude.shape}x{da.longitude.shape} does not match "
                f"the surface grid {np.shape(lat_coords)}x{np.shape(lon_coords)}; "
                f"refusing to interpolate (check {precip_glob!r} is the right file set)."
            )
        da = da.assign_coords(latitude=lat_coords, longitude=lon_coords)

    da = da.assign_coords(time=_as_day_index(da['time'].values))
    da = da.groupby('time').mean()
    da = da.reindex(time=_as_day_index(time_coords)).compute()

    values = da.transpose('time', 'latitude', 'longitude').values.astype(np.float32)
    covered = np.isfinite(values).all(axis=(1, 2))
    return values, covered


def align_target_to_days(target_y: xr.DataArray, time_coords, context_label: str = "context"):
    """Select a (time, point) target on a context's day axis, by DATE.

    The context axis is not always every day of the requested year range -- a
    source whose series stops short (ERA5-Land precip) trims it -- so the target
    and the context can no longer be assumed equal just because both were built
    from the same year bounds. Selecting by date rather than by position means a
    mismatch surfaces as an error instead of a silent one-day shift in the truth.

    Returns the target unchanged when the axes already agree.
    """
    ctx_days = _as_day_index(time_coords)
    tgt_days = _as_day_index(target_y["time"].values)
    if tgt_days.equals(ctx_days):
        return target_y
    missing = ctx_days.difference(tgt_days)
    if len(missing):
        raise ValueError(
            f"target is missing {len(missing)} of the {context_label} days "
            f"(e.g. {missing[0].date()}).")
    print(f"trimming target {len(tgt_days)} -> {len(ctx_days)} days to match the {context_label}")
    return target_y.assign_coords(time=tgt_days).sel(time=ctx_days)


def load_era5_precip_channel(
    precip_glob: Path | str,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    time_coords,
    var_name: str = 'tp',
    channel_name: str = 'tp',
    stats_out: list | None = None,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, list[str], slice]:
    """ERA5-Land daily precipitation as one z-scored context channel.

    Unlike every other context source, the ERA5-Land precipitation series can stop
    short of the requested day axis (its files are labelled by accumulation window,
    so the 2020-2023 set ends 2023-12-30). Rather than inventing a field for the
    uncovered days -- a spatially constant or persisted frame is an input the
    encoder never otherwise sees -- this returns the channel over the CONTIGUOUS
    covered span plus the slice into ``time_coords`` it corresponds to. The caller
    trims every other source to match, so the model only ever trains on real data.

    If ``stats_out`` is a list, the per-channel normalization stats are appended to
    it as ``{'channel', 'mean', 'std'}``, the same contract as
    ``load_era5_surface_levels``.

    Returns:
        tensor: (time, 1, lat, lon) float32, z-scored over the covered span.
        channel_names: ``[channel_name]``.
        span: slice into the requested ``time_coords``.
    """
    if device is None:
        device = torch.device('cpu')

    values, covered = load_era5_precip_aligned(
        precip_glob, lat_coords, lon_coords, time_coords, var_name=var_name)
    target_days = _as_day_index(time_coords)

    if not covered.any():
        raise ValueError(
            f"ERA5 precip {precip_glob!r} covers none of the requested days "
            f"({target_days[0].date()}..{target_days[-1].date()}).")

    idx = np.flatnonzero(covered)
    span = slice(int(idx[0]), int(idx[-1]) + 1)
    if not covered[span].all():
        gaps = target_days[span][~covered[span]]
        raise ValueError(
            f"ERA5 precip has {len(gaps)} interior gap(s) inside its covered span "
            f"(e.g. {gaps[0].date()}); an interior hole is a data defect, not "
            "something to slice around.")

    dropped = int(len(target_days) - (span.stop - span.start))
    if dropped:
        print(f"ERA5 precip covers {span.stop - span.start}/{len(target_days)} requested days "
              f"({target_days[span][0].date()}..{target_days[span][-1].date()}); "
              f"trimming {dropped} uncovered day(s) from the run.")

    sub = values[span]
    mean_val = float(np.mean(sub))
    std_val = float(np.std(sub))
    if std_val == 0:
        raise ValueError(
            f"ERA5 precip channel {channel_name!r} has zero standard deviation "
            "(constant field); cannot normalize.")
    sub = (sub - mean_val) / std_val

    if stats_out is not None:
        stats_out.append({'channel': channel_name, 'mean': mean_val, 'std': std_val})

    out = torch.from_numpy(sub).to(device).unsqueeze(1)
    print(f"\nERA5 precip channel {channel_name!r}: {tuple(out.shape)} "
          f"(mean {mean_val:.4f} mm, std {std_val:.4f} mm)")
    return out, [channel_name], span


# Load high-resolution topography data
def load_high_res_topography(
    topography_path: Path | str,
    dem_var: str = 'DEM',
    tpi_var: str = 'TPI_500M',
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Loads high-resolution topography data (DEM and TPI) from a Zarr file.

    Note: the dataset we used mixed Zarr format 3 and Zarr format 2 and that
    caused trouble. Renaming zarr.json to zarr.json.bak makes the dataset
    fully loadable in format 2.

    Args:
        topography_path: Path to the Zarr directory containing topography data
        dem_var: Name of the DEM variable (default 'DEM')
        tpi_var: Name of the TPI variable (default 'TPI_500M')

    Returns:
        Tuple of (dem_data, tpi_data) as xarray.DataArrays
    """
    print(f"Loading high-resolution topography from Zarr: {topography_path}")
    topo_ds = xr.open_zarr(str(topography_path))
    print(f"Topography dataset dimensions: {topo_ds.dims}")

    # Load DEM
    if dem_var in topo_ds:
        dem_data = topo_ds[dem_var]
    else:
        first_var = list(topo_ds.data_vars)[0]
        print(f"Warning: '{dem_var}' variable not found. Using first variable '{first_var}' instead.")
        dem_data = topo_ds[first_var]
    print(f"DEM shape: {dem_data.shape}, dtype: {dem_data.dtype}")

    # Load TPI
    if tpi_var in topo_ds:
        tpi_data = topo_ds[tpi_var]
        print(f"TPI ({tpi_var}) shape: {tpi_data.shape}, dtype: {tpi_data.dtype}")
    else:
        print(f"Warning: '{tpi_var}' variable not found. TPI will be None.")
        tpi_data = None

    return dem_data, tpi_data


def prepare_meteoswiss_targets(
    meteo_swiss_glob: str,
    normalization_stats: Era5Metadata,
    data_var: str = 'TmaxD',
    grid_elevation: xr.DataArray | None = None,
    hi_res_elevation: xr.DataArray | None = None,
    hi_res_tpi: xr.DataArray | None = None,
    convert_to_kelvin: bool = False,
    normalize_targets: bool = True,
    year_start: int | None = None,
    year_end: int | None = None,
    device: torch.device | None = None,
) -> Tuple[xr.DataArray, xr.DataArray, torch.Tensor]:
    """
    Transforms the MeteoSwiss Dataset into the Target Tensors (x, y, e).

    Args:
        meteo_swiss_glob: Glob pattern for MeteoSwiss NetCDF files
        normalization_stats: Stats from input for normalization
        data_var: Name of temperature variable (default 'TmaxD')
        grid_elevation: xarray.DataArray with ERA5 grid elevation (optional)
        hi_res_elevation: xarray.DataArray with high-res topography (optional)
        hi_res_tpi: xarray.DataArray with high-res TPI data (optional, e.g. TPI_2000M)
        convert_to_kelvin: Set True if MeteoSwiss is Celsius and input is Kelvin
        year_start: Optional first year to include (inclusive)
        year_end: Optional last year to include (inclusive)
        device: Torch device to load elevation tensor to (defaults to CPU)

    Returns:
        X (Locations): xr.DataArray (point, 2) -> [Lat_norm, Lon_norm]
        Y (Truth):     xr.DataArray (time, point) -> [Temp_norm]
        E (Topo):      torch.Tensor (point, 3) -> [True_Elev, Elev_Diff, mTPI]
    """
    ds = xr.open_mfdataset(str(meteo_swiss_glob), combine='by_coords', data_vars='all')
    t_start = f'{year_start}-01-01' if year_start is not None else None
    t_end = f'{year_end}-12-31' if year_end is not None else None
    if t_start is not None or t_end is not None:
        ds = ds.sel(time=slice(t_start, t_end))

    print(f"--- Preparing MeteoSwiss Targets (x, y, e) ---")

    # 1. Flatten the Grid
    # Your dataset has dims (time, N, E).
    # We stack N and E to create a list of points.
    print(f"Flattening ({ds.sizes['N']}, {ds.sizes['E']}) grid to (point)...")

    # This operation is lazy in xarray/dask.
    # It creates a MultiIndex, but doesn't move data yet.
    ds_flat = ds.stack(point=("N", "E"))

    # 2. Prepare Target Locations (Tensor x)
    print("Creating Target Coordinate Tensor (x)...")

    # Extract the 2D lat/lon (now flattened to 1D)
    # Note: dask arrays remain lazy.
    lat_flat = ds_flat['lat']
    lon_flat = ds_flat['lon']

    # open_mfdataset(..., data_vars='all') can broadcast the static lat/lon onto
    # the concat (time) axis when combining yearly files (seen with the OGD
    # RhiresD/TmaxD archives), which would leave tensor_x as (coord, time, point)
    # and break the transpose below. The geographic coords are time-invariant, so
    # collapse any stray time dim back to per-point. No-op when absent.
    if 'time' in lat_flat.dims:
        lat_flat = lat_flat.isel(time=0, drop=True)
    if 'time' in lon_flat.dims:
        lon_flat = lon_flat.isel(time=0, drop=True)

    # === COORDINATE VERIFICATION ===
    # Check that MeteoSwiss lat/lon are in degrees (WGS84), not LV95 meters
    lat_sample = lat_flat.values[:5] if hasattr(lat_flat.values, '__len__') else lat_flat.compute().values[:5]
    lon_sample = lon_flat.values[:5] if hasattr(lon_flat.values, '__len__') else lon_flat.compute().values[:5]

    print(f"\n=== COORDINATE VERIFICATION ===")
    print(f"MeteoSwiss lat (first 5 points): {lat_sample}")
    print(f"MeteoSwiss lon (first 5 points): {lon_sample}")
    print(f"MeteoSwiss lat range: [{float(lat_flat.min().compute()):.4f}, {float(lat_flat.max().compute()):.4f}]")
    print(f"MeteoSwiss lon range: [{float(lon_flat.min().compute()):.4f}, {float(lon_flat.max().compute()):.4f}]")
    print(f"\nERA5 normalization bounds (should overlap with MeteoSwiss if both in degrees):")
    print(f"ERA5 lat range: [{normalization_stats.lat_min:.4f}, {normalization_stats.lat_max:.4f}]")
    print(f"ERA5 lon range: [{normalization_stats.lon_min:.4f}, {normalization_stats.lon_max:.4f}]")

    # Sanity check: if MeteoSwiss coords are in LV95 meters, they'll be ~1e6, not ~45-48 degrees
    if lat_sample.mean() > 1000:
        print(f"\n*** WARNING: MeteoSwiss lat values ({lat_sample.mean():.0f}) look like LV95 meters, not WGS84 degrees! ***")
        print(f"*** Expected lat in range ~45-48 for Switzerland ***")
    if lon_sample.mean() > 1000:
        print(f"\n*** WARNING: MeteoSwiss lon values ({lon_sample.mean():.0f}) look like LV95 meters, not WGS84 degrees! ***")
        print(f"*** Expected lon in range ~5-11 for Switzerland ***")
    print(f"=== END COORDINATE VERIFICATION ===\n")

    # Normalize using input boundaries
    lat_range = normalization_stats.lat_max - normalization_stats.lat_min
    lon_range = normalization_stats.lon_max - normalization_stats.lon_min
    if lat_range == 0 or lon_range == 0:
        raise ValueError(f"Normalization coordinate range is zero (lat_range={lat_range}, lon_range={lon_range}); cannot normalize.")
    lat_norm = (lat_flat - normalization_stats.lat_min) / lat_range
    lon_norm = (lon_flat - normalization_stats.lon_min) / lon_range

    # Stack into (point, 2)
    tensor_x = xr.concat([lat_norm, lon_norm], dim="coord")
    tensor_x = tensor_x.transpose("point", "coord")
    tensor_x = tensor_x.assign_coords(coord=["lat", "lon"])

    # 3. Prepare Ground Truth Data (Tensor y)
    print(f"Creating Ground Truth Tensor (y) for {data_var}...")

    data_flat = ds_flat[data_var]

    # --- UNIT CONVERSION CHECK ---
    if convert_to_kelvin:
        print(f"  -> Converting Celsius to Kelvin (+{KELVIN_OFFSET}) before normalization")
        data_flat = data_flat + KELVIN_OFFSET

    # Normalize using stats (Z-Score)
    # Truth = (Temp - Input_Mean) / Input_Std
    # For strictly-non-negative targets with a Bernoulli-Gamma likelihood (e.g.
    # precipitation) z-scoring would push zeros/values negative and break the
    # Gamma log-prob, so callers can keep the raw amounts via normalize_targets=False.
    if normalize_targets:
        if normalization_stats.data_std == 0:
            raise ValueError("Normalization data_std is zero (constant field); cannot normalize targets.")
        y_norm = (data_flat - normalization_stats.data_mean) / normalization_stats.data_std
    else:
        y_norm = data_flat

    tensor_y = y_norm.transpose("time", "point")

    # 4. Prepare Topography (Tensor e)
    # The paper expects 3 features: [True Elevation, Elevation Diff, mTPI]
    print("Creating Elevation Tensor (e) [3 features]...")

    feature_names = ['true_elev', 'elev_diff', 'mTPI']

    if hi_res_elevation is not None and grid_elevation is not None:
        # Get target coordinates as numpy arrays
        lat_vals = lat_flat.values if hasattr(lat_flat, 'values') else lat_flat.compute().values
        lon_vals = lon_flat.values if hasattr(lon_flat, 'values') else lon_flat.compute().values

        # Convert WGS84 lat/lon to Swiss LV95 (x, y) for hi-res DEM lookup
        print("  -> Converting target coordinates to Swiss LV95 for hi-res DEM lookup...")
        target_x_lv95, target_y_lv95 = wgs84_to_lv95(lon_vals, lat_vals)

        # Interpolate hi-res DEM at target points using LV95 coordinates
        # The hi-res DEM has dimensions (y, x) with x, y as the index coordinates
        print("  -> Interpolating hi-res DEM at target points...")
        true_elev = hi_res_elevation.interp(
            x=xr.DataArray(target_x_lv95, dims='point'),
            y=xr.DataArray(target_y_lv95, dims='point'),
            method='linear'
        )

        # Interpolate ERA5 grid elevation at target points using lat/lon
        # ERA5 grid elevation has dimensions (latitude, longitude)
        print("  -> Interpolating ERA5 grid elevation at target points...")
        grid_elev = grid_elevation.interp(
            latitude=xr.DataArray(lat_vals, dims='point'),
            longitude=xr.DataArray(lon_vals, dims='point'),
            method='linear'
        )

        # Compute elevation difference
        elev_diff = true_elev - grid_elev

        print(f"  -> True elevation range: [{float(true_elev.min()):.1f}, {float(true_elev.max()):.1f}] m")
        print(f"  -> Grid elevation range: [{float(grid_elev.min()):.1f}, {float(grid_elev.max()):.1f}] m")
        print(f"  -> Elevation diff range: [{float(elev_diff.min()):.1f}, {float(elev_diff.max()):.1f}] m")

        # Interpolate TPI at target points if provided
        if hi_res_tpi is not None:
            print("  -> Interpolating TPI at target points...")
            tpi = hi_res_tpi.interp(
                x=xr.DataArray(target_x_lv95, dims='point'),
                y=xr.DataArray(target_y_lv95, dims='point'),
                method='linear'
            )
            print(f"  -> TPI range: [{float(tpi.min()):.1f}, {float(tpi.max()):.1f}] m")
        else:
            print("  -> TPI data not provided. Filling with zeros.")
            tpi = xr.zeros_like(true_elev)

        # Stack: [True Elev, Elev Diff, mTPI]
        tensor_e = xr.concat([true_elev, elev_diff, tpi], dim="feature", coords='minimal')
    else:
        print("  -> Warning: Elevation data not provided. Filling ALL 3 features with zeros.")
        zeros = xr.zeros_like(lat_flat)
        tensor_e = xr.concat([zeros, zeros, zeros], dim="feature")

    # Assign coordinates and transpose to (point, feature)
    tensor_e = tensor_e.assign_coords(feature=feature_names)
    tensor_e = tensor_e.transpose("point", "feature")

    # Convert elevation to torch tensor
    if device is None:
        device = torch.device('cpu')
    tensor_e_torch = torch.from_numpy(tensor_e.values.astype(np.float32)).to(device)

    print(f"Final X (Locations): {tensor_x.sizes}")
    print(f"Final Y (Targets):   {tensor_y.sizes}")
    print(f"Final E (Topo):      {tensor_e_torch.shape}, dtype: {tensor_e_torch.dtype}")

    return tensor_x, tensor_y, tensor_e_torch


def calculate_dists_meteoswiss(
    normalization_stats: Era5Metadata,
    meteoswiss_targets: xr.DataArray,
    device: torch.device | None = None
) -> torch.Tensor:
    """
    Get the distances between the ERA5 grid points and MeteoSwiss target points.

    Args:
        normalization_stats: Stats containing ERA5 grid coordinates and normalization bounds
        meteoswiss_targets: xarray.DataArray with normalized target coordinates (target_x)
        device: Torch device to load tensor to (defaults to CPU)

    Returns:
        Tensor of shape (n_target_points, n_lat, n_lon) with squared distances
    """
    if device is None:
        device = torch.device('cpu')

    # Build ERA5 grid from stored coordinates
    era5_lon_grid, era5_lat_grid = np.meshgrid(
        normalization_stats.lon_coords,
        normalization_stats.lat_coords
    )
    era5_lat_grid = torch.from_numpy(era5_lat_grid).float().to(device)
    era5_lon_grid = torch.from_numpy(era5_lon_grid).float().to(device)

    # De-normalize MeteoSwiss coordinates back to degrees
    lat_norm = meteoswiss_targets.sel(coord='lat').values
    lon_norm = meteoswiss_targets.sel(coord='lon').values

    lat = lat_norm * (normalization_stats.lat_max - normalization_stats.lat_min) + normalization_stats.lat_min
    lon = lon_norm * (normalization_stats.lon_max - normalization_stats.lon_min) + normalization_stats.lon_min

    # Combine into coordinate array for get_dists
    meteoswiss_coords = np.stack([lat, lon], axis=1)

    # get_dists returns a CPU tensor, move to requested device
    dists = get_dists(meteoswiss_coords, era5_lat_grid, era5_lon_grid)
    return dists.to(device)


def get_meteoswiss_grid_shape(meteo_swiss_glob: str) -> Tuple[int, int]:
    """
    Get the original grid shape (N, E) from MeteoSwiss data.

    Args:
        meteo_swiss_glob: Glob pattern for MeteoSwiss NetCDF files

    Returns:
        Tuple of (N, E) grid dimensions
    """
    ds = xr.open_mfdataset(meteo_swiss_glob, combine='by_coords', data_vars='all')
    return ds.sizes['N'], ds.sizes['E']


def compute_seasonal_features(
    time_coords: np.ndarray,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute seasonal features (cos/sin of day-of-year) from time coordinates.

    The seasonal encoding allows the elevation MLP to learn season-dependent
    corrections, e.g., different lapse rates in winter (inversions) vs summer.

    Args:
        time_coords: Array of datetime64 values (from load_era5_data)
        device: Torch device to load tensor to (defaults to CPU)

    Returns:
        Tensor of shape (n_times, 2) with [cos(day_of_year), sin(day_of_year)]
    """
    if device is None:
        device = torch.device('cpu')

    # Convert datetime64 to day of year
    times_pd = pd.to_datetime(time_coords)
    day_of_year = times_pd.dayofyear.values  # 1-366

    # Convert to radians (0 to 2*pi over the year)
    time_rads = (day_of_year - 1) / 365.0 * 2 * np.pi

    # Compute cos and sin encoding
    cos_doy = np.cos(time_rads)
    sin_doy = np.sin(time_rads)

    # Stack into (n_times, 2) tensor
    seasonal = np.stack([cos_doy, sin_doy], axis=1)
    seasonal_tensor = torch.from_numpy(seasonal.astype(np.float32)).to(device)

    print(f"Computed seasonal features: {seasonal_tensor.shape}, dtype: {seasonal_tensor.dtype}")

    return seasonal_tensor


# Scaffold channels are the non-physical context layers (grid geometry, season,
# topography, and the optional surface variable itself); they form their own
# group for the targeted encoder.
SCAFFOLD_CHANNEL_NAMES = {'data', 'lat', 'lon', 'cos_time', 'sin_time', 'elevation'}


def channel_variable_group(name: str) -> str:
    """Map a context channel name to its physical-variable group label.

    Scaffold channels (lat/lon/season/elevation/data) -> 'scaffold'. Atmospheric
    channels named '<var><level>_<hour>' or '<var>_<hour>' (e.g. 't850_12',
    't2m_06', 'tp_12') -> their leading-letter variable ('t', 'z', 'q', 'tp').
    """
    if name in SCAFFOLD_CHANNEL_NAMES:
        return 'scaffold'
    m = re.match(r'^([A-Za-z]+)', name)
    return m.group(1) if m else 'other'


def channel_groups_by_variable(channel_names: list[str]) -> "OrderedDict[str, list[int]]":
    """Group channel indices by physical variable.

    Returns an ordered mapping {group_label: [channel indices]} preserving first
    appearance order, used by the targeted (grouped) encoders.
    """
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, name in enumerate(channel_names):
        groups.setdefault(channel_variable_group(name), []).append(i)
    return groups


def build_atmospheric_native_context(
    pressure_level_dir: Path | str,
    time_coords: np.ndarray,
    era5_metadata: Era5Metadata,
    grid_elevation: xr.DataArray | None = None,
    atmos_variables: list[str] = ('z', 't', 'q'),
    atmos_levels: list[int] = (1000, 925, 850, 700, 500, 300),
    atmos_hours: list[str] = ('00', '06', '12', '15', '18'),
    include_seasonal: bool = True,
    include_elevation: bool = True,
    surface_dir: Path | str | None = None,
    atmos_sfc_variables: list[str] = ('t2m', 'tp'),
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, list[str], Era5Metadata, dict]:
    """Assemble the atmospheric context on its native coarse (0.25 deg) grid.

    The atmospheric pressure-level channels are kept at their native resolution
    (no upsampling to the fine surface grid) and a matching coarse
    lat/lon/seasonal/elevation scaffold is built on the same grid. Atmospheric
    channels come first so channel 0 is a real atmospheric field (not the lat
    scaffold) -- fixing the historical channel-order ambiguity.

    Args:
        pressure_level_dir: Base dir for ERA5 pressure-level files.
        time_coords: Daily datetime64 axis (from load_era5_data) to align to.
        era5_metadata: Metadata from the fine ERA5 load; supplies the target
            normalization bounds (lat/lon min/max, data mean/std). Reused so the
            MeteoSwiss targets and the returned coarse distances stay consistent.
        grid_elevation: ERA5 altitude DataArray (latitude, longitude) to sample
            onto the coarse grid for the elevation scaffold channel.
        include_seasonal/include_elevation: scaffold toggles.
        surface_dir: If given (USE_SFC_ATMOS), append ERA5 single-level surface
            anchor channels on the same coarse grid.
        device: Torch device for the returned tensor.

    Returns:
        context: torch.Tensor (time, channel, lat_native, lon_native).
        channel_names: list[str] in tensor channel order (atmospheric first,
            then scaffold, then optional surface anchor).
        coarse_metadata: Era5Metadata copy whose lat_coords/lon_coords are the
            native coarse grid (bounds inherited from era5_metadata), for
            calculate_dists_meteoswiss.
        normalization: {channel_name: {'mean', 'std'}} for every atmospheric /
            surface-anchor / coarse-elevation channel, so the same per-channel
            transform can be reapplied to unseen data.
    """
    if device is None:
        device = torch.device('cpu')

    atmos_stats: list = []
    atmos, atmos_names, coarse_lat, coarse_lon = load_era5_pressure_levels(
        pressure_level_dir,
        lat_coords=era5_metadata.lat_coords,
        lon_coords=era5_metadata.lon_coords,
        time_coords=time_coords,
        variables=list(atmos_variables),
        levels=list(atmos_levels),
        hours=list(atmos_hours),
        native_grid=True,
        stats_out=atmos_stats,
        device=device,
    )
    # Collect the normalization actually applied, so the same transform can be
    # reapplied to unseen data (see Trainer's manifest.json).
    normalization = { s['channel']: {'mean': s['mean'], 'std': s['std']} for s in atmos_stats}

    n_time = atmos.shape[0]
    n_lat = len(coarse_lat)
    n_lon = len(coarse_lon)

    lat_range = era5_metadata.lat_max - era5_metadata.lat_min
    lon_range = era5_metadata.lon_max - era5_metadata.lon_min
    if lat_range == 0 or lon_range == 0:
        raise ValueError("Era5Metadata coordinate range is zero; cannot build coarse scaffold.")

    lat_norm = (np.asarray(coarse_lat, dtype=np.float32) - era5_metadata.lat_min) / lat_range
    lon_norm = (np.asarray(coarse_lon, dtype=np.float32) - era5_metadata.lon_min) / lon_range
    lat_grid = np.broadcast_to(lat_norm[:, None], (n_lat, n_lon))
    lon_grid = np.broadcast_to(lon_norm[None, :], (n_lat, n_lon))
    lat_ch = torch.from_numpy(np.broadcast_to(lat_grid, (n_time, n_lat, n_lon)).copy()).to(device)
    lon_ch = torch.from_numpy(np.broadcast_to(lon_grid, (n_time, n_lat, n_lon)).copy()).to(device)

    scaffold = [lat_ch, lon_ch]
    scaffold_names = ['lat', 'lon']

    if include_seasonal:
        seasonal = compute_seasonal_features(time_coords, device=device)  # (time, 2)
        cos_ch = seasonal[:, 0].view(n_time, 1, 1).expand(n_time, n_lat, n_lon)
        sin_ch = seasonal[:, 1].view(n_time, 1, 1).expand(n_time, n_lat, n_lon)
        scaffold.extend([cos_ch, sin_ch])
        scaffold_names.extend(['cos_time', 'sin_time'])

    if include_elevation and grid_elevation is not None:
        lat_clamped = np.clip(coarse_lat, float(grid_elevation.latitude.min()), float(grid_elevation.latitude.max()))
        lon_clamped = np.clip(coarse_lon, float(grid_elevation.longitude.min()), float(grid_elevation.longitude.max()))
        elev = grid_elevation.interp(latitude=lat_clamped, longitude=lon_clamped, method='linear')
        elev = elev.assign_coords(latitude=coarse_lat, longitude=coarse_lon)
        elev_vals = elev.transpose('latitude', 'longitude').values.astype(np.float32)
        emean = float(np.nanmean(elev_vals))
        estd = float(np.nanstd(elev_vals))
        if estd == 0:
            raise ValueError("Coarse-grid elevation has zero standard deviation; cannot normalize.")
        elev_norm = (elev_vals - emean) / estd
        elev_ch = torch.from_numpy(elev_norm).to(device).unsqueeze(0).expand(n_time, n_lat, n_lon)
        scaffold.append(elev_ch)
        scaffold_names.append('elevation')
        normalization['elevation'] = {'mean': emean, 'std': estd}

    scaffold_tensor = torch.stack(scaffold, dim=1)  # (time, n_scaffold, lat, lon)

    parts = [atmos, scaffold_tensor]
    channel_names = list(atmos_names) + scaffold_names

    if surface_dir is not None:
        sfc_stats: list = []
        sfc, sfc_names = load_era5_surface_levels(
            surface_dir,
            lat_coords=np.asarray(coarse_lat),
            lon_coords=np.asarray(coarse_lon),
            time_coords=time_coords,
            variables=list(atmos_sfc_variables),
            hours=list(atmos_hours),
            stats_out=sfc_stats,
            device=device,
        )
        parts.append(sfc)
        channel_names.extend(sfc_names)
        for s in sfc_stats:
            normalization[s['channel']] = {'mean': s['mean'], 'std': s['std']}

    context = torch.cat(parts, dim=1)
    coarse_metadata = dataclasses.replace(
        era5_metadata,
        lat_coords=np.asarray(coarse_lat),
        lon_coords=np.asarray(coarse_lon),
    )

    print(f"Native-grid atmospheric context (time, channel, lat, lon): {tuple(context.shape)}; "
          f"channels: {len(channel_names)} (atmospheric {len(atmos_names)} + "
          f"scaffold {len(scaffold_names)}); channel 0 = '{channel_names[0]}'")

    return context, channel_names, coarse_metadata, normalization


def sample_era5_grid_cells(
    era5_shape: Tuple[int, int],
    n_points: int | None = None,
    frac: float | None = None,
    seed: int | None = None,
) -> np.ndarray | None:
    """
    Sample random ERA5 grid cell indices for sparse masking.

    Args:
        era5_shape: (n_lat, n_lon) shape of the ERA5 grid.
        n_points: Absolute number of cells to keep (mutually exclusive with frac).
        frac: Fraction of cells to keep (mutually exclusive with n_points).
        seed: If provided, calls set_seed for reproducibility.

    Returns:
        Sorted flat indices of kept cells, or None if all cells are kept.
    """
    assert (n_points is None) != (frac is None), "Specify exactly one of n_points or frac"

    n_lat, n_lon = era5_shape
    total = n_lat * n_lon

    if frac is not None:
        n_keep = max(1, int(total * frac))
    else:
        n_keep = n_points

    if n_keep >= total:
        print(f"Keeping all {total} ERA5 grid cells (no sparsity)")
        return None

    if seed is not None:
        set_seed(seed)
    indices = np.random.choice(total, size=n_keep, replace=False)
    print(f"Sampled {n_keep} of {total} ERA5 grid cells")
    return np.sort(indices)


def build_era5_sparse_mask(
    era5_shape: Tuple[int, int],
    n_channels: int,
    cell_indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a sparse mask for ERA5 grid cells.

    Args:
        era5_shape: (n_lat, n_lon) shape of the ERA5 grid.
        n_channels: Number of channels in the context tensor.
        cell_indices: Flat indices of cells to keep (from sample_era5_grid_cells).
        device: Torch device.

    Returns:
        Mask tensor of shape (1, channels, lat, lon) with 1s at kept cells.
    """
    n_lat, n_lon = era5_shape
    mask_2d = torch.zeros(n_lat, n_lon, device=device)
    lat_idx = torch.from_numpy(cell_indices // n_lon).long()
    lon_idx = torch.from_numpy(cell_indices % n_lon).long()
    mask_2d[lat_idx, lon_idx] = 1.0
    return mask_2d.unsqueeze(0).unsqueeze(0).expand(1, n_channels, -1, -1).contiguous()


def sample_meteoswiss_points(
    target_y_tensor: torch.Tensor,
    n_points: int | None = None,
    frac: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample random MeteoSwiss point indices that have valid (non-NaN) data.

    Uses the first time step to identify valid grid cells (inside Switzerland).

    Args:
        target_y_tensor: (n_times, n_points) MeteoSwiss target values (normalized).
        n_points: Absolute number of points to sample (mutually exclusive with frac).
        frac: Fraction of valid points to sample (mutually exclusive with n_points).
        seed: If provided, calls set_seed for reproducibility.

    Returns:
        Sorted array of sampled point indices.
    """
    assert (n_points is None) != (frac is None), "Specify exactly one of n_points or frac"

    valid_mask = ~torch.isnan(target_y_tensor[0])
    valid_indices = torch.where(valid_mask)[0].cpu().numpy()

    if frac is not None:
        n_points = max(1, int(len(valid_indices) * frac))

    if seed is not None:
        set_seed(seed)
    sampled = np.random.choice(valid_indices, size=min(n_points, len(valid_indices)), replace=False)
    sampled = np.sort(sampled)

    print(f"Sampled {len(sampled)} MeteoSwiss points from {len(valid_indices)} valid points")
    return sampled


def find_nearest_era5_indices(
    target_x: xr.DataArray,
    metadata: Era5Metadata,
    point_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map sampled MeteoSwiss point indices to their nearest ERA5 grid cell indices.

    Args:
        target_x: (n_points, 2) normalized [lat, lon] coordinates of all MeteoSwiss points.
        metadata: Era5Metadata with lat_coords, lon_coords, and normalization bounds.
        point_indices: Indices of sampled MeteoSwiss points.

    Returns:
        Tuple of (lat_indices, lon_indices), each (n_sampled,) arrays of ERA5 grid indices.
    """
    lat_norm = target_x.sel(coord='lat').values[point_indices]
    lon_norm = target_x.sel(coord='lon').values[point_indices]
    lat_deg = lat_norm * (metadata.lat_max - metadata.lat_min) + metadata.lat_min
    lon_deg = lon_norm * (metadata.lon_max - metadata.lon_min) + metadata.lon_min

    lat_indices = np.argmin(
        np.abs(metadata.lat_coords[np.newaxis, :] - lat_deg[:, np.newaxis]), axis=1
    )
    lon_indices = np.argmin(
        np.abs(metadata.lon_coords[np.newaxis, :] - lon_deg[:, np.newaxis]), axis=1
    )

    n_unique = len(set(zip(lat_indices, lon_indices)))
    print(f"Mapped {len(point_indices)} MeteoSwiss points to {n_unique} unique ERA5 grid cells")
    return lat_indices, lon_indices


def build_sparse_context(
    target_y_tensor: torch.Tensor,
    era5_template: torch.Tensor,
    point_indices: np.ndarray,
    era5_lat_idx: np.ndarray,
    era5_lon_idx: np.ndarray,
    day_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a sparse context grid from MeteoSwiss observations for one day.

    Places sampled MeteoSwiss temperature observations at their nearest ERA5 grid cells.
    Coordinate and seasonal channels come from the ERA5 template. Unobserved cells have
    NaN in channel 0 — the model derives its mask from this NaN pattern. When multiple
    points map to the same cell, their values are averaged.

    Args:
        target_y_tensor: (n_times, n_points) normalized MeteoSwiss temperatures.
        era5_template: (n_times, channels, lat, lon) ERA5 data (for coordinate channels).
        point_indices: Sampled MeteoSwiss point indices.
        era5_lat_idx: ERA5 latitude indices for each sampled point.
        era5_lon_idx: ERA5 longitude indices for each sampled point.
        day_idx: Time index for this day.
        device: Torch device.

    Returns:
        Context tensor (1, channels, lat, lon) with sparse MeteoSwiss temps in channel 0
        and NaN at unobserved cells.
    """
    _, channels, n_lat, n_lon = era5_template.shape

    # Clone ERA5 template for coordinate/seasonal channels
    context = era5_template[day_idx:day_idx + 1].clone()
    context[0, 0, :, :] = float('nan')  # NaN everywhere in temp channel

    day_values = target_y_tensor[day_idx, point_indices]

    # Accumulate on grid (averaging when multiple points hit the same cell)
    counts = torch.zeros(n_lat, n_lon, device=device)
    temp_sum = torch.zeros(n_lat, n_lon, device=device)

    for i, (lat_i, lon_i) in enumerate(zip(era5_lat_idx, era5_lon_idx)):
        val = day_values[i]
        if not torch.isnan(val):
            temp_sum[lat_i, lon_i] += val.item()
            counts[lat_i, lon_i] += 1

    observed = counts > 0
    context[0, 0][observed] = temp_sum[observed] / counts[observed]

    return context


def interpolate_era5_to_targets(
    y_context: torch.Tensor,
    target_coords: xr.DataArray,
    metadata: Era5Metadata,
) -> torch.Tensor:
    """
    Bilinearly interpolate ERA5 temperature (channel 0) to target point locations.

    This provides the reference (baseline) predictions for skill score calculation:
    the best you could do by simply reading off the coarse ERA5 grid at each station.

    Args:
        y_context: ERA5 input tensor of shape (time, channels, lat, lon).
                   Channel 0 is the normalized temperature field.
        target_coords: xarray.DataArray of shape (point, coord) with normalized
                       [lat, lon] coordinates in [0, 1] (from prepare_meteoswiss_targets).
        metadata: Era5Metadata with lat_coords to determine grid orientation.

    Returns:
        Tensor of shape (time, n_points) with interpolated normalized temperatures.
    """
    n_time = y_context.shape[0]
    temp_field = y_context[:, 0:1, :, :]  # (time, 1, lat, lon)

    # Extract normalized target coordinates
    lat_norm = torch.from_numpy(
        target_coords.sel(coord='lat').values.astype(np.float32)
    )
    lon_norm = torch.from_numpy(
        target_coords.sel(coord='lon').values.astype(np.float32)
    )

    # Convert [0, 1] normalized coords to grid_sample's [-1, 1] range.
    # grid_sample convention: -1 maps to index 0, +1 maps to last index.
    # Longitude (W dimension): lon increases with index, so grid_x = 2*lon_norm - 1
    grid_x = 2.0 * lon_norm - 1.0

    # Latitude (H dimension): check ordering of the ERA5 grid.
    # If lat_coords is descending (north-to-south), index 0 = lat_max (norm=1),
    # so grid_y = -(2*lat_norm - 1) = 1 - 2*lat_norm.
    # If ascending (south-to-north), index 0 = lat_min (norm=0),
    # so grid_y = 2*lat_norm - 1.
    lat_descending = metadata.lat_coords[0] > metadata.lat_coords[-1]
    if lat_descending:
        grid_y = 1.0 - 2.0 * lat_norm
    else:
        grid_y = 2.0 * lat_norm - 1.0

    # grid_sample expects grid of shape (N, H_out, W_out, 2) with (x, y) ordering.
    # For point sampling: H_out=1, W_out=n_points
    n_points = len(lat_norm)
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (n_points, 2)
    grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, n_points, 2)
    grid = grid.expand(n_time, -1, -1, -1)  # (time, 1, n_points, 2)

    # Move to same device as input
    grid = grid.to(temp_field.device)

    # Bilinear interpolation
    interpolated = F.grid_sample(
        temp_field, grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    # interpolated shape: (time, 1, 1, n_points) -> squeeze to (time, n_points)
    return interpolated.squeeze(1).squeeze(1)
