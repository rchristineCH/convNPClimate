#!/usr/bin/env python3
"""
Headless, class-based training framework for convNPClimate.

This is a script-based reimplementation of ``training_notebook.ipynb``. It loads
the data once, then trains one independent model per requested variable
(currently ``tmax``; the design supports adding others -- see VARIABLE_SPECS and
model_factory.LIKELIHOODS) with K-fold cross-validation, evaluating each epoch on
the fine-resolution MeteoSwiss reference. Instead of relying on notebook cell
outputs it emits a single comprehensive ``run_summary.md`` (plus the existing
per-epoch CSVs and PNG training curves) under
``trained_models/<TRIAL_NAME>/<variable>/``.

It reuses the existing backend wherever possible (``params``, ``datasets``,
``model_factory``, ``convCNP.training.training_elev.train_elev``,
``visualization``) and adds:

  * native coarse-grid atmospheric encoding (``--atmos-native-grid``)
  * targeted grouped encoders (``--encoder grouped_setconv|channel_attention``)
  * GPU + multicore use (auto device selection, thread config, optional
    multi-GPU ``DataParallel``)

Example
-------
    python train.py --variables tmax --use-surface \
        --data-year-start 2023 --n-epochs 2 --n-folds 2 --trial-name smoke_surface
"""

import argparse
import atexit
import dataclasses
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Configure a non-interactive matplotlib backend before importing visualization
# (which imports matplotlib at module load) so plots render headless.
import matplotlib
matplotlib.use("Agg")

import params
import datasets as ds
import model_factory
import visualization as vis
import cloud_sync
from convCNP.training.training_elev import train_elev


logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Per-variable configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class VariableSpec:
    """How to load and normalize the MeteoSwiss target for one variable."""
    name: str
    meteoswiss_glob_attr: str  # attribute name on DataPaths
    data_var: str
    convert_to_kelvin: bool
    normalize_targets: bool
    grad_clip: float | None  # default gradient-norm clip (stabilizes Gamma NLL)


VARIABLE_SPECS: dict[str, VariableSpec] = {
    "tmax": VariableSpec("tmax", "METEO_SWISS_MAX_TEMP_GLOB", "TmaxD", True, True, None),
    # Precipitation: non-negative raw mm targets (no Kelvin conversion, no
    # z-scoring -- z-scoring would push zeros negative and break the Gamma
    # log-prob). The Gamma NLL is stabilised by a default gradient-norm clip
    # of 1.0. The output distribution is bernoulli_gamma (see model_factory).
    "precip": VariableSpec("precip", "METEO_SWISS_PRECIP_GLOB", "RhiresD", False, False, 1.0),
}


# ---------------------------------------------------------------------------
# Multi-GPU wrapper
# ---------------------------------------------------------------------------
class ConvCNPDataParallel(torch.nn.DataParallel):
    """DataParallel that scatters only the batch-aligned inputs.

    The model's forward is ``model(x, mask, dists, elev, seasonal=...)``. Of
    these, ``x``/``mask``/``seasonal`` are batched along dim 0 and must be
    scattered across GPUs, but ``dists`` (n_points, lat, lon) and ``elev``
    (n_points, 3) are indexed by target point, not batch, and must be replicated
    to every replica unchanged. Vanilla ``nn.DataParallel`` would wrongly split
    them along their first dim.
    """

    def scatter(self, inputs, kwargs, device_ids):
        x, mask, dists, elev = inputs
        x_s = torch.nn.parallel.scatter(x, device_ids, dim=0)
        mask_s = torch.nn.parallel.scatter(mask, device_ids, dim=0)
        used = device_ids[: len(x_s)]

        seasonal = kwargs.get("seasonal", None)
        seasonal_s = (
            torch.nn.parallel.scatter(seasonal, device_ids, dim=0)
            if seasonal is not None
            else None
        )

        new_inputs, new_kwargs = [], []
        for i, dev in enumerate(used):
            d_dev = f"cuda:{dev}"
            new_inputs.append((x_s[i], mask_s[i], dists.to(d_dev), elev.to(d_dev)))
            new_kwargs.append(
                {"seasonal": seasonal_s[i] if seasonal_s is not None else None}
            )
        return new_inputs, tuple(new_kwargs)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
class TrainingConfig:
    """Resolves CLI arguments into a base ``Params`` plus run-level settings."""

    def __init__(self, args: argparse.Namespace):
        params.configure_renku_cuda()
        self.device = (
            torch.device(args.device) if args.device else params.select_device()
        )

        self.variables = args.variables
        for v in self.variables:
            if v not in VARIABLE_SPECS:
                raise ValueError(f"Unknown variable {v!r}; known: {list(VARIABLE_SPECS)}")

        self.data_parallel = args.data_parallel
        self.grad_clip_override = args.grad_clip
        self.base_dir = Path(args.base_dir)
        self.data_dir = Path(args.data_dir)

        # The configuration is always encoded into the trial name so runs are
        # self-describing and don't overwrite each other. If --trial-name is
        # given it is used as a prefix; otherwise the slug alone is the name.
        slug = self._config_slug(args)
        self.trial_name = f"{args.trial_name}__{slug}" if args.trial_name else slug

        self.output_root = self.base_dir / args.models_subdir / self.trial_name
        self.data_paths = ds.build_data_paths(self.data_dir)

        year_start = None if args.all_years else args.data_year_start
        year_end = None if args.all_years else (args.data_year_end or args.data_year_start)

        # The base Params; VARIABLE is overridden per training run. USE_SURFACE /
        # USE_ATMOSPHERIC are mutually exclusive (enforced in Params).
        use_atmospheric = args.use_atmospheric
        use_surface = not use_atmospheric

        self.distribution = args.distribution

        # Only override the atmospheric variable set when --atmos-variables is
        # given; otherwise fall back to the Params default (['z','t','q']) so
        # existing atmospheric runs are unchanged.
        atmos_kw = (
            {"ATMOS_VARIABLES": list(args.atmos_variables)}
            if getattr(args, "atmos_variables", None) else {}
        )

        base = params.Params(
            VARIABLE=self.variables[0],
            DISTRIBUTION=args.distribution,
            **atmos_kw,
            DATA_YEAR_START=year_start,
            DATA_YEAR_END=year_end,
            N_EPOCHS=args.n_epochs,
            N_FOLDS=args.n_folds,
            BATCH_SIZE=args.batch_size,
            LR=args.lr,
            LENGTH_SCALE=args.length_scale,
            PATIENCE=args.patience,
            SEED=args.seed,
            TRIAL_NAME=self.trial_name,
            RUN_TYPE="cloud" if params.is_renku() else "local",
            DEVICE=str(self.device),
            SEASONAL_FEATURES=not args.no_seasonal,
            SEASONAL_FEATURES_IN_MLP=not args.no_seasonal,
            USE_SURFACE=use_surface,
            USE_ATMOSPHERIC=use_atmospheric,
            USE_SFC_ATMOS=args.use_sfc_atmos,
            USE_SURFACE_PRECIP=args.use_surface_precip,
            ATMOS_NATIVE_GRID=args.atmos_native_grid,
            USE_ELEVATION=not args.no_elevation,
            USE_MTPI=not args.no_elevation,
            USE_ELEVATION_CHANNEL=not args.no_geopotential,
            ENCODER=args.encoder,
        )
        self.base_params = base.with_data_paths(self.data_paths)

    @staticmethod
    def _config_slug(args: argparse.Namespace) -> str:
        """Build a compact, filesystem-safe slug encoding the run configuration."""
        enc_abbr = {"flat": "flat", "grouped_setconv": "gsc", "channel_attention": "catt"}
        dist_abbr = {
            "gaussian": "gauss",
            "bernoulli_gamma": "bg",
            "bernoulli_gamma_crps": "bgcrps",
        }
        parts = [
            "-".join(args.variables),
        ]
        # Encode the distribution only when explicitly overridden, so existing
        # default runs (tmax->gaussian, precip->bernoulli_gamma) keep their slug
        # unchanged while the two precip distributions get distinct dirs.
        if args.distribution is not None:
            parts.append(dist_abbr.get(args.distribution, args.distribution))
        parts.append("atm" if args.use_atmospheric else "sfc")
        if args.use_atmospheric:
            parts.append("natg" if args.atmos_native_grid else "regrid")
            if args.use_sfc_atmos:
                parts.append("sfcanc")
            # Mark a non-default atmospheric variable set (e.g. wind added) so
            # runs stay uniquely named and don't collide with the z/t/q baseline.
            av = getattr(args, "atmos_variables", None)
            if av and list(av) != list(params.Params().ATMOS_VARIABLES):
                parts.append("av-" + "".join(av))
        elif getattr(args, "use_surface_precip", False):
            # Mark the extra ERA5-Land precip channel so a surface precip run
            # cannot silently land in the same directory as a plain surface run.
            parts.append("tp")
        parts.append(enc_abbr[args.encoder])
        if args.all_years:
            parts.append("yall")
        elif args.data_year_end is not None:
            parts.append(f"y{args.data_year_start}-{args.data_year_end}")
        else:
            parts.append(f"y{args.data_year_start}")
        parts.append(f"e{args.n_epochs}f{args.n_folds}")
        parts.append(f"b{args.batch_size}")
        if args.no_seasonal:
            parts.append("noseas")
        if args.no_elevation:
            parts.append("noelev")
        if getattr(args, "no_geopotential", False):
            parts.append("nogeo")
        return "_".join(parts)

    def params_for(self, variable: str, in_channels: int) -> params.Params:
        """Per-variable Params with VARIABLE and IN_CHANNELS set."""
        return dataclasses.replace(
            self.base_params, VARIABLE=variable, IN_CHANNELS=in_channels
        )


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------
class DataBundle:
    """Loads the shared input context, scaffold, topography and channel groups.

    The predictor context (surface or atmospheric) is shared across variables
    (independent models, shared predictors); only the MeteoSwiss target and the
    target-point distances differ per variable and are prepared in ``Trainer``.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        p = config.base_params
        device = config.device

        # 1. Fine ERA5 surface load -> metadata (normalization bounds + stats),
        #    grid elevation, and the daily time axis. The data channel itself is
        #    gated by USE_SURFACE; the lat/lon/season/elevation scaffold is always
        #    built.
        fine_context, self.era5_metadata, self.grid_elevation, self.time_coords = (
            ds.load_era5_data(
                p.ERA5_MAX_TEMP_GLOB,
                var_name="t2m_max",
                year_start=p.DATA_YEAR_START,
                year_end=p.DATA_YEAR_END,
                geopotential_glob=p.ERA5_GEOPOTENTIAL_GLOB,
                geopotential_var_name="z",
                include_seasonal_embeddings=p.SEASONAL_FEATURES,
                include_data_channel=p.USE_SURFACE,
                include_elevation_channel=p.USE_ELEVATION_CHANNEL,
                device=device,
            )
        )

        # 2. Assemble the predictor context + channel names + the metadata to use
        #    for target-point distances (must match the context's grid).
        self.channel_normalization: dict = {}
        if p.USE_ATMOSPHERIC and p.ATMOS_NATIVE_GRID:
            self.context, self.channel_names, self.dists_metadata, self.channel_normalization = (
                ds.build_atmospheric_native_context(
                    p.ERA5_PRESSURE_LEVEL_DIR,
                    time_coords=self.time_coords,
                    era5_metadata=self.era5_metadata,
                    grid_elevation=self.grid_elevation if p.USE_ELEVATION else None,
                    atmos_variables=p.ATMOS_VARIABLES,
                    atmos_levels=p.ATMOS_LEVELS,
                    atmos_hours=p.ATMOS_HOURS,
                    include_seasonal=p.SEASONAL_FEATURES,
                    include_elevation=p.USE_ELEVATION,
                    surface_dir=p.ERA5_SURFACE_DIR if p.USE_SFC_ATMOS else None,
                    atmos_sfc_variables=p.ATMOS_SFC_VARIABLES,
                    device=device,
                )
            )
            self.grid_mode = "native coarse atmospheric"
        else:
            scaffold_names = self._surface_scaffold_names(p, fine_context.shape[1])
            context = fine_context
            channel_names = list(scaffold_names)
            if p.USE_ATMOSPHERIC:
                atmos_stats: list = []
                atmos, atmos_names = ds.load_era5_pressure_levels(
                    p.ERA5_PRESSURE_LEVEL_DIR,
                    lat_coords=self.era5_metadata.lat_coords,
                    lon_coords=self.era5_metadata.lon_coords,
                    time_coords=self.time_coords,
                    variables=p.ATMOS_VARIABLES,
                    levels=p.ATMOS_LEVELS,
                    hours=p.ATMOS_HOURS,
                    native_grid=False,
                    stats_out=atmos_stats,
                    device=device,
                )
                context = torch.cat([context, atmos], dim=1)
                channel_names += atmos_names
                if p.USE_SFC_ATMOS:
                    sfc, sfc_names = ds.load_era5_surface_levels(
                        p.ERA5_SURFACE_DIR,
                        lat_coords=self.era5_metadata.lat_coords,
                        lon_coords=self.era5_metadata.lon_coords,
                        time_coords=self.time_coords,
                        variables=p.ATMOS_SFC_VARIABLES,
                        hours=p.ATMOS_HOURS,
                        stats_out=atmos_stats,
                        device=device,
                    )
                    context = torch.cat([context, sfc], dim=1)
                    channel_names += sfc_names
                self.channel_normalization = {
                    s['channel']: {'mean': s['mean'], 'std': s['std']} for s in atmos_stats
                }
                self.grid_mode = "atmospheric regridded to fine surface"
            elif p.USE_SURFACE_PRECIP:
                # Append the coarse ERA5-Land precipitation field. Its series can
                # stop short of the requested year range (the files are labelled by
                # accumulation window, so the 2020-2023 set ends 2023-12-30), so the
                # loader returns the contiguous covered span and every other source
                # is trimmed to match -- the model trains on real fields only, with
                # nothing filled or persisted.
                tp_stats: list = []
                tp_ctx, tp_names, span = ds.load_era5_precip_channel(
                    p.ERA5_PRECIP_GLOB,
                    lat_coords=self.era5_metadata.lat_coords,
                    lon_coords=self.era5_metadata.lon_coords,
                    time_coords=self.time_coords,
                    stats_out=tp_stats,
                    device=device,
                )
                if span.start != 0 or span.stop != len(self.time_coords):
                    context = context[span]
                    self.time_coords = self.time_coords[span]
                context = torch.cat([context, tp_ctx], dim=1)
                channel_names += tp_names
                self.channel_normalization.update(
                    {s['channel']: {'mean': s['mean'], 'std': s['std']} for s in tp_stats}
                )
                self.grid_mode = "fine surface + ERA5-Land precip"
            else:
                self.grid_mode = "fine surface"
            self.context = context
            self.channel_names = channel_names
            self.dists_metadata = self.era5_metadata

        self.in_channels = self.context.shape[1]
        if len(self.channel_names) != self.in_channels:
            logger.warning(
                "channel name count (%d) != context channels (%d); using generic names",
                len(self.channel_names), self.in_channels,
            )
            self.channel_names = [f"ch{i}" for i in range(self.in_channels)]

        # The day axis the run actually uses. Normally this is every day of the
        # requested year range, but a source whose series stops short (ERA5-Land
        # precip) trims it, so record it rather than letting a reader assume.
        _days = pd.to_datetime(np.asarray(self.time_coords))
        self.time_span = {
            "start": str(_days[0].date()),
            "end": str(_days[-1].date()),
            "n_days": int(len(_days)),
            "limited_by": "era5_land_precip" if p.USE_SURFACE_PRECIP else None,
        }
        logger.info("run day axis: %s..%s (%d days)",
                    self.time_span["start"], self.time_span["end"], self.time_span["n_days"])

        # 3. Seasonal features for the elevation MLP.
        self.seasonal_features = (
            ds.compute_seasonal_features(self.time_coords, device=device)
            if p.SEASONAL_FEATURES
            else None
        )

        # 4. High-res topography (shared across variables).
        self.hi_res_elevation, self.hi_res_tpi = ds.load_high_res_topography(
            p.HI_RES_TOPOGRAPHY_ZARR_PATH
        )

        # 5. Channel groups (by physical variable) for the targeted encoders.
        self.channel_groups = ds.channel_groups_by_variable(self.channel_names)

    def normalization_manifest(self) -> dict:
        """Everything needed to reapply the model to unseen data.

        Captures the exact per-channel normalization, channel order/groups, the
        grid the model expects, and the coordinate bounds used to normalize
        target locations. Saved as manifest.json next to the checkpoints.
        """
        m = self.era5_metadata
        dm = self.dists_metadata
        return {
            "grid_mode": self.grid_mode,
            "in_channels": self.in_channels,
            "channel_names": list(self.channel_names),
            "channel_groups": {k: list(v) for k, v in self.channel_groups.items()},
            "time_span": dict(self.time_span),
            "seasonal_features": self.config.base_params.SEASONAL_FEATURES,
            "normalization": {
                # surface temperature channel + tmax target denormalization
                "surface_data": {"mean": m.data_mean, "std": m.data_std},
                # geopotential-derived elevation channel (surface mode)
                "surface_elevation": {"mean": m.elev_mean, "std": m.elev_std},
                # lat/lon scaffold channels are normalized into [0,1] with these bounds
                "lat_bounds": [m.lat_min, m.lat_max],
                "lon_bounds": [m.lon_min, m.lon_max],
                # atmospheric / surface-anchor / coarse-elevation per-channel stats
                "per_channel": self.channel_normalization,
                # seasonal channels (cos/sin day-of-year) are deterministic from dates
                "seasonal": "deterministic: cos/sin(2*pi*(doy-1)/365)",
            },
            "dists_grid": {
                "lat_coords": np.asarray(dm.lat_coords).tolist(),
                "lon_coords": np.asarray(dm.lon_coords).tolist(),
                "lat_bounds": [dm.lat_min, dm.lat_max],
                "lon_bounds": [dm.lon_min, dm.lon_max],
            },
        }

    @staticmethod
    def _surface_scaffold_names(p: params.Params, n_channels: int) -> list[str]:
        """Reconstruct the channel names produced by load_era5_data."""
        names: list[str] = []
        if p.USE_SURFACE:
            names.append("data")
        names += ["lat", "lon"]
        if p.SEASONAL_FEATURES:
            names += ["cos_time", "sin_time"]
        # load_era5_data appends 'elevation' whenever a geopotential glob is given
        # (build_data_paths always provides one) AND the geopotential input channel
        # is enabled. --no-geopotential (USE_ELEVATION_CHANNEL=False) drops it.
        if p.ERA5_GEOPOTENTIAL_GLOB is not None and p.USE_ELEVATION_CHANNEL:
            names.append("elevation")
        if len(names) != n_channels:
            logger.warning(
                "surface scaffold name count (%d) != channels (%d)",
                len(names), n_channels,
            )
        return names


# ---------------------------------------------------------------------------
# Per-variable trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Trains one variable's model with K-fold CV and writes its artifacts."""

    STATS_HEADER = (
        "Fold,Mean absolute error,Pearson correlation,"
        "Spearman correlation,Epoch,train NLL,test NLL,test CRPS\n"
    )

    def __init__(self, config: TrainingConfig, data: DataBundle, variable: str,
                 realtime: "Optional[RealtimeLogger]" = None):
        self.config = config
        self.data = data
        self.variable = variable
        self.spec = VARIABLE_SPECS[variable]
        self.output_dir = config.output_root / variable
        self.params = config.params_for(variable, data.in_channels)
        self.realtime = realtime

    def run(self) -> dict:
        cfg, data, p = self.config, self.data, self.params
        device = cfg.device
        self.output_dir.mkdir(parents=True, exist_ok=True)

        meteoswiss_glob = getattr(cfg.data_paths, self.spec.meteoswiss_glob_attr)

        # Persist config + metadata + a complete normalization manifest, so the
        # trained model can be reapplied to unseen data (new dates/region).
        p.save_json(self.output_dir / "params.json")
        ds.save_metadata_json(data.dists_metadata, self.output_dir / "metadata.json")
        manifest = data.normalization_manifest()
        manifest["variable"] = self.variable
        manifest["distribution"] = model_factory.resolve_distribution(p)
        manifest["encoder"] = p.ENCODER
        manifest["target"] = {
            "data_var": self.spec.data_var,
            "meteoswiss_glob": meteoswiss_glob,
            "convert_to_kelvin": self.spec.convert_to_kelvin,
            "normalize_targets": self.spec.normalize_targets,
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Per-variable MeteoSwiss target + target-point distances.
        target_x, target_y, target_topo = ds.prepare_meteoswiss_targets(
            meteoswiss_glob,
            normalization_stats=data.era5_metadata,
            data_var=self.spec.data_var,
            grid_elevation=data.grid_elevation if p.USE_ELEVATION else None,
            hi_res_elevation=data.hi_res_elevation if p.USE_ELEVATION else None,
            hi_res_tpi=data.hi_res_tpi if p.USE_MTPI else None,
            convert_to_kelvin=self.spec.convert_to_kelvin,
            normalize_targets=self.spec.normalize_targets,
            year_start=p.DATA_YEAR_START,
            year_end=p.DATA_YEAR_END,
            device=device,
        )
        # The context day axis can be shorter than the requested year range when a
        # source's series stops short (see DataBundle.time_span), so select the
        # target on the context's actual days BY DATE rather than trusting both to
        # have been derived from the same year bounds.
        target_y = ds.align_target_to_days(
            target_y, data.time_coords, f"{self.variable} context")

        dists = ds.calculate_dists_meteoswiss(data.dists_metadata, target_x, device=device)
        target_tensor = torch.from_numpy(target_y.values.astype(np.float32)).to(device)
        if target_tensor.shape[0] != data.context.shape[0]:
            raise ValueError(
                f"target days ({target_tensor.shape[0]}) != context days "
                f"({data.context.shape[0]}) for variable {self.variable!r}.")
        self.n_target_points = int(target_tensor.shape[1])

        channel_groups = data.channel_groups if p.ENCODER != "flat" else None
        grad_clip = (
            self.config.grad_clip_override
            if self.config.grad_clip_override is not None
            else self.spec.grad_clip
        )

        epoch_cb = self.realtime.make_callback(self.variable) if self.realtime else None
        if self.realtime:
            self.realtime.start_variable(self.variable)

        fold_times = []
        n_model_params = None
        for fold in range(p.N_FOLDS):
            logger.info("[%s] starting fold %d/%d", self.variable, fold + 1, p.N_FOLDS)
            if self.realtime:
                self.realtime.start_fold(self.variable, fold)
            fold_stats_file = self.output_dir / f"stats_fold{fold}.csv"
            # Only reuse an existing per-fold CSV if it starts with the *current*
            # header. A stale file from an earlier run (e.g. a pre-CRPS 7-column
            # header, or a partial run with a different column set) would otherwise
            # be silently appended to — mixing column counts in one file and later
            # crashing _merge_fold_stats' pd.read_csv. Reset it instead.
            existing_header = (
                fold_stats_file.read_text().split("\n", 1)[0] + "\n"
                if fold_stats_file.exists()
                else None
            )
            if existing_header != self.STATS_HEADER:
                if existing_header is not None:
                    logger.warning(
                        "[%s] stats_fold%d.csv has a stale/mismatched header — "
                        "resetting it (was: %r)",
                        self.variable, fold, existing_header.strip(),
                    )
                fold_stats_file.write_text(self.STATS_HEADER)

            params.set_seed(p.SEED + fold)
            model, loss_fn, get_value_fn = model_factory.build_model(
                p, channel_groups=channel_groups
            )
            if n_model_params is None:
                n_model_params = sum(
                    q.numel() for q in model.parameters() if q.requires_grad
                )
            model = model.to(device)
            model = self._maybe_parallelize(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=p.LR)

            t0 = time.time()
            train_elev(
                model=model,
                opt=optimizer,
                ll=loss_fn,
                elev=target_topo,
                dists=dists,
                y_context=data.context,
                y_target=target_tensor,
                output_dir=str(self.output_dir),
                y_target_t=None,
                get_value=get_value_fn,
                fold=fold,
                n_folds=p.N_FOLDS,
                n_epochs=p.N_EPOCHS,
                batch_size=p.BATCH_SIZE,
                patience=p.PATIENCE,
                stats_file=str(fold_stats_file),
                seasonal=data.seasonal_features,
                device=device,
                grad_clip=grad_clip,
                epoch_callback=epoch_cb,
                crps_fn=model_factory.build_crps_diagnostic(p),
            )
            fold_times.append(time.time() - t0)
            if self.realtime:
                self.realtime.finish_fold(self.variable, fold)

        if self.realtime:
            self.realtime.finish_variable(self.variable)

        # Consolidate per-fold CSVs and plot.
        stats = self._merge_fold_stats()
        self._plot(stats)

        return {
            "variable": self.variable,
            "distribution": model_factory.resolve_distribution(p),
            "n_model_params": n_model_params,
            "n_target_points": self.n_target_points,
            "grad_clip": grad_clip,
            "fold_times": fold_times,
            "per_fold": self._summarize(stats),
            "output_dir": self.output_dir,
        }

    def _maybe_parallelize(self, model):
        if (
            self.config.data_parallel
            and self.config.device.type == "cuda"
            and torch.cuda.device_count() > 1
        ):
            logger.info(
                "wrapping model in DataParallel across %d GPUs",
                torch.cuda.device_count(),
            )
            return ConvCNPDataParallel(model)
        return model

    def _merge_fold_stats(self) -> pd.DataFrame:
        fold_csvs = [
            self.output_dir / f"stats_fold{f}.csv" for f in range(self.params.N_FOLDS)
        ]
        frames = [pd.read_csv(c) for c in fold_csvs if c.exists()]
        stats = pd.concat(frames) if frames else pd.DataFrame()
        stats.to_csv(self.output_dir / "stats.csv", index=False)
        return stats

    def _plot(self, stats: pd.DataFrame):
        if stats.empty:
            return
        try:
            vis.plot_training_curves(
                stats, save_path=str(self.output_dir / "trainingstats")
            )
        except Exception as exc:  # plotting must never abort a training run
            logger.warning("plot_training_curves failed: %s", exc)

    @staticmethod
    def _summarize(stats: pd.DataFrame) -> list[dict]:
        """Per-fold best-epoch (min test NLL) metrics."""
        if stats.empty:
            return []
        has_crps = "test CRPS" in stats.columns
        out = []
        for fold, grp in stats.groupby("Fold"):
            best = grp.loc[grp["test NLL"].idxmin()]
            row = {
                "fold": int(fold),
                "best_epoch": int(best["Epoch"]),
                "test_nll": float(best["test NLL"]),
                "train_nll": float(best["train NLL"]),
                "mae": float(best["Mean absolute error"]),
                "pearson": float(best["Pearson correlation"]),
                "spearman": float(best["Spearman correlation"]),
                "epochs_run": int(grp["Epoch"].max()) + 1,
            }
            if has_crps:
                row["crps"] = float(best["test CRPS"])
            out.append(row)
        return out


# ---------------------------------------------------------------------------
# Comprehensive run log
# ---------------------------------------------------------------------------
class RunLogger:
    """Writes the comprehensive ``run_summary.md`` for a full run."""

    def __init__(self, config: TrainingConfig, data: DataBundle):
        self.config = config
        self.data = data
        self.start_time = time.time()
        self.lines: list[str] = []

    def _git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self.config.base_dir),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return "unknown"

    def write(self, results: list[dict]):
        cfg, data = self.config, self.data
        device = cfg.device
        L = self.lines.append

        L(f"# Training run: {cfg.base_params.TRIAL_NAME}\n")
        L(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        L(f"- Git commit: `{self._git_commit()}`")
        L(f"- Host: {socket.gethostname()} ({platform.platform()})")
        L(f"- Python: {platform.python_version()} | torch: {torch.__version__} "
          f"| CUDA: {torch.version.cuda}")
        L(f"- Device: `{device}`")
        if device.type == "cuda":
            gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            L(f"- GPUs ({len(gpus)}): {', '.join(gpus)}")
            L(f"- DataParallel: {cfg.data_parallel and len(gpus) > 1}")
            peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
            L(f"- Peak GPU memory: {peak_mb:.0f} MB")
        L(f"- CPU cores: {os.cpu_count()} | torch threads: {torch.get_num_threads()}")
        total_wall = time.time() - self.start_time
        L(f"- Total wall time: {self._fmt(total_wall)}\n")

        # Config
        L("## Configuration\n")
        L("| Param | Value |")
        L("| --- | --- |")
        for f in dataclasses.fields(cfg.base_params):
            if f.name.endswith("_GLOB") or f.name.endswith("_DIR") or f.name.endswith("_PATH"):
                continue  # keep the table readable; paths are in params.json
            # IN_CHANNELS is 0 on the base params (set per variable from the data);
            # show the resolved value.
            value = data.in_channels if f.name == "IN_CHANNELS" else getattr(cfg.base_params, f.name)
            L(f"| {f.name} | {value} |")
        L("")

        # Data summary
        L("## Data\n")
        L(f"- Grid mode: **{data.grid_mode}**")
        ctx = data.context
        L(f"- Context tensor (time, channel, lat, lon): `{tuple(ctx.shape)}`")
        L(f"- Input channels: {data.in_channels}")
        L(f"- Time steps: {len(data.time_coords)} "
          f"({pd.to_datetime(data.time_coords).min().date()} -> "
          f"{pd.to_datetime(data.time_coords).max().date()})")
        m = data.era5_metadata
        L(f"- Normalization: data_mean={m.data_mean:.4f}, data_std={m.data_std:.4f}, "
          f"lat=[{m.lat_min:.3f}, {m.lat_max:.3f}], lon=[{m.lon_min:.3f}, {m.lon_max:.3f}]")
        L(f"- Encoder: **{cfg.base_params.ENCODER}**")
        L("\n### Channel groups (by physical variable)\n")
        L("| Group | #channels | Members |")
        L("| --- | --- | --- |")
        for label, idxs in data.channel_groups.items():
            members = ", ".join(data.channel_names[i] for i in idxs)
            if len(members) > 90:
                members = members[:87] + "..."
            L(f"| `{label}` | {len(idxs)} | {members} |")
        L("")

        # Per-variable results
        L("## Results\n")
        for res in results:
            L(f"### {res['variable']} ({res['distribution']})\n")
            L(f"- Model trainable params: {res['n_model_params']:,}")
            L(f"- MeteoSwiss target points: {res['n_target_points']:,}")
            L(f"- Gradient clip: {res['grad_clip']}")
            ft = res["fold_times"]
            if ft:
                L(f"- Fold wall times: "
                  + ", ".join(self._fmt(t) for t in ft)
                  + f"  (total {self._fmt(sum(ft))})")
            per_fold = res["per_fold"]
            if not per_fold:
                L("- No metrics recorded.\n")
                continue
            has_crps = any("crps" in r for r in per_fold)
            crps_h = " CRPS |" if has_crps else ""
            crps_sep = " --- |" if has_crps else ""
            L(f"\n| Fold | best epoch | epochs | test NLL | train NLL |{crps_h} MAE | Pearson | Spearman |")
            L(f"| --- | --- | --- | --- | --- |{crps_sep} --- | --- | --- |")
            for r in per_fold:
                crps_c = f" {r['crps']:.4f} |" if has_crps else ""
                L(f"| {r['fold']} | {r['best_epoch']} | {r['epochs_run']} | "
                  f"{r['test_nll']:.4f} | {r['train_nll']:.4f} |{crps_c} {r['mae']:.4f} | "
                  f"{r['pearson']:.4f} | {r['spearman']:.4f} |")
            L(self._aggregate_row(per_fold))
            L("")

            # Artifacts
            L("**Artifacts:** "
              + ", ".join(
                  f"`{pth.name}`"
                  for pth in [
                      res["output_dir"] / "stats.csv",
                      res["output_dir"] / "params.json",
                      res["output_dir"] / "metadata.json",
                      res["output_dir"] / "trainingstats_all.png",
                  ]
              )
              + f" (in `{res['output_dir']}`)\n")

        out_path = self.config.output_root / "run_summary.md"
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        logger.info("wrote run summary: %s", out_path)
        return out_path

    @staticmethod
    def _aggregate_row(per_fold: list[dict]) -> str:
        def ms(key):
            vals = [r[key] for r in per_fold]
            return float(np.mean(vals)), float(np.std(vals))
        tn = ms("test_nll"); mae = ms("mae"); pe = ms("pearson"); sp = ms("spearman")
        crps_c = ""
        if any("crps" in r for r in per_fold):
            cr = ms("crps")
            crps_c = f" {cr[0]:.4f}±{cr[1]:.4f} |"
        return (
            f"| **mean±std** | | | {tn[0]:.4f}±{tn[1]:.4f} | |{crps_c} "
            f"{mae[0]:.4f}±{mae[1]:.4f} | {pe[0]:.4f}±{pe[1]:.4f} | "
            f"{sp[0]:.4f}±{sp[1]:.4f} |"
        )

    @staticmethod
    def _fmt(seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"


# ---------------------------------------------------------------------------
# Realtime status report
# ---------------------------------------------------------------------------
def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _fmt_crps_cell(value, has_crps: bool) -> str:
    """Render a CRPS table cell: empty when the column is absent, ``-`` for a
    NaN CRPS (e.g. a tmax row in a mixed run), else the formatted value."""
    if not has_crps:
        return ""
    if value is None or np.isnan(value):
        return " - |"
    return f" {value:.4f} |"


class RealtimeLogger:
    """Maintains a live, comprehensive ``status.md`` updated after every epoch.

    Unlike ``RunLogger`` (written once at the end), this rewrites a status file
    continuously during training so a long run can be monitored in real time:
    overall progress + ETA, the currently-running variable/fold/epoch, a per-fold
    status grid, best-so-far metrics per variable, and a rolling table of recent
    epochs. The file is written atomically (temp + rename) so a reader never sees
    a half-written report.
    """

    MAX_RECENT = 15

    def __init__(self, config: TrainingConfig, variables: list[str], n_folds: int, n_epochs: int):
        self.config = config
        self.variables = variables
        self.n_folds = n_folds
        self.n_epochs = n_epochs
        self.path = config.output_root / "status.md"
        self.start = time.time()
        self.planned_epochs = max(1, len(variables) * n_folds * n_epochs)
        self.completed_epochs = 0
        self.epoch_durations: list[float] = []
        self.recent: list[dict] = []
        self.best: dict[str, dict] = {}
        self.fold_status: dict[tuple, str] = {
            (v, f): "pending" for v in variables for f in range(n_folds)
        }
        self.current = {"variable": None, "fold": None, "epoch": None}
        self.phase = "initializing"
        config.output_root.mkdir(parents=True, exist_ok=True)
        self._write()

    # -- lifecycle hooks ---------------------------------------------------
    def start_variable(self, variable: str):
        self.phase = f"training {variable}"
        self.current.update(variable=variable, fold=0, epoch=None)
        self._write()

    def start_fold(self, variable: str, fold: int):
        self.fold_status[(variable, fold)] = "running"
        self.current.update(variable=variable, fold=fold, epoch=None)
        self._write()

    def finish_fold(self, variable: str, fold: int):
        if self.fold_status.get((variable, fold)) != "early-stopped":
            self.fold_status[(variable, fold)] = "done"
        self._write()

    def finish_variable(self, variable: str):
        self.phase = f"finished {variable}"
        self._write()

    def finish_run(self):
        self.phase = "completed"
        self.current.update(variable=None, fold=None, epoch=None)
        self._write()

    def make_callback(self, variable: str):
        """Return an epoch callback bound to ``variable`` for ``train_elev``."""
        def cb(info: dict):
            if info.get("early_stopped"):
                self.fold_status[(variable, info["fold"])] = "early-stopped"
                self._write()
                return
            self.completed_epochs += 1
            self.epoch_durations.append(info.get("epoch_duration", 0.0))
            row = dict(info)
            row["variable"] = variable
            row["time"] = time.strftime("%H:%M:%S")
            self.recent.append(row)
            self.recent = self.recent[-self.MAX_RECENT:]
            self.current.update(variable=variable, fold=info["fold"], epoch=info["epoch"])

            prev = self.best.get(variable)
            if prev is None or info["test_nll"] < prev["test_nll"]:
                self.best[variable] = {
                    "test_nll": info["test_nll"], "mae": info["mae"],
                    "pearson": info["pearson"], "spearman": info["spearman"],
                    "crps": info.get("crps", float("nan")),
                    "fold": info["fold"], "epoch": info["epoch"],
                }
            self._write()
        return cb

    # -- rendering ---------------------------------------------------------
    def _eta(self) -> Optional[float]:
        if not self.epoch_durations:
            return None
        avg = sum(self.epoch_durations) / len(self.epoch_durations)
        return avg * max(0, self.planned_epochs - self.completed_epochs)

    def _write(self):
        lines: list[str] = []
        L = lines.append
        pct = 100.0 * self.completed_epochs / self.planned_epochs
        filled = int(round(pct / 5))
        bar = "█" * filled + "░" * (20 - filled)
        eta = self._eta()

        L(f"# Live status: {self.config.base_params.TRIAL_NAME}\n")
        L(f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        L(f"- Phase: **{self.phase}**")
        cur = self.current
        if cur["variable"] is not None:
            fold_str = "-" if cur["fold"] is None else f"{cur['fold'] + 1}/{self.n_folds}"
            epoch_str = "-" if cur["epoch"] is None else f"{cur['epoch'] + 1}/{self.n_epochs}"
            L(f"- Current: variable **{cur['variable']}**, fold {fold_str}, epoch {epoch_str}")
        L(f"- Progress: `{bar}` {pct:.0f}%  ({self.completed_epochs}/{self.planned_epochs} "
          f"epochs, upper bound)")
        L(f"- Elapsed: {_fmt_duration(time.time() - self.start)}"
          + (f" | est. remaining: {_fmt_duration(eta)}" if eta is not None else ""))
        L("")

        # Fold status grid
        L("## Fold status\n")
        L("| Variable | " + " | ".join(f"fold {f}" for f in range(self.n_folds)) + " |")
        L("| --- " * (self.n_folds + 1) + "|")
        icons = {"pending": "·", "running": "▶ running", "done": "✓ done",
                 "early-stopped": "✓ early-stop"}
        for v in self.variables:
            cells = " | ".join(icons.get(self.fold_status[(v, f)], "?") for f in range(self.n_folds))
            L(f"| {v} | {cells} |")
        L("")

        # Best-so-far
        if self.best:
            has_crps = any(not np.isnan(b.get("crps", float("nan")))
                           for b in self.best.values())
            crps_h = " CRPS |" if has_crps else ""
            crps_sep = " --- |" if has_crps else ""
            L("## Best so far (min test NLL)\n")
            L(f"| Variable | fold | epoch | test NLL |{crps_h} MAE | Pearson | Spearman |")
            L(f"| --- | --- | --- | --- |{crps_sep} --- | --- | --- |")
            for v, b in self.best.items():
                crps_c = _fmt_crps_cell(b.get("crps"), has_crps)
                L(f"| {v} | {b['fold']} | {b['epoch']} | {b['test_nll']:.4f} |{crps_c} "
                  f"{b['mae']:.4f} | {b['pearson']:.4f} | {b['spearman']:.4f} |")
            L("")

        # Recent epochs
        if self.recent:
            has_crps = any(not np.isnan(r.get("crps", float("nan")))
                           for r in self.recent)
            crps_h = " CRPS |" if has_crps else ""
            crps_sep = " --- |" if has_crps else ""
            L(f"## Recent epochs (last {len(self.recent)})\n")
            L(f"| time | variable | fold | epoch | test NLL | train NLL |{crps_h} MAE | Pearson | Spearman | best? |")
            L(f"| --- | --- | --- | --- | --- | --- |{crps_sep} --- | --- | --- | --- |")
            for r in reversed(self.recent):
                star = "★" if r.get("improved") else ""
                crps_c = _fmt_crps_cell(r.get("crps"), has_crps)
                L(f"| {r['time']} | {r['variable']} | {r['fold']} | {r['epoch']} | "
                  f"{r['test_nll']:.4f} | {r['train_nll']:.4f} |{crps_c} {r['mae']:.4f} | "
                  f"{r['pearson']:.4f} | {r['spearman']:.4f} | {star} |")
            L("")

        tmp = self.path.with_suffix(".md.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variables", nargs="+", default=["tmax"],
                    choices=list(VARIABLE_SPECS), help="Variables to train (independent models).")
    ap.add_argument("--distribution", default=None,
                    choices=list(model_factory.LIKELIHOODS),
                    help="Output distribution (overrides the per-variable default in "
                         "model_factory.VARIABLE_TO_DISTRIBUTION). E.g. for precip use "
                         "'bernoulli_gamma' (default). Applies to all --variables.")
    ap.add_argument("--trial-name", default=None,
                    help="Optional trial-name prefix. The run configuration is always "
                         "appended as a slug (e.g. '<prefix>__tmax_sfc_flat_y2023_e30f5_b8'); "
                         "if omitted, the slug alone is used.")
    ap.add_argument("--base-dir", default=".", help="Repo base dir (outputs under <base>/<models-subdir>).")
    ap.add_argument("--models-subdir", default="trained_models",
                    help="Directory under --base-dir that holds the trained-model trials "
                         "(default 'trained_models'; e.g. 'CLEAN_trained_models').")
    ap.add_argument("--data-dir", default="./datasets", help="Dataset base dir.")
    ap.add_argument("--remote-dataset-dir", default="../datasets-chr",
                    help="Remote/mounted dataset dir to sync from when running on cloud.")
    ap.add_argument("--data-year-start", type=int, default=2023, help="First year of data to load (inclusive).")
    ap.add_argument("--data-year-end", type=int, default=None, help="Last year of data to load (inclusive). Defaults to same as --data-year-start.")
    ap.add_argument("--all-years", action="store_true", help="Use all available years (ignore --data-year-start/--data-year-end).")

    ap.add_argument("--n-epochs", type=int, default=30)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="Max gradient norm (overrides per-variable default; precip defaults to 1.0).")
    ap.add_argument("--length-scale", type=float, default=params.Params.LENGTH_SCALE,
                    help="Initial RBF length scale of the final layer (paper default 0.1). "
                         "Exposed for hyperparameter studies.")

    # Input source (mutually exclusive; surface is the default).
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--use-surface", dest="use_atmospheric", action="store_false",
                     help="Use the ERA5-Land surface field (default).")
    src.add_argument("--use-atmospheric", dest="use_atmospheric", action="store_true",
                     help="Use ERA5 pressure-level (atmospheric) fields.")
    ap.set_defaults(use_atmospheric=False)
    ap.add_argument("--atmos-native-grid", action=argparse.BooleanOptionalAction, default=True,
                    help="Encode atmospheric input at its native coarse grid (default on).")
    ap.add_argument("--use-sfc-atmos", action="store_true",
                    help="Append ERA5 single-level surface anchor channels (requires --use-atmospheric).")
    ap.add_argument("--use-surface-precip", action="store_true",
                    help="Append the ERA5-Land daily precipitation field (tp, mm) as an extra "
                         "surface context channel next to t2m_max (requires --use-surface). "
                         "Same grid and daily axis, no regridding. The run's day axis is "
                         "trimmed to the days tp actually covers.")
    ap.add_argument("--atmos-variables", nargs="+", default=None,
                    choices=["z", "t", "q", "r", "u", "v"],
                    help="ERA5 pressure-level variables to use as atmospheric input "
                         "channels (requires --use-atmospheric). z=geopotential, "
                         "t=temperature, q=specific humidity, r=relative humidity, "
                         "u/v=wind components. Default (unset): z t q. The matching "
                         "files must be downloaded (see datasets/download_era5.py "
                         "--pl-fields).")

    ap.add_argument("--encoder", default="flat",
                    choices=["flat", "grouped_setconv", "channel_attention"],
                    help="Input encoder variant.")
    ap.add_argument("--no-seasonal", action="store_true", help="Disable seasonal features.")
    ap.add_argument("--no-elevation", action="store_true", help="Disable elevation/TPI features.")
    ap.add_argument("--no-geopotential", action="store_true",
                    help="Drop the geopotential-derived coarse elevation input channel from the "
                         "surface encoder context. Geopotential is still loaded so the elevation "
                         "MLP (DEM/TPI/elev_diff) keeps working — this ablates only the input "
                         "channel. Surface mode only.")

    ap.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True,
                    help="Write a live status.md report during training (default on).")
    ap.add_argument("--data-parallel", action=argparse.BooleanOptionalAction, default=True,
                    help="Use multi-GPU DataParallel when >1 GPU is available.")
    ap.add_argument("--device", default=None, help="Override device (e.g. 'cuda', 'cpu').")
    return ap


class _Tee:
    """Duplicate writes to an underlying stream and a log file.

    Used to mirror everything printed to the console (bare ``print()`` calls,
    ``logging`` output, and uncaught-exception tracebacks) into ``run.log``.
    Non-write attribute access is delegated to the wrapped stream so it stays a
    drop-in replacement for ``sys.stdout`` / ``sys.stderr``.
    """

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _setup_run_log(path: Path):
    """Tee stdout/stderr to ``path`` so the full console output is persisted.

    Must be called before ``configure_runtime`` (which points logging at
    ``sys.stdout``) so the logging handler picks up the teed stream and we avoid
    writing log records twice.
    """
    # encoding is explicit: without it Python uses the locale, which is ASCII in a
    # non-interactive launch (nohup/cron/CI) and blows up on the non-ASCII glyphs
    # the progress output carries.
    fh = open(path, "a", buffering=1, encoding="utf-8")  # line-buffered: survives crashes
    fh.write(f"\n{'=' * 79}\n=== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{'=' * 79}\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    atexit.register(fh.flush)


def configure_runtime():
    """Set up logging and multicore thread usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    n_cores = os.cpu_count() or 1
    torch.set_num_threads(n_cores)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_cores))


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = TrainingConfig(args)
    config.output_root.mkdir(parents=True, exist_ok=True)
    _setup_run_log(config.output_root / "run.log")
    configure_runtime()

    logger.info("device=%s | variables=%s | encoder=%s | trial=%s",
                config.device, config.variables, args.encoder, config.trial_name)

    params.set_seed(config.base_params.SEED)
    if config.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    cloud_sync.sync_from_cloud_if_needed(
        Path(args.remote_dataset_dir), config.data_dir, config.base_params.RUN_TYPE
    )

    run_logger = RunLogger(config, data=None)  # data attached after load
    data = DataBundle(config)
    run_logger.data = data

    realtime = None
    if args.realtime:
        realtime = RealtimeLogger(
            config, config.variables,
            config.base_params.N_FOLDS, config.base_params.N_EPOCHS,
        )
        logger.info("live status report: %s", realtime.path)

    results = []
    for variable in config.variables:
        logger.info("=== Training variable: %s ===", variable)
        trainer = Trainer(config, data, variable, realtime=realtime)
        results.append(trainer.run())

    if realtime:
        realtime.finish_run()

    summary_path = run_logger.write(results)
    logger.info("Done. Summary at %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
