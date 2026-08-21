#!/usr/bin/env python3
"""
Gaussian (tmax) evaluation under the two regimes the precip pipeline established.

There are exactly two honest ways to score a model trained on 2020-2023, and this
script implements both — mirroring ``eval_precip.py``, which is the reference:

**CV holdout (default, no --eval-year).** The 5 cross-validation folds partition the
1461-day training axis into contiguous blocks. Each fold predicts *only* the block it
was held out from, so every day is predicted exactly once, by the one model that never
saw it. This is NOT a fold ensemble; per-fold numbers land in ``per_fold``.

**Holdout year (--eval-year 2024).** A year no fold saw. All folds predict all days and
are combined as a Gaussian mixture moment match. Inputs are normalised with the
*training-frozen* statistics in ``manifest.json`` — never re-fit on the holdout year.

Both regimes render the same report (figures + ``report_metrics.json`` + ``report.md``)
via ``report.render_evaluation``, and both are served from a persisted prediction bundle
under ``<model_dir>/pred_cache/`` so GPU inference happens at most once per
(model, regime). Everything downstream — ``error_analysis.py``, ``station_analysis.py``,
figure re-renders — then costs no GPU at all.

NOTE: needs the ERA5 *inputs* for the period (surface + pressure-level z/t/q for the
atmospheric model) plus the MeteoSwiss truth; see download_era5.py /
datasets/download_meteoswiss.py.

Examples
--------
    # 2020-2023 cross-validation holdout
    python evaluate.py --model-dir CLEAN_trained_models/<trial>/tmax
    # genuine 2024 holdout year
    python evaluate.py --model-dir CLEAN_trained_models/<trial>/tmax --eval-year 2024
    # force re-inference instead of reusing the cached bundle
    python evaluate.py --model-dir <...>/tmax --eval-year 2024 --refresh-cache
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr

import params as params_mod
import datasets as ds
import model_factory
import predict  # context builders, targets, ensemble inference helpers
import report  # render_evaluation (shared renderer)
import metrics as metrics_mod
from convCNP.training.training_elev import get_fold_holdout_indices
from infer import _resolve_model_dir

logger = logging.getLogger("evaluate")

# The tmax target lattice: MeteoSwiss TmaxD is 240 (N) x 370 (E) = 88800 points, of
# which 46718 fall inside Switzerland. (98050 is the *precip* RhiresD grid — different
# variable, different lattice. Do not cross the two.)

def _targets_with_truth(manifest, p, device, year_start=None, year_end=None):
    """Like predict.load_meteoswiss_targets, but also returns the truth DataArray.

    Returns (target_x, target_y_all, target_topo, lat_arr, lon_arr) where
    target_y_all is the MeteoSwiss truth (time, point), loaded exactly as training did
    for ``p.VARIABLE``: z-scored Kelvin for tmax, raw mm for precip (z-scoring would
    push zeros negative and break the Gamma log-prob).

    ``year_start``/``year_end`` bound the truth years loaded. They default to None
    (every year in the glob, 1971-2025) because ``feature_importance.py`` imports this
    helper and relies on that; the evaluation paths always pass explicit bounds so the
    truth for six decades is not materialised to score four years.
    """
    # Local import: train.py does not import evaluate, so this cannot cycle.
    from train import VARIABLE_SPECS
    spec = VARIABLE_SPECS[p.VARIABLE]
    dists_meta = predict.manifest_to_dists_metadata(manifest)
    hi_res_elevation, hi_res_tpi = ds.load_high_res_topography(p.HI_RES_TOPOGRAPHY_ZARR_PATH)
    grid_elev = None
    if p.ERA5_GEOPOTENTIAL_GLOB:
        geo = xr.open_mfdataset(p.ERA5_GEOPOTENTIAL_GLOB, combine="by_coords")
        geo = geo.rename({"latitude": "longitude", "longitude": "latitude"})
        alt = geo["z"] / 9.80665
        if "time" in alt.dims:
            alt = alt.isel(time=0)
        grid_elev = alt
    target_x, target_y, target_topo = ds.prepare_meteoswiss_targets(
        getattr(p, spec.meteoswiss_glob_attr),
        normalization_stats=dists_meta,
        data_var=spec.data_var,
        grid_elevation=grid_elev if p.USE_ELEVATION else None,
        hi_res_elevation=hi_res_elevation if p.USE_ELEVATION else None,
        hi_res_tpi=hi_res_tpi if p.USE_MTPI else None,
        convert_to_kelvin=spec.convert_to_kelvin,
        normalize_targets=spec.normalize_targets,
        year_start=year_start,
        year_end=year_end,
        device=device,
    )
    lat_arr = (target_x.sel(coord="lat").values
               * (dists_meta.lat_max - dists_meta.lat_min) + dists_meta.lat_min)
    lon_arr = (target_x.sel(coord="lon").values
               * (dists_meta.lon_max - dists_meta.lon_min) + dists_meta.lon_min)
    return target_x, target_y, target_topo, lat_arr, lon_arr


def era5_reference_degC(context, p, target_x, dists_meta, dates, device) -> np.ndarray:
    """Bilinear-ERA5 surface-tmax skill baseline at the targets, in Celsius, (T, P).

    This is the deterministic "what ERA5 alone says here" reference the skill score is
    measured against. It spans an arbitrary ``dates`` index, so the same function serves
    the CV regime (1461 days across four years) and a single holdout year — replacing the
    three near-identical per-year copies that used to live in ``evaluate``, ``report`` and
    ``station_analysis``.

    Surface model: context channel 0 already holds the surface tmax field.
    Atmospheric model: channel 0 is a pressure-level field, so ERA5 surface tmax is loaded
    separately and aligned onto ``dates``.
    """
    if p.USE_SURFACE:
        ref_norm = ds.interpolate_era5_to_targets(context, target_x, dists_meta)
        return dists_meta.denormalize(ref_norm.cpu().numpy()) - ds.KELVIN_OFFSET

    dates = pd.DatetimeIndex(dates).normalize()
    year_start, year_end = int(dates[0].year), int(dates[-1].year)
    logger.info("atmospheric model: loading ERA5 surface tmax %d-%d for the skill baseline",
                year_start, year_end)
    sfc, ref_meta, _, tc = ds.load_era5_data(
        p.ERA5_MAX_TEMP_GLOB, var_name="t2m_max",
        year_start=year_start, year_end=year_end,
        geopotential_glob=None, include_seasonal_embeddings=False,
        include_data_channel=True, device=device,
    )
    # Align on calendar day rather than assuming the ERA5 file starts where we do.
    src_days = pd.DatetimeIndex(pd.to_datetime(tc)).normalize()
    pos = pd.Series(np.arange(len(src_days)), index=src_days)
    pos = pos[~pos.index.duplicated(keep="first")]
    idx = pos.reindex(dates).to_numpy()
    if np.isnan(idx).any():
        missing = dates[np.isnan(idx)]
        raise RuntimeError(
            f"ERA5 surface tmax is missing {len(missing)} day(s) needed for the skill "
            f"reference, first {missing[0].date()} — the baseline would be silently NaN.")
    ref_norm = ds.interpolate_era5_to_targets(sfc[idx.astype(int)], target_x, ref_meta)
    return ref_meta.denormalize(ref_norm.cpu().numpy()) - ds.KELVIN_OFFSET


def build_eval_context(model_dir: Path, dates: pd.DatetimeIndex,
                       device: torch.device) -> SimpleNamespace:
    """Everything both regimes need, built once: context, dists, targets, truth, reference.

    The context is normalised with the TRAINING-frozen statistics in ``manifest.json``,
    which is what makes a holdout year honest — the 2024 inputs are never re-standardised
    on 2024's own distribution.
    """
    manifest = predict.load_manifest(model_dir)
    p = params_mod.Params.load_json(model_dir / "params.json")
    p.DEVICE = str(device)
    dists_meta = predict.manifest_to_dists_metadata(manifest)
    dates = pd.DatetimeIndex(dates)

    if p.USE_ATMOSPHERIC and p.ATMOS_NATIVE_GRID:
        context = predict.build_atmospheric_context(manifest, p, dates, device)
    else:
        context = predict.build_surface_context(manifest, p, dates, device)
    T = int(context.shape[0])
    if T != len(dates):
        raise RuntimeError(
            f"context has {T} days but {len(dates)} were requested — fold-holdout "
            "indices are derived from the day count and would silently shift.")

    target_x, target_y_all, target_topo, lat_arr, lon_arr = _targets_with_truth(
        manifest, p, device,
        year_start=int(dates[0].year), year_end=int(dates[-1].year))
    dists = ds.calculate_dists_meteoswiss(dists_meta, target_x, device=device)
    seasonal = (ds.compute_seasonal_features(dates.values.astype("datetime64[ns]"),
                                             device=device) if p.SEASONAL_FEATURES else None)

    # Align truth onto the context days by calendar day. A mismatch here would silently
    # score predictions against the wrong days, so it is fatal rather than a warning.
    want = dates.normalize()
    truth_days = pd.DatetimeIndex(pd.to_datetime(target_y_all.time.values)).normalize()
    pos = pd.Series(np.arange(len(truth_days)), index=truth_days)
    pos = pos[~pos.index.duplicated(keep="first")]
    idx = pos.reindex(want).to_numpy()
    if np.isnan(idx).any():
        missing = want[np.isnan(idx)]
        raise RuntimeError(
            f"MeteoSwiss truth is missing {len(missing)} of {len(want)} requested days, "
            f"first {missing[0].date()}.")
    truth_norm = target_y_all.values[idx.astype(int)].astype(np.float32)   # (T, P)

    era5_ref = era5_reference_degC(context, p, target_x, dists_meta, dates, device)

    grid_N, grid_E = ds.get_meteoswiss_grid_shape(p.METEO_SWISS_MAX_TEMP_GLOB)
    n_points = int(truth_norm.shape[1])
    if n_points != grid_N * grid_E:
        raise RuntimeError(
            f"target points ({n_points}) != grid N*E ({grid_N}*{grid_E}); the report "
            "renderer reshapes to this lattice and would produce garbage maps.")

    truths_c = predict.denormalize(truth_norm, dists_meta)
    valid_mask = ~np.isnan(truths_c[0].reshape(grid_N, grid_E))
    channel_groups = (ds.channel_groups_by_variable(manifest["channel_names"])
                      if p.ENCODER != "flat" else None)
    grid_mode = manifest.get("grid_mode") or (
        "native coarse atmospheric" if p.USE_ATMOSPHERIC else "fine surface")

    return SimpleNamespace(
        manifest=manifest, p=p, dists_meta=dists_meta, context=context, dists=dists,
        target_x=target_x, target_topo=target_topo, seasonal=seasonal,
        truths_c=truths_c, era5_ref=era5_ref, dates=dates,
        lat_arr=np.asarray(lat_arr), lon_arr=np.asarray(lon_arr),
        grid_shape=(grid_N, grid_E), valid_mask=valid_mask, grid_mode=grid_mode,
        channel_groups=channel_groups, n_times=T,
    )


def _bundle_from_context(C, preds_c, sigmas_c, *, day_mask, per_fold_blocks, fold_epochs,
                         eval_year, prediction_mode, n_folds,
                         sigma_within_c=None, sigma_between_c=None) -> SimpleNamespace:
    """Pack a regime's degC arrays into the namespace every consumer already speaks."""
    return SimpleNamespace(
        errors_c=preds_c - C.truths_c, preds_c=preds_c, sigmas_c=sigmas_c,
        truths_c=C.truths_c, era5_ref=C.era5_ref,
        target_topo=C.target_topo.cpu().numpy(), target_topo_t=C.target_topo,
        lat_arr=C.lat_arr, lon_arr=C.lon_arr, dates=np.asarray(C.dates),
        grid_shape=C.grid_shape, valid_mask=C.valid_mask, p=C.p,
        grid_mode=C.grid_mode, n_folds=n_folds, n_times=C.n_times,
        day_mask=day_mask, per_fold_blocks=per_fold_blocks, fold_epochs=fold_epochs,
        eval_year=eval_year, prediction_mode=prediction_mode,
        sigma_within_c=sigma_within_c, sigma_between_c=sigma_between_c,
    )


def predict_cv(model_dir: Path, device: torch.device, folds: Optional[list[int]] = None,
               day_batch: int = 1) -> SimpleNamespace:
    """2020-2023 cross-validation holdout — each fold scores ONLY its own held-out block.

    The folds partition the training day axis into contiguous blocks
    (``get_fold_holdout_indices``), so concatenating them reconstructs the full period
    with **every day predicted exactly once, by the one model that never saw it**. This is
    deliberately not a fold ensemble: ensembling here would let models that trained on a
    day help predict it. Mirrors ``eval_precip.predict_all_folds``.
    """
    p = params_mod.Params.load_json(model_dir / "params.json")
    if p.DATA_YEAR_START is None or p.DATA_YEAR_END is None:
        raise ValueError(
            "the CV regime needs a bounded training span; this run has "
            f"DATA_YEAR_START={p.DATA_YEAR_START}, DATA_YEAR_END={p.DATA_YEAR_END}.")
    dates = predict._date_range(f"{p.DATA_YEAR_START}-01-01", f"{p.DATA_YEAR_END}-12-31")
    logger.info("=== CV holdout %s | %d-%d (%d days) ===",
                model_dir, p.DATA_YEAR_START, p.DATA_YEAR_END, len(dates))
    C = build_eval_context(model_dir, dates, device)

    n_times, n_points = C.n_times, C.truths_c.shape[1]
    preds_n = np.full((n_times, n_points), np.nan, dtype=np.float32)
    sig_n = np.full((n_times, n_points), np.nan, dtype=np.float32)
    day_mask = np.zeros(n_times, dtype=bool)
    per_fold_blocks: dict[int, tuple] = {}
    fold_epochs: dict[int, int] = {}

    for fold in (folds if folds is not None else list(range(C.p.N_FOLDS))):
        ckpt = model_dir / f"model_fold_{fold}"
        if not ckpt.exists():
            logger.warning("checkpoint %s missing; skipping fold %d", ckpt, fold)
            continue
        model, epoch = model_factory.load_model_checkpoint(
            ckpt, C.p, device, channel_groups=C.channel_groups)
        start, end = get_fold_holdout_indices(fold, C.p.N_FOLDS, n_times)
        logger.info("fold %d: predicting held-out days [%d, %d) = %s..%s (ckpt epoch %d)",
                    fold, start, end, C.dates[start].date(), C.dates[end - 1].date(), epoch)
        pf, sf = predict.predict_all_days(
            model, C.context, C.dists, C.target_topo, C.seasonal, device,
            day_batch=day_batch, day_start=start, day_end=end)
        preds_n[start:end] = pf
        sig_n[start:end] = sf
        day_mask[start:end] = True
        per_fold_blocks[fold] = (start, end)
        fold_epochs[fold] = int(epoch)

    if not per_fold_blocks:
        raise RuntimeError(f"No fold checkpoints found under {model_dir}.")

    preds_c = predict.denormalize(preds_n, C.dists_meta)
    sigmas_c = sig_n * C.dists_meta.data_std
    return _bundle_from_context(
        C, preds_c, sigmas_c, day_mask=day_mask, per_fold_blocks=per_fold_blocks,
        fold_epochs=fold_epochs, eval_year=None, prediction_mode="fold_holdout",
        n_folds=len(per_fold_blocks))


def predict_holdout_year(model_dir: Path, device: torch.device, year: int,
                         folds: Optional[list[int]] = None,
                         day_batch: int = 1) -> SimpleNamespace:
    """A genuinely unseen year: every fold predicts every day, combined as a mixture.

    Unlike precip — which averages its (rho, alpha, beta) elementwise — a Gaussian
    ensemble must be combined by moment matching: ``mu = mean_k mu_k`` and
    ``sigma^2 = mean_k sigma_k^2 + var_k mu_k``. Averaging the sigmas instead would throw
    away the between-fold disagreement, which is exactly the part of the uncertainty that
    grows on unseen data. The two components are returned separately so the split is
    reportable.
    """
    dates = predict._date_range(f"{year}-01-01", f"{year}-12-31")
    logger.info("=== holdout year %s | %d (%d days) ===", model_dir, year, len(dates))
    C = build_eval_context(model_dir, dates, device)

    fold_ids = folds if folds is not None else list(range(C.p.N_FOLDS))
    all_preds_n, all_sigmas_n, fold_epochs = [], [], {}
    for fold in fold_ids:
        ckpt = model_dir / f"model_fold_{fold}"
        if not ckpt.exists():
            logger.warning("checkpoint %s missing; skipping fold %d", ckpt, fold)
            continue
        model, epoch = model_factory.load_model_checkpoint(
            ckpt, C.p, device, channel_groups=C.channel_groups)
        logger.info("fold %d (epoch %d): %d days x %d points ...",
                    fold, epoch, C.n_times, len(C.lat_arr))
        pf, sf = predict.predict_all_days(
            model, C.context, C.dists, C.target_topo, C.seasonal, device,
            day_batch=day_batch)
        all_preds_n.append(pf)
        all_sigmas_n.append(sf)
        fold_epochs[fold] = int(epoch)

    if not all_preds_n:
        raise RuntimeError(f"No fold checkpoints found under {model_dir}.")

    mu_k = np.array(all_preds_n)
    within_var = np.mean(np.array(all_sigmas_n) ** 2, axis=0)
    between_var = np.var(mu_k, axis=0)
    preds_mean_n = np.mean(mu_k, axis=0)

    sd = C.dists_meta.data_std
    preds_c = predict.denormalize(preds_mean_n, C.dists_meta)
    return _bundle_from_context(
        C, preds_c, np.sqrt(within_var + between_var) * sd,
        day_mask=np.ones(C.n_times, dtype=bool), per_fold_blocks={},
        fold_epochs=fold_epochs, eval_year=year, prediction_mode="fold_ensemble",
        n_folds=len(all_preds_n),
        sigma_within_c=np.sqrt(within_var) * sd,
        sigma_between_c=np.sqrt(between_var) * sd)


# ---------------------------------------------------------------------------
# Prediction-bundle cache
# ---------------------------------------------------------------------------
# One inference pass per (model, regime), persisted so that metrics, figures,
# error_analysis and station_analysis are all plot-only re-runs. This mirrors
# eval_precip's pred_cache, with two tmax-specific differences:
#
#   * only the 46718 valid grid points are stored, not all 88800 (a 1.9x saving);
#     load_bundle re-expands to the full lattice with NaN, which is exactly what
#     the truth carries at those points anyway.
#   * the bundle holds *denormalised degC*, because for a Gaussian the fold
#     ensemble has to be moment-matched before it can be cached. That makes the
#     file valid only for the manifest that produced it, so load_bundle checks a
#     hash of manifest.json. Precip caches raw distribution params and is immune.
CACHE_SCHEMA_VERSION = 1
CACHE_DIRNAME = "pred_cache"   # gitignored: ~1.1 GB (CV) / ~0.4 GB (2024) per model


def bundle_cache_path(model_dir: Path, eval_year: Optional[int] = None,
                      kind: str = "grid") -> Path:
    stem = "cv" if eval_year is None else f"holdout_{eval_year}"
    prefix = "" if kind == "grid" else f"{kind}_"
    return Path(model_dir) / CACHE_DIRNAME / f"{prefix}{stem}.npz"


def _manifest_hash(model_dir: Path) -> str:
    path = Path(model_dir) / "manifest.json"
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_bundle(B: SimpleNamespace, path: Path, model_dir: Path) -> None:
    """Persist a prediction bundle as a pickle-free npz (atomic write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.asarray(B.valid_mask).reshape(-1)
    fold_ids = sorted(B.per_fold_blocks)
    epoch_ids = sorted(B.fold_epochs)
    compact = lambda x: np.asarray(x)[:, valid].astype(np.float32)   # noqa: E731

    arrays = dict(
        schema_version=np.int64(CACHE_SCHEMA_VERSION),
        preds_degC=compact(B.preds_c), sigmas_degC=compact(B.sigmas_c),
        truths_degC=compact(B.truths_c), era5_ref_degC=compact(B.era5_ref),
        target_topo=np.asarray(B.target_topo, dtype=np.float32),
        lat=np.asarray(B.lat_arr, dtype=np.float32),
        lon=np.asarray(B.lon_arr, dtype=np.float32),
        valid_mask=np.asarray(B.valid_mask, dtype=bool),
        dates=np.asarray(B.dates).astype("datetime64[ns]"),
        day_mask=np.asarray(B.day_mask, dtype=bool),
        fold_ids=np.asarray(fold_ids, dtype=np.int64),
        fold_starts=np.asarray([B.per_fold_blocks[f][0] for f in fold_ids], dtype=np.int64),
        fold_ends=np.asarray([B.per_fold_blocks[f][1] for f in fold_ids], dtype=np.int64),
        epoch_fold_ids=np.asarray(epoch_ids, dtype=np.int64),
        epoch_values=np.asarray([B.fold_epochs[f] for f in epoch_ids], dtype=np.int64),
        grid_shape=np.asarray(B.grid_shape, dtype=np.int64),
        n_times=np.int64(B.n_times), n_folds=np.int64(B.n_folds),
        eval_year=np.int64(-1 if B.eval_year is None else B.eval_year),
        prediction_mode=np.asarray(B.prediction_mode),
        grid_mode=np.asarray(str(B.grid_mode)),
        manifest_sha256=np.asarray(_manifest_hash(model_dir)),
        model_dir=np.asarray(str(model_dir)),
    )
    for key, src in (("sigma_within_degC", B.sigma_within_c),
                     ("sigma_between_degC", B.sigma_between_c)):
        if src is not None:
            arrays[key] = compact(src)

    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)
    logger.info("cached inference: %s (%.2f GB)", path, path.stat().st_size / 1e9)


def load_bundle(path: Path, model_dir: Path) -> Optional[SimpleNamespace]:
    """Rebuild a bundle from disk, or None if it is stale/unreadable (=> recompute)."""
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("could not read %s (%r); re-running inference", path, exc)
        return None

    if int(z["schema_version"]) != CACHE_SCHEMA_VERSION:
        logger.warning("%s has schema v%d, expected v%d; re-running inference",
                       path, int(z["schema_version"]), CACHE_SCHEMA_VERSION)
        return None
    want = _manifest_hash(model_dir)
    if want and str(z["manifest_sha256"]) != want:
        logger.warning("%s was built against a different manifest.json; re-running "
                       "inference (the bundle stores degC, so its values would be wrong)",
                       path)
        return None

    valid = z["valid_mask"].reshape(-1)
    n_days, n_points = int(z["preds_degC"].shape[0]), int(valid.size)

    def expand(compact):
        full = np.full((n_days, n_points), np.nan, dtype=np.float32)
        full[:, valid] = compact
        return full

    preds_c, truths_c = expand(z["preds_degC"]), expand(z["truths_degC"])
    target_topo = z["target_topo"]
    p = params_mod.Params.load_json(Path(model_dir) / "params.json")
    eval_year = int(z["eval_year"])
    per_fold_blocks = {int(f): (int(s), int(e)) for f, s, e
                       in zip(z["fold_ids"], z["fold_starts"], z["fold_ends"])}
    logger.info("reusing cached inference: %s (no GPU pass)", path)
    return SimpleNamespace(
        errors_c=preds_c - truths_c, preds_c=preds_c, sigmas_c=expand(z["sigmas_degC"]),
        truths_c=truths_c, era5_ref=expand(z["era5_ref_degC"]),
        target_topo=target_topo, target_topo_t=torch.from_numpy(target_topo),
        lat_arr=z["lat"], lon_arr=z["lon"], dates=z["dates"],
        grid_shape=tuple(int(v) for v in z["grid_shape"]), valid_mask=z["valid_mask"],
        p=p, grid_mode=str(z["grid_mode"]), n_folds=int(z["n_folds"]),
        n_times=int(z["n_times"]), day_mask=z["day_mask"],
        per_fold_blocks=per_fold_blocks,
        fold_epochs={int(f): int(v) for f, v in zip(z["epoch_fold_ids"], z["epoch_values"])},
        eval_year=None if eval_year < 0 else eval_year,
        prediction_mode=str(z["prediction_mode"]),
        sigma_within_c=expand(z["sigma_within_degC"]) if "sigma_within_degC" in z else None,
        sigma_between_c=expand(z["sigma_between_degC"]) if "sigma_between_degC" in z else None,
        served_from_cache=True,
    )


def load_or_predict(model_dir: Path, device: torch.device,
                    eval_year: Optional[int] = None,
                    folds: Optional[list[int]] = None,
                    refresh: bool = False, use_cache: bool = True,
                    day_batch: int = 1) -> SimpleNamespace:
    """The single entry point for every tmax consumer of model predictions.

    Everything downstream — reports, error_analysis, station_analysis — comes through
    here, so GPU inference happens at most once per (model, regime). An explicit fold
    subset bypasses the cache in both directions: it does not represent the full run and
    must never be persisted as if it did.
    """
    model_dir = Path(model_dir)
    cache = bundle_cache_path(model_dir, eval_year)
    if folds is not None:
        logger.info("explicit fold subset requested: bypassing the prediction cache")
    elif use_cache and not refresh and cache.exists():
        B = load_bundle(cache, model_dir)
        if B is not None:
            return B

    if eval_year is not None:
        B = predict_holdout_year(model_dir, device, eval_year, folds=folds,
                                 day_batch=day_batch)
    else:
        B = predict_cv(model_dir, device, folds=folds, day_batch=day_batch)
    B.served_from_cache = False
    if use_cache and folds is None:
        save_bundle(B, cache, model_dir)
    return B


def pooled_scalars(B, sl: slice) -> dict:
    """Pooled headline metrics over one day slice, defined exactly as the overall ones.

    Reuses metrics.compute_perpixel_metrics so a per-fold number and the overall number
    can never drift apart in definition.
    """
    m = metrics_mod.compute_perpixel_metrics(
        all_errors=B.errors_c[sl], all_preds=B.preds_c[sl], all_sigmas=B.sigmas_c[sl],
        all_truths=B.truths_c[sl], all_ref_preds=B.era5_ref[sl],
        grid_shape=B.grid_shape, valid_mask=B.valid_mask,
    )
    z = (B.errors_c[sl] / B.sigmas_c[sl]).ravel()
    z = z[np.isfinite(z)]
    return {
        "n_days": int(B.errors_c[sl].shape[0]),
        "mae_degC": float(np.nanmean(m.mae_grid)),
        "rmse_degC": float(np.nanmean(m.rmse_grid)),
        "bias_degC": float(np.nanmean(m.bias_grid)),
        "crps_degC": float(m.overall_crps),
        "skill_score": float(m.global_skill),
        "z_std": float(np.std(z)) if z.size else float("nan"),
        "coverage_90": float(np.mean(np.abs(z) <= 1.6448536269514722)) if z.size else float("nan"),
    }


CV_NOTE = (
    "Cross-validation holdout over the **training span {ys}-{ye}**. Each of the {k} folds "
    "predicts **only** the contiguous block it was held out from during training; the blocks "
    "tile the {n}-day axis, so **every day is predicted exactly once, by the one model that "
    "never saw it**. This is not a fold ensemble — per-fold numbers are in `per_fold`. These "
    "are training-span days scored out-of-sample, *not* an unseen year."
)
HOLDOUT_NOTE = (
    "Genuine **holdout year {year}**: no fold saw any {year} day. Inputs are normalised with "
    "the **training-frozen** statistics in `manifest.json` — never re-fit on {year}. All {k} "
    "folds predict all {n} days and are combined as a Gaussian-mixture moment match "
    "(mu = mean mu_k; sigma^2 = mean sigma_k^2 + var mu_k), so sigma carries both the "
    "within-fold and the between-fold uncertainty (see `sigma_decomposition`)."
)


def evaluate_regime(model_dir: Path, device: torch.device,
                    eval_year: Optional[int] = None,
                    folds: Optional[list[int]] = None,
                    output_dir: Optional[Path] = None,
                    refresh_cache: bool = False, use_cache: bool = True,
                    day_batch: int = 1) -> dict:
    """Score one regime and render its report. Cache-served unless forced otherwise."""
    model_dir = Path(model_dir)
    B = load_or_predict(model_dir, device, eval_year=eval_year, folds=folds,
                        refresh=refresh_cache, use_cache=use_cache, day_batch=day_batch)
    p = B.p
    is_cv = eval_year is None
    out_dir = Path(output_dir) if output_dir else (
        model_dir / ("eval_cv" if is_cv else f"eval_{eval_year}"))

    # CV leaves days uncovered only if a fold subset was requested; score what exists.
    keep = np.asarray(B.day_mask, dtype=bool)
    sl = slice(None) if keep.all() else keep
    dates = np.asarray(B.dates)[sl]

    extra = {
        "eval_regime": "cv_holdout" if is_cv else f"holdout_year_{eval_year}",
        "eval_year": None if is_cv else int(eval_year),
        "prediction_mode": B.prediction_mode,
        "fold_ensemble": not is_cv,
        "eval_days": int(np.asarray(B.errors_c[sl]).shape[0]),
        "n_valid_points": int(np.asarray(B.valid_mask).sum()),
        "cache": {
            "path": str(bundle_cache_path(model_dir, eval_year)),
            "schema_version": CACHE_SCHEMA_VERSION,
            "served_from_cache": bool(getattr(B, "served_from_cache", False)),
        },
    }
    if is_cv:
        extra["fold_blocks"] = {
            str(f): {"start": int(s), "end": int(e),
                     "first_date": str(pd.Timestamp(np.asarray(B.dates)[s]).date()),
                     "last_date": str(pd.Timestamp(np.asarray(B.dates)[e - 1]).date())}
            for f, (s, e) in sorted(B.per_fold_blocks.items())}
        extra["per_fold"] = {
            str(f): {**pooled_scalars(B, slice(s, e)),
                     "checkpoint_epoch": int(B.fold_epochs.get(f, -1))}
            for f, (s, e) in sorted(B.per_fold_blocks.items())}
        period_label = "CV-holdout"
        note = CV_NOTE.format(ys=p.DATA_YEAR_START, ye=p.DATA_YEAR_END,
                              k=len(B.per_fold_blocks), n=extra["eval_days"])
    else:
        if B.sigma_within_c is not None:
            extra["sigma_decomposition"] = {
                "mean_sigma_within_degC": float(np.nanmean(B.sigma_within_c[sl])),
                "mean_sigma_between_degC": float(np.nanmean(B.sigma_between_c[sl])),
                "mean_sigma_total_degC": float(np.nanmean(B.sigmas_c[sl])),
            }
        period_label = f"{eval_year} holdout"
        note = HOLDOUT_NOTE.format(year=eval_year, k=B.n_folds, n=extra["eval_days"])

    result = report.render_evaluation(
        out_dir,
        all_errors=B.errors_c[sl], all_preds=B.preds_c[sl], all_sigmas=B.sigmas_c[sl],
        all_truths=B.truths_c[sl], all_dates=dates, all_ref=B.era5_ref[sl],
        grid_shape=B.grid_shape, target_topo=B.target_topo_t, valid_mask=B.valid_mask,
        variable=p.VARIABLE, encoder=p.ENCODER, grid_mode=B.grid_mode,
        n_folds_evaluated=B.n_folds, model_dir=model_dir,
        period_label=period_label, md_single_day=False,
        md_fold_ids=sorted(B.per_fold_blocks) if is_cv else (),
        extra=extra, regime_note=note,
    )
    o, cal = result["overall"], result["calibration"]
    logger.info("[%s] MAE %.3f | RMSE %.3f | CRPS %.3f | skill %.3f | z-std %.3f | cov90 %.1f%%",
                extra["eval_regime"], o["mae_degC"], o["rmse_degC"], o["crps_degC"],
                o["skill_score"], cal["z_std"], 100 * cal["coverage_90"])
    return result


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", help="Dir with params.json / manifest.json / model_fold_*.")
    src.add_argument("--trial-dir", help="Trial dir; its single variable subdir is used.")
    ap.add_argument("--eval-year", type=int, default=None,
                    help="Holdout year (e.g. 2024). Omit for the 2020-2023 CV holdout.")
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="Evaluate only these folds. Bypasses the prediction cache.")
    ap.add_argument("--device", default=None, help="Override device (cuda/cpu).")
    ap.add_argument("--output-dir", default=None,
                    help="Write the report here instead of <model_dir>/eval_{cv,<year>}.")
    ap.add_argument("--day-batch", type=int, default=1,
                    help="Days per forward pass (results are day-independent).")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Re-run inference and overwrite the cached bundle.")
    ap.add_argument("--no-cache", dest="use_cache", action="store_false",
                    help="Neither read nor write the prediction bundle.")
    ap.add_argument("--years", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.years:
        ap.error(
            "--years is gone. tmax now has exactly two regimes: omit --eval-year for the "
            "2020-2023 cross-validation holdout (each fold scores only the block it was "
            "held out from), or pass --eval-year 2024 for the genuine holdout year. "
            "2020-2023 was never 'unseen' — it is the training span, and evaluating the "
            "full fold ensemble on it is an in-sample score.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout,
    )
    params_mod.configure_renku_cuda()
    device = torch.device(args.device) if args.device else params_mod.select_device()
    model_dir = _resolve_model_dir(args)

    out_dir = Path(args.output_dir) if args.output_dir else (
        model_dir / ("eval_cv" if args.eval_year is None else f"eval_{args.eval_year}"))
    result = evaluate_regime(
        model_dir, device, eval_year=args.eval_year, folds=args.folds,
        output_dir=out_dir, refresh_cache=args.refresh_cache,
        use_cache=args.use_cache, day_batch=args.day_batch)

    o = result["overall"]
    print(f"\n=== {model_dir} | {result['eval_regime']} "
          f"({result['eval_days']} days, {result['prediction_mode']}) ===")
    print(f"  MAE {o['mae_degC']:.3f} | RMSE {o['rmse_degC']:.3f} | "
          f"bias {o['bias_degC']:+.3f} | CRPS {o['crps_degC']:.3f} | "
          f"skill {o['skill_score']:.3f}")
    for fold, fm in sorted(result.get("per_fold", {}).items()):
        print(f"    fold {fold}: {fm['n_days']:>4d} d | MAE {fm['mae_degC']:.3f} | "
              f"CRPS {fm['crps_degC']:.3f} | skill {fm['skill_score']:.3f}")
    print(f"  report: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
