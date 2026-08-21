#!/usr/bin/env python3
"""
Precipitation-specific evaluation for trained convNPClimate models.

``infer.py`` evaluates the Gaussian temperature model (degC, CRPS, coverage).
Precipitation needs different, Vaughan-et-al-2022-style *physical* metrics for
the ``bernoulli_gamma`` distribution.

Given a trained precip model directory (``params.json``, ``metadata.json``,
``model_fold_*``), this reconstructs the exact training data pipeline (reusing
``train.DataBundle`` + ``train.VARIABLE_SPECS``), predicts each fold's held-out
days, and reports, pooled over held-out days x target points (raw mm):

  * wet-day frequency (observed vs. predicted); the prediction is FULLY
    PROBABILISTIC -- the expected exceedance P(Y>=1mm) = rho*(1-GammaCDF(1)),
    with no fixed rho>=0.5 wet/dry classification -- and the relative wet-day
    frequency R01;
  * SDII   - mean wet-day accumulation (observed vs. the analytic wet intensity
    E[Y*1{Y>=1mm}]/P(Y>=1mm));
  * R10    - relative frequency of days > 10 mm (observed vs. sampled);
  * P98    - 98th-percentile wet-day accumulation (observed vs. sampled);
  * MAE / bias of the predicted mean accumulation (mm);
  * the model's own mean held-out NLL (for reference only).

Extreme metrics (R10, P98) are computed by sampling from the predicted
distribution (Bernoulli draw for wet/dry; Gamma sampling for the wet-day
amounts) because the predicted *mean* alone cannot represent the tail.

Example
-------
    python eval_precip.py --model-dir trained_models/<run>/precip
    python eval_precip.py --trial-dir trained_models/<run> --folds 0 1
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from torch.distributions.gamma import Gamma
from scipy import stats as scipy_stats

import params as params_mod
import datasets as ds
import model_factory
import precip_baseline
import predict  # _date_range + build_atmospheric_context (manifest-based year eval)
from convCNP.training.training_elev import get_fold_holdout_indices
from convCNP.training.utils import generate_context_mask
import train  # reuse DataBundle + VARIABLE_SPECS so inference matches training exactly


logger = logging.getLogger("eval_precip")

# A day is counted "wet" (for observed frequency / SDII / P98) at this threshold
# in mm. The model's predicted wet-day frequency is the analytic exceedance
# P(Y>=WET_THRESHOLD_MM) -- no fixed rho threshold (see _bg_exceedance_prob).
WET_THRESHOLD_MM = 1.0
R10_THRESHOLD_MM = 10.0
# A day is "no rain" below this accumulation (mm) -- the complement of the lowest
# threshold row in the spatial wet-day frequency figure. The model's predicted
# dry-day frequency is the analytic 1 - P(Y >= DRY_DAY_THRESHOLD_MM).
DRY_DAY_THRESHOLD_MM = 0.1
# Below this accumulation (mm) an observation is treated as the dry atom for the
# randomized PIT (RhiresD records exact zeros; this absorbs floating-point noise).
DRY_OBS_THRESHOLD_MM = 0.05
# Number of samples drawn per (day, point) for the sampling-based extreme metrics.
N_SAMPLES = 20
# Number of equal-width bins for the PIT histogram (Vaughan et al. 2022, Fig. 10).
PIT_BINS = 20
# Cap the number of (day, point) pairs used for the PIT histogram so the scipy
# Gamma-CDF call stays bounded on the ~23M-point pooled set; subsample is seeded.
PIT_MAX_SAMPLES = 1_000_000
PIT_SEED = 42


# ---------------------------------------------------------------------------
# Per-fold parameter prediction (distribution-agnostic)
# ---------------------------------------------------------------------------
def predict_params_fold(model, context, start, end, dists, elev, seasonal, device):
    """Predict the raw distribution-parameter tensor for one held-out fold.

    Mirrors the forward call used in training/eval (all-ones context mask) and
    returns ``(holdout_days, n_points, n_params)`` on CPU.
    """
    model.eval()
    outs = []
    with torch.no_grad():
        for day in range(start, end):
            day_ctx = context[day:day + 1]
            b, c, x, y = day_ctx.shape
            mask = generate_context_mask(b, c, x, y, device=device)
            day_seasonal = seasonal[day:day + 1] if seasonal is not None else None
            out = model(day_ctx, mask, dists, elev, seasonal=day_seasonal)
            outs.append(out.cpu())
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Distribution-specific samplers -> accumulation samples (mm)
# ---------------------------------------------------------------------------
def _sample_bernoulli_gamma(p, n_samples):
    """Sample accumulations from Bernoulli-Gamma params (N, 3) -> (N, n_samples)."""
    rho, alpha, beta = p[:, 0], p[:, 1], p[:, 2]
    wet = (torch.rand(p.shape[0], n_samples) < rho.unsqueeze(1)).float()
    gamma = Gamma(concentration=alpha.unsqueeze(1).expand(-1, n_samples),
                  rate=beta.unsqueeze(1).expand(-1, n_samples))
    return wet * gamma.sample()


_SAMPLERS = {
    "bernoulli_gamma": _sample_bernoulli_gamma,
    # Same Bernoulli-Gamma output; only the training loss differs (CRPS vs NLL),
    # so the physical-metric sampling is identical.
    "bernoulli_gamma_crps": _sample_bernoulli_gamma,
}


# ---------------------------------------------------------------------------
# Analytic Bernoulli-Gamma functionals (fully probabilistic; no rho threshold)
# ---------------------------------------------------------------------------
# The head is Gamma(concentration=alpha, rate=beta): wet-day mean alpha/beta,
# scale 1/beta. These give the model's exceedance probability and expected wet
# accumulation without ever classifying a day wet/dry via a fixed rho cut.
def _bg_exceedance_prob(params_np, t):
    """P(Y >= t) per element for BG params (..., 3), t in mm.

    rho * (1 - GammaCDF(t; alpha, scale=1/beta)); = rho for t <= 0.
    """
    rho, alpha, beta = params_np[..., 0], params_np[..., 1], params_np[..., 2]
    if t <= 0:
        return rho.astype(float)
    return rho * (1.0 - scipy_stats.gamma.cdf(t, a=alpha, scale=1.0 / beta))


def _bg_expected_wet_accum(params_np, t):
    """E[Y * 1{Y >= t}] per element for BG params (..., 3), t in mm.

    Uses E[G*1{G>=t}] = (alpha/beta)*(1 - GammaCDF(t; alpha+1, scale=1/beta)) for
    the Gamma bulk, weighted by rho:
    rho*(alpha/beta)*(1 - GammaCDF(t; alpha+1, scale=1/beta)); = rho*alpha/beta for t<=0.
    """
    rho, alpha, beta = params_np[..., 0], params_np[..., 1], params_np[..., 2]
    mean_full = rho * (alpha / beta)
    if t <= 0:
        return mean_full
    return mean_full * (1.0 - scipy_stats.gamma.cdf(t, a=alpha + 1.0, scale=1.0 / beta))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_precip_metrics(params_t, truths, distribution, get_value_fn):
    """Compute the precip metric suite over a (days, points, K) param block.

    truths: (days, points) observed accumulation in mm (may contain NaN).
    """
    days, points, _ = params_t.shape
    mean_pred = get_value_fn(params_t).reshape(-1).numpy()  # full BG mean rho*alpha/beta
    obs = truths.reshape(-1)
    valid = ~np.isnan(obs)
    obs = obs[valid]
    mean_pred = mean_pred[valid]

    obs_wet = obs >= WET_THRESHOLD_MM

    # Fully probabilistic predicted wet-day frequency / SDII (no rho>=0.5 rule):
    # wet frequency = mean predicted exceedance P(Y>=1mm); SDII = analytic wet-day
    # intensity E[Y*1{Y>=1mm}] / P(Y>=1mm), both matching the observed 1 mm cut.
    flat_t = params_t.reshape(-1, params_t.shape[-1])[torch.from_numpy(valid)]
    flat = flat_t.numpy()
    exc = _bg_exceedance_prob(flat, WET_THRESHOLD_MM)           # P(Y>=1mm) per obs
    ewa = _bg_expected_wet_accum(flat, WET_THRESHOLD_MM)        # E[Y*1{Y>=1mm}] per obs

    wetfreq_obs = float(np.mean(obs_wet))
    wetfreq_pred = float(np.mean(exc))
    sdii_obs = float(np.mean(obs[obs_wet])) if obs_wet.any() else float("nan")
    sdii_pred = float(np.sum(ewa) / np.sum(exc)) if np.sum(exc) > 0 else float("nan")

    # Dry-day ("no rain") frequency: complement of the 0.1 mm exceedance, same
    # definition as the leading row of the spatial threshold figure.
    dryfreq_obs = float(np.mean(obs < DRY_DAY_THRESHOLD_MM))
    dryfreq_pred = float(np.mean(1.0 - _bg_exceedance_prob(flat, DRY_DAY_THRESHOLD_MM)))

    # Sampling-based extremes.
    sampler = _SAMPLERS[distribution]
    samples = sampler(flat_t, N_SAMPLES).numpy().reshape(-1)  # (n_valid * S,)
    r10_obs = float(np.mean(obs > R10_THRESHOLD_MM))
    r10_pred = float(np.mean(samples > R10_THRESHOLD_MM))
    wet_samples = samples[samples >= WET_THRESHOLD_MM]
    p98_obs = float(np.percentile(obs[obs_wet], 98)) if obs_wet.any() else float("nan")
    p98_pred = float(np.percentile(wet_samples, 98)) if wet_samples.size else float("nan")

    err = mean_pred - obs
    spearman, pearson = _pooled_correlations(obs, mean_pred)
    return {
        "n_obs": int(valid.sum()),
        "wetday_freq_obs": wetfreq_obs,
        "wetday_freq_pred": wetfreq_pred,
        "dryday_freq_obs": dryfreq_obs,
        "dryday_freq_pred": dryfreq_pred,
        "R01_rel_wetday_freq": (wetfreq_pred / wetfreq_obs) if wetfreq_obs > 0 else float("nan"),
        "SDII_obs_mm": sdii_obs,
        "SDII_pred_mm": sdii_pred,
        "SDII_bias_mm": sdii_pred - sdii_obs,
        "R10_freq_obs": r10_obs,
        "R10_freq_pred": r10_pred,
        "P98_obs_mm": p98_obs,
        "P98_pred_mm": p98_pred,
        "P98_abs_bias_mm": abs(p98_pred - p98_obs),
        "mae_mm": float(np.mean(np.abs(err))),
        "bias_mm": float(np.mean(err)),
        # Vaughan Table 2 'sp': Spearman corr (here pooled over day x point; the
        # per-station/per-point time-series version is in metrics["per_point"]).
        "spearman_pooled": spearman,
        "pearson_pooled": pearson,
    }


_CORR_MAX_SAMPLES = 2_000_000


def _pooled_correlations(obs: np.ndarray, pred: np.ndarray):
    """Spearman/Pearson over the pooled (day x point) cloud (subsampled for speed)."""
    n = obs.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    if n > _CORR_MAX_SAMPLES:
        rng = np.random.default_rng(PIT_SEED)
        idx = rng.choice(n, size=_CORR_MAX_SAMPLES, replace=False)
        obs, pred = obs[idx], pred[idx]
    # A degenerate (constant) prediction makes correlation undefined -> nan.
    if np.ptp(pred) == 0 or np.ptp(obs) == 0:
        return float("nan"), float("nan")
    sp = float(scipy_stats.spearmanr(obs, pred).correlation)
    pr = float(scipy_stats.pearsonr(obs, pred)[0])
    return sp, pr


def compare_threshold_vs_probabilistic(params_t, truths):
    """Quantify the effect of dropping the fixed rho>=0.5 threshold.

    Computes each affected metric BOTH ways over the same pooled (day x point)
    valid set: the legacy fixed-threshold rule vs the fully probabilistic
    definition now used by ``compute_precip_metrics``. Returns a dict
    {metric: {"threshold": old, "probabilistic": new, "delta": new-old}} plus the
    observed references, so the evaluation can report and plot the difference.

    Legacy (threshold) definitions reproduced here:
      * mean_pred = (alpha/beta) zeroed where rho<=0.5  (old ``_get_value_precip``)
      * wet freq  = mean(rho>=0.5);  R01 = that / obs wet freq
      * SDII      = mean(mean_pred over days with rho>=0.5)
    Probabilistic definitions:
      * mean_pred = rho*alpha/beta (full mean)
      * wet freq  = mean(P(Y>=1mm));  SDII = E[Y*1{Y>=1mm}]/P(Y>=1mm)
    """
    obs = truths.reshape(-1)
    valid = ~np.isnan(obs)
    obs = obs[valid]
    flat = params_t.reshape(-1, params_t.shape[-1])[torch.from_numpy(valid)].numpy()
    rho, alpha, beta = flat[:, 0], flat[:, 1], flat[:, 2]
    obs_wet = obs >= WET_THRESHOLD_MM
    wetfreq_obs = float(np.mean(obs_wet))
    sdii_obs = float(np.mean(obs[obs_wet])) if obs_wet.any() else float("nan")

    # --- legacy fixed-threshold quantities ---
    pred_wet = rho >= model_factory.DRY_PROBABILITY_THRESHOLD
    mean_thr = (alpha / beta)
    mean_thr = np.where(rho <= model_factory.DRY_PROBABILITY_THRESHOLD, 0.0, mean_thr)
    wetfreq_thr = float(np.mean(pred_wet))
    sdii_thr = float(np.mean(mean_thr[pred_wet])) if pred_wet.any() else float("nan")
    sp_thr, pr_thr = _pooled_correlations(obs, mean_thr)
    err_thr = mean_thr - obs

    # --- probabilistic quantities ---
    mean_prob = rho * (alpha / beta)
    exc = _bg_exceedance_prob(flat, WET_THRESHOLD_MM)
    ewa = _bg_expected_wet_accum(flat, WET_THRESHOLD_MM)
    wetfreq_prob = float(np.mean(exc))
    sdii_prob = float(np.sum(ewa) / np.sum(exc)) if np.sum(exc) > 0 else float("nan")
    sp_prob, pr_prob = _pooled_correlations(obs, mean_prob)
    err_prob = mean_prob - obs

    def _row(old, new):
        return {"threshold": float(old), "probabilistic": float(new),
                "delta": float(new - old)}

    return {
        "_obs_reference": {"wetday_freq_obs": wetfreq_obs, "SDII_obs_mm": sdii_obs},
        "wetday_freq_pred": _row(wetfreq_thr, wetfreq_prob),
        "R01_rel_wetday_freq": _row(
            wetfreq_thr / wetfreq_obs if wetfreq_obs > 0 else float("nan"),
            wetfreq_prob / wetfreq_obs if wetfreq_obs > 0 else float("nan")),
        "SDII_pred_mm": _row(sdii_thr, sdii_prob),
        "mae_mm": _row(np.mean(np.abs(err_thr)), np.mean(np.abs(err_prob))),
        "bias_mm": _row(np.mean(err_thr), np.mean(err_prob)),
        "spearman_pooled": _row(sp_thr, sp_prob),
        "pearson_pooled": _row(pr_thr, pr_prob),
    }


# ---------------------------------------------------------------------------
# Shared inference: reconstruct the pipeline and predict every held-out fold
# ---------------------------------------------------------------------------
def _load_params(model_dir: Path, device: torch.device,
                 precip_glob: Optional[str] = None) -> params_mod.Params:
    """Load a run's Params, optionally repointing the ERA5-Land tp INPUT channel.

    ``precip_glob`` overrides ``ERA5_PRECIP_GLOB`` for inference only; the stored
    ``params.json`` is left alone, so the run keeps its training provenance. This
    is the model's input, NOT the scoring reference -- that is ``--baseline-glob``,
    handled separately in ``compute_baseline_skill``. A no-op for models without
    ``USE_SURFACE_PRECIP``, which never read the field.
    """
    p = params_mod.Params.load_json(model_dir / "params.json")
    p.DEVICE = str(device)
    if precip_glob:
        p = dataclasses.replace(p, ERA5_PRECIP_GLOB=precip_glob)
        p.DEVICE = str(device)
        logger.info("tp INPUT channel overridden: %s", precip_glob)
    return p


def predict_all_folds(model_dir: Path, device: torch.device,
                      folds: Optional[list[int]] = None,
                      precip_glob: Optional[str] = None) -> SimpleNamespace:
    """Reconstruct the training pipeline and predict each fold's held-out days.

    The cross-validation folds partition the time axis, so concatenating each
    fold's held-out predictions reconstructs a per-point series over the whole
    evaluation period. Returns everything both ``evaluate`` (metrics) and
    ``eval_precip_figures`` (plots) need, so inference happens in exactly one
    place and the two agree by construction.
    """
    p = _load_params(model_dir, device, precip_glob)
    distribution = model_factory.resolve_distribution(p)
    if distribution not in _SAMPLERS:
        raise ValueError(
            f"eval_precip only supports precipitation distributions {list(_SAMPLERS)}; "
            f"got {distribution!r} (variable={p.VARIABLE!r}). Use infer.py for tmax."
        )

    spec = train.VARIABLE_SPECS.get(p.VARIABLE)
    if spec is None:
        raise ValueError(f"Variable {p.VARIABLE!r} not enabled in train.VARIABLE_SPECS.")

    logger.info("reconstructing data pipeline (variable=%s, distribution=%s, encoder=%s)...",
                p.VARIABLE, distribution, p.ENCODER)
    data = train.DataBundle(SimpleNamespace(base_params=p, device=device))

    meteoswiss_glob = getattr(p, spec.meteoswiss_glob_attr)
    target_x, target_y, target_topo = ds.prepare_meteoswiss_targets(
        meteoswiss_glob,
        normalization_stats=data.era5_metadata,
        data_var=spec.data_var,
        grid_elevation=data.grid_elevation if p.USE_ELEVATION else None,
        hi_res_elevation=data.hi_res_elevation if p.USE_ELEVATION else None,
        hi_res_tpi=data.hi_res_tpi if p.USE_MTPI else None,
        convert_to_kelvin=spec.convert_to_kelvin,
        normalize_targets=spec.normalize_targets,
        year_start=p.DATA_YEAR_START,
        year_end=p.DATA_YEAR_END,
        device=device,
    )
    target_y = ds.align_target_to_days(target_y, data.time_coords, "CV context")
    dists = ds.calculate_dists_meteoswiss(data.dists_metadata, target_x, device=device)
    target_tensor = torch.from_numpy(target_y.values.astype(np.float32)).to(device)

    n_times = data.context.shape[0]
    if target_tensor.shape[0] != n_times:
        raise RuntimeError(
            f"target days ({target_tensor.shape[0]}) != context days ({n_times}).")
    n_points = int(target_tensor.shape[1])
    n_folds = p.N_FOLDS
    fold_ids = folds if folds is not None else list(range(n_folds))
    channel_groups = data.channel_groups if p.ENCODER != "flat" else None
    get_value_fn = model_factory.LIKELIHOODS[distribution].get_value_fn
    n_params = model_factory.LIKELIHOODS[distribution].n_params

    # Full (time, point, K) param block; days not covered by an evaluated fold
    # stay flagged off in day_mask and are excluded everywhere downstream.
    params_full = torch.full((n_times, n_points, n_params), float("nan"))
    day_mask = np.zeros(n_times, dtype=bool)
    per_fold_blocks: dict[int, tuple] = {}
    fold_epochs: dict[int, int] = {}
    for fold in fold_ids:
        ckpt = model_dir / f"model_fold_{fold}"
        if not ckpt.exists():
            logger.warning("checkpoint %s missing; skipping fold %d", ckpt, fold)
            continue
        model, epoch = model_factory.load_model_checkpoint(
            ckpt, p, device, channel_groups=channel_groups
        )
        start, end = get_fold_holdout_indices(fold, n_folds, n_times)
        logger.info("fold %d: predicting holdout days [%d, %d) (ckpt epoch %d)...",
                    fold, start, end, epoch)
        params_t = predict_params_fold(
            model, data.context, start, end, dists, target_topo,
            data.seasonal_features, device,
        )  # (end-start, n_points, K) on CPU
        params_full[start:end] = params_t
        day_mask[start:end] = True
        per_fold_blocks[fold] = (start, end)
        fold_epochs[fold] = int(epoch)

    if not per_fold_blocks:
        raise RuntimeError("No fold checkpoints found to evaluate.")

    # Real-world target coordinates (un-normalise target_x back to degrees) and
    # the (N, E) grid shape, so figures can lay points on a map.
    m = data.era5_metadata
    lat_norm = np.asarray(target_x.sel(coord="lat").values, dtype=float)
    lon_norm = np.asarray(target_x.sel(coord="lon").values, dtype=float)
    target_lat = lat_norm * (m.lat_max - m.lat_min) + m.lat_min
    target_lon = lon_norm * (m.lon_max - m.lon_min) + m.lon_min
    grid_shape = _infer_grid_shape(target_y, n_points)
    time_dates = np.asarray(getattr(data, "time_coords", np.arange(n_times)))

    return SimpleNamespace(
        params=p, distribution=distribution, get_value_fn=get_value_fn,
        encoder=p.ENCODER, grid_mode=data.grid_mode,
        params_full=params_full, day_mask=day_mask,
        truth_full=target_tensor.cpu().numpy(),
        per_fold_blocks=per_fold_blocks, fold_epochs=fold_epochs,
        target_lat=target_lat, target_lon=target_lon, grid_shape=grid_shape,
        time_dates=time_dates, n_target_points=n_points,
    )


def _infer_grid_shape(target_y, n_points: int):
    """Recover the (N, E) MeteoSwiss grid shape from the stacked point index."""
    try:
        n_n = int(np.unique(np.asarray(target_y["N"].values)).size)
        n_e = int(np.unique(np.asarray(target_y["E"].values)).size)
        if n_n * n_e == n_points:
            return (n_n, n_e)
    except Exception as exc:  # pragma: no cover - fall back to scatter plots
        logger.warning("could not infer (N,E) grid shape: %r; figures will scatter", exc)
    return None


def predict_holdout_year(model_dir: Path, device: torch.device, year: int,
                         folds: Optional[list[int]] = None,
                         precip_glob: Optional[str] = None) -> SimpleNamespace:
    """Ensemble every available fold over a genuinely unseen YEAR (e.g. 2024).

    Reuses the model's TRAINING-time normalization -- built by ``DataBundle`` on
    the training years and frozen into a manifest -- and applies it to ``year``
    via ``predict.build_atmospheric_context``, so the holdout inputs are NOT
    renormalized on the holdout year's own statistics. Every fold predicts every
    day (there is no CV holdout split for an unseen year) and the fold precip
    params are averaged (ensemble). Returns a bundle shaped exactly like
    ``predict_all_folds`` so all metric/figure code is reused unchanged.
    """
    p = _load_params(model_dir, device, precip_glob)
    distribution = model_factory.resolve_distribution(p)
    if distribution not in _SAMPLERS:
        raise ValueError(
            f"eval_precip only supports precipitation distributions {list(_SAMPLERS)}; "
            f"got {distribution!r}.")
    if getattr(p, "USE_ATMOSPHERIC", False) and not getattr(p, "ATMOS_NATIVE_GRID", False):
        raise NotImplementedError(
            "predict_holdout_year supports native-grid atmospheric or surface precip "
            "models; the regridded-atmospheric variant is not reconstructible from the "
            "manifest alone.")
    spec = train.VARIABLE_SPECS[p.VARIABLE]

    logger.info("reconstructing training-time (%s-%s) normalization for the %d holdout...",
                p.DATA_YEAR_START, p.DATA_YEAR_END, year)
    data = train.DataBundle(SimpleNamespace(base_params=p, device=device))
    manifest = data.normalization_manifest()

    dates = predict._date_range(f"{year}-01-01", f"{year}-12-31")
    # Score the days the tp input actually covers (a calendar year is typically
    # short its 31 Dec), the same trim training applies. No-op for other models.
    dates = predict.precip_covered_dates(p, dates, label=f" of {year}")
    logger.info("building %d-day ERA5 context for %d (training normalization)...", len(dates), year)
    if p.USE_ATMOSPHERIC and p.ATMOS_NATIVE_GRID:
        context = predict.build_atmospheric_context(manifest, p, dates, device)
    else:
        context = predict.build_surface_context(manifest, p, dates, device)
    T = int(context.shape[0])
    seasonal = (ds.compute_seasonal_features(dates.values.astype("datetime64[ns]"), device=device)
                if p.SEASONAL_FEATURES else None)

    meteoswiss_glob = getattr(p, spec.meteoswiss_glob_attr)
    target_x, target_y, target_topo = ds.prepare_meteoswiss_targets(
        meteoswiss_glob,
        normalization_stats=data.era5_metadata,     # training stats, NOT year stats
        data_var=spec.data_var,
        grid_elevation=data.grid_elevation if p.USE_ELEVATION else None,
        hi_res_elevation=data.hi_res_elevation if p.USE_ELEVATION else None,
        hi_res_tpi=data.hi_res_tpi if p.USE_MTPI else None,
        convert_to_kelvin=spec.convert_to_kelvin,
        normalize_targets=spec.normalize_targets,    # precip -> raw mm
        year_start=year, year_end=year, device=device,
    )
    # The context can be a trimmed subset of the calendar year (see the
    # precip-coverage trim above), so select the target on its days by date.
    target_y = ds.align_target_to_days(target_y, dates, f"{year} context")
    dists = ds.calculate_dists_meteoswiss(data.dists_metadata, target_x, device=device)
    target_tensor = torch.from_numpy(target_y.values.astype(np.float32))
    n_points = int(target_tensor.shape[1])
    if target_tensor.shape[0] != T:
        raise RuntimeError(
            f"target days ({target_tensor.shape[0]}) != ERA5 context days ({T}) for {year}; "
            "check for missing dates in either source.")

    channel_groups = data.channel_groups if p.ENCODER != "flat" else None
    get_value_fn = model_factory.LIKELIHOODS[distribution].get_value_fn
    fold_ids = folds if folds is not None else list(range(p.N_FOLDS))
    fold_params, fold_epochs = [], {}
    for fold in fold_ids:
        ckpt = model_dir / f"model_fold_{fold}"
        if not ckpt.exists():
            logger.warning("checkpoint %s missing; skipping fold %d", ckpt, fold)
            continue
        model, epoch = model_factory.load_model_checkpoint(
            ckpt, p, device, channel_groups=channel_groups)
        logger.info("fold %d (epoch %d): predicting %d days of %d...", fold, epoch, T, year)
        params_t = predict_params_fold(
            model, context, 0, T, dists, target_topo, seasonal, device)
        fold_params.append(params_t.numpy())
        fold_epochs[fold] = int(epoch)
    if not fold_params:
        raise RuntimeError("No fold checkpoints found to evaluate.")
    params_full = torch.from_numpy(
        np.mean(np.stack(fold_params), axis=0).astype(np.float32))  # (T, P, K) ensemble

    m = data.era5_metadata
    lat_norm = np.asarray(target_x.sel(coord="lat").values, dtype=float)
    lon_norm = np.asarray(target_x.sel(coord="lon").values, dtype=float)
    target_lat = lat_norm * (m.lat_max - m.lat_min) + m.lat_min
    target_lon = lon_norm * (m.lon_max - m.lon_min) + m.lon_min
    grid_shape = _infer_grid_shape(target_y, n_points)

    return SimpleNamespace(
        params=p, distribution=distribution, get_value_fn=get_value_fn,
        encoder=p.ENCODER, grid_mode=data.grid_mode,
        params_full=params_full, day_mask=np.ones(T, dtype=bool),
        truth_full=target_tensor.cpu().numpy(),
        per_fold_blocks={}, fold_epochs=fold_epochs,
        target_lat=target_lat, target_lon=target_lon, grid_shape=grid_shape,
        time_dates=np.asarray(dates.values), n_target_points=n_points,
        eval_year=year,
    )


# ---------------------------------------------------------------------------
# Prediction-bundle cache (inference state persisted to disk)
# ---------------------------------------------------------------------------
CACHE_SCHEMA_VERSION = 1
CACHE_DIRNAME = "pred_cache"  # gitignored: ~2.3 GB (CV) / ~0.6 GB (2024) per model


def bundle_cache_path(model_dir: Path, eval_year: Optional[int] = None) -> Path:
    name = "cv.npz" if eval_year is None else f"holdout_{eval_year}.npz"
    return Path(model_dir) / CACHE_DIRNAME / name


def save_bundle(B: SimpleNamespace, path: Path) -> None:
    """Persist a prediction bundle as a pickle-free npz (atomic write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fold_ids = sorted(B.per_fold_blocks)
    epoch_ids = sorted(B.fold_epochs)
    eval_year = getattr(B, "eval_year", None)
    arrays = dict(
        schema_version=np.int64(CACHE_SCHEMA_VERSION),
        params_full=B.params_full.numpy().astype(np.float32, copy=False),
        day_mask=np.asarray(B.day_mask, dtype=bool),
        truth_full=np.asarray(B.truth_full, dtype=np.float32),
        target_lat=np.asarray(B.target_lat, dtype=np.float64),
        target_lon=np.asarray(B.target_lon, dtype=np.float64),
        grid_shape=np.asarray(B.grid_shape if B.grid_shape is not None else (-1, -1),
                              dtype=np.int64),
        time_dates=np.asarray(B.time_dates),
        fold_ids=np.asarray(fold_ids, dtype=np.int64),
        fold_starts=np.asarray([B.per_fold_blocks[f][0] for f in fold_ids], dtype=np.int64),
        fold_ends=np.asarray([B.per_fold_blocks[f][1] for f in fold_ids], dtype=np.int64),
        epoch_fold_ids=np.asarray(epoch_ids, dtype=np.int64),
        epoch_values=np.asarray([B.fold_epochs[f] for f in epoch_ids], dtype=np.int64),
        distribution=np.asarray(B.distribution),
        encoder=np.asarray(B.encoder),
        grid_mode=np.asarray(str(B.grid_mode)),
        eval_year=np.int64(-1 if eval_year is None else eval_year),
        n_target_points=np.int64(B.n_target_points),
    )
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)
    logger.info("saved prediction bundle cache: %s (%.2f GB)",
                path, path.stat().st_size / 1e9)


def load_bundle(path: Path, model_dir: Path) -> Optional[SimpleNamespace]:
    """Rebuild a bundle from cache without touching checkpoints; None if stale."""
    with np.load(path) as z:
        if int(z["schema_version"]) != CACHE_SCHEMA_VERSION:
            logger.warning("cache %s: schema %d != %d; recomputing",
                           path, int(z["schema_version"]), CACHE_SCHEMA_VERSION)
            return None
        p = params_mod.Params.load_json(Path(model_dir) / "params.json")
        distribution = str(z["distribution"][()])
        if distribution != model_factory.resolve_distribution(p):
            logger.warning("cache %s: distribution %r != params.json %r; recomputing",
                           path, distribution, model_factory.resolve_distribution(p))
            return None
        gs = z["grid_shape"]
        B = SimpleNamespace(
            params=p, distribution=distribution,
            get_value_fn=model_factory.LIKELIHOODS[distribution].get_value_fn,
            encoder=str(z["encoder"][()]), grid_mode=str(z["grid_mode"][()]),
            params_full=torch.from_numpy(z["params_full"]),
            day_mask=np.asarray(z["day_mask"], dtype=bool),
            truth_full=z["truth_full"],
            per_fold_blocks={int(f): (int(s), int(e)) for f, s, e in
                             zip(z["fold_ids"], z["fold_starts"], z["fold_ends"])},
            fold_epochs={int(f): int(v) for f, v in
                         zip(z["epoch_fold_ids"], z["epoch_values"])},
            target_lat=z["target_lat"], target_lon=z["target_lon"],
            grid_shape=None if int(gs[0]) < 0 else (int(gs[0]), int(gs[1])),
            time_dates=z["time_dates"],
            n_target_points=int(z["n_target_points"]),
        )
        eval_year = int(z["eval_year"])
        if eval_year >= 0:
            B.eval_year = eval_year  # CV bundles must NOT carry this attribute
        return B


def load_or_predict(model_dir: Path, device: torch.device,
                    eval_year: Optional[int] = None,
                    folds: Optional[list[int]] = None,
                    refresh: bool = False, use_cache: bool = True,
                    precip_glob: Optional[str] = None) -> SimpleNamespace:
    """Return a prediction bundle, serving the persisted inference state if present.

    Every downstream evaluation (metrics, figures, skill timeseries) comes
    through here, so GPU inference happens at most once per (model, regime).
    An explicit fold subset bypasses the cache — it would not represent the
    full run and must never be persisted as if it did.

    A ``precip_glob`` override also bypasses the cache: the bundle path keys only
    on the regime, so a cached bundle built from the default tp field would
    otherwise be served for a run that asked for a different input.
    """
    model_dir = Path(model_dir)
    cache = bundle_cache_path(model_dir, eval_year)
    if folds is not None:
        logger.info("explicit fold subset requested: bypassing prediction cache")
    elif precip_glob:
        logger.info("tp input override requested: bypassing prediction cache")
    elif use_cache and not refresh and cache.exists():
        B = load_bundle(cache, model_dir)
        if B is not None:
            logger.info("reusing cached inference: %s", cache)
            return B
    if eval_year is not None:
        B = predict_holdout_year(model_dir, device, eval_year, folds=folds,
                                 precip_glob=precip_glob)
    else:
        B = predict_all_folds(model_dir, device, folds=folds,
                              precip_glob=precip_glob)
    # Never persist a bundle built from a non-default tp input: the cache path
    # cannot express which field produced it, so a later default run would
    # silently reuse it.
    if use_cache and folds is None and not precip_glob:
        save_bundle(B, cache)
    return B


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------
def evaluate(model_dir: Path, device: torch.device, folds: Optional[list[int]] = None,
             baseline_glob: Optional[str] = None, eval_year: Optional[int] = None,
             refresh_cache: bool = False, use_cache: bool = True,
             precip_glob: Optional[str] = None) -> dict:
    B = load_or_predict(model_dir, device, eval_year=eval_year, folds=folds,
                        refresh=refresh_cache, use_cache=use_cache,
                        precip_glob=precip_glob)
    distribution, get_value_fn = B.distribution, B.get_value_fn

    # Per-fold pooled metrics (CV holdout only; a holdout year is fold-ensembled,
    # so per_fold_blocks is empty and per_fold stays {}).
    per_fold = {}
    for fold, (start, end) in B.per_fold_blocks.items():
        params_t = B.params_full[start:end]
        truths = B.truth_full[start:end]
        per_fold[fold] = compute_precip_metrics(params_t, truths, distribution, get_value_fn)
        per_fold[fold]["checkpoint_epoch"] = B.fold_epochs[fold]

    eval_params = B.params_full[B.day_mask]
    eval_truth = B.truth_full[B.day_mask]
    overall = compute_precip_metrics(eval_params, eval_truth, distribution, get_value_fn)

    # New Vaughan-aligned blocks.
    per_point = per_point_summary(B)
    pit = compute_pit(B)
    # Attempt the baseline for every regime; compute_baseline_skill keys on the
    # holdout year and degrades gracefully to available=False when the tp-<year>.nc
    # field is not present yet (so 2024 skill turns on automatically once it lands).
    baseline = compute_baseline_skill(B, overall, baseline_glob=baseline_glob)

    # Effect of the fully-probabilistic refactor (old rho>=0.5 vs new definitions).
    threshold_vs_prob = compare_threshold_vs_probabilistic(eval_params, eval_truth)

    return {
        "model_dir": str(model_dir),
        "variable": B.params.VARIABLE,
        "distribution": distribution,
        "encoder": B.encoder,
        "grid_mode": B.grid_mode,
        "eval_regime": (f"holdout_year_{eval_year}" if eval_year is not None
                        else "cv_holdout"),
        "eval_year": eval_year,
        # What this result was actually computed from: the tp field fed to the
        # model as an INPUT (None = the run's own params.json), and the tp field
        # used as the scoring REFERENCE (None = precip_baseline's default).
        "precip_input_glob": precip_glob,
        "baseline_glob": baseline_glob,
        "n_folds_evaluated": len(per_fold) if eval_year is None else int(len(B.fold_epochs)),
        "fold_ensemble": eval_year is not None,
        "n_target_points": B.n_target_points,
        "wet_threshold_mm": WET_THRESHOLD_MM,
        "dry_day_threshold_mm": DRY_DAY_THRESHOLD_MM,
        "n_samples": N_SAMPLES,
        "nll_comparable_across_distributions": False,
        "overall": overall,
        "per_fold": per_fold,
        "per_point": per_point,
        "pit": pit,
        "baseline": baseline,
        "threshold_vs_probabilistic": threshold_vs_prob,
    }


# ---------------------------------------------------------------------------
# Per-point (per-"station") metrics  -> Vaughan-style median + IQR across points
# ---------------------------------------------------------------------------
def _pearson_along_time(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation along axis 0 (no NaNs); (D,P)->(P,)."""
    am = a - a.mean(axis=0)
    bm = b - b.mean(axis=0)
    num = (am * bm).sum(axis=0)
    den = np.sqrt((am ** 2).sum(axis=0) * (bm ** 2).sum(axis=0))
    out = np.full(a.shape[1], np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def _spearman_along_time(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Column-wise Spearman correlation along axis 0 (rank then Pearson)."""
    try:
        ro = scipy_stats.rankdata(obs, axis=0)
        rp = scipy_stats.rankdata(pred, axis=0)
    except TypeError:  # older scipy without axis= support
        ro = np.apply_along_axis(scipy_stats.rankdata, 0, obs)
        rp = np.apply_along_axis(scipy_stats.rankdata, 0, pred)
    return _pearson_along_time(ro, rp)


def per_point_metrics(B: SimpleNamespace) -> dict:
    """Per-target-point metrics over the reconstructed held-out series.

    Mirrors Vaughan's per-station evaluation: each MeteoSwiss grid point gets
    its own mb/MAE/Spearman/Pearson/wet-day/SDII over the held-out days. The
    target domain mask is static (ocean / outside-Switzerland points are NaN on
    every day), so a point is "valid" when finite on all evaluated days.
    Returns arrays of length n_points (NaN at invalid points) + a validity mask.
    """
    eval_params = B.params_full[B.day_mask]          # torch (D, P, K)
    obs = B.truth_full[B.day_mask]                   # np   (D, P)
    mean_pred = B.get_value_fn(eval_params).numpy()  # (D, P) full BG mean
    P = obs.shape[1]
    point_valid = np.isfinite(obs).all(axis=0)

    keys = ["mean_obs", "mean_pred", "bias", "mae", "spearman", "pearson",
            "wetfreq_obs", "wetfreq_pred", "r01", "sdii_obs", "sdii_pred", "sdii_bias"]
    out = {k: np.full(P, np.nan) for k in keys}
    if not point_valid.any():
        out["point_valid"] = point_valid
        return out

    o = obs[:, point_valid]
    pr = mean_pred[:, point_valid]
    obs_wet = o >= WET_THRESHOLD_MM
    # Fully probabilistic per-point wet-day freq / SDII (no rho>=0.5 rule):
    # exceedance P(Y>=1mm) and expected wet accumulation E[Y*1{Y>=1mm}] per day.
    pv = eval_params[:, torch.from_numpy(point_valid), :].numpy()  # (D, Pv, K)
    exc = _bg_exceedance_prob(pv, WET_THRESHOLD_MM)                # (D, Pv)
    ewa = _bg_expected_wet_accum(pv, WET_THRESHOLD_MM)            # (D, Pv)

    out["mean_obs"][point_valid] = o.mean(axis=0)
    out["mean_pred"][point_valid] = pr.mean(axis=0)
    out["bias"][point_valid] = pr.mean(axis=0) - o.mean(axis=0)
    out["mae"][point_valid] = np.abs(pr - o).mean(axis=0)
    out["spearman"][point_valid] = _spearman_along_time(o, pr)
    out["pearson"][point_valid] = _pearson_along_time(o, pr)
    wfo = obs_wet.mean(axis=0)
    wfp = exc.mean(axis=0)                                         # expected wet-day freq
    out["wetfreq_obs"][point_valid] = wfo
    out["wetfreq_pred"][point_valid] = wfp
    with np.errstate(divide="ignore", invalid="ignore"):
        out["r01"][point_valid] = np.where(wfo > 0, wfp / wfo, np.nan)
        oc = obs_wet.sum(axis=0)
        exc_sum = exc.sum(axis=0)
        sdii_o = np.where(oc > 0, (o * obs_wet).sum(axis=0) / oc, np.nan)
        sdii_p = np.where(exc_sum > 0, ewa.sum(axis=0) / exc_sum, np.nan)
    out["sdii_obs"][point_valid] = sdii_o
    out["sdii_pred"][point_valid] = sdii_p
    out["sdii_bias"][point_valid] = sdii_p - sdii_o
    out["point_valid"] = point_valid
    return out


def per_point_summary(B: SimpleNamespace) -> dict:
    """Median + IQR (q25/q75) across points for each per-point metric (JSON)."""
    pp = per_point_metrics(B)
    summary = {"n_valid_points": int(np.count_nonzero(pp["point_valid"]))}
    for key, arr in pp.items():
        if key == "point_valid":
            continue
        vals = arr[np.isfinite(arr)]
        if vals.size:
            summary[key] = {
                "median": float(np.median(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
            }
        else:
            summary[key] = {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return summary


# ---------------------------------------------------------------------------
# Randomized PIT calibration (Vaughan et al. 2022, Fig. 10)
# ---------------------------------------------------------------------------
def compute_pit(B: SimpleNamespace) -> dict:
    """Randomized probability-integral-transform histogram for Bernoulli-Gamma.

    The predictive law is mixed: a point mass (1-rho) at zero plus a Gamma bulk
    with weight rho. The randomized PIT (Gneiting et al., 2007) handles the dry
    atom: dry observations map to U*(1-rho); wet observations to
    (1-rho) + rho*GammaCDF(y; alpha, 1/beta). A calibrated model gives a uniform
    histogram; mass piled near 0 indicates over-prediction of accumulation.
    """
    if B.distribution not in ("bernoulli_gamma", "bernoulli_gamma_crps"):
        return {"available": False,
                "note": f"randomized PIT implemented for the Bernoulli-Gamma family only "
                        f"(distribution={B.distribution})"}
    eval_params = B.params_full[B.day_mask]
    obs = B.truth_full[B.day_mask].reshape(-1)
    flat = eval_params.reshape(-1, eval_params.shape[-1]).numpy()
    valid = np.isfinite(obs)
    obs = obs[valid]
    flat = flat[valid]
    n = obs.shape[0]
    if n == 0:
        return {"available": False, "note": "no valid observations"}
    if n > PIT_MAX_SAMPLES:
        rng = np.random.default_rng(PIT_SEED)
        idx = rng.choice(n, size=PIT_MAX_SAMPLES, replace=False)
        obs, flat = obs[idx], flat[idx]
    rng = np.random.default_rng(PIT_SEED)
    rho, alpha, beta = flat[:, 0], flat[:, 1], flat[:, 2]
    pit = np.empty(obs.shape[0])
    dry = obs <= DRY_OBS_THRESHOLD_MM
    pit[dry] = rng.uniform(size=int(dry.sum())) * (1.0 - rho[dry])
    wet = ~dry
    cdf = scipy_stats.gamma.cdf(obs[wet], a=alpha[wet], scale=1.0 / beta[wet])
    pit[wet] = (1.0 - rho[wet]) + rho[wet] * cdf
    density, edges = np.histogram(pit, bins=PIT_BINS, range=(0.0, 1.0), density=True)
    return {
        "available": True,
        "bins": PIT_BINS,
        "density": density.tolist(),
        "bin_edges": edges.tolist(),
        "n_used": int(obs.shape[0]),
        "mean_pit": float(np.mean(pit)),
        "frac_pit_below_0.1": float(np.mean(pit < 0.1)),
        "frac_pit_above_0.9": float(np.mean(pit > 0.9)),
        "dry_obs_fraction": float(np.mean(dry)),
    }


# ---------------------------------------------------------------------------
# Bilinear-ERA5 baseline + skill score (Vaughan-style baseline comparison)
# ---------------------------------------------------------------------------
def compute_baseline_skill(B: SimpleNamespace, overall: dict,
                           baseline_glob: Optional[str] = None) -> dict:
    """Compare the model against a bilinear-ERA5-precip baseline.

    Skill = 1 - MAE_model / MAE_baseline on the shared valid mask (matching the
    tmax CRPS-skill convention in the report's methodology chapter, where for
    a deterministic baseline CRPS reduces to MAE).
    """
    # Key the baseline on the evaluated year: for a holdout-year run use that year
    # (so tp-<year>.nc is picked up automatically the moment it is available),
    # otherwise the training start year.
    base_year = int(getattr(B, "eval_year", None) or B.params.DATA_YEAR_START)
    base, source = precip_baseline.bilinear_era5_precip(
        B.target_lat, B.target_lon, B.time_dates,
        year=base_year, glob=baseline_glob,
    )
    if base is None:
        return {"available": False,
                "note": f"no ERA5 precip baseline field for {base_year}; skipped "
                        "(will be computed once the tp field is present)"}

    base_eval = base[B.day_mask]                          # (D, P) mm
    obs = B.truth_full[B.day_mask]                        # (D, P)
    pred = B.get_value_fn(B.params_full[B.day_mask]).numpy()
    mask = np.isfinite(obs) & np.isfinite(base_eval)
    if not mask.any():
        return {"available": False, "note": "baseline/obs masks do not overlap"}

    o = obs[mask]
    b = base_eval[mask]
    pm = pred[mask]
    base_mae = float(np.mean(np.abs(b - o)))
    model_mae = float(np.mean(np.abs(pm - o)))
    bwet = b >= WET_THRESHOLD_MM
    owet = o >= WET_THRESHOLD_MM
    wf_o = float(np.mean(owet))
    # The baseline's own correlation with the observations, through the same subsampled
    # estimator the model uses (_pooled_correlations under PIT_SEED), so the two are
    # directly comparable and one can be drawn as a reference line for the other. No model
    # term enters it -- only the interpolated field and the truth.
    base_sp, base_pr = _pooled_correlations(o, b)
    return {
        "available": True,
        "source": source,
        "n_obs": int(mask.sum()),
        "baseline_mae_mm": base_mae,
        "baseline_bias_mm": float(np.mean(b - o)),
        "baseline_R01_rel_wetday_freq": (float(np.mean(bwet)) / wf_o) if wf_o > 0 else float("nan"),
        "baseline_SDII_mm": float(np.mean(b[bwet])) if bwet.any() else float("nan"),
        "baseline_spearman_pooled": base_sp,
        "baseline_pearson_pooled": base_pr,
        "model_mae_mm_common_mask": model_mae,
        "skill_mae": (1.0 - model_mae / base_mae) if base_mae > 0 else float("nan"),
    }


def print_report(result: dict):
    o = result["overall"]
    print()
    print(f"=== Precip evaluation: {result['model_dir']} ===")
    regime = result.get("eval_regime", "cv_holdout")
    ens = " (fold-ensemble)" if result.get("fold_ensemble") else ""
    print(f"regime={regime}{ens} | distribution={result['distribution']} | "
          f"encoder={result['encoder']} | grid={result['grid_mode']} | "
          f"folds={result['n_folds_evaluated']} | points={result['n_target_points']:,}")
    print("Predicted wet-day freq / SDII are FULLY PROBABILISTIC (no rho>=0.5 rule).")
    print("NOTE: NLL is not comparable across distributions; compare on the metrics below.")
    print()
    rows = [
        ("Wet-day freq (obs)", "wetday_freq_obs"),
        ("Wet-day freq (pred)", "wetday_freq_pred"),
        ("R01 relative wet-day freq", "R01_rel_wetday_freq"),
        ("SDII obs (mm)", "SDII_obs_mm"),
        ("SDII pred (mm)", "SDII_pred_mm"),
        ("SDII bias (mm)", "SDII_bias_mm"),
        ("R10 freq obs", "R10_freq_obs"),
        ("R10 freq pred", "R10_freq_pred"),
        ("P98 obs (mm)", "P98_obs_mm"),
        ("P98 pred (mm)", "P98_pred_mm"),
        ("P98 abs bias (mm)", "P98_abs_bias_mm"),
        ("MAE mean (mm)", "mae_mm"),
        ("Bias mean (mm)", "bias_mm"),
        ("Spearman (pooled)", "spearman_pooled"),
        ("Pearson (pooled)", "pearson_pooled"),
    ]
    print(f"{'Metric':<30} {'Overall':>12}")
    print("-" * 44)
    for label, key in rows:
        if key in o and o[key] is not None:
            print(f"{label:<30} {o[key]:>12.4f}")
    print()

    pp = result.get("per_point")
    if pp:
        print(f"Per-point (median [q25, q75] across {pp.get('n_valid_points', 0):,} points):")
        for label, key in [("MAE (mm)", "mae"), ("Bias (mm)", "bias"),
                           ("Spearman", "spearman"), ("R01", "r01"),
                           ("SDII bias (mm)", "sdii_bias")]:
            s = pp.get(key)
            if s:
                print(f"  {label:<16} {s['median']:>8.3f}  [{s['q25']:.3f}, {s['q75']:.3f}]")
        print()

    base = result.get("baseline")
    if base and base.get("available"):
        print(f"Baseline ({base['source']}): MAE {base['baseline_mae_mm']:.3f} mm | "
              f"model MAE {base['model_mae_mm_common_mask']:.3f} mm | "
              f"skill(MAE) {base['skill_mae']:.3f}")
    pit = result.get("pit")
    if pit and pit.get("available"):
        print(f"PIT: mean {pit['mean_pit']:.3f} (ideal 0.5) | "
              f"frac<0.1 {pit['frac_pit_below_0.1']:.3f} | frac>0.9 {pit['frac_pit_above_0.9']:.3f}")
    print()

    tvp = result.get("threshold_vs_probabilistic")
    if tvp:
        ref = tvp.get("_obs_reference", {})
        print("Effect of removing the fixed rho>=0.5 threshold "
              "(threshold -> probabilistic):")
        print(f"  {'Metric':<22} {'threshold':>11} {'probabilistic':>14} {'delta':>10}")
        print("  " + "-" * 59)
        for label, key in [("Wet-day freq (pred)", "wetday_freq_pred"),
                           ("R01", "R01_rel_wetday_freq"),
                           ("SDII pred (mm)", "SDII_pred_mm"),
                           ("MAE (mm)", "mae_mm"), ("Bias (mm)", "bias_mm"),
                           ("Spearman", "spearman_pooled"), ("Pearson", "pearson_pooled")]:
            r = tvp.get(key)
            if r:
                print(f"  {label:<22} {r['threshold']:>11.4f} "
                      f"{r['probabilistic']:>14.4f} {r['delta']:>+10.4f}")
        if ref:
            print(f"  (obs wet-day freq {ref.get('wetday_freq_obs', float('nan')):.4f}, "
                  f"obs SDII {ref.get('SDII_obs_mm', float('nan')):.3f} mm)")
        print()


def _resolve_model_dir(args) -> Path:
    if args.model_dir:
        return Path(args.model_dir)
    trial = Path(args.trial_dir)
    candidates = [d for d in trial.iterdir() if (d / "params.json").exists()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"No <variable>/params.json found under {trial}")
    raise SystemExit(
        f"Multiple variables under {trial}: {[c.name for c in candidates]}; "
        "pass --model-dir to choose one."
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", help="Dir with params.json/metadata.json/model_fold_*.")
    src.add_argument("--trial-dir", help="Trial dir; its single variable subdir is used.")
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="Subset of folds to evaluate.")
    ap.add_argument("--device", default=None, help="Override device (e.g. 'cuda', 'cpu').")
    ap.add_argument("--output", default=None, help="JSON output path (default <model_dir>/eval_precip_metrics.json).")
    ap.add_argument("--baseline-glob", default=None,
                    help="Override glob for the ERA5 daily precip baseline field "
                         "(default datasets/ERA5_Land/precipitation/tp-<year>.nc). "
                         "This is the SCORING reference only.")
    ap.add_argument("--precip-glob", default=None,
                    help="Override the ERA5-Land tp field fed to the model as an INPUT "
                         "channel (ERA5_PRECIP_GLOB), for USE_SURFACE_PRECIP runs. The "
                         "stored params.json is not modified, and the prediction cache "
                         "is bypassed so a default-input bundle is never reused.")
    ap.add_argument("--eval-year", type=int, default=None,
                    help="Evaluate a genuinely unseen holdout YEAR (e.g. 2024): fold-ensemble "
                         "with training-time normalization. Writes eval_precip_metrics_<year>.json.")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Re-run inference even if a pred_cache bundle exists, then overwrite it.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Neither read nor write the pred_cache prediction bundle.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    params_mod.configure_renku_cuda()
    device = torch.device(args.device) if args.device else params_mod.select_device()

    model_dir = _resolve_model_dir(args)
    result = evaluate(model_dir, device, folds=args.folds, baseline_glob=args.baseline_glob,
                      eval_year=args.eval_year, refresh_cache=args.refresh_cache,
                      use_cache=not args.no_cache, precip_glob=args.precip_glob)

    default_name = ("eval_precip_metrics.json" if args.eval_year is None
                    else f"eval_precip_metrics_{args.eval_year}.json")
    out_path = Path(args.output) if args.output else (model_dir / default_name)
    out_path.write_text(json.dumps(result, indent=2))
    print_report(result)
    print(f"Wrote metrics: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
