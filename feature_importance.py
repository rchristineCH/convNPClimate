#!/usr/bin/env python3
"""
Feature-importance study for the *atmospheric* tmax ConvCNP.

Quantifies how much each ERA5 input channel (and named physical group of channels)
contributes to downscaling skill, using three complementary, self-implemented methods:

  * PFI  -- Permutation Feature Importance (global, model-agnostic): permute a
            channel across the day axis and measure the metric degradation.
  * SHAP -- channel-coalition Shapley values (global): mask channels (replace by
            their temporal-mean field) and attribute the metric change. Exact for
            small groupings, sampled KernelSHAP for the 95 individual channels.
  * LIME -- local linear surrogate (local): for a few representative days, fit a
            distance-weighted Ridge over channel masks to get per-day attributions.

Importance is attributed per variable, always vs the MeteoSwiss truth: for tmax three
metrics in degC (MAE, Gaussian NLL, CRPS), for precip two in raw mm (MAE, Bernoulli-Gamma
NLL). Higher importance = removing/scrambling the feature hurts that metric more.

Three model types are supported (auto-detected from params.json / joint_meta.json):
  * standalone atmospheric **tmax** model (manifest-based) — the three tmax metrics;
  * standalone **precip** model (Bernoulli-Gamma, manifest-based) — importance is
    attributed to MAE (mm) and the Bernoulli-Gamma NLL. Runs trained on the CRPS loss
    (``bernoulli_gamma_crps``) are scored through the same plain BG value/likelihood
    functions, so importances stay comparable across the NLL / CRPS / fine-tuned runs;
  * **joint two-stage** tmax+precip model (``joint_meta.json`` present, e.g.
    ``trained_models/joint_my__..._y2020-2023.../joint_two_stage``) — no manifest, so
    context/normalization/targets are rebuilt from raw data like ``eval_joint``. Both
    heads are scored from ONE perturbation sweep and reported into ``tmax/`` and
    ``precip/`` subdirs; precip importance is attributed to MAE (mm) + Bernoulli-Gamma NLL.

The model is a black box here: every method only ever calls ``score(context)`` which
re-runs the fold ensemble on a (possibly perturbed) copy of the context tensor. This
reuses ``predict.predict_all_days`` and the ensemble/denormalise logic from
``evaluate.build_eval_context``.

Examples
--------
    # fast local smoke test (few days, one fold)
    python feature_importance.py --method pfi \
        --model-dir trained_models/tmax_atm_natg_flat_y2023_e30f5_b8/tmax \
        --year 2022 --n-days 10 --folds 1 --n-repeats 2

    # full study, all methods (HEAVY -- run on the cluster)
    python feature_importance.py --method all \
        --model-dir trained_models/tmax_atm_natg_flat_y2023_e30f5_b8/tmax --year 2022
"""

import argparse
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import properscoring as ps
import torch

import params as params_mod
import datasets as ds
import model_factory
import predict
from evaluate import _targets_with_truth
from infer import _resolve_model_dir
from convCNP.training.utils import (get_value_tmax, get_sigma_tmax,
                                    generate_context_mask)

logger = logging.getLogger("feature_importance")

METRICS = ("mae", "nll", "crps")  # tmax, all in degC; lower is better, importance = perturbed - baseline
PRECIP_METRICS = ("pr_mae", "pr_nll")  # precip: MAE in mm + Bernoulli-Gamma NLL (both lower is better)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics_degC(preds_n: np.ndarray, sigma_n: np.ndarray, truth_n: np.ndarray,
                  dists_meta: ds.Era5Metadata) -> dict[str, float]:
    """MAE / NLL / CRPS in degC from normalized ensemble outputs and truth.

    Shapes: preds_n, sigma_n, truth_n are (T, P) in normalized units. NaN truth
    points (off-grid MeteoSwiss cells) are ignored via nan-aware reductions.
    """
    sd = dists_meta.data_std
    mu_c = predict.denormalize(preds_n, dists_meta)
    truth_c = predict.denormalize(truth_n, dists_meta)
    sigma_c = np.maximum(sigma_n * sd, 1e-6)

    err = mu_c - truth_c
    mae = float(np.nanmean(np.abs(err)))
    nll = float(np.nanmean(
        0.5 * np.log(2.0 * np.pi * sigma_c ** 2) + (err ** 2) / (2.0 * sigma_c ** 2)))
    crps = float(np.nanmean(ps.crps_gaussian(truth_c, mu=mu_c, sig=sigma_c)))
    return {"mae": mae, "nll": nll, "crps": crps}


def _predict_joint_all_days(model, context, dists, elev, seasonal, device, day_batch: int = 1):
    """Run a joint two-head model over all days → (tmax_preds_n, tmax_sig_n, precip_params).

    Mirrors ``predict.predict_all_days`` but for ``JointConditionalConvCNP`` /
    two-stage models whose ``forward → (out_tmax, out_precip)``. Uses the same
    all-ones context mask as ``eval_joint`` so predictions match that evaluator.
    Returns numpy arrays: tmax mean/sigma (T, P) in normalized units, precip
    Bernoulli-Gamma params (T, P, 3).
    """
    model.eval()
    T = context.shape[0]
    step = max(1, day_batch)
    preds, sigs, prparams = [], [], []
    with torch.no_grad():
        for s in range(0, T, step):
            e = min(s + step, T)
            day_ctx = context[s:e]
            b, c, x, y = day_ctx.shape
            mask = generate_context_mask(b, c, x, y, device=device)
            seas = seasonal[s:e] if seasonal is not None else None
            out_t, out_p = model(day_ctx, mask, dists, elev, seasonal=seas)
            preds.append(get_value_tmax(out_t).cpu().numpy())
            sigs.append(get_sigma_tmax(out_t).cpu().numpy())
            prparams.append(out_p.cpu().numpy())
    return (np.concatenate(preds, axis=0), np.concatenate(sigs, axis=0),
            np.concatenate(prparams, axis=0))


def _predict_precip_params_all_days(model, context, dists, elev, seasonal, device,
                                    day_batch: int = 1) -> np.ndarray:
    """Run a STANDALONE Bernoulli-Gamma precip model over all days → (T, P, 3) params.

    The single-head counterpart of ``_predict_joint_all_days``: ``predict.predict_all_days``
    cannot be reused because it extracts Gaussian mean/sigma, whereas the precip head
    emits the raw ``[rho, alpha, beta]`` triple that the BG metrics need.
    """
    model.eval()
    T = context.shape[0]
    step = max(1, day_batch)
    out = []
    with torch.no_grad():
        for s in range(0, T, step):
            e = min(s + step, T)
            day_ctx = context[s:e]
            b, c, x, y = day_ctx.shape
            mask = generate_context_mask(b, c, x, y, device=device)
            seas = seasonal[s:e] if seasonal is not None else None
            out.append(model(day_ctx, mask, dists, elev, seasonal=seas).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Backbone: load once, score many perturbed contexts
# ---------------------------------------------------------------------------

@dataclass
class Backbone:
    """Everything needed to score a (perturbed) context for one model+year."""
    context: torch.Tensor                # (T, C, lat, lon) on device
    mean_field: torch.Tensor             # (1, C, lat, lon): per-channel temporal mean
    dists: torch.Tensor
    target_topo: torch.Tensor
    seasonal: Optional[torch.Tensor]
    truth_n: np.ndarray                  # (T, P) normalized MeteoSwiss truth
    dists_meta: ds.Era5Metadata
    models: list                         # loaded fold ensemble (already on device, eval)
    channel_names: list[str]
    device: torch.device
    rng: np.random.Generator
    baseline: dict[str, float] = field(default_factory=dict)
    day_batch: int = 1                   # days per forward pass (>1 batches the GPU)
    # --- precip extensions; unset for the standalone tmax model ----------------
    metrics: tuple[str, ...] = METRICS   # metric keys score() returns (PFI/SHAP iterate this)
    is_joint: bool = False               # True → models are JointConditional/two-stage modules
    is_precip: bool = False              # True → models are standalone Bernoulli-Gamma precip
    truth_precip: Optional[np.ndarray] = None    # (T, P) raw-mm precip truth
    precip_value_fn: Optional[Callable] = None   # params(...,3) → predicted accumulation (mm)
    precip_loss_fn: Optional[Callable] = None    # Bernoulli-Gamma log-likelihood (gamma_ll)

    @property
    def n_channels(self) -> int:
        return self.context.shape[1]

    @property
    def lime_targets(self) -> tuple[str, ...]:
        """Model outputs LIME fits a surrogate for (one per head)."""
        if self.is_joint:
            return (LIME_TARGET, LIME_TARGET_PRECIP)
        return (LIME_TARGET_PRECIP,) if self.is_precip else (LIME_TARGET,)

    def score(self, context: Optional[torch.Tensor] = None) -> dict[str, float]:
        """Run the fold ensemble on ``context`` (default: unperturbed) → metrics dict.

        Standalone tmax: 3 Gaussian metrics (mae/nll/crps, degC). Standalone precip:
        pr_mae (mm) + pr_nll (Bernoulli-Gamma). Joint two-head: both sets from the
        same forwards.
        """
        ctx = self.context if context is None else context
        if self.is_precip:
            return self._score_precip(ctx)
        if not self.is_joint:
            all_preds, all_sig = [], []
            for model in self.models:
                preds_n, sig_n = predict.predict_all_days(
                    model, ctx, self.dists, self.target_topo, self.seasonal, self.device,
                    day_batch=self.day_batch)
                all_preds.append(preds_n)
                all_sig.append(sig_n)
            within_var = np.mean(np.array(all_sig) ** 2, axis=0)
            between_var = np.var(np.array(all_preds), axis=0)
            preds_mean_n = np.mean(all_preds, axis=0)
            sigma_total_n = np.sqrt(within_var + between_var)
            return _metrics_degC(preds_mean_n, sigma_total_n, self.truth_n, self.dists_meta)

        # Joint: one two-head forward per fold gives tmax (Gaussian) + precip (BG) params.
        all_preds, all_sig, fold_pr_mm, fold_pr_nll = [], [], [], []
        truth_pr_t = torch.from_numpy(self.truth_precip)
        for model in self.models:
            preds_n, sig_n, pr_params = _predict_joint_all_days(
                model, ctx, self.dists, self.target_topo, self.seasonal, self.device,
                day_batch=self.day_batch)
            all_preds.append(preds_n)
            all_sig.append(sig_n)
            pr_t = torch.from_numpy(pr_params)                       # (T, P, 3)
            fold_pr_mm.append(self.precip_value_fn(pr_t).numpy())    # (T, P) mm
            fold_pr_nll.append(-float(self.precip_loss_fn(truth_pr_t, pr_t)))
        # tmax: ensemble mixture (same as the standalone path).
        within_var = np.mean(np.array(all_sig) ** 2, axis=0)
        between_var = np.var(np.array(all_preds), axis=0)
        preds_mean_n = np.mean(all_preds, axis=0)
        sigma_total_n = np.sqrt(within_var + between_var)
        out = _metrics_degC(preds_mean_n, sigma_total_n, self.truth_n, self.dists_meta)
        # precip: ensemble-mean accumulation for MAE(mm); mean per-fold NLL.
        pr_mm_mean = np.mean(fold_pr_mm, axis=0)                     # (T, P)
        out["pr_mae"] = float(np.nanmean(np.abs(pr_mm_mean - self.truth_precip)))
        out["pr_nll"] = float(np.mean(fold_pr_nll))
        return out

    def _score_precip(self, ctx: torch.Tensor) -> dict[str, float]:
        """Standalone Bernoulli-Gamma head → pr_mae (mm) + pr_nll.

        Mirrors the precip half of the joint path and ``eval_precip``: the point
        prediction is the FULL BG mean ``rho*alpha/beta`` (no rho>=0.5 rule), MAE is
        taken on the fold-ensemble mean accumulation, and the NLL is averaged over
        folds (a per-fold likelihood does not ensemble by averaging parameters).
        """
        truth_pr_t = torch.from_numpy(self.truth_precip)
        fold_mm, fold_nll = [], []
        for model in self.models:
            pr_params = _predict_precip_params_all_days(
                model, ctx, self.dists, self.target_topo, self.seasonal, self.device,
                day_batch=self.day_batch)                             # (T, P, 3)
            pr_t = torch.from_numpy(pr_params)
            fold_mm.append(self.precip_value_fn(pr_t).numpy())        # (T, P) mm
            fold_nll.append(-float(self.precip_loss_fn(truth_pr_t, pr_t)))
        pr_mm_mean = np.mean(fold_mm, axis=0)
        return {"pr_mae": float(np.nanmean(np.abs(pr_mm_mean - self.truth_precip))),
                "pr_nll": float(np.mean(fold_nll))}

    # --- perturbation builders ------------------------------------------------
    def permuted(self, channels: list[int]) -> torch.Tensor:
        """Copy of context with the given channels shuffled along the day axis."""
        return self.permuted_with(channels, self.rng.permutation(self.context.shape[0]))

    def permuted_with(self, channels: list[int], perm: np.ndarray) -> torch.Tensor:
        """Like ``permuted`` but with a caller-supplied day permutation, so the draw can
        happen serially in the parent and be applied in a worker (deterministic PFI)."""
        ctx = self.context.clone()
        reordered = self.context[torch.as_tensor(perm, device=ctx.device)]
        for c in channels:
            ctx[:, c] = reordered[:, c]
        return ctx

    def masked(self, present: np.ndarray) -> torch.Tensor:
        """Copy of context where channels with present==0 are set to their mean field.

        ``present`` is a 0/1 vector of length n_channels. Masked channels lose their
        day-to-day (synoptic) signal but keep the spatial climatology.
        """
        ctx = self.context.clone()
        absent = np.where(present == 0)[0]
        if len(absent):
            idx = torch.as_tensor(absent, device=ctx.device, dtype=torch.long)
            ctx[:, idx] = self.mean_field[:, idx]
        return ctx


def _build_backbone(model_dir: Path, year: int, device: torch.device, *,
                    day_stride: int, n_days: Optional[int], folds: Optional[int],
                    seed: int, day_batch: int = 1,
                    precip_glob: Optional[str] = None) -> Backbone:
    manifest = predict.load_manifest(model_dir)
    p = params_mod.Params.load_json(model_dir / "params.json")
    p.DEVICE = str(device)
    # Override the tp INPUT channel for inference only, leaving params.json alone
    # (same contract as eval_precip._load_params). Needed whenever the tp file the
    # run was trained against is not the one that should drive it now -- e.g. the
    # defective tp-2024.nc, which must not reach the encoder.
    if precip_glob:
        p = dataclasses.replace(p, ERA5_PRECIP_GLOB=precip_glob)
        logger.info("tp INPUT channel overridden: %s", precip_glob)
    distribution = model_factory.resolve_distribution(p)
    # Gaussian tmax and Bernoulli-Gamma precip share this whole builder: context,
    # targets, dists and the fold ensemble are distribution-agnostic. Only the truth
    # semantics (z-scored Kelvin vs raw mm, handled by _targets_with_truth) and the
    # metric set differ. The CRPS-trained precip runs emit the same [rho, alpha, beta]
    # triple, so they ride the same path.
    is_precip = distribution.startswith("bernoulli_gamma")
    if distribution != "gaussian" and not is_precip:
        raise ValueError(
            f"feature_importance.py standalone mode supports the Gaussian tmax model "
            f"and the Bernoulli-Gamma precip models; got distribution {distribution!r}")
    dists_meta = predict.manifest_to_dists_metadata(manifest)

    dates = predict._date_range(f"{year}-01-01", f"{year}-12-31")
    # A tp-channel model can only be driven on the days ERA5-Land precip covers
    # (a calendar year is typically short its 31 Dec). The truth is re-aligned to
    # the context days a few lines below, so trimming here is sufficient.
    dates = predict.precip_covered_dates(p, dates, label=f" of {year}")
    # Rebuild the context exactly as predict.py does: native-grid atmospheric via
    # build_atmospheric_context, everything else (surface / regridded) via
    # build_surface_context — so PFI/SHAP/LIME work on surface models too.
    if p.USE_ATMOSPHERIC and p.ATMOS_NATIVE_GRID:
        context = predict.build_atmospheric_context(manifest, p, dates, device)  # (T,C,lat,lon)
    else:
        context = predict.build_surface_context(manifest, p, dates, device)      # (T,C,lat,lon)
    T_full = context.shape[0]

    # Targets + truth, aligned to context days (mirrors evaluate.build_eval_context).
    target_x, target_y_all, target_topo, _, _ = _targets_with_truth(
        manifest, p, device, year_start=year, year_end=year)
    dists = ds.calculate_dists_meteoswiss(dists_meta, target_x, device=device)
    seasonal = (ds.compute_seasonal_features(dates.values.astype("datetime64[ns]"), device=device)
                if p.SEASONAL_FEATURES else None)

    truth_year = target_y_all.sel(time=target_y_all.time.dt.year == year)
    if truth_year.sizes["time"] != T_full:
        ctx_days = pd.to_datetime(dates).normalize()
        truth_days = pd.to_datetime(truth_year.time.values).normalize()
        truth_year = truth_year.isel(time=np.where(truth_days.isin(ctx_days))[0])
    truth_n_full = truth_year.values.astype(np.float32)  # (T, P)

    # Optional day subsampling to keep the perturbation loops tractable.
    sel = np.arange(0, T_full, max(1, day_stride))
    if n_days is not None:
        sel = sel[:n_days]
    if len(sel) != T_full:
        logger.info("subsampling %d of %d days (stride=%d)", len(sel), T_full, day_stride)
        context = context[sel]
        truth_n_full = truth_n_full[sel]
        if seasonal is not None:
            seasonal = seasonal[sel]

    # Fold ensemble.
    channel_groups = (ds.channel_groups_by_variable(manifest["channel_names"])
                      if p.ENCODER != "flat" else None)
    ckpts = sorted(model_dir.glob("model_fold_*"))
    if not ckpts:
        raise FileNotFoundError(f"No model_fold_* checkpoints in {model_dir}")
    if folds is not None:
        ckpts = ckpts[:folds]
    models = []
    for ckpt in ckpts:
        model, epoch = model_factory.load_model_checkpoint(
            ckpt, p, device, channel_groups=channel_groups)
        model.eval()
        models.append(model)
    logger.info("loaded %d fold(s); context %s; %d target points",
                len(models), tuple(context.shape), truth_n_full.shape[1])

    extra = {}
    if is_precip:
        # Score every precip run through the plain Bernoulli-Gamma value/likelihood
        # functions, whatever loss it was TRAINED with — the parameterization is the
        # same [rho, alpha, beta], so this keeps pr_mae/pr_nll (and therefore the
        # importances) comparable across the NLL-, CRPS- and fine-tuned runs. Same
        # choice the joint path makes.
        spec = model_factory.LIKELIHOODS["bernoulli_gamma"]
        extra = dict(metrics=PRECIP_METRICS, is_precip=True,
                     truth_precip=truth_n_full,   # precip truth is raw mm, never z-scored
                     precip_value_fn=spec.get_value_fn, precip_loss_fn=spec.loss_fn)

    bb = Backbone(
        context=context, mean_field=context.mean(dim=0, keepdim=True),
        dists=dists, target_topo=target_topo, seasonal=seasonal,
        truth_n=truth_n_full, dists_meta=dists_meta, models=models,
        channel_names=list(manifest["channel_names"]), device=device,
        rng=np.random.default_rng(seed), day_batch=max(1, day_batch),
        **extra,
    )
    bb.baseline = bb.score()
    if is_precip:
        logger.info("baseline precip: MAE(mm) %.4f | NLL %.4f",
                    bb.baseline["pr_mae"], bb.baseline["pr_nll"])
    else:
        logger.info("baseline: MAE %.4f | NLL %.4f | CRPS %.4f",
                    bb.baseline["mae"], bb.baseline["nll"], bb.baseline["crps"])
    return bb


def _build_backbone_joint(model_dir: Path, device: torch.device, *,
                          day_stride: int, n_days: Optional[int], folds: Optional[int],
                          seed: int, day_batch: int = 1) -> Backbone:
    """Backbone for a JOINT two-stage (tmax + precip) model, e.g.
    ``trained_models/joint_my__..._y2020-2023.../joint_two_stage``.

    The joint run writes no manifest, so context/normalization/grid/targets are
    reconstructed from the raw data exactly like ``eval_joint.predict_all_folds_joint``
    (``train.DataBundle`` + ``train_joint.prepare_joint_targets``), guaranteeing the
    target point set / ``dists`` match the checkpoints. Both heads are scored from one
    perturbation sweep: tmax (Gaussian, degC) + precip (Bernoulli-Gamma, mm).
    """
    import train
    import train_joint as tj
    from types import SimpleNamespace

    p = params_mod.Params.load_json(model_dir / "params.json")
    p.DEVICE = str(device)
    if not (p.USE_ATMOSPHERIC and p.ATMOS_NATIVE_GRID):
        raise ValueError("feature_importance.py targets the atmospheric native-grid model")
    meta = json.loads((model_dir / "joint_meta.json").read_text())
    mode, in_channels = meta["mode"], int(meta["in_channels"])

    logger.info("reconstructing joint pipeline (mode=%s, encoder=%s)...", mode, p.ENCODER)
    config = SimpleNamespace(base_params=p, device=device)
    data = train.DataBundle(config)
    (target_x, dists, target_topo, y_tmax, y_precip, data_std, grid_shape) = \
        tj.prepare_joint_targets(config, data, device,
                                 keep_precip_buffer=meta.get("keep_precip_buffer", False))

    context = data.context.to(device)
    seasonal = (data.seasonal_features.to(device)
                if data.seasonal_features is not None else None)
    dists_meta = data.era5_metadata
    channel_names = list(data.channel_names)
    truth_n_full = y_tmax.cpu().numpy()          # (T, P) z-scored Kelvin tmax truth
    truth_precip_full = y_precip.cpu().numpy()   # (T, P) raw-mm precip truth
    T_full = context.shape[0]

    # Optional day subsampling (mirrors _build_backbone), applied to every day-indexed array.
    sel = np.arange(0, T_full, max(1, day_stride))
    if n_days is not None:
        sel = sel[:n_days]
    if len(sel) != T_full:
        logger.info("subsampling %d of %d days (stride=%d)", len(sel), T_full, day_stride)
        context = context[sel]
        truth_n_full = truth_n_full[sel]
        truth_precip_full = truth_precip_full[sel]
        if seasonal is not None:
            seasonal = seasonal[sel]

    # Fold ensemble of joint modules (built + loaded exactly as eval_joint does).
    ckpts = sorted(model_dir.glob("model_fold_*"))
    if not ckpts:
        raise FileNotFoundError(f"No model_fold_* checkpoints in {model_dir}")
    if folds is not None:
        ckpts = ckpts[:folds]
    models = []
    for ckpt in ckpts:
        if mode == "two_stage":
            tmax = tj.build_tmax_model(p, in_channels)
            model = tj.build_conditional_precip_model(p, in_channels, tmax)
        else:
            model = tj.build_joint_model(p, in_channels)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()
        models.append(model)
    logger.info("loaded %d joint fold(s); context %s; %d target points",
                len(models), tuple(context.shape), truth_n_full.shape[1])

    spec = model_factory.LIKELIHOODS["bernoulli_gamma"]
    bb = Backbone(
        context=context, mean_field=context.mean(dim=0, keepdim=True),
        dists=dists, target_topo=target_topo, seasonal=seasonal,
        truth_n=truth_n_full, dists_meta=dists_meta, models=models,
        channel_names=channel_names, device=device,
        rng=np.random.default_rng(seed), day_batch=max(1, day_batch),
        metrics=METRICS + PRECIP_METRICS, is_joint=True,
        truth_precip=truth_precip_full,
        precip_value_fn=spec.get_value_fn, precip_loss_fn=spec.loss_fn,
    )
    bb.baseline = bb.score()
    logger.info("baseline tmax: MAE %.4f NLL %.4f CRPS %.4f | precip: MAE(mm) %.4f NLL %.4f",
                bb.baseline["mae"], bb.baseline["nll"], bb.baseline["crps"],
                bb.baseline["pr_mae"], bb.baseline["pr_nll"])
    return bb


# ---------------------------------------------------------------------------
# Channel name parsing / groupings
# ---------------------------------------------------------------------------

SCAFFOLD = {"lat", "lon", "cos_time", "sin_time", "elevation"}
# Surface anchor channels (--atmos-surface-anchors): hourly, no pressure level.
SURFACE_ANCHORS = ("t2m", "tp")

# 'data' is an internal bookkeeping name (datasets.py's generic surface-channel slot),
# not the physical variable -- it is always ERA5-Land t2m_max regardless of target (see
# README gotchas). Grouping/parsing logic below stays keyed on the raw name 'data' (it
# has to match bb.channel_names); this is applied only where a name becomes a figure label.
# The label differs by target: for precip, 'data' is a distinct auxiliary field, so its
# own name (t2m_max) reads fine; for tmax, the model's own output IS t2m_max, so labelling
# the input channel 't2m_max' too would read as if the target were feeding itself -- 'surface
# temp' says what the channel is without colliding with the output name.
DISPLAY_NAME = {
    "tmax": {"data": "surface temp"},
    "precip": {"data": "t2m_max"},
}


def _display(name: str, variable: str) -> str:
    return DISPLAY_NAME.get(variable, {}).get(name, name)


def _parse_channel(name: str) -> dict[str, str]:
    """'t850_12' -> {var:'t', level:'850', hour:'12'}; scaffold names -> group 'scaffold';
    the surface 'data' channel -> its own atomic group (no level/hour).

    Surface anchors are matched before the generic split: 't2m_12' would otherwise
    come out as var 't', level '2m' and 'tp_12' as var 't', level 'p', which lumps
    both in with atmospheric temperature.
    """
    if name in SCAFFOLD:
        return {"var": "scaffold", "level": name, "hour": "-"}
    if name == "data":
        return {"var": "data", "level": "-", "hour": "-"}
    for anchor in SURFACE_ANCHORS:
        # Hourly as an atmospheric anchor ('tp_12'), or a single surface channel ('tp',
        # ERA5-Land precipitation under --use-surface-precip).
        if name == anchor:
            return {"var": anchor, "level": "-", "hour": "-"}
        if name.startswith(f"{anchor}_"):
            return {"var": anchor, "level": "-", "hour": name.split("_", 1)[1]}
    var = name[0]
    rest = name[1:]
    level, hour = (rest.split("_") + [""])[:2]
    return {"var": var, "level": level, "hour": hour}


def _groupings(channel_names: list[str]) -> dict[str, dict[str, list[int]]]:
    """Return {grouping_name: {group_label: [channel indices]}} for var/level/hour."""
    out: dict[str, dict[str, list[int]]] = {"variable": {}, "level": {}, "hour": {}}
    for i, name in enumerate(channel_names):
        meta = _parse_channel(name)
        out["variable"].setdefault(meta["var"], []).append(i)
        if meta["var"] not in ("scaffold", "data"):
            if meta["level"] != "-":      # anchors are hourly but carry no level
                out["level"].setdefault(meta["level"], []).append(i)
            out["hour"].setdefault(meta["hour"], []).append(i)
    return out


# ---------------------------------------------------------------------------
# Method 1: Permutation Feature Importance
# ---------------------------------------------------------------------------

def run_pfi(bb: Backbone, n_repeats: int, n_workers: Optional[int] = 1,
            log_every: int = 10) -> pd.DataFrame:
    """Per-channel and grouped PFI. Importance = mean over repeats of (perturbed - baseline)."""
    # Units in their natural order: each channel, then each group of each grouping.
    units = [(name, "channel", [i]) for i, name in enumerate(bb.channel_names)]
    for gname, groups in _groupings(bb.channel_names).items():
        for label, chans in groups.items():
            units.append((f"{gname}={label}", gname, chans))
    logger.info("PFI: %d units x %d repeats", len(units), n_repeats)

    # Draw the day-permutations serially in the parent (preserving the original draw
    # order) so the parallel result is identical to the serial one; score in workers.
    T = bb.context.shape[0]
    jobs = []  # (unit_idx, (channels, perm))
    for u_idx, (_, _, channels) in enumerate(units):
        for _ in range(n_repeats):
            jobs.append((u_idx, (channels, bb.rng.permutation(T))))
    scores = _parallel_eval(bb, [spec for _, spec in jobs], _perm_score_worker, n_workers,
                            log_every=log_every, label="PFI evals")

    deltas = [{m: [] for m in bb.metrics} for _ in units]
    for (u_idx, _), s in zip(jobs, scores):
        for m in bb.metrics:
            deltas[u_idx][m].append(s[m] - bb.baseline[m])
    rows = []
    for u_idx, (label, grouping, channels) in enumerate(units):
        for m in bb.metrics:
            arr = np.array(deltas[u_idx][m])
            rows.append(dict(method="pfi", grouping=grouping, feature=label, metric=m,
                             importance=float(arr.mean()), std=float(arr.std()),
                             n_channels=len(channels)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parallel per-evaluation scoring (CPU)
# ---------------------------------------------------------------------------
#
# PFI, SHAP and LIME all reduce to "evaluate the fold ensemble on many perturbed
# contexts" -- per-coalition masks (SHAP), per-channel day-permutations (PFI), or
# per-mask single-day contexts (LIME). Each evaluation is an independent forward
# pass, so we fan them out across worker processes via fork: children inherit the
# already-loaded backbone copy-on-write (no pickling of the models/context), with
# torch.set_num_threads(1) per worker so N workers don't oversubscribe the cores.
#
# Determinism is preserved because every random draw (coalition masks, PFI day
# permutations, LIME masks) happens serially in the parent BEFORE the fan-out;
# workers only consume the pre-drawn specs and never touch bb.rng. So results are
# identical regardless of the worker count. Forking after CUDA init is unsafe, so
# non-CPU devices fall back to the serial loop (this is the CPU path; use the GPU
# for the other levers).

_FI_BB: Optional["Backbone"] = None


def _worker_init():
    # One torch/BLAS thread per worker so N workers don't oversubscribe the cores.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _resolve_workers(n_workers: Optional[int]) -> int:
    """0/negative -> auto (cpu_count-1); else the requested count."""
    if n_workers is None or n_workers <= 0:
        return max(1, (os.cpu_count() or 1) - 1)
    return n_workers


def _fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS (or M:SS)."""
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _parallel_eval(bb: "Backbone", items: list, worker: Callable, n_workers: Optional[int],
                   *, log_every: int = 200, label: str = "evals") -> list:
    """Apply ``worker(item)`` to every item; parallel across processes on CPU, else serial.

    ``worker`` is a module-level fn that reads the shared backbone from the global
    ``_FI_BB`` (set here for the duration of the call, in both paths). Result order
    matches ``items``.

    Progress is logged for the first 10 items (immediate confirmation + early pace),
    then every ``log_every``, then the final item -- each line carries elapsed time
    and a live ETA extrapolated from the rate so far.
    """
    global _FI_BB
    n = len(items)
    workers = _resolve_workers(n_workers)
    parallel = workers > 1 and n > 1 and bb.device.type == "cpu"
    if workers > 1 and not parallel and bb.device.type != "cpu":
        logger.warning("--workers=%d ignored on %s device; running serial "
                       "(fork after CUDA init is unsafe).", workers, bb.device.type)
    t0 = time.time()

    def _log(done: int, suffix: str = ""):
        if not (done <= 10 or done % log_every == 0 or done == n):
            return
        elapsed = time.time() - t0
        eta = elapsed / done * (n - done) if done else 0.0
        logger.info("  %d/%d %s | elapsed %s, ETA %s%s",
                    done, n, label, _fmt_hms(elapsed), _fmt_hms(eta), suffix)

    _FI_BB = bb
    try:
        if parallel:
            results: list = [None] * n
            suffix = f" ({workers} workers)"
            pool = mp.get_context("fork").Pool(processes=workers, initializer=_worker_init)
            with pool:
                for i, r in enumerate(pool.imap(worker, items, chunksize=1)):
                    results[i] = r
                    _log(i + 1, suffix)
            return results
        results = []
        for i, item in enumerate(items):
            results.append(worker(item))
            _log(i + 1)
        return results
    finally:
        _FI_BB = None


# Worker fns -- each reads the shared backbone from _FI_BB and returns a small,
# picklable result (a metrics dict or a float).
def _score_worker(present: np.ndarray) -> dict[str, float]:
    return _FI_BB.score(_FI_BB.masked(present))


def _perm_score_worker(spec) -> dict[str, float]:
    channels, perm = spec
    return _FI_BB.score(_FI_BB.permuted_with(channels, perm))


def _lime_predict_worker(spec) -> dict[str, float]:
    d, mask = spec
    bb = _FI_BB
    day_ctx0 = bb.context[d:d + 1]
    day_seasonal = bb.seasonal[d:d + 1] if bb.seasonal is not None else None
    absent = np.where(mask == 0)[0]
    ctx = day_ctx0.clone()
    if len(absent):
        idx = torch.as_tensor(absent, device=ctx.device, dtype=torch.long)
        ctx[:, idx] = bb.mean_field[:, idx]
    return _predict_daymean(bb, ctx, day_seasonal)


def _parallel_scores(bb: "Backbone", present_list: list[np.ndarray], n_workers: Optional[int],
                     *, log_every: int = 200, label: str = "coalitions") -> list[dict]:
    """SHAP convenience wrapper over _parallel_eval (masked-context scoring)."""
    return _parallel_eval(bb, present_list, _score_worker, n_workers,
                          log_every=log_every, label=label)


# ---------------------------------------------------------------------------
# Method 2: SHAP (channel-coalition Shapley values)
# ---------------------------------------------------------------------------
#
# Value function f(S) = error metric using ONLY the channels in S "present"; the
# rest are masked to their temporal-mean field. Adding a useful channel lowers the
# error, so its Shapley value phi is <= 0. We report importance = -phi >= 0 (the
# error reduction the channel contributes), keeping the same sign convention as PFI
# (larger = more important). Efficiency: sum_i (-phi_i) = f(empty) - f(all).

def _shap_values_for_subsets(cache: dict, labels: list[str], metric: str) -> dict[str, float]:
    """Exact Shapley values from a full {frozenset(present_labels): metrics} cache."""
    from itertools import combinations
    from math import factorial
    G = len(labels)
    phi = {}
    for i, li in enumerate(labels):
        rest = [l for l in labels if l != li]
        val = 0.0
        for k in range(len(rest) + 1):
            w = factorial(k) * factorial(G - k - 1) / factorial(G)
            for S in combinations(rest, k):
                fS = cache[frozenset(S)][metric]
                fSi = cache[frozenset(S + (li,))][metric]
                val += w * (fSi - fS)
        phi[li] = val
    return phi


def run_shap_grouped(bb: Backbone, grouping: str, groups: dict[str, list[int]],
                     n_workers: Optional[int] = 1) -> pd.DataFrame:
    """Exact Shapley values over the (few) groups of a grouping (variable/level/hour)."""
    labels = list(groups)
    G = len(labels)
    logger.info("SHAP exact (%s): %d groups -> %d coalitions", grouping, G, 2 ** G)
    present_keys, present_list = [], []
    for bits in range(2 ** G):
        present_groups = [labels[k] for k in range(G) if (bits >> k) & 1]
        present = np.zeros(bb.n_channels, dtype=int)
        for g in present_groups:
            present[groups[g]] = 1
        present_keys.append(frozenset(present_groups))
        present_list.append(present)
    scores = _parallel_scores(bb, present_list, n_workers,
                              log_every=2 ** 30, label=f"{grouping} coalitions")
    cache: dict[frozenset, dict[str, float]] = dict(zip(present_keys, scores))

    rows = []
    full, empty = frozenset(labels), frozenset()
    for m in bb.metrics:
        phi = _shap_values_for_subsets(cache, labels, m)
        total = cache[empty][m] - cache[full][m]            # = sum(-phi)
        recon = -sum(phi.values())
        if abs(recon - total) > 1e-6 * (1 + abs(total)):
            logger.warning("SHAP efficiency off for %s/%s: sum(-phi)=%.5f vs f(0)-f(all)=%.5f",
                           grouping, m, recon, total)
        for g in labels:
            rows.append(dict(method="shap", grouping=grouping, feature=f"{grouping}={g}",
                             metric=m, importance=-phi[g], std=np.nan, n_channels=len(groups[g])))
    return pd.DataFrame(rows)


def _kernel_shap(Z: np.ndarray, w: np.ndarray, y: np.ndarray,
                 f_empty: float, f_full: float, ridge: float = 1e-6) -> np.ndarray:
    """Constrained weighted least squares (KernelSHAP) for one metric.

    Minimise ||sqrt(w)(Z phi - (y - f_empty))||^2 s.t. sum(phi) = f_full - f_empty.
    Solved via the KKT system. Returns phi (M,).
    """
    M = Z.shape[1]
    W = w / w.sum()
    yp = y - f_empty
    A = Z.T @ (W[:, None] * Z) + ridge * np.eye(M)
    c = Z.T @ (W * yp)
    a = np.ones(M)
    b = f_full - f_empty
    KKT = np.zeros((M + 1, M + 1))
    KKT[:M, :M] = A
    KKT[:M, M] = a
    KKT[M, :M] = a
    rhs = np.concatenate([c, [b]])
    sol = np.linalg.solve(KKT, rhs)
    return sol[:M]


def run_shap_channels(bb: Backbone, n_samples: int, n_workers: Optional[int] = 1,
                      log_every: int = 10) -> pd.DataFrame:
    """Per-channel Shapley values via sampled KernelSHAP (all 95 channels)."""
    M = bb.n_channels
    f_full = bb.baseline                                    # all present
    f_empty = bb.score(bb.masked(np.zeros(M, dtype=int)))   # all masked (climatology)
    # Report f(empty) on the head's primary metric — "mae" does not exist for a
    # precip-only backbone, whose metrics are (pr_mae, pr_nll).
    primary = bb.metrics[0]
    logger.info("SHAP KernelSHAP: %d channels, %d coalitions; f(empty) %s %.4f",
                M, n_samples, primary.upper(), f_empty[primary])

    # SHAP kernel: sample coalition sizes k in 1..M-1 with weight (M-1)/(k(M-k)).
    sizes = np.arange(1, M)
    size_w = (M - 1) / (sizes * (M - sizes))
    size_w /= size_w.sum()
    Z = np.zeros((n_samples, M), dtype=float)
    kw = np.zeros(n_samples)
    for n in range(n_samples):
        k = bb.rng.choice(sizes, p=size_w)
        idx = bb.rng.choice(M, size=k, replace=False)
        Z[n, idx] = 1.0
        kw[n] = (M - 1) / (k * (M - k))                     # per-coalition SHAP weight

    present_list = [Z[n].astype(int) for n in range(n_samples)]
    scores = _parallel_scores(bb, present_list, n_workers, log_every=log_every, label="coalitions")
    yvals = {m: np.array([s[m] for s in scores], dtype=float) for m in bb.metrics}

    rows = []
    for m in bb.metrics:
        phi = _kernel_shap(Z, kw, yvals[m], f_empty[m], f_full[m])
        for i, name in enumerate(bb.channel_names):
            rows.append(dict(method="shap", grouping="channel", feature=name, metric=m,
                             importance=-phi[i], std=np.nan, n_channels=1))
    return pd.DataFrame(rows)


def run_shap(bb: Backbone, n_samples: int, n_workers: Optional[int] = 1,
             log_every: int = 10) -> pd.DataFrame:
    parts = [run_shap_channels(bb, n_samples, n_workers, log_every=log_every)]
    for gname, groups in _groupings(bb.channel_names).items():
        parts.append(run_shap_grouped(bb, gname, groups, n_workers))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Method 3: LIME (local linear surrogate)
# ---------------------------------------------------------------------------
#
# For a few representative days, fit a distance-weighted Ridge of the model's
# predicted spatial-mean tmax (degC) against binary channel masks. The surrogate
# coefficients are the LOCAL effect of each channel being present on that day's
# prediction. Aggregated across days (mean +- std), they give a global LIME view.

LIME_TARGET = "pred_tmax_degC"
LIME_TARGET_PRECIP = "pred_precip_mm"


def _predict_daymean(bb: Backbone, day_ctx: torch.Tensor,
                     day_seasonal: Optional[torch.Tensor]) -> dict[str, float]:
    """Ensemble-mean spatial-mean model output(s) for a single-day context.

    Returns predicted tmax (degC), or — for a standalone precip model — the predicted
    accumulation (mm); the joint model returns both from the same forwards.
    """
    if bb.is_precip:
        vals = []
        for model in bb.models:
            pr_params = _predict_precip_params_all_days(
                model, day_ctx, bb.dists, bb.target_topo, day_seasonal, bb.device)
            vals.append(bb.precip_value_fn(torch.from_numpy(pr_params)).numpy())  # (1, P) mm
        return {LIME_TARGET_PRECIP: float(np.nanmean(np.mean(vals, axis=0)))}

    if not bb.is_joint:
        preds = []
        for model in bb.models:
            p, _ = predict.predict_all_days(
                model, day_ctx, bb.dists, bb.target_topo, day_seasonal, bb.device)
            preds.append(p)
        pc = predict.denormalize(np.mean(preds, axis=0), bb.dists_meta)  # (1, P) degC
        return {LIME_TARGET: float(np.nanmean(pc))}

    tvals, pvals = [], []
    for model in bb.models:
        preds_n, _sig, pr_params = _predict_joint_all_days(
            model, day_ctx, bb.dists, bb.target_topo, day_seasonal, bb.device)
        tvals.append(preds_n)
        pvals.append(bb.precip_value_fn(torch.from_numpy(pr_params)).numpy())  # (1, P) mm
    tc = predict.denormalize(np.mean(tvals, axis=0), bb.dists_meta)            # (1, P) degC
    return {LIME_TARGET: float(np.nanmean(tc)),
            LIME_TARGET_PRECIP: float(np.nanmean(np.mean(pvals, axis=0)))}


def run_lime(bb: Backbone, n_samples: int, n_days: int,
             ridge_alpha: float = 1.0, kernel_width: float = 0.25,
             n_workers: Optional[int] = 1, log_every: int = 10) -> pd.DataFrame:
    from sklearn.linear_model import Ridge

    M = bb.n_channels
    T = bb.context.shape[0]
    targets = bb.lime_targets
    day_idxs = np.unique(np.linspace(0, T - 1, min(n_days, T), dtype=int))
    logger.info("LIME: %d day(s) x %d masks; targets %s", len(day_idxs), n_samples, targets)

    # Draw all masks serially in the parent (per-day order preserved) then predict
    # every (day, mask) in workers, so the parallel result matches the serial one.
    day_Z = {}
    jobs = []  # (day_idx, mask_row)
    for d in day_idxs:
        Z = bb.rng.integers(0, 2, size=(n_samples, M)).astype(float)
        Z[0] = 1.0  # anchor on the true instance
        day_Z[int(d)] = Z
        for n in range(n_samples):
            jobs.append((int(d), Z[n]))
    preds = _parallel_eval(bb, jobs, _lime_predict_worker, n_workers,
                           log_every=log_every, label="LIME masks")

    # Per target: (day → predicted-output vector over that day's masks).
    day_y = {t: {int(d): np.zeros(n_samples) for d in day_idxs} for t in targets}
    counters = {int(d): 0 for d in day_idxs}
    for (d, _), pv in zip(jobs, preds):
        j = counters[d]
        for t in targets:
            day_y[t][d][j] = pv[t]
        counters[d] += 1

    groupings = _groupings(bb.channel_names)
    rows = []
    for t in targets:
        per_day = []  # (n_days, M) local coefficients for this target
        for d in day_idxs:
            Z, y = day_Z[int(d)], day_y[t][int(d)]
            # Cosine distance of each mask to the all-present instance -> LIME kernel weight.
            cos_sim = np.sqrt(np.clip(Z.mean(axis=1), 0, 1))     # sqrt(k/M)
            dist = 1.0 - cos_sim
            w = np.exp(-(dist ** 2) / (kernel_width ** 2))
            reg = Ridge(alpha=ridge_alpha).fit(Z, y, sample_weight=w)
            per_day.append(reg.coef_)
            logger.info("  [%s] day %d: pred %.3f, R2(weighted) %.3f",
                        t, int(d), y[0], reg.score(Z, y, sample_weight=w))
        coefs = np.array(per_day)                                 # (D, M)
        for i, name in enumerate(bb.channel_names):
            rows.append(dict(method="lime", grouping="channel", feature=name, metric=t,
                             importance=float(coefs[:, i].mean()),
                             std=float(coefs[:, i].std()), n_channels=1))
        for gname, groups in groupings.items():
            for label, chans in groups.items():
                g = coefs[:, chans].sum(axis=1)                   # per-day group coefficient
                rows.append(dict(method="lime", grouping=gname, feature=f"{gname}={label}",
                                 metric=t, importance=float(g.mean()),
                                 std=float(g.std()), n_channels=len(chans)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_csv(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    logger.info("wrote %s (%d rows)", path, len(df))
    return path


def _plot_topn(df: pd.DataFrame, out_dir: Path, method: str, grouping: str, variable: str,
               topn: int = 20, metrics: tuple[str, ...] = METRICS):
    """Horizontal bar chart of top-N |importance| per metric for one grouping."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[(df["method"] == method) & (df["grouping"] == grouping)]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), max(4, 0.3 * topn)))
    if len(metrics) == 1:
        axes = [axes]
    def _label(feature: str) -> str:
        # Group rows are 'grouping=name' (e.g. 'variable=data'); channel rows are the
        # bare name. Either way, only the trailing name part needs the display mapping.
        prefix, sep, name = feature.rpartition("=")
        return f"{prefix}{sep}{_display(name, variable)}"

    for ax, m in zip(axes, metrics):
        d = sub[sub["metric"] == m].copy()
        d["abs"] = d["importance"].abs()
        d = d.sort_values("abs", ascending=True).tail(topn)
        ax.barh(d["feature"].map(_label), d["importance"],
                xerr=d["std"] if d["std"].notna().any() else None,
                color="#3878c7")
        ax.axvline(0, color="k", lw=0.7)
        ax.set_title(f"{method.upper()} {m.upper()}")
        ax.set_xlabel(f"importance [{m}]")
    fig.suptitle(f"{method.upper()} importance — grouping: {grouping}")
    fig.tight_layout()
    path = out_dir / f"{method}_{grouping}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", path)


# Fixed symmetric colour limit for the per-channel heatmaps, one per target variable.
# Hardcoded on purpose: a scale derived per model makes the same importance render deep
# red in one figure and pale in another, so no two figures can be compared by eye.
#
# ±0.4 holds the bulk of both variables (over all CLEAN models, 2026-07-31, at most 4
# channels per figure fall outside it). What it deliberately does not stretch to is the
# handful of dominating channels — the single 'data' channel of the surface-input tmax
# models reaches 8.26 under PFI, and a scale covering that would flatten every other
# figure to nothing. Those cells saturate and carry their value as a label instead.
HEATMAP_VMAX = {"tmax": 0.4, "precip": 0.4}
# Colour bands across the full -vmax..+vmax range. Discrete rather than continuous, so a
# cell reads as a class off the legend instead of a shade to be eyeballed.
HEATMAP_BINS = 16
# Every cell is labelled with its value. Medium grey keeps the numbers legible without
# competing with the colour; past ±HEATMAP_STRONG the cell matters enough to read first,
# so it switches to solid black. A value that rounds to zero at HEATMAP_DECIMALS leaves
# its cell white.
HEATMAP_STRONG = 0.2
HEATMAP_INK = "#777777"
HEATMAP_DECIMALS = 3

# Row order for the auxiliary panel; anything else present is appended.
AUX_ORDER = ("data", "tp", "lat", "lon", "cos_time", "sin_time", "elevation")
PREFERRED_VARS = ("z", "t", "q", "u", "v")


def _heatmap_panels(sub: pd.DataFrame) -> list[tuple[str, list[str], list[str], str]]:
    """Panels for one figure as (title, row labels, column labels, y-axis label).

    Three channel families, each with its own grid: the pressure-level variables
    (level × hour), the surface anchors t2m/tp (variable × hour, no level), and the
    auxiliary channels — the scaffold plus the surface models' single 'data' channel —
    which have neither level nor hour and so form a one-column strip.
    """
    present = set(sub["var"])
    hours = sorted(h for h in sub["hour"].unique() if h != "-")
    panels = []

    levels = sorted((lv for lv in sub["level"].unique() if str(lv).isdigit()),
                    key=int, reverse=True)
    physvars = ([v for v in PREFERRED_VARS if v in present]
                + [v for v in sorted(present)
                   if v not in PREFERRED_VARS and v not in SURFACE_ANCHORS
                   and v not in ("scaffold", "data")])
    if levels and hours:
        for v in physvars:
            panels.append((v, levels, hours, "level hPa"))

    anchors = [a for a in SURFACE_ANCHORS if a in present]
    if anchors and hours:
        panels.append(("surface anchors", anchors, hours, ""))

    atomic = {c for c, m in zip(sub["channel"], sub["feature"].map(_parse_channel))
              if _is_auxiliary(m)}
    aux = _aux_order(atomic)
    if aux:
        panels.append(("auxiliary", aux, [""], ""))
    return panels


def _panel_grid(sub: pd.DataFrame, title: str, rows: list[str],
                cols: list[str]) -> np.ndarray:
    """Fill one panel's (row, column) grid from the per-channel importances."""
    grid = np.full((len(rows), len(cols)), np.nan)
    if title == "auxiliary":
        for _, r in sub.iterrows():
            if r["channel"] in rows:
                grid[rows.index(r["channel"]), 0] = r["importance"]
    elif title == "surface anchors":
        for _, r in sub.iterrows():
            if r["var"] in rows and r["hour"] in cols:
                grid[rows.index(r["var"]), cols.index(r["hour"])] = r["importance"]
    else:
        dv = sub[sub["var"] == title]
        for _, r in dv.iterrows():
            if r["level"] in rows and r["hour"] in cols:
                grid[rows.index(r["level"]), cols.index(r["hour"])] = r["importance"]
    return grid


def _group_values(df: pd.DataFrame, method: str, metric: str) -> dict[str, dict[str, float]]:
    """{grouping: {label: importance}} from the stored variable/level/hour group rows.

    These are measured by permuting a whole group at once, which is not the sum over its
    channels: with 6 levels x 5 hours of correlated inputs, permuting one channel leaves
    its information reachable through the rest, so the parts always understate the whole.
    """
    out: dict[str, dict[str, float]] = {"variable": {}, "level": {}, "hour": {}}
    rows = df[(df["method"] == method) & (df["metric"] == metric)
              & (df["grouping"].isin(out))]
    for r in rows.itertuples():
        out[r.grouping][str(r.feature).split("=", 1)[-1]] = float(r.importance)
    return out


# Fixed x-limit per (variable, method) and grouping, so the same bar length means the same
# importance in every model's figure. From the observed maxima over all CLEAN models
# (2026-07-31). 'auxiliary' deliberately excludes the surface 'data' channel, which reaches
# 8.26 and would flatten the scaffold bars; it overflows and is labelled instead.
GROUP_XMAX = {
    ("tmax", "pfi"):    {"variable": 8.5, "level": 4.0, "hour": 3.0, "auxiliary": 0.6},
    ("tmax", "lime"):   {"variable": 1.6, "level": 0.9, "hour": 0.8, "auxiliary": 0.6},
    ("precip", "pfi"):  {"variable": 26.0, "level": 2.6, "hour": 4.6, "auxiliary": 0.12},
    ("precip", "lime"): {"variable": 3.3, "level": 0.9, "hour": 1.6, "auxiliary": 0.15},
}
GROUP_PANELS = ("variable", "level", "hour", "auxiliary")


# Where the pre-fix parser filed each surface anchor: it keyed the variable on the first
# character, so t2m_* and tp_* were pulled into 'variable=t' — but their *level* came out
# as '2m' and 'p', which is by accident exactly the right set of channels. Those level
# rows are therefore the anchors' own group permutation under a different name, and can be
# read back as independent measurements instead of being written off.
LEGACY_ANCHOR_LEVEL = {"2m": "t2m", "p": "tp"}


def _is_auxiliary(meta: dict) -> bool:
    """A channel with no level/hour grid: the scaffold, the surface 'data', a single 'tp'.

    Defined by shape, not by a list of names, so a channel added later cannot be dropped
    from the figures without anyone noticing. The scaffold is called out separately
    because it stores its own name in the level field.
    """
    return meta["var"] == "scaffold" or (meta["level"] == "-" and meta["hour"] == "-")


def _aux_order(aux) -> list[str]:
    """Auxiliary channels in reading order, with anything unlisted kept rather than lost."""
    return [c for c in AUX_ORDER if c in aux] + sorted(set(aux) - set(AUX_ORDER))


def _anchor_groups(parsed: dict, groups: dict) -> tuple[dict[str, float], list[str]]:
    """({anchor variable: importance}, [variables folded into 'variable=t']).

    Empty for a CSV written after the parser fix, which carries the anchors as ordinary
    variable groups. The second element only affects labelling: 'variable=t' really was
    measured with the anchors in it, so it is not atmospheric temperature alone.
    """
    missing = {m["var"] for m in parsed.values()} - set(groups["variable"])
    anchors = {var: groups["level"][lvl]
               for lvl, var in LEGACY_ANCHOR_LEVEL.items()
               if var in missing and lvl in groups["level"]}
    folded = sorted(v for v in missing if v[0] in groups["variable"])
    return anchors, folded


def _group_bar_color(value: float, xmax: float):
    """Bar fill from the heatmap's colormap, so both figures read the same way."""
    import matplotlib.pyplot as plt
    frac = 0.5 + 0.5 * float(np.clip(value / xmax, -1.0, 1.0))
    return plt.get_cmap("RdBu_r")(frac)


def _bar_panel(ax, labels: list[str], values: list[float], xmax: float, title: str):
    """One horizontal bar panel of a group figure, on a fixed symmetric x scale.

    PFI importances are conventionally >=0 (a permutation can only hurt or leave a metric
    unchanged), but LIME coefficients are signed, and the colour scale (_group_bar_color)
    already diverges around 0 -- so the axis has to show both sides, not just [0, xmax].
    """
    pos = range(len(labels))
    ax.barh(pos, values, height=0.75,
            color=[_group_bar_color(v, xmax) for v in values],
            edgecolor="#555555", linewidth=0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(-xmax, xmax)
    ax.axvline(0, color="#555555", lw=0.6)
    for i, v in enumerate(values):
        # A bar past the scale is labelled inside its own end, matching how the heatmap
        # treats a saturated cell. Sign decides which side of 0 the label sits on.
        sign = -1.0 if v < 0 else 1.0
        over = abs(v) > xmax
        x = sign * (xmax * 0.97 if over else min(abs(v) + xmax * 0.03, xmax * 0.97))
        ha = ("left" if over else "right") if sign < 0 else ("right" if over else "left")
        ax.text(x, i, f"{v:.{HEATMAP_DECIMALS}f}", va="center",
                ha=ha, fontsize=7,
                color="white" if over else "#333333",
                fontweight="bold" if over or abs(v) > HEATMAP_STRONG else "normal")
    ax.set_yticks(pos, labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0)


def _plot_group_importance(df: pd.DataFrame, out_dir: Path, method: str, metric: str,
                           variable: str):
    """Grouped importances for one model: by variable, level, hour and auxiliary channel.

    The companion to the per-channel heatmap, and the figure the argument should rest on.
    With 6 levels x 5 hours of correlated inputs, permuting one channel leaves its
    information reachable through its neighbours, so every individual cell under-attributes
    — these are the whole group permuted at once, which is what the model actually loses.
    The auxiliary panel is per channel rather than grouped, since the scaffold is only five
    channels and 'scaffold' as a single bar hides which of them matters. A surface-input run
    has no pressure levels or hourly channels — every channel is auxiliary — so the
    variable/level/hour panels would just repeat 'by auxiliary' under a different name;
    only 'by auxiliary' is drawn for that case.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = _group_values(df, method, metric)
    chan = df[(df["method"] == method) & (df["grouping"] == "channel")
              & (df["metric"] == metric)]
    if chan.empty:
        return
    parsed = {c: _parse_channel(c) for c in chan["feature"]}
    anchors, folded = _anchor_groups(parsed, groups)

    aux = {c: float(r) for c, r in zip(chan["feature"], chan["importance"])
           if _is_auxiliary(parsed[c])}
    # Variables: the stored groups, with the mixed one named for what it actually is — the
    # three permuted together — and the anchors themselves added as their own bars. It is
    # not 't plus extras': tp is total precipitation, nothing to do with temperature.
    joint = f"{'+'.join([folded[0][0]] + folded)} (joint)" if folded else None
    var_labels = [joint if folded and v == folded[0][0] else _display(v, variable) for v in groups["variable"]]
    var_values = [float(v) for v in groups["variable"].values()]
    for name, value in anchors.items():
        var_labels.append(name)
        var_values.append(float(value))
    # Levels: pressure first, then one 'surface' entry — what distinguishes the anchors
    # here is that they sit at the surface, not which of them they are.
    lv_labels = sorted((lv for lv in groups["level"] if lv.isdigit()), key=int, reverse=True)
    lv_values = [float(groups["level"][lv]) for lv in lv_labels]
    if anchors:
        lv_labels.append("surface")
        lv_values.append(float(sum(anchors.values())))

    aux_order = _aux_order(aux)
    content = {
        "variable": (var_labels, var_values),
        "level": (lv_labels, lv_values),
        "hour": (sorted(groups["hour"]),
                 [float(groups["hour"][h]) for h in sorted(groups["hour"])]),
        "auxiliary": ([_display(c, variable) for c in aux_order], [float(aux[c]) for c in aux_order]),
    }
    # A surface-input run (no pressure levels, no hourly atmospheric channels) has every
    # channel auxiliary, so 'by variable'/'by level'/'by hour' are not aggregating anything
    # — 'by variable' just repeats each auxiliary channel as its own singleton group (plus
    # 'scaffold', already broken out per-channel in 'by auxiliary'), and 'by hour' collapses
    # to one bar keyed '-' that is really just one channel's own group permuted again. Only
    # 'by auxiliary' carries information here, so it is the only panel shown.
    group_panels = GROUP_PANELS if any(not _is_auxiliary(m) for m in parsed.values()) else ("auxiliary",)
    xmax = GROUP_XMAX.get((variable, method), {})
    panels = []
    for name in group_panels:
        labels, values = content[name]
        if not labels:
            continue
        panels.append((name, labels, values,
                       xmax.get(name) or (max(map(abs, values)) * 1.15 or 1.0)))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(3.1 * len(panels), 3.6))
    for ax, (name, labels, values, lim) in zip(axes[0], panels):
        _bar_panel(ax, labels, values, lim, f"by {name}")
    axes[0][0].set_ylabel("group")
    # 'surface' is the two anchor groups added up, not a single permutation of both.
    note = (f" · {joint.split(' ')[0]} permuted as one group; surface = "
            f"{' + '.join(anchors)}" if folded else "")
    fig.suptitle(f"{method.upper()} grouped importance — {metric} · whole group permuted{note}")
    fig.tight_layout()
    path = out_dir / f"{method}_groups_{metric}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)


def _plot_channel_heatmap(df: pd.DataFrame, out_dir: Path, method: str, metric: str,
                          variable: str, vmax: float | None = None, scale_note: str = ""):
    """Per-channel importance heatmap: pressure levels, surface anchors and auxiliaries.

    Every channel the model actually receives appears somewhere, so the figure accounts
    for the whole input rather than the pressure-level subset — which is also what gives
    the surface-input models a heatmap at all.

    ``vmax`` sets the symmetric colour limit; pass one of ``HEATMAP_VMAX`` so every
    figure for a variable shares a legend. The range is split into ``HEATMAP_BINS``
    discrete bands, and a dominating channel outside the range takes the colourbar's
    over/under cap and is labelled with its value — so the two are never confused.
    Falling back to this model's own maximum (``vmax=None``) is only for one-off
    inspection: it makes the figure incomparable to the others.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    sub = df[(df["method"] == method) & (df["grouping"] == "channel") & (df["metric"] == metric)]
    if sub.empty:
        return
    meta = sub["feature"].map(_parse_channel)
    sub = sub.assign(channel=sub["feature"], var=[m["var"] for m in meta],
                     level=[m["level"] for m in meta], hour=[m["hour"] for m in meta])
    panels = _heatmap_panels(sub)
    if not panels:
        return
    vmax = float(vmax if vmax is not None else sub["importance"].abs().max()) or 1.0

    # Discrete bands, with the two end colours reserved for the over/under caps so a
    # dominating channel is visibly outside the scale rather than merely at its end.
    # Even bands over the range, with the two end colours reserved for the over/under
    # caps so a dominating channel is visibly outside the scale, not merely at its end.
    bounds = np.linspace(-vmax, vmax, HEATMAP_BINS + 1)
    band_colors = plt.get_cmap("RdBu_r")(np.linspace(0, 1, HEATMAP_BINS + 2))
    cmap = ListedColormap(band_colors[1:-1])
    cmap.set_bad("white")   # rounds-to-zero cells, and channels a model does not have
    cmap.set_under(band_colors[0])
    cmap.set_over(band_colors[-1])
    norm = BoundaryNorm(bounds, cmap.N)

    widths = [max(len(cols), 0.9) for _, _, cols, _ in panels]
    fig, axes = plt.subplots(
        1, len(panels), squeeze=False,
        # The auxiliary panel's row labels are long ('cos_time', 'elevation') and sit to
        # the left of a narrow panel, so the columns need more room than the default.
        gridspec_kw={"width_ratios": widths, "wspace": 0.45},
        figsize=(0.62 * sum(widths) + 1.5 * len(panels), 4.2))
    for panel_i, (ax, (title, rows, cols, ylabel)) in enumerate(zip(axes[0], panels)):
        grid = _panel_grid(sub, title, rows, cols)
        # A cell that rounds to zero carries no signal, so it stays white rather than
        # taking the innermost band's tint. Masking it hands it to the colormap's "bad"
        # colour, which is where absent channels already land.
        rounds_to_zero = np.isfinite(grid) & (np.round(grid, HEATMAP_DECIMALS) == 0)
        im = ax.imshow(np.ma.masked_where(rounds_to_zero, grid), cmap=cmap, norm=norm,
                       aspect="auto")
        # Every cell carries its value, so a colour never has to be read off the legend
        # to two decimals — and a dominating channel past the scale still shows its
        # number rather than only the cap colour.
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isfinite(grid[i, j]):
                    continue
                # 0.0 rather than the raw value, so a tiny negative does not print as
                # '-0.000' on a cell the figure is calling zero.
                value = 0.0 if rounds_to_zero[i, j] else grid[i, j]
                strong = abs(value) > HEATMAP_STRONG
                # Past the scale the cell takes the darkest cap colour, where black
                # would not read.
                saturated = abs(value) > vmax
                ax.text(j, i, f"{value:.{HEATMAP_DECIMALS}f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if saturated else "black" if strong else HEATMAP_INK,
                        fontweight="bold" if strong else "normal")
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(rows)), [_display(r, variable) for r in rows], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("hour UTC" if cols != [""] else "")
        if ylabel and panel_i == 0:   # the level panels share one axis label
            ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, boundaries=bounds,
                        ticks=bounds, extend="both", spacing="proportional",
                        label=f"importance [{metric}]")
    cbar.ax.set_yticklabels([f"{b:+.2f}" for b in bounds], fontsize=8)
    fig.suptitle(f"{method.upper()} per-channel importance — {metric}{scale_note}")
    path = out_dir / f"{method}_heatmap_{metric}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)


def _top_features(df: pd.DataFrame, method: str, grouping: str, metric: str, n: int = 8):
    d = df[(df["method"] == method) & (df["grouping"] == grouping) & (df["metric"] == metric)].copy()
    d = d.reindex(d["importance"].abs().sort_values(ascending=False).index).head(n)
    return [(r["feature"], r["importance"]) for _, r in d.iterrows()]


@dataclass
class _Head:
    """One reporting head (model output). Standalone → tmax only; joint → tmax + precip."""
    name: str                    # 'tmax' | 'precip'
    subdir: str                  # output subdir under feature_importance ('.' = in place)
    metrics: tuple[str, ...]     # PFI/SHAP metric keys for this head
    primary: str                 # metric used for ranking / tables / heatmap
    lime_target: str             # LIME surrogate target name for this head
    unit: str                    # display unit for the baseline line

    def owns(self, metric: str) -> bool:
        return metric in self.metrics or metric == self.lime_target

    def metric_for(self, method: str) -> str:
        return self.lime_target if method == "lime" else self.primary


def _report_heads(bb: Backbone) -> list[_Head]:
    if bb.is_joint:
        return [
            _Head("tmax", "tmax", METRICS, "mae", LIME_TARGET, "degC"),
            _Head("precip", "precip", PRECIP_METRICS, "pr_mae", LIME_TARGET_PRECIP, "mm"),
        ]
    if bb.is_precip:
        return [_Head("precip", ".", PRECIP_METRICS, "pr_mae", LIME_TARGET_PRECIP, "mm")]
    return [_Head("tmax", ".", METRICS, "mae", LIME_TARGET, "degC")]


def write_summary(summaries: dict[str, pd.DataFrame], bb: Backbone, out_dir: Path,
                  head: _Head, span: str) -> Path:
    """Cross-method summary.md for one head: rankings + grouped tables + heatmaps."""
    for method, df in summaries.items():
        _plot_channel_heatmap(df, out_dir, method, head.metric_for(method), head.name,
                              vmax=HEATMAP_VMAX[head.name],
                              scale_note=f" · fixed {head.name} scale ±{HEATMAP_VMAX[head.name]:g}")
        _plot_group_importance(df, out_dir, method, head.metric_for(method), head.name)

    n_pts = (bb.truth_precip.shape[1] if head.name == "precip" and bb.truth_precip is not None
             else bb.truth_n.shape[1])
    base = " | ".join(f"{m.upper()} {bb.baseline[m]:.3f}" for m in head.metrics)
    lines = [f"# Feature importance — {head.name} ({span})", ""]
    lines += [f"- Target points: {n_pts}  |  days scored: {bb.context.shape[0]}  "
              f"|  folds: {len(bb.models)}",
              f"- Baseline (all channels): {base}  [{head.unit}]", ""]

    lines += ["## Top channels (|importance|)", ""]
    header = "| rank | " + " | ".join(
        f"{mname.upper()} ({'pred' if mname == 'lime' else head.metric_for(mname)})"
        for mname in summaries) + " |"
    lines += [header, "|" + "---|" * (len(summaries) + 1)]
    tops = {mname: _top_features(df, mname, "channel", head.metric_for(mname), 8)
            for mname, df in summaries.items()}
    for i in range(8):
        cells = []
        for mname in summaries:
            if i < len(tops[mname]):
                f, v = tops[mname][i]
                cells.append(f"{_display(f, head.name)} ({v:+.3f})")
            else:
                cells.append("")
        lines.append(f"| {i + 1} | " + " | ".join(cells) + " |")
    lines.append("")

    for grouping in ("variable", "hour", "level"):
        lines += [f"## By {grouping}", ""]
        for mname, df in summaries.items():
            metric = head.metric_for(mname)
            items = df[(df.method == mname) & (df.grouping == grouping) & (df.metric == metric)]
            items = items.sort_values("importance", key=lambda s: s.abs(), ascending=False)
            txt = ", ".join(f"{_display(r['feature'].split('=')[1], head.name)} ({r['importance']:+.3f})"
                            for _, r in items.iterrows())
            lines.append(f"- **{mname.upper()}** ({metric}): {txt}")
        lines.append("")

    lines += ["## Figures", ""]
    for mname in summaries:
        for g in ("channel", "variable", "level", "hour"):
            lines.append(f"- `{mname}_{g}.png`")
        lines.append(f"- `{mname}_heatmap_*.png` (level × hour)")
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines))
    logger.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", help="Dir with params.json / manifest.json / model_fold_*.")
    src.add_argument("--trial-dir", help="Trial dir; its single variable subdir is used.")
    ap.add_argument("--method", default="pfi",
                    help="Comma-separated subset of pfi,shap,lime — or 'all' for every "
                         "method. e.g. --method pfi,lime to skip the ~30 h SHAP pass.")
    ap.add_argument("--year", type=int, default=2022,
                    help="Evaluation year for the standalone tmax model (default 2022). "
                         "Ignored for a joint model (uses its DATA_YEAR_START..END span).")
    ap.add_argument("--device", default=None, help="Override device (cuda/cpu).")
    ap.add_argument("--precip-glob", default=None,
                    help="Override ERA5_PRECIP_GLOB (the tp input channel) for this run only; "
                         "params.json is untouched. Use when the tp file the model was trained "
                         "against should not drive it now (e.g. the defective tp-2024.nc).")
    ap.add_argument("--output-dir", default=None,
                    help="Output dir (default <model_dir>/feature_importance).")
    # compute controls
    ap.add_argument("--day-stride", type=int, default=1, help="Use every Nth day.")
    ap.add_argument("--n-days", type=int, default=None, help="Cap number of (strided) days.")
    ap.add_argument("--folds", type=int, default=None, help="Use only the first K folds.")
    ap.add_argument("--n-repeats", type=int, default=5, help="PFI permutation repeats.")
    ap.add_argument("--shap-nsamples", type=int, default=2000, help="KernelSHAP coalitions.")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel CPU worker processes for per-evaluation scoring across "
                         "PFI/SHAP/LIME (0 = auto = cpu_count-1; 1 = serial). Ignored on GPU.")
    ap.add_argument("--shap-workers", type=int, default=None,
                    help="Deprecated alias for --workers (kept for back-compat).")
    ap.add_argument("--log-every", type=int, default=10,
                    help="Progress-log cadence: log the first 10 evals, then every N, then the "
                         "last (each line shows elapsed + live ETA). Default 10.")
    ap.add_argument("--day-batch", type=int, default=1,
                    help="Days processed per forward pass. 1 = one-day-at-a-time (CPU default); "
                         "larger batches the day axis to feed the GPU (e.g. 64-365 on CUDA). "
                         "Results are identical to day-batch=1 up to float tolerance.")
    ap.add_argument("--lime-nsamples", type=int, default=1000, help="LIME masks per day.")
    ap.add_argument("--lime-days", type=int, default=4, help="Representative days for LIME.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    params_mod.configure_renku_cuda()
    device = torch.device(args.device) if args.device else params_mod.select_device()
    model_dir = _resolve_model_dir(args)
    out_dir = Path(args.output_dir) if args.output_dir else (model_dir / "feature_importance")

    # A joint two-stage model (tmax + precip) is detected by joint_meta.json; it has
    # no manifest and is scored via _build_backbone_joint. Otherwise the standalone
    # (manifest-based) path, which auto-detects tmax (Gaussian) vs precip
    # (Bernoulli-Gamma) from params.json. --year applies only to the standalone model.
    is_joint = (model_dir / "joint_meta.json").exists()
    if is_joint:
        p_meta = params_mod.Params.load_json(model_dir / "params.json")
        span = f"{p_meta.DATA_YEAR_START}-{p_meta.DATA_YEAR_END or p_meta.DATA_YEAR_START}"
        logger.info("joint two-stage model detected → scoring tmax + precip heads (%s)", span)
        bb = _build_backbone_joint(model_dir, device,
                                   day_stride=args.day_stride, n_days=args.n_days,
                                   folds=args.folds, seed=args.seed, day_batch=args.day_batch)
    else:
        span = str(args.year)
        bb = _build_backbone(model_dir, args.year, device,
                             day_stride=args.day_stride, n_days=args.n_days,
                             folds=args.folds, seed=args.seed, day_batch=args.day_batch,
                             precip_glob=args.precip_glob)

    if args.method == "all":
        methods = ["pfi", "shap", "lime"]
    else:
        methods = [m.strip() for m in args.method.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ("pfi", "shap", "lime")]
    if unknown:
        raise SystemExit(f"unknown --method {unknown}; choose from pfi, shap, lime, all")
    workers = args.shap_workers if args.shap_workers is not None else args.workers

    # Each method costs hours (PFI ~5-7 h, SHAP ~15 h), so persist its raw frame the
    # moment it finishes and reuse it on a re-run. Previously every result was held in
    # memory until ALL methods completed, so interrupting a run — or not wanting the
    # method that happened to be last — threw away everything already computed.
    #
    # The reuse is guarded by a provenance record: a frame is only reused if it was
    # produced by the same backbone AND the same parameters for that method. Without
    # that, a cheap smoke run (--n-days 1 --folds 1) would leave a file that a later
    # full run silently adopts, yielding results that look legitimate and are not.
    out_dir.mkdir(parents=True, exist_ok=True)
    runners = {
        "pfi": lambda: run_pfi(bb, args.n_repeats, workers, log_every=args.log_every),
        "shap": lambda: run_shap(bb, args.shap_nsamples, workers, log_every=args.log_every),
        "lime": lambda: run_lime(bb, args.lime_nsamples, args.lime_days, n_workers=workers,
                                 log_every=args.log_every),
    }
    # Speed-only knobs (day_batch, workers, log_every) are deliberately excluded: they
    # do not change results, so including them would force needless recomputation.
    backbone_fp = {
        "model_dir": str(model_dir), "span": span,
        "day_stride": args.day_stride, "n_days": args.n_days, "folds": args.folds,
        "seed": args.seed, "n_channels": int(bb.n_channels),
        "n_days_scored": int(bb.context.shape[0]),
        "baseline": {k: round(float(v), 9) for k, v in bb.baseline.items()},
    }
    method_fp = {
        "pfi": {"n_repeats": args.n_repeats},
        "shap": {"shap_nsamples": args.shap_nsamples},
        "lime": {"lime_nsamples": args.lime_nsamples, "lime_days": args.lime_days},
    }

    full: dict[str, pd.DataFrame] = {}
    for method in methods:
        raw, meta_path = out_dir / f"_raw_{method}.csv", out_dir / f"_raw_{method}.meta.json"
        want = {"backbone": backbone_fp, "method": method_fp[method]}
        if raw.exists():
            try:
                have = json.load(open(meta_path))
            except Exception:                                          # noqa: BLE001
                have = None
            if have == want:
                full[method] = pd.read_csv(raw)
                logger.info("reusing completed %s from %s (%d rows) — delete it to recompute",
                            method.upper(), raw, len(full[method]))
                continue
            differs = ("no/unreadable provenance record" if have is None else
                       [k for k in want if have.get(k) != want[k]])
            logger.warning("%s exists but was produced by a different run (%s); recomputing",
                           raw, differs)
        full[method] = runners[method]()
        full[method].to_csv(raw, index=False)
        json.dump(want, open(meta_path, "w"), indent=2)
        logger.info("persisted %s to %s (%d rows) — safe from here on",
                    method.upper(), raw, len(full[method]))

    json.dump(bb.baseline, open(out_dir / "baseline.json", "w"), indent=2)

    print(f"\nFeature-importance outputs in: {out_dir}")
    for head in _report_heads(bb):
        head_dir = out_dir if head.subdir == "." else (out_dir / head.subdir)
        summaries: dict[str, pd.DataFrame] = {}
        for method, df in full.items():
            keep = df[df["metric"].apply(head.owns)].copy()
            if keep.empty:
                continue
            _save_csv(keep, head_dir, method)
            metrics = (head.lime_target,) if method == "lime" else head.metrics
            for g in ("channel", "variable", "level", "hour"):
                _plot_topn(keep, head_dir, method, g, head.name, metrics=metrics)
            summaries[method] = keep
        if summaries:
            write_summary(summaries, bb, head_dir, head, span)
        for name, df in summaries.items():
            metric = head.metric_for(name)
            d = df[(df.grouping == "channel") & (df.metric == metric)]
            top = d.reindex(d["importance"].abs().sort_values(ascending=False).index).head(5)
            print(f"  [{head.name}] {name.upper()} top-5 channels by |importance|({metric}): "
                  f"{list(top['feature'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
