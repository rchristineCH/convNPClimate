#!/usr/bin/env python3
"""
Inference + evaluation for trained convNPClimate models.

Given a trained model directory (``params.json``, ``metadata.json``,
``model_fold_*``), this reconstructs the exact training data pipeline, predicts
each fold's held-out days at the fine-resolution MeteoSwiss reference, and reports
a comprehensive set of evaluation metrics:

  * deterministic: MAE, RMSE, bias (degC)
  * association:   per-station Pearson / Spearman (median) + pooled
  * probabilistic: Gaussian NLL (normalized + degC), CRPS (degC)
  * calibration:   coverage at 1 sigma (~0.68) and 1.96 sigma (~0.95)
  * skill:         CRPS skill vs a baseline (bilinear ERA5 in surface mode,
                   else climatology)

Metrics are written to ``<model_dir>/eval_metrics.json`` and printed as a table.

Example
-------
    python infer.py --model-dir trained_models/my_run/tmax
    python infer.py --trial-dir trained_models/my_run --folds 0 1
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import properscoring as ps
from scipy import stats as sstats

import params as params_mod
import datasets as ds
import model_factory
from inference import predict_holdout_fold
from convCNP.training.training_elev import get_fold_holdout_indices
import train  # reuse DataBundle + VARIABLE_SPECS so inference matches training exactly


logger = logging.getLogger("infer")

LOG2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _pointwise_correlations(preds: np.ndarray, truths: np.ndarray):
    """Per-station Pearson/Spearman over time; returns (median_pearson, median_spearman)."""
    n_points = preds.shape[1]
    pear = np.full(n_points, np.nan)
    spear = np.full(n_points, np.nan)
    for st in range(n_points):
        t = truths[:, st]
        p = preds[:, st]
        valid = ~np.isnan(t)
        if valid.sum() < 2:
            continue
        tv, pv = t[valid], p[valid]
        if np.std(tv) == 0 or np.std(pv) == 0:
            continue
        pear[st] = sstats.pearsonr(pv, tv)[0]
        spear[st] = sstats.spearmanr(pv, tv).correlation
    return float(np.nanmedian(pear)), float(np.nanmedian(spear))


def compute_metrics(
    preds_c: np.ndarray,      # (days, points)  predicted mean, degC
    sigmas_c: np.ndarray,     # (days, points)  predicted sigma, degC
    truths_c: np.ndarray,     # (days, points)  observed, degC
    data_std: float,
    ref_c: Optional[np.ndarray] = None,   # (days, points) baseline mean, degC
    ref_name: str = "none",
) -> dict:
    """Compute the full metric suite over a (days, points) prediction block."""
    err = preds_c - truths_c
    valid = ~np.isnan(truths_c)
    e = err[valid]
    p_c = preds_c[valid]
    s_c = np.clip(sigmas_c[valid], 1e-6, None)
    t_c = truths_c[valid]

    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    bias = float(np.mean(e))

    # Gaussian NLL in degC and in the normalized training units.
    nll_c = float(np.mean(0.5 * LOG2PI + np.log(s_c) + 0.5 * (e / s_c) ** 2))
    e_n = e / data_std
    s_n = s_c / data_std
    nll_norm = float(np.mean(0.5 * LOG2PI + np.log(s_n) + 0.5 * (e_n / s_n) ** 2))

    crps = float(np.mean(ps.crps_gaussian(t_c, mu=p_c, sig=s_c)))

    cov_1sig = float(np.mean(np.abs(e) <= s_c))
    cov_95 = float(np.mean(np.abs(e) <= 1.96 * s_c))

    med_pear, med_spear = _pointwise_correlations(preds_c, truths_c)
    pooled_pear = float(sstats.pearsonr(p_c, t_c)[0])

    out = {
        "n_obs": int(valid.sum()),
        "mae_degC": mae,
        "rmse_degC": rmse,
        "bias_degC": bias,
        "nll_degC": nll_c,
        "nll_normalized": nll_norm,
        "crps_degC": crps,
        "coverage_1sigma": cov_1sig,
        "coverage_95pct": cov_95,
        "median_pearson": med_pear,
        "median_spearman": med_spear,
        "pooled_pearson": pooled_pear,
        "skill_reference": ref_name,
    }

    if ref_c is not None:
        crps_ref = float(np.mean(np.abs((ref_c - truths_c)[valid])))  # deterministic CRPS = MAE
        out["baseline_mae_degC"] = crps_ref
        out["crps_skill"] = (1.0 - crps / crps_ref) if crps_ref > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------
def evaluate(model_dir: Path, device: torch.device, folds: Optional[list[int]] = None) -> dict:
    p = params_mod.Params.load_json(model_dir / "params.json")
    p.DEVICE = str(device)

    spec = train.VARIABLE_SPECS.get(p.VARIABLE)
    if spec is None:
        raise ValueError(
            f"Variable {p.VARIABLE!r} is not enabled in train.VARIABLE_SPECS; "
            "re-enable it there to evaluate."
        )

    # Rebuild the exact training context/dists/scaffold (reuses DataBundle).
    logger.info("reconstructing data pipeline (variable=%s, encoder=%s)...", p.VARIABLE, p.ENCODER)
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
    dists = ds.calculate_dists_meteoswiss(data.dists_metadata, target_x, device=device)
    target_tensor = torch.from_numpy(target_y.values.astype(np.float32)).to(device)
    metadata = data.dists_metadata

    n_times = data.context.shape[0]
    n_folds = p.N_FOLDS
    fold_ids = folds if folds is not None else list(range(n_folds))
    channel_groups = data.channel_groups if p.ENCODER != "flat" else None

    # ERA5 bilinear baseline is only meaningful when channel 0 is the temperature
    # field, i.e. surface mode. In atmospheric mode fall back to climatology.
    use_era5_baseline = p.USE_SURFACE
    ref_name = "bilinear ERA5 (channel 0)" if use_era5_baseline else "climatology (per-station mean)"
    if use_era5_baseline:
        ref_norm_all = ds.interpolate_era5_to_targets(data.context, target_x, metadata)

    blocks = {"preds": [], "sigmas": [], "truths": [], "ref": []}
    per_fold = {}
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
        res = predict_holdout_fold(
            model, data.context, start, end, dists, target_topo,
            data.seasonal_features, target_tensor, metadata, device,
        )
        sigmas_c = res.sigmas  # already denormalized to degC by predict_holdout_fold

        if use_era5_baseline:
            ref_c = (metadata.denormalize(ref_norm_all[start:end].cpu().numpy())
                     - ds.KELVIN_OFFSET)
        else:
            clim = np.nanmean(res.truths, axis=0, keepdims=True)  # per-station mean over fold
            ref_c = np.broadcast_to(clim, res.truths.shape)

        per_fold[fold] = compute_metrics(
            res.preds, sigmas_c, res.truths, metadata.data_std, ref_c, ref_name
        )
        per_fold[fold]["checkpoint_epoch"] = int(epoch)

        blocks["preds"].append(res.preds)
        blocks["sigmas"].append(sigmas_c)
        blocks["truths"].append(res.truths)
        blocks["ref"].append(ref_c)

    if not per_fold:
        raise RuntimeError("No fold checkpoints found to evaluate.")

    preds = np.concatenate(blocks["preds"], axis=0)
    sigmas = np.concatenate(blocks["sigmas"], axis=0)
    truths = np.concatenate(blocks["truths"], axis=0)
    ref = np.concatenate(blocks["ref"], axis=0)

    overall = compute_metrics(preds, sigmas, truths, metadata.data_std, ref, ref_name)

    # mean +/- std across folds for the headline metrics
    agg = {}
    keys = ["mae_degC", "rmse_degC", "bias_degC", "crps_degC", "nll_normalized",
            "median_pearson", "median_spearman"]
    if use_era5_baseline or True:
        keys.append("crps_skill")
    for k in keys:
        vals = [m[k] for m in per_fold.values() if k in m and not np.isnan(m[k])]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "model_dir": str(model_dir),
        "variable": p.VARIABLE,
        "encoder": p.ENCODER,
        "grid_mode": data.grid_mode,
        "n_folds_evaluated": len(per_fold),
        "n_target_points": int(target_tensor.shape[1]),
        "skill_reference": ref_name,
        "overall": overall,
        "per_fold": per_fold,
        "fold_aggregate": agg,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(result: dict):
    o = result["overall"]
    print()
    print(f"=== Evaluation: {result['model_dir']} ===")
    print(f"variable={result['variable']} | encoder={result['encoder']} | "
          f"grid={result['grid_mode']} | folds={result['n_folds_evaluated']} | "
          f"points={result['n_target_points']:,}")
    print(f"skill reference: {result['skill_reference']}")
    print()
    rows = [
        ("MAE (degC)", "mae_degC"),
        ("RMSE (degC)", "rmse_degC"),
        ("Bias (degC)", "bias_degC"),
        ("CRPS (degC)", "crps_degC"),
        ("NLL (normalized)", "nll_normalized"),
        ("NLL (degC)", "nll_degC"),
        ("Median Pearson", "median_pearson"),
        ("Median Spearman", "median_spearman"),
        ("Pooled Pearson", "pooled_pearson"),
        ("Coverage @1sigma (~0.68)", "coverage_1sigma"),
        ("Coverage @1.96sigma (~0.95)", "coverage_95pct"),
        ("CRPS skill vs baseline", "crps_skill"),
    ]
    print(f"{'Metric':<30} {'Overall':>12} {'Fold mean+/-std':>22}")
    print("-" * 66)
    agg = result["fold_aggregate"]
    for label, key in rows:
        if key not in o:
            continue
        val = o[key]
        a = agg.get(key)
        astr = f"{a['mean']:.4f}+/-{a['std']:.4f}" if a else ""
        print(f"{label:<30} {val:>12.4f} {astr:>22}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_model_dir(args) -> Path:
    if args.model_dir:
        return Path(args.model_dir)
    trial = Path(args.trial_dir)
    # Pick the single variable subdir that contains params.json.
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
    ap.add_argument("--output", default=None, help="JSON output path (default <model_dir>/eval_metrics.json).")
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
    result = evaluate(model_dir, device, folds=args.folds)

    out_path = Path(args.output) if args.output else (model_dir / "eval_metrics.json")
    out_path.write_text(json.dumps(result, indent=2))
    print_report(result)
    print(f"Wrote metrics: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
