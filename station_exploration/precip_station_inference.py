#!/usr/bin/env python
"""Bernoulli-Gamma inference at the SMN rain gauges, for the two selected precip runs.

The gridded precip evaluation scores RhiresD, which is itself interpolated from
these gauges; this decodes the same trained models directly at the gauge
coordinates, so the daily point observation can be scored with nothing
interpolated in between. The final RBF layer takes arbitrary target points, so
the machinery is ``station_analysis``'s target construction plus
``eval_precip``'s fold-exact parameter prediction:

  * CV holdout  — each fold predicts only its own held-out block
    (``get_fold_holdout_indices`` on the SAME day axis as training: for SFC-TP
    the tp channel stops at 2023-12-30, so the axis is the 1460-day covered
    span, exactly as ``train.DataBundle`` trimmed it).
  * 2024 holdout — all five folds, ``(rho, alpha, beta)`` averaged elementwise
    (the Bernoulli-Gamma ensemble convention; the moment-match is Gaussian-only).

SFC-TP's 2024 tp INPUT comes from ``datasets/ERA5_Land/tp_hourly/tp-*.nc`` (the
rebuilt series continuous with training), never the defective ``tp-2024.nc`` —
the same override ``scripts/reeval_precip_w0606.sh`` uses. The scoring
REFERENCE for both models and regimes is the RhiresD-aligned 06-06 rebuild
``tp_hourly/w0606``, bilinearly interpolated to the gauges.

Writes one bundle per model x regime to station_exploration/bundles/
(``<label>_{cv,2024}.npz``): params (T,N,3), truth_mm, base_mm, dates,
day_mask, station meta columns.

Usage:  python station_exploration/precip_station_inference.py [--models ft-crps sfc-tp]
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import params as params_mod                      # noqa: E402
import datasets as ds                            # noqa: E402
import model_factory                             # noqa: E402
import predict                                   # noqa: E402
import precip_baseline                           # noqa: E402
from eval_precip import predict_params_fold      # noqa: E402
from station_analysis import build_station_targets  # noqa: E402
from convCNP.training.training_elev import get_fold_holdout_indices  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("precip_stations")

MODELS = {
    "ft-crps": REPO / "CLEAN_trained_models/clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps/precip",
    "sfc-tp": REPO / "CLEAN_trained_models/clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8/precip",
}
# 00-00 series continuous with SFC-TP's training input; replaces the defective
# tp-2024.nc for the 2024 leg (reeval_precip_w0606.sh's SFCTP_INPUT_GLOB).
SFCTP_2024_INPUT_GLOB = str(REPO / "datasets/ERA5_Land/tp_hourly/tp-*.nc")
# 06-06 UTC RhiresD-aligned rebuild: the scoring reference, both regimes.
BASELINE_GLOB = str(REPO / "datasets/ERA5_Land/tp_hourly/w0606/tp-*.nc")
CACHE = HERE / "cache"
BUNDLES = HERE / "bundles"


def load_gauges() -> pd.DataFrame:
    meta = pd.read_csv(CACHE / "smn_stations.csv")
    return meta.sort_values("stn").reset_index(drop=True)


def gauge_truth(meta: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    obs = pd.read_csv(CACHE / "smn_precip_daily.csv", parse_dates=["date"])
    wide = obs.pivot_table(index="date", columns="stn", values="precip_mm")
    wide = wide.reindex(index=pd.DatetimeIndex(dates).normalize(),
                        columns=meta["stn"].tolist())
    return wide.to_numpy(dtype=np.float32)


def station_context(tgts, p, dates, device):
    if p.USE_ATMOSPHERIC and p.ATMOS_NATIVE_GRID:
        context = predict.build_atmospheric_context(tgts.manifest, p, dates, device)
    else:
        context = predict.build_surface_context(tgts.manifest, p, dates, device)
    seasonal = (ds.compute_seasonal_features(dates.values.astype("datetime64[ns]"),
                                             device=device)
                if p.SEASONAL_FEATURES else None)
    groups = (ds.channel_groups_by_variable(tgts.manifest["channel_names"])
              if p.ENCODER != "flat" else None)
    return context, seasonal, groups


def run_one(label: str, model_dir: Path, meta: pd.DataFrame, eval_year,
            device: torch.device, day_batch: int) -> None:
    out = BUNDLES / f"{label}_{'cv' if eval_year is None else eval_year}.npz"
    if out.exists():
        logger.info("%s exists; skipping", out.name)
        return
    t0 = time.time()

    p = params_mod.Params.load_json(model_dir / "params.json")
    p.DEVICE = str(device)
    if eval_year is not None and getattr(p, "USE_SURFACE_PRECIP", False):
        p = dataclasses.replace(p, ERA5_PRECIP_GLOB=SFCTP_2024_INPUT_GLOB)
        p.DEVICE = str(device)
        logger.info("tp INPUT override for %s %s: %s", label, eval_year,
                    SFCTP_2024_INPUT_GLOB)
    n_params = model_factory.LIKELIHOODS[model_factory.resolve_distribution(p)].n_params

    tgts = build_station_targets(
        meta.rename(columns={"stn": "stn_abbr"}), model_dir, device)

    if eval_year is None:
        dates = predict._date_range(f"{p.DATA_YEAR_START}-01-01",
                                    f"{p.DATA_YEAR_END}-12-31")
    else:
        dates = predict._date_range(f"{eval_year}-01-01", f"{eval_year}-12-31")
    # SFC-TP's day axis is the tp-covered span — the SAME trim training applied,
    # or the CV fold blocks would not be the blocks the folds actually held out.
    dates = predict.precip_covered_dates(p, dates, label=f" for {label}")
    n_times, n = len(dates), tgts.n

    context, seasonal, groups = station_context(tgts, p, dates, device)
    truth = gauge_truth(meta, dates)

    params_full = np.full((n_times, n, n_params), np.nan, dtype=np.float32)
    day_mask = np.zeros(n_times, dtype=bool)
    fold_params = []
    for fold in range(p.N_FOLDS):
        ckpt = model_dir / f"model_fold_{fold}"
        model, epoch = model_factory.load_model_checkpoint(
            ckpt, p, device, channel_groups=groups)
        if eval_year is None:
            a, b = get_fold_holdout_indices(fold, p.N_FOLDS, n_times)
            pt = predict_params_fold(model, context, a, b, tgts.dists,
                                     tgts.target_topo, seasonal, device)
            params_full[a:b] = pt.numpy()
            day_mask[a:b] = True
            logger.info("  fold %d (epoch %d): days [%d, %d)", fold, epoch, a, b)
        else:
            pt = predict_params_fold(model, context, 0, n_times, tgts.dists,
                                     tgts.target_topo, seasonal, device)
            fold_params.append(pt.numpy())
            logger.info("  fold %d (epoch %d): all %d days", fold, epoch, n_times)
    if eval_year is not None:
        # Bernoulli-Gamma ensemble = elementwise parameter mean (see README gotchas).
        params_full = np.mean(np.stack(fold_params), axis=0).astype(np.float32)
        day_mask[:] = True

    base_year = int(eval_year or p.DATA_YEAR_START)
    base, source = precip_baseline.bilinear_era5_precip(
        tgts.lats, tgts.lons, np.array(dates), year=base_year, glob=BASELINE_GLOB)
    if base is None:
        raise RuntimeError(f"no baseline field: {source}")
    logger.info("baseline: %s", source)

    BUNDLES.mkdir(exist_ok=True)
    np.savez_compressed(
        out, params=params_full, truth_mm=truth, base_mm=base.astype(np.float32),
        dates=np.array(dates, dtype="datetime64[D]"), day_mask=day_mask,
        stn=meta["stn"].to_numpy(), name=meta["name"].to_numpy(),
        lat=meta["lat"].to_numpy(), lon=meta["lon"].to_numpy(),
        elev_m=meta["elev_m"].to_numpy(), is_nbcn=meta["is_nbcn"].to_numpy(),
        regime="cv_holdout" if eval_year is None else f"holdout_year_{eval_year}",
        model_label=label, model_dir=str(model_dir),
        baseline_glob=BASELINE_GLOB,
    )
    logger.info("wrote %s (%.0f s)", out.name, time.time() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS))
    ap.add_argument("--day-batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    meta = load_gauges()
    logger.info("%d gauges (%d NBCN)", len(meta), int(meta["is_nbcn"].sum()))
    for label in args.models:
        for eval_year in (None, 2024):
            run_one(label, MODELS[label], meta, eval_year, device, args.day_batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
