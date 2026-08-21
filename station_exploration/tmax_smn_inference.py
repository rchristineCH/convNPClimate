#!/usr/bin/env python
"""The selected tmax model at the SMN temperature cohort (~147 stations).

The committed station analysis verifies at the 28 NBCN sites, whose ``ths200dx``
is homogenised. The SMN daily files carry ``tre200dx`` — the OPERATIONAL daily
maximum — at five times as many stations, on the same coordinates the precip
verification uses. This reuses station_analysis's target construction and
fold-exact prediction unchanged (only the metadata/observation tables differ)
and writes bundles shaped exactly like ``pred_cache/stations_*.npz`` to
station_exploration/bundles/tmax-smn_{cv,2024}.npz.

The day-label convention of ``tre200dx`` vs TmaxD was checked before use
(check_alignment.py): lag 0 at all 147 stations, median r 0.999.

Usage:  python station_exploration/tmax_smn_inference.py
"""
from __future__ import annotations

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

import station_analysis as sa  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("tmax_smn")

MODEL_DIR = (REPO / "CLEAN_trained_models/"
             "lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8/tmax")
CACHE = HERE / "cache"
BUNDLES = HERE / "bundles"


def load_cohort():
    """(station_meta in station_analysis's column convention, daily obs table)."""
    meta = pd.read_csv(CACHE / "smn_stations.csv")
    daily = pd.read_csv(CACHE / "smn_tmax_daily.csv", parse_dates=["date"])
    stns = sorted(set(daily["stn"]))
    meta = (meta[meta["stn"].isin(stns)].sort_values("stn").reset_index(drop=True)
            .rename(columns={"stn": "stn_abbr"}))
    daily = daily.rename(columns={"stn": "stn_abbr"})
    return meta, daily


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, daily = load_cohort()
    logger.info("%d SMN temperature stations (%d NBCN among them)",
                len(meta), int(meta["is_nbcn"].sum()))
    tgts = sa.build_station_targets(meta, MODEL_DIR, device)

    BUNDLES.mkdir(exist_ok=True)
    for eval_year in (None, 2024):
        out = BUNDLES / f"tmax-smn_{'cv' if eval_year is None else eval_year}.npz"
        if out.exists():
            logger.info("%s exists; skipping", out.name)
            continue
        t0 = time.time()
        if eval_year is None:
            r = sa.predict_stations_cv(MODEL_DIR, tgts, meta, daily, device,
                                       day_batch=8)
        else:
            r = sa.predict_stations_holdout_year(MODEL_DIR, tgts, meta, daily,
                                                 eval_year, device, day_batch=8)
        np.savez_compressed(
            out, preds_C=r.preds_C, sigmas_C=r.sigmas_C, truth_C=r.truth_C,
            era5_C=r.era5_C, dates=np.array(r.dates, dtype="datetime64[D]"),
            day_mask=r.day_mask, topo=tgts.target_topo.cpu().numpy(),
            stn=meta["stn_abbr"].to_numpy(), name=meta["name"].to_numpy(),
            lat=meta["lat"].to_numpy(), lon=meta["lon"].to_numpy(),
            elev_m=meta["elev_m"].to_numpy(), is_nbcn=meta["is_nbcn"].to_numpy(),
            regime=r.regime, prediction_mode=r.prediction_mode,
            obs_source="tre200dx (operational, not homogenised)",
        )
        ok = np.isfinite(r.truth_C[r.day_mask]) & np.isfinite(r.preds_C[r.day_mask])
        mae = float(np.mean(np.abs((r.preds_C - r.truth_C)[r.day_mask][ok])))
        logger.info("wrote %s (MAE %.3f degC over %d station-days, %.0f s)",
                    out.name, mae, int(ok.sum()), time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
