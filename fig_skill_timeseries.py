#!/usr/bin/env python3
"""Daily-precipitation skill graph: ERA5-Land vs MeteoSwiss vs model prediction.

Over the CV period (2020-2023, the model's held-out folds concatenated) this
plots the Switzerland-domain-mean DAILY accumulated precipitation (mm/day) for:
  * MeteoSwiss RhiresD (truth),
  * the bilinear-ERA5-Land baseline (coarse reference interpolated to the target
    grid; precip_baseline.bilinear_era5_precip), and
  * each trained model's predicted mean (rho*alpha/beta) -- NLL- and CRPS-trained.
It annotates the skill score  skill = 1 - MAE_model / MAE_ERA5-Land  (pooled over
day x point on the common valid mask; eval_precip.compute_baseline_skill), the
repo's Vaughan-style baseline skill. Output: precip_processing_comparison/.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Resolve the repo from this file's location rather than a fixed cluster path, so
# the same script runs on the cluster and locally.
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import params as params_mod
import eval_precip as ep
import precip_baseline

ROOT = REPO
OUT = ROOT / "CLEAN_trained_models/precip_processing_comparison"   # the tracked dir
OUT.mkdir(parents=True, exist_ok=True)
# Full-period ERA5-Land baseline (all training years, not just DATA_YEAR_START).
#
# Points at the corrected, RhiresD-aligned rebuild (06-06 UTC), not
# datasets/ERA5_Land/precipitation/, where tp-2024.nc is ~1.53x inflated and
# day-shifted -- a skill timeseries drawn against that field is measuring the
# defect. Override with PRECIP_BASELINE_GLOB to reproduce an older figure.
BASELINE_GLOB = os.environ.get("PRECIP_BASELINE_GLOB", "tp_hourly/w0606/tp-*.nc")
# All five CLEAN precip runs. The figure used to cover only three, which left the
# CRPS fine-tune and the surface-tp run out of the one plot that plots skill
# directly -- and the surface-tp run is the closest of the five to the baseline.
MODELS = {
    "NLL": ROOT / "CLEAN_trained_models/clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8/precip",
    "NLL+ftCRPS": ROOT / "CLEAN_trained_models/clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps/precip",
    "CRPS": ROOT / "CLEAN_trained_models/clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8/precip",
    "WIND": ROOT / "CLEAN_trained_models/clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8/precip",
    "SFC-TP": ROOT / "CLEAN_trained_models/clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8/precip",
}
COLORS = {"MeteoSwiss": "k", "ERA5-Land": "#d95f02", "NLL": "C0", "NLL+ftCRPS": "C1",
          "CRPS": "C3", "WIND": "C2", "SFC-TP": "C4"}


def _domain_mean(arr2d, mask_pts):
    """Domain-mean per day over the shared valid points -> (n_days,)."""
    return np.nanmean(arr2d[:, mask_pts], axis=1)


def _bundle(mdir, device, eval_year):
    return ep.load_or_predict(mdir, device, eval_year=eval_year)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-year", type=int, default=None,
                    help="Genuine holdout YEAR (e.g. 2024); default = 2020-2023 CV holdout. "
                         "The ERA5-Land line/skill appear automatically once tp-<year>.nc exists.")
    args = ap.parse_args()

    params_mod.configure_renku_cuda()
    device = params_mod.select_device()
    year = args.eval_year
    # One multi-year glob for BOTH regimes. It used to fall back to None for a
    # holdout year, which resolved to datasets/ERA5_Land/precipitation/tp-<year>.nc
    # -- the defective 2024 file. bilinear_era5_precip reindexes onto the model's
    # own day axis, so the non-evaluated years simply select nothing.
    base_glob = BASELINE_GLOB
    regime = f"holdout_{year}" if year is not None else "cv_holdout_2020_2023"
    period = str(year) if year is not None else "2020-2023 CV holdout"

    series, skill = {}, {}
    dates = obs_series = base_series = common_pts = base_source = None

    for name, mdir in MODELS.items():
        print(f"[{name}] running inference ({regime}) ...", flush=True)
        B = _bundle(mdir, device, year)
        obs = B.truth_full[B.day_mask]                                   # (D,P) mm
        pred = B.get_value_fn(B.params_full[B.day_mask]).numpy()         # full BG mean
        base_year = int(getattr(B, "eval_year", None) or B.params.DATA_YEAR_START)
        base, source = precip_baseline.bilinear_era5_precip(
            B.target_lat, B.target_lon, B.time_dates, year=base_year, glob=base_glob)
        # Shared spatial domain = Swiss points valid on every day (obs) AND covered
        # by the baseline on at least one day (`any`, so a single date-alignment gap
        # -- ERA5 tp is labelled with a 1-day accumulation shift -- does not void the
        # whole domain). Per-day nanmean below tolerates the odd all-NaN base day.
        obs_valid = np.isfinite(obs).all(axis=0)
        base_cover = np.isfinite(base).any(axis=0) if base is not None else obs_valid
        pts = obs_valid & base_cover
        if common_pts is None:
            common_pts = pts
            dates = pd.to_datetime(np.asarray(B.time_dates))
            obs_series = _domain_mean(obs, common_pts)
            if base is not None:
                base_series = _domain_mean(base, common_pts)
                base_source = source
            else:
                print(f"  ERA5-Land baseline pending: {source}", flush=True)
            print(f"  {common_pts.sum():,} shared points", flush=True)
        series[name] = _domain_mean(pred, common_pts)
        skill[name] = ep.compute_baseline_skill(B, {}, baseline_glob=base_glob)

    have_base = base_series is not None
    all_series = {"MeteoSwiss": obs_series}
    if have_base:
        all_series["ERA5-Land"] = base_series
    all_series.update(series)

    # ---- figure ----
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [3.1, 1.0]})
    order = [k for k in ["MeteoSwiss", "ERA5-Land", *MODELS] if k in all_series]
    for lab in order:
        y = all_series[lab]
        roll = pd.Series(y, index=dates).rolling(30, center=True, min_periods=5).mean()
        ax0.plot(dates, y, color=COLORS[lab], lw=0.5, alpha=0.25)
        ax0.plot(dates, roll.values, color=COLORS[lab], lw=1.8,
                 label=(lab if lab not in MODELS else f"model ({lab})"))
    ax0.set_ylabel("domain-mean daily precip (mm/day)")
    ax0.set_title(f"Daily accumulated precipitation over Switzerland ({period})\n"
                  "thin = daily, thick = 30-day rolling mean")
    ax0.legend(fontsize=9, ncol=2)
    ax0.grid(alpha=0.3)

    if have_base:
        names = list(MODELS)
        labels = ["ERA5-Land"] + names
        maes = ([skill[names[0]]["baseline_mae_mm"]] +
                [skill[n]["model_mae_mm_common_mask"] for n in names])
        colors = [COLORS[lab] for lab in labels]
        x = np.arange(len(labels))
        ax1.bar(x, maes, color=colors)
        ax1.set_xticks(x); ax1.set_xticklabels(labels)
        ax1.set_ylabel("pooled MAE vs MeteoSwiss (mm)")
        ax1.set_title("Skill vs ERA5-Land\nskill = 1 - MAE_model / MAE_ERA5-Land")
        for xi, m in zip(x, maes):
            ax1.text(xi, m + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
        for xi, name in enumerate(names, start=1):
            s = skill[name]["skill_mae"]
            ax1.text(xi, maes[xi] * 0.5, f"skill\n{s:+.3f}", ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold")
        ax1.grid(alpha=0.3, axis="y")
    else:
        ax1.axis("off")
        ax1.text(0.5, 0.5, f"ERA5-Land baseline for {year}\nnot available yet "
                 f"(tp-{year}.nc missing).\nSkill panel appears once it lands.",
                 ha="center", va="center", fontsize=10, color="0.4")

    fig.tight_layout()
    suffix = "" if year is None else f"_{year}"
    fig.savefig(OUT / f"skill_timeseries{suffix}.png", dpi=120)
    plt.close(fig)

    summary = {
        "regime": regime,
        "baseline_source": base_source or f"pending (tp-{year}.nc)",
        "baseline_available": have_base,
        "skill_mae_vs_era5land": {k: skill[k].get("skill_mae") for k in MODELS},
        "pooled_mae_mm": {
            "ERA5-Land": skill["NLL"].get("baseline_mae_mm"),
            **{k: skill[k].get("model_mae_mm_common_mask") for k in MODELS}},
        "domain_mean_daily_mm": {k: float(np.nanmean(v)) for k, v in all_series.items()},
    }
    (OUT / f"skill_timeseries{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote skill_timeseries{suffix}.png + .json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
