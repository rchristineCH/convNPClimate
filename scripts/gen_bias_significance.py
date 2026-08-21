#!/usr/bin/env python3
"""Is the domain-mean bias of a tmax model different from zero?

The field is smooth and successive days share their weather, so pooling every
point and every day would hand a t test tens of millions of nominally
independent errors and make any offset "significant". This script does what
the methodology chapter describes instead: collapse each day to its domain
mean, then charge the remaining serial correlation to the sample size via the
effective sample size of Wilks (2019),

    n_eff = T * (1 - r1) / (1 + r1),

with r1 the lag-1 autocorrelation of the daily series. Confidence intervals
and the t test use n_eff in place of T, which widens the standard error by
sqrt((1 + r1) / (1 - r1)).

Predictions come from ``evaluate.load_or_predict``, i.e. the cached bundle, so
once ``evaluate.py`` has run for a regime this script costs no GPU at all.

Outputs
-------
    <model_dir>/bias_significance_{cv,<year>}.json

Example
-------
    python scripts/gen_bias_significance.py --model-dir CLEAN_trained_models/<trial>/tmax
    python scripts/gen_bias_significance.py --model-dir CLEAN_trained_models/<trial>/tmax --eval-year 2024
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import params as params_mod
from infer import _resolve_model_dir
import evaluate as ev

logger = logging.getLogger("bias_significance")

BEST_ATMOS = "CLEAN_trained_models/lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8"


def daily_domain_mean_bias(model_dir: Path, device, eval_year=None,
                           refresh: bool = False, use_cache: bool = True):
    """One value per day: the domain mean of (prediction - truth), in degrees C."""
    B = ev.load_or_predict(model_dir, device, eval_year=eval_year,
                           refresh=refresh, use_cache=use_cache)
    keep = np.asarray(B.day_mask, dtype=bool)
    sl = slice(None) if keep.all() else keep
    errors = np.asarray(B.errors_c)[sl]          # (days, points), degC
    dates = np.asarray(B.dates)[sl]
    return np.nanmean(errors, axis=1), dates


def bias_significance(daily: np.ndarray, alpha: float = 0.05) -> dict:
    """Effective-sample-size t test on a daily series of domain-mean biases.

    The series is chronological, so the lag-1 autocorrelation is taken directly
    on consecutive entries. n_eff is clipped to at least 2 so the t
    distribution stays defined even for a pathologically persistent series.
    """
    b = np.asarray(daily, dtype=float)
    b = b[np.isfinite(b)]
    T = b.size
    if T < 3:
        raise ValueError(f"need at least 3 days to test a bias, got {T}")

    mean = float(b.mean())
    s = float(b.std(ddof=1))
    r1 = float(np.corrcoef(b[:-1], b[1:])[0, 1])

    # A negative lag-1 correlation would inflate n_eff above T, which the
    # correction is not meant to do; floor the ratio at 1 day of information.
    n_eff = float(np.clip(T * (1.0 - r1) / (1.0 + r1), 2.0, float(T)))
    inflation = float(np.sqrt((1.0 + r1) / (1.0 - r1))) if r1 < 1.0 else float("inf")

    se = s / np.sqrt(T)
    se_adj = s / np.sqrt(n_eff)
    dof = n_eff - 1.0
    t_stat = mean / se_adj if se_adj > 0 else float("nan")
    p_value = float(2.0 * sstats.t.sf(abs(t_stat), df=dof))
    t_crit = float(sstats.t.ppf(1.0 - alpha / 2.0, df=dof))
    half_width = t_crit * se_adj

    return {
        "n_days": int(T),
        "mean_bias_degC": mean,
        "sd_of_daily_means_degC": s,
        "lag1_autocorr": r1,
        "n_eff": n_eff,
        "n_eff_fraction": n_eff / T,
        "se_naive_degC": float(se),
        "se_adjusted_degC": float(se_adj),
        "interval_inflation": inflation,
        "dof": dof,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "ci95_low_degC": float(mean - half_width),
        "ci95_high_degC": float(mean + half_width),
        "significant_at_5pct": bool(p_value < alpha),
    }


def run(model_dir: Path, device, out_path: Path, eval_year=None,
        refresh_cache: bool = False, use_cache: bool = True) -> dict:
    daily, dates = daily_domain_mean_bias(model_dir, device, eval_year=eval_year,
                                          refresh=refresh_cache, use_cache=use_cache)
    result = bias_significance(daily)
    result["regime"] = "cv_holdout" if eval_year is None else f"holdout_year_{eval_year}"
    result["eval_year"] = eval_year
    result["model_dir"] = str(model_dir)
    result["first_day"] = str(np.asarray(dates)[0])
    result["last_day"] = str(np.asarray(dates)[-1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"{result['regime']}: bias {result['mean_bias_degC']:+.3f} degC "
          f"[{result['ci95_low_degC']:+.3f}, {result['ci95_high_degC']:+.3f}] 95% CI | "
          f"r1 {result['lag1_autocorr']:.3f} | "
          f"n_eff {result['n_eff']:.0f} of {result['n_days']} days "
          f"({100 * result['n_eff_fraction']:.0f}%) | "
          f"intervals x{result['interval_inflation']:.2f} | "
          f"p {result['p_value']:.3g} "
          f"({'significant' if result['significant_at_5pct'] else 'not significant'} at 5%)")
    print("wrote", out_path)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--model-dir", help="Dir with params.json / manifest.json / model_fold_*.")
    src.add_argument("--trial-dir", default=BEST_ATMOS,
                     help=f"Trial dir (default the selected model: {BEST_ATMOS}).")
    ap.add_argument("--eval-year", type=int, default=None,
                    help="Holdout year (e.g. 2024). Omit for the 2020-2023 CV holdout.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None,
                    help="Output JSON (default <model_dir>/bias_significance_{cv,<year>}.json).")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Re-run inference instead of reusing evaluate.py's bundle.")
    ap.add_argument("--no-cache", dest="use_cache", action="store_false",
                    help="Neither read nor write the prediction bundle.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    params_mod.configure_renku_cuda()
    device = torch.device(args.device) if args.device else params_mod.select_device()
    model_dir = _resolve_model_dir(args)
    out_path = Path(args.out) if args.out else (
        model_dir / ("bias_significance_cv.json" if args.eval_year is None
                     else f"bias_significance_{args.eval_year}.json"))
    run(model_dir, device, out_path, eval_year=args.eval_year,
        refresh_cache=args.refresh_cache, use_cache=args.use_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
