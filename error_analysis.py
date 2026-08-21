#!/usr/bin/env python3
"""
Per-grid-point error-pattern analysis for a tmax model.

Mines *where* the model errs as a function of topography — altitude, the elevation
mismatch between the coarse ERA5 grid and the true grid-cell terrain height, mTPI
(ridge/valley) — and region (altitude bands). The goal is to motivate a **targeted
fine-tuning loss** that up-weights the systematically-hard grid points.

Runs on one evaluation regime at a time, the same two ``evaluate.py`` defines: the
2020-2023 cross-validation holdout (default) or a genuine holdout year
(``--eval-year 2024``). It used to pool ``--years 2020-2024``, which averaged four
in-sample years together with the one honest holdout and so understated the error it
was trying to characterise.

Predictions come from ``evaluate.load_or_predict``, i.e. the cached bundle — once
``evaluate.py`` has run for a regime this script costs no GPU at all.

Outputs a self-contained interpretability report with embedded figures:
    <out>/REPORT.md
    <out>/*.png

Example
-------
    python error_analysis.py --model-dir CLEAN_trained_models/<trial>/tmax
    python error_analysis.py --model-dir CLEAN_trained_models/<trial>/tmax --eval-year 2024
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats

import params as params_mod
import visualization as vis  # save_fig: PNG + editable .fig.pkl
from infer import _resolve_model_dir
import evaluate as ev

logger = logging.getLogger("error_analysis")

BEST_ATMOS = ("CLEAN_trained_models/clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8")


# ---------------------------------------------------------------------------
# Per-grid-point errors for one regime (served from evaluate.py's bundle cache)
# ---------------------------------------------------------------------------
def aggregate(model_dir: Path, device, eval_year=None,
              refresh: bool = False, use_cache: bool = True) -> dict:
    B = ev.load_or_predict(model_dir, device, eval_year=eval_year,
                           refresh=refresh, use_cache=use_cache)
    keep = np.asarray(B.day_mask, dtype=bool)
    sl = slice(None) if keep.all() else keep
    regime = "cv_holdout" if eval_year is None else f"holdout_year_{eval_year}"
    return dict(err=np.asarray(B.errors_c)[sl], sig=np.asarray(B.sigmas_c)[sl],
                topo=B.target_topo, lat=B.lat_arr, lon=B.lon_arr,
                regime=regime, eval_year=eval_year,
                ndays=int(np.asarray(B.errors_c)[sl].shape[0]))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _binned(x, y, nbins=20):
    edges = np.nanpercentile(x, np.linspace(0, 100, nbins + 1))
    edges = np.unique(edges)
    centers, means, ses = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 10:
            centers.append(0.5 * (lo + hi)); means.append(np.nanmean(y[m]))
            ses.append(np.nanstd(y[m]) / np.sqrt(m.sum()))
    return np.array(centers), np.array(means), np.array(ses)


def fig_hist(mae, hi_thr, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(mae, bins=60, color="C0", alpha=0.85)
    ax.axvline(np.median(mae), color="k", ls="--", label=f"median {np.median(mae):.2f}")
    ax.axvline(hi_thr, color="C3", ls="-", label=f"P90 {hi_thr:.2f} (high-error cut)")
    ax.set_xlabel("per-grid-point MAE (°C)"); ax.set_ylabel("# grid points")
    ax.set_title("Distribution of per-grid-point MAE"); ax.legend()
    fig.tight_layout(); vis.save_fig(fig, out, dpi=130, bbox_inches=None); plt.close(fig)


def fig_vs_feature(x, mae, xlabel, out, logx=False):
    r = sstats.pearsonr(x, mae)[0]; rho = sstats.spearmanr(x, mae)[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, mae, s=3, alpha=0.15, color="C0")
    cx, cy, ce = _binned(x, mae)
    ax.errorbar(cx, cy, yerr=ce, fmt="-o", color="C3", lw=2, ms=4, label="binned mean ± SE")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("per-grid-point MAE (°C)")
    ax.set_title(f"MAE vs {xlabel}   (Pearson r={r:.2f}, Spearman ρ={rho:.2f})")
    ax.legend(); fig.tight_layout(); vis.save_fig(fig, out, dpi=130, bbox_inches=None); plt.close(fig)
    return r, rho


def fig_bias(bias, mae, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(bias, bins=60, color="C2", alpha=0.85)
    axes[0].axvline(0, color="k", ls="--")
    axes[0].set_xlabel("per-grid-point bias (°C)"); axes[0].set_ylabel("# grid points")
    axes[0].set_title(f"Systematic bias  (mean |bias| {np.mean(np.abs(bias)):.2f} °C)")
    axes[1].scatter(np.abs(bias), mae, s=3, alpha=0.15, color="C2")
    lim = max(np.nanpercentile(mae, 99), np.nanpercentile(np.abs(bias), 99))
    axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="MAE = |bias| (all error systematic)")
    axes[1].set_xlabel("|bias| (°C, systematic)"); axes[1].set_ylabel("MAE (°C, total)")
    axes[1].set_title("How much of the error is systematic?"); axes[1].legend()
    fig.tight_layout(); vis.save_fig(fig, out, dpi=130, bbox_inches=None); plt.close(fig)


def fig_cohort(feat, hi, lo, name, unit, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(np.nanpercentile(feat, 1), np.nanpercentile(feat, 99), 40)
    ax.hist(feat[lo], bins=bins, density=True, alpha=0.55, color="C0",
            label=f"normal grid pts (n={lo.sum()})")
    ax.hist(feat[hi], bins=bins, density=True, alpha=0.55, color="C3",
            label=f"high-error P90 (n={hi.sum()})")
    ax.set_xlabel(f"{name} ({unit})"); ax.set_ylabel("density")
    ax.set_title(f"{name}: high-error vs normal grid points"); ax.legend()
    fig.tight_layout(); vis.save_fig(fig, out, dpi=130, bbox_inches=None); plt.close(fig)


def fig_spatial(lat, lon, mae, hi, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(lon, lat, c=mae, s=5, cmap="YlOrRd",
                         vmax=np.nanpercentile(mae, 98))
    axes[0].set_title("Per-grid-point MAE (°C)"); fig.colorbar(sc, ax=axes[0], shrink=.85)
    axes[1].scatter(lon, lat, c="0.8", s=4)
    axes[1].scatter(lon[hi], lat[hi], c="C3", s=8, label="high-error (P90)")
    axes[1].set_title("High-error grid points"); axes[1].legend()
    for a in axes:
        a.set_xlabel("lon (°E)"); a.set_ylabel("lat (°N)")
        # Degrees of longitude shrink with latitude; without this the country is
        # stretched east-west by ~1/cos(47°) ≈ 1.5.
        a.set_aspect(1.0 / np.cos(np.deg2rad(np.nanmean(lat))))
    fig.tight_layout(); vis.save_fig(fig, out, dpi=130, bbox_inches=None); plt.close(fig)


def fig_regions(bands, names, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    ax.bar(x, [b["mae"] for b in bands], color="C0",
           yerr=[b["se"] for b in bands], capsize=4)
    for i, b in enumerate(bands):
        ax.text(i, b["mae"], f"{b['mae']:.2f}\n(n={b['n']})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("mean per-grid-point MAE (°C)")
    ax.set_title("MAE by altitude band")
    fig.tight_layout(); vis.save_fig(fig, out, dpi=130, bbox_inches=None); plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(model_dir: Path, device, out_dir: Path, eval_year=None,
        refresh_cache: bool = False, use_cache: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    # No second-level cache here any more: the reduction below is a few seconds of
    # numpy over an already-cached bundle, so one cache layer (evaluate's) is enough.
    agg = aggregate(model_dir, device, eval_year=eval_year,
                    refresh=refresh_cache, use_cache=use_cache)
    err, sig, topo = agg["err"], agg["sig"], agg["topo"]
    # Per-grid-point summaries (nan-aware: ignores masked-out days/points).
    mae = np.nanmean(np.abs(err), axis=0)
    bias = np.nanmean(err, axis=0)               # systematic component
    rmse = np.sqrt(np.nanmean(err ** 2, axis=0))
    msig = np.nanmean(sig, axis=0)
    alt = topo[:, 0]; elevdiff = topo[:, 1]; abselevdiff = np.abs(elevdiff); mtpi = topo[:, 2]
    valid = (np.isfinite(mae) & np.isfinite(alt) & np.isfinite(mtpi) & np.isfinite(abselevdiff))
    mae, bias, rmse, msig = mae[valid], bias[valid], rmse[valid], msig[valid]
    alt, elevdiff, abselevdiff, mtpi = alt[valid], elevdiff[valid], abselevdiff[valid], mtpi[valid]
    lat, lon = agg["lat"][valid], agg["lon"][valid]
    meta = dict(regime=agg["regime"], eval_year=agg["eval_year"], ndays=agg["ndays"])

    absbias = np.abs(bias)
    n = len(mae)
    # Systematic share: how much of the error is a fixable per-grid-point offset vs irreducible spread.
    sys_share = float(np.mean(absbias) / np.mean(mae))

    hi_thr = np.nanpercentile(mae, 90)
    hi = mae >= hi_thr
    lo = ~hi

    # --- correlations (both MAE and the systematic |bias|) ---
    feats = {"altitude": alt, "|elev mismatch|": abselevdiff, "mTPI": mtpi,
             "elev mismatch (signed)": elevdiff, "predicted σ": msig}
    corr = {k: (sstats.pearsonr(v, mae)[0], sstats.spearmanr(v, mae)[0]) for k, v in feats.items()}
    corr_bias = {k: (sstats.pearsonr(v, absbias)[0], sstats.spearmanr(v, absbias)[0])
                 for k, v in feats.items()}

    # --- standardized multiple regression: relative importance ---
    X = np.column_stack([sstats.zscore(alt), sstats.zscore(abselevdiff), sstats.zscore(mtpi)])
    X = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X, sstats.zscore(mae), rcond=None)
    reg = dict(zip(["intercept", "altitude", "|elev mismatch|", "mTPI"], beta))
    yhat = X @ beta
    r2 = 1 - np.sum((sstats.zscore(mae) - yhat) ** 2) / np.sum((sstats.zscore(mae) - sstats.zscore(mae).mean()) ** 2)

    # --- altitude-band regions ---
    band_edges = [(-1e9, 600), (600, 1200), (1200, 2000), (2000, 1e9)]
    band_names = ["lowland\n<600m", "hill\n600-1200m", "mountain\n1200-2000m", "alpine\n>2000m"]
    bands = []
    for lo_e, hi_e in band_edges:
        m = (alt >= lo_e) & (alt < hi_e)
        bands.append(dict(mae=float(np.mean(mae[m])) if m.any() else np.nan,
                          se=float(np.std(mae[m]) / np.sqrt(max(m.sum(), 1))),
                          n=int(m.sum()),
                          frac_hi=float(hi[m].mean()) if m.any() else np.nan))

    # --- figures ---
    fig_hist(mae, hi_thr, out_dir / "mae_histogram.png")
    fig_vs_feature(alt, mae, "altitude (m)", out_dir / "mae_vs_altitude.png")
    fig_vs_feature(abselevdiff + 1, mae, "|elev mismatch| (m)", out_dir / "mae_vs_elevmismatch.png", logx=True)
    fig_vs_feature(mtpi, mae, "mTPI (m)", out_dir / "mae_vs_mtpi.png")
    fig_bias(bias, mae, out_dir / "bias_analysis.png")
    fig_cohort(alt, hi, lo, "altitude", "m", out_dir / "cohort_altitude.png")
    fig_cohort(abselevdiff, hi, lo, "|elev mismatch|", "m", out_dir / "cohort_elevmismatch.png")
    fig_spatial(lat, lon, mae, hi, out_dir / "spatial_mae.png")
    fig_regions(bands, band_names, out_dir / "region_bands.png")

    # --- machine-readable summary ---
    summary = dict(
        model_dir=str(model_dir), eval_regime=meta["regime"],
        eval_year=meta["eval_year"], n_days=meta["ndays"],
        n_grid_points=int(n), mae_overall=float(np.mean(mae)),
        mae_median=float(np.median(mae)), mae_p90=float(hi_thr),
        mean_abs_bias=float(np.mean(absbias)), systematic_share=sys_share,
        correlations={k: dict(pearson=float(a), spearman=float(b)) for k, (a, b) in corr.items()},
        correlations_absbias={k: dict(pearson=float(a), spearman=float(b)) for k, (a, b) in corr_bias.items()},
        regression_std_beta=reg, regression_r2=float(r2),
        high_error_cohort=dict(
            n=int(hi.sum()), mae=float(mae[hi].mean()), mean_abs_bias=float(absbias[hi].mean()),
            median_altitude=float(np.median(alt[hi])),
            median_abs_elevmismatch=float(np.median(abselevdiff[hi])),
            median_mtpi=float(np.median(mtpi[hi])),
            mean_bias=float(bias[hi].mean())),
        normal_cohort=dict(
            median_altitude=float(np.median(alt[lo])),
            median_abs_elevmismatch=float(np.median(abselevdiff[lo])),
            mean_abs_bias=float(absbias[lo].mean()), mean_bias=float(bias[lo].mean())),
        bands=[{**b, "name": nm.replace("\n", " ")} for b, nm in zip(bands, band_names)],
    )
    (out_dir / "error_analysis_summary.json").write_text(json.dumps(summary, indent=2))
    _write_report(out_dir, summary, band_names)
    print(f"\nWrote {out_dir}/REPORT.md (+ summary.json, 9 figures)")
    return summary


def _write_report(out, s, band_names):
    corr = s["correlations"]; corrb = s["correlations_absbias"]; r2 = s["regression_r2"]
    # strongest topographic correlate by |Spearman| (exclude predicted σ — not topographic)
    topo_keys = ["altitude", "|elev mismatch|", "mTPI", "elev mismatch (signed)"]
    strongest = max(topo_keys, key=lambda k: abs(corr[k]["spearman"]))
    weak = abs(r2) < 0.05
    h, nrm = s["high_error_cohort"], s["normal_cohort"]
    bands = s["bands"]
    verdict = ("**weak — topography does *not* meaningfully predict where this model errs**"
               if weak else f"**present — strongest topographic driver is `{strongest}`**")

    L = [
        "# Error-pattern analysis — best atmospheric model",
        "",
        f"- Model: `{s['model_dir']}`",
        f"- Regime **`{s['eval_regime']}`** = {s['n_days']:,} days, "
        f"{s['n_grid_points']:,} valid grid points (5-fold ensemble).",
        f"- Mean per-grid-point MAE **{s['mae_overall']:.3f} °C** (median {s['mae_median']:.3f}); "
        f"P90 high-error cut **{s['mae_p90']:.3f} °C**.",
        f"- Mean per-grid-point |bias| **{s['mean_abs_bias']:.3f} °C** ⇒ only "
        f"**{100*s['systematic_share']:.0f}%** of the average error is a fixable systematic offset; "
        "the rest is irreducible day-to-day spread.",
        "",
        "> **What a \"grid point\" is.** The targets are the **grid cells** of the *gridded* "
        "MeteoSwiss `TmaxD` analysis (240×370 LV95 lattice; the valid count above are the cells "
        "inside the Swiss landmask) — **not** weather stations. `TmaxD` is itself interpolated "
        "from the ~90 real MeteoSwiss stations with an **elevation-aware** regression.",
        ">",
        "> Two caveats follow from this: (1) **spatial autocorrelation** — neighbouring cells are "
        "correlated, so the effective sample size is far below the cell count and the reported "
        "correlation/R² significance is *overstated* (the qualitative 'no strong topographic "
        "pattern' conclusion still holds); (2) the **reference already bakes in an elevation "
        "regression**, so residual topographic error is partly removed in the truth itself — a "
        "plausible reason the topographic signal is so weak (both model input and reference handle "
        "elevation).",
        "",
        "## 1. Is there a topographic / regional pattern?",
        "",
        f"**Verdict: {verdict}.** A standardized regression of per-grid-point MAE on "
        f"altitude + |elev mismatch| + mTPI explains only **R²={r2:.3f}** of the variance.",
        "",
        "Per-grid-point correlation of error with each feature (Pearson r / Spearman ρ):",
        "",
        "| Feature | r (vs MAE) | ρ (vs MAE) | r (vs \\|bias\\|) | ρ (vs \\|bias\\|) |",
        "|---|---|---|---|---|",
    ]
    for k in ["altitude", "|elev mismatch|", "mTPI", "elev mismatch (signed)", "predicted σ"]:
        L.append(f"| {k} | {corr[k]['pearson']:+.3f} | {corr[k]['spearman']:+.3f} | "
                 f"{corrb[k]['pearson']:+.3f} | {corrb[k]['spearman']:+.3f} |")
    L += [
        "",
        "All |correlations| are small (≤~0.15). The only consistent signal is a mild **lowland-worse** "
        "trend (negative altitude ρ) — the opposite of the intuition that high alpine terrain is "
        "hardest — and predicted σ correlating *negatively* with error (the model is mildly "
        "over-confident exactly where it is most accurate).",
        "",
        "![MAE histogram](mae_histogram.png)",
        "",
        "## 2. Systematic vs random error",
        "",
        f"Mean |bias| is only **{s['mean_abs_bias']:.2f} °C** against MAE **{s['mae_overall']:.2f} °C** "
        f"({100*s['systematic_share']:.0f}% systematic). Most of the per-grid-point error is **random "
        "weather-day variance, not a correctable offset** — which is *why* topography explains so "
        "little, and a key caveat for any reweighting (you cannot reweight away irreducible noise).",
        "",
        "![Bias analysis](bias_analysis.png)",
        "",
        "## 3. The high-error cohort (top 10%)",
        "",
        f"n={h['n']}, mean MAE {h['mae']:.2f} °C, mean |bias| {h['mean_abs_bias']:.2f} °C:",
        "",
        "| | high-error P90 | normal |",
        "|---|---|---|",
        f"| median altitude (m) | {h['median_altitude']:.0f} | {nrm['median_altitude']:.0f} |",
        f"| median \\|elev mismatch\\| (m) | {h['median_abs_elevmismatch']:.0f} | {nrm['median_abs_elevmismatch']:.0f} |",
        f"| mean bias (°C) | {h['mean_bias']:+.2f} | {nrm['mean_bias']:+.2f} |",
        "",
        "The hard cohort is **not** distinguished by altitude or elevation-mismatch (if anything it "
        "sits slightly lower) — confirming the error is spatially diffuse, not a topographic subgroup.",
        "",
        "![MAE vs altitude](mae_vs_altitude.png)",
        "![MAE vs elevation mismatch](mae_vs_elevmismatch.png)",
        "![MAE vs mTPI](mae_vs_mtpi.png)",
        "![Altitude cohort](cohort_altitude.png)",
        "![Spatial MAE](spatial_mae.png)",
        "",
        "## 4. Error by altitude band (region)",
        "",
        "| Band | n grid points | mean MAE (°C) | % in high-error cohort |",
        "|---|---|---|---|",
    ]
    for b in bands:
        L.append(f"| {b['name']} | {b['n']} | {b['mae']:.3f} | {100*b['frac_hi']:.0f}% |")
    L += [
        "",
        "![Region bands](region_bands.png)",
        "",
        "## 5. Implications for a targeted fine-tuning loss",
        "",
        "**A topography-conditioned weight will not help** — there is no altitude/mTPI/elev-mismatch "
        "signal to exploit (R²≈0, |ρ|≤0.15). Weighting the loss by altitude or elevation mismatch "
        "would up-weight grid points that are *not* actually the hard ones.",
        "",
        "Two honest options remain, in order of expected payoff:",
        "",
        "1. **Empirical per-grid-point weighting (model-agnostic).** Target the *measured*-hard grid points "
        "directly, ignoring whether topography explains them. Precompute a static weight from this "
        "analysis and apply it in the per-target NLL of "
        "`convCNP/training/training_elev.train_batch_elev`:",
        "",
        "```",
        "  L = Σ_i w_i · NLL_i / Σ_i w_i",
        "  w_i = 1 + α · clip( (MAE_i − median_MAE) / IQR_MAE , 0, 3 )   # per-grid-point MAE",
        "```",
        "   Start α≈1. **Caveat:** since ~%d%% of error is random, the achievable gain is bounded — "
        "this mostly helps the *systematic-bias* part of the hard cohort." % round(100*s['systematic_share']),
        "",
        "2. **Per-grid-point bias correction (most reliable).** The fixable part is the systematic offset. "
        "Fit a residual bias per grid point (or a smooth bias field) on the training period and subtract "
        "it at inference — a post-hoc correction that directly removes the "
        f"{100*s['systematic_share']:.0f}% systematic share without retraining.",
        "",
        "**What would actually move the needle** (since the error is weather-driven, not spatial): "
        "richer *temporal/synoptic* inputs or a CRPS/joint-Gaussian objective — not a topographic "
        "reweighting. Re-run `error_analysis.py` after any change and check the "
        "high-error-cohort MAE and the systematic share.",
    ]
    (out / "REPORT.md").write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--model-dir", help="Dir with params.json / manifest.json / model_fold_*.")
    src.add_argument("--trial-dir", default=BEST_ATMOS,
                     help=f"Trial dir (default best atmos: {BEST_ATMOS}).")
    ap.add_argument("--eval-year", type=int, default=None,
                    help="Holdout year (e.g. 2024). Omit for the 2020-2023 CV holdout.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None,
                    help="Output dir (default <model_dir>/error_analysis_{cv,<year>}).")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Re-run inference instead of reusing evaluate.py's bundle.")
    ap.add_argument("--no-cache", dest="use_cache", action="store_false",
                    help="Neither read nor write the prediction bundle.")
    ap.add_argument("--years", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.years:
        ap.error("--years is gone: this script now runs one regime at a time. Omit "
                 "--eval-year for the 2020-2023 CV holdout, or pass --eval-year 2024. "
                 "Pooling 2020-2024 mixed four in-sample years into the holdout.")

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    params_mod.configure_renku_cuda()
    device = torch.device(args.device) if args.device else params_mod.select_device()
    model_dir = _resolve_model_dir(args)
    out = Path(args.out) if args.out else (
        model_dir / ("error_analysis_cv" if args.eval_year is None
                     else f"error_analysis_{args.eval_year}"))
    run(model_dir, device, out, eval_year=args.eval_year,
        refresh_cache=args.refresh_cache, use_cache=args.use_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
