#!/usr/bin/env python3
"""Compare the five CLEAN solo tmax models across the two evaluation regimes.

Mirrors ``compare_precip_models.py``. Reads the ``report_metrics.json`` that
``evaluate.py`` writes for each model x regime and produces, under
``CLEAN_trained_models/tmax_model_comparison/``:

  * ``metrics_comparison.{md,json,csv}`` — every headline metric for each model in
    both regimes, plus the CV → holdout delta (the honest generalization cost).
  * ``regime_comparison.png`` — grouped bars, CV vs 2024 holdout, per metric.
  * ``perfold_spread.png`` — the per-fold CV spread, which only exists because the CV
    regime scores each fold on its own held-out block.

The two regimes:
  CV 2020-2023   each fold predicts only the contiguous block it was held out from,
                 so every day is scored by a model that never saw it.
  2024 holdout   a year no fold saw; all folds ensembled by Gaussian moment matching
                 with training-frozen normalization.

The previous version of this script compared "2023 holdout" against "<year> unseen"
for four runs trained on 2023 alone. Those runs are retired, and the
framing no longer applies: with the CV regime in place, the in-sample ensemble score
it used as a reference is not something we compute any more.

Skill vs ERA5 IS comparable across all five models — verified on every run, not assumed.
``evaluate.era5_reference_degC`` branches on ``USE_SURFACE``, but both branches end up
sampling the same ERA5-Land surface tmax file (``ERA5_MAX_TEMP_GLOB``, identical in all
five ``params.json``) at the same MeteoSwiss target points, denormalised with the same
ERA5-Land domain bounds (``manifest.dists_grid.lat_bounds/lon_bounds``). The atmospheric
runs' coarse 0.25° ``lat_coords`` are never used for the reference. Skill is therefore
``1 - CRPS/CRPS_ref`` with one shared constant per regime: a strictly monotone rescaling
of CRPS that ranks the models identically.

Rather than trust that, this script re-derives each model's reference as
``crps_degC / (1 - skill_score)`` — recoverable from the report alone, no bundle access —
and requires the models to agree to ``REF_TOL_DEGC``. They currently do, to ~1e-15 (CV
2.560 °C, 2024 2.543 °C). If a future run changes ``ERA5_MAX_TEMP_GLOB`` or its manifest
bounds, the check fails loudly and the skill panel and table are relabelled as not
comparable.

(An earlier version of this docstring claimed the opposite — that surface and atmospheric
models did not share a reference. That was wrong: the reference arrays are bit-identical
for the CV regime and agree to 3e-05 °C on 2024 float32 round-trip. The *input* grids do
differ, 29x61 vs 11x23; the skill reference does not. Do not re-derive the old claim from
the input-grid difference.)

Example
-------
    python compare_evaluations.py
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "CLEAN_trained_models" / "tmax_model_comparison"

# label -> model dir, in the order they should appear. Each gradient-clipped 60-epoch
# retrain sits immediately after the 30-epoch unclipped run it should be read against —
# that pairing is the experiment (`lbclip_*`, which flipped the wind verdict).
DEFAULT_MODELS = {
    "atm":             "CLEAN_trained_models/clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8/tmax",
    "atm-clip60":      "CLEAN_trained_models/lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8/tmax",
    "atm+wind":        "CLEAN_trained_models/clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8/tmax",
    "atm+wind-clip60": "CLEAN_trained_models/lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8/tmax",
    "atm+sfcanchors":  "CLEAN_trained_models/sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8/tmax",
    "sfc":             "CLEAN_trained_models/baseline__tmax_sfc_flat_y2020-2023_e30f5_b8/tmax",
    "sfc-nogeo":       "CLEAN_trained_models/baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo/tmax",
}

# regime key -> (report dir name, display label)
REGIMES = [("cv", "eval_cv", "CV 2020-2023"),
           ("2024", "eval_2024", "2024 holdout")]

# Shared model styling — imported by scripts/overlay_tmax_distributions.py so a model
# looks the same in every figure in the report. Hue = input family; the variant within a
# family (clipped retrain, or surface-without-geopotential) is a dashed line / hatched
# bar, never a second shade of the same hue. Four hues is the most that clears the
# all-pairs colour gates (worst CVD ΔE 9.2, worst normal-vision ΔE 16.3, OKLab×100);
# a fifth hue fails whichever one is picked, which is why the variant is a texture.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
# label -> (color, linestyle for lines, hatch for bars)
MODEL_STYLE = {
    "atm":             (BLUE,   "-",  ""),
    "atm-clip60":      (BLUE,   "--", "//"),
    "atm+wind":        (AQUA,   "-",  ""),
    "atm+wind-clip60": (AQUA,   "--", "//"),
    "atm+sfcanchors":  (VIOLET, "-",  ""),
    "sfc":             (ORANGE, "-",  ""),
    "sfc-nogeo":       (ORANGE, "--", "//"),
}
FALLBACK_COLOR = "#7a7a7a"
# In the bar charts the x-axis already names the model, so the hue is free to carry model
# identity (matching the overlays) and the regime is the hatch instead.
REGIME_HATCH = {"cv": "", "2024": "///"}


def model_color(label: str) -> str:
    return MODEL_STYLE.get(label, (FALLBACK_COLOR,))[0]

# (id, json section, key, display label, "good" direction)
METRICS = [
    ("mae",         "overall",     "mae_degC",    "MAE (°C)",           "lower"),
    ("rmse",        "overall",     "rmse_degC",   "RMSE (°C)",          "lower"),
    ("bias",        "overall",     "bias_degC",   "Bias (°C)",          "zero"),
    ("crps",        "overall",     "crps_degC",   "CRPS (°C)",          "lower"),
    ("skill",       "overall",     "skill_score", "Skill vs ERA5",      "higher"),
    ("z_std",       "calibration", "z_std",       "z-std (→1)",         "one"),
    ("coverage_90", "calibration", "coverage_90", "Coverage@90 (→0.9)", "p90"),
    ("mace",        "reliability", "mace",        "MACE (→0)",          "zero"),
]
# How far the per-model ERA5 reference may drift before skill stops being comparable.
# Observed agreement is ~1e-15 °C; a changed source file or domain would move it by >0.01.
REF_TOL_DEGC = 1e-3
STATION_REF_TOL_DEGC = 1e-3

# Station-level pooled metrics, from station_analysis.py's summary.json.
# (id, summary.json key, display label)
STATION_METRICS = [
    ("st_mae",       "mae_overall",      "Station MAE (°C)"),
    ("st_crps",      "crps_overall",     "Station CRPS (°C)"),
    ("st_skill",     "skill_overall",    "Skill vs ERA5 (pooled)"),
    ("st_skill_low", "skill_lowland",    "Skill, lowland <1000 m"),
    ("st_skill_med", "skill_median",     "Skill, per-station median"),
    ("st_bias",      "mean_bias",        "Mean bias (°C)"),
    ("st_abs_bias",  "mean_abs_bias",    "Mean |per-station bias| (°C)"),
    ("st_sys_share", "systematic_share", "Systematic share of MAE"),
    ("st_mae_era5",  "mae_era5_overall", "ERA5 baseline MAE (°C)"),
]


def _read(path: Path) -> dict | None:
    # json.loads (unlike a strict parser) accepts the bare NaN literals that
    # station_analysis.py writes for stations with too few valid days.
    return json.loads(path.read_text()) if path.exists() else None


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _baseline_check(rows, key: str, tol: float, what: str) -> dict[str, dict]:
    """Per regime: do all models share the same skill reference?

    Returns {regime_key: {value, spread, consistent, per_model, n, unverified}}. This is a
    check, not an assumption — the whole point is that a future model changing its ERA5
    source or domain bounds must be caught rather than silently ranked against a different
    baseline. Deliberately not an `assert`: `python -O` strips those, and the tables stay
    valid either way; only skill's *cross-model* reading changes.
    """
    out: dict[str, dict] = {}
    for rkey, _dirname, disp in REGIMES:
        per_model = {r["model"]: r.get(key) for r in rows if r["regime"] == rkey}
        vals = {m: v for m, v in per_model.items() if _finite(v)}
        n = len(vals)
        spread = (max(vals.values()) - min(vals.values())) if n else float("nan")
        consistent = (spread <= tol) if n >= 2 else True
        out[rkey] = {
            "value": (sum(vals.values()) / n) if n else None,
            "spread": spread, "consistent": consistent, "n": n,
            "unverified": n < 2, "per_model": per_model,
        }
        if not consistent:
            print(f"[WARN] {what}: models do NOT share a reference in the {disp} regime "
                  f"(spread {spread:.4f} °C > tol {tol}).", file=sys.stderr)
            for m, v in per_model.items():
                print(f"[WARN]   {m:16s} {v}", file=sys.stderr)
            print("[WARN]   check ERA5_MAX_TEMP_GLOB in each params.json and "
                  "dists_grid.lat_bounds/lon_bounds in each manifest.json.", file=sys.stderr)
        elif n == 1:
            print(f"[warn] {what}: only one model in the {disp} regime — nothing to "
                  f"cross-check, so comparability is unverified (not confirmed).")
    return out


def collect(models: dict[str, Path]) -> list[dict]:
    """One row per (model, regime)."""
    rows = []
    for label, mdir in models.items():
        mdir = Path(mdir)
        for key, dirname, disp in REGIMES:
            j = _read(mdir / dirname / "report_metrics.json")
            if not j:
                print(f"[warn] {label}: no {dirname}/report_metrics.json under {mdir} — "
                      f"run `python evaluate.py --model-dir {mdir}"
                      f"{'' if key == 'cv' else ' --eval-year ' + key}`")
                continue
            row = {"model": label, "regime": key, "regime_label": disp,
                   "eval_days": j.get("eval_days") or j.get("holdout_days"),
                   "eval_regime": j.get("eval_regime", "?"),
                   "prediction_mode": j.get("prediction_mode", "?"),
                   "per_fold": j.get("per_fold", {})}
            row.update({mid: j.get(section, {}).get(k)
                        for mid, section, k, _, _ in METRICS})
            # skill = 1 - crps/crps_ref, so the reference the skill was measured against is
            # recoverable from the report alone — no access to the gitignored bundles.
            crps, skill = row.get("crps"), row.get("skill")
            if _finite(crps) and _finite(skill) and abs(1.0 - skill) > 1e-3:
                row["ref_mae_degC"] = crps / (1.0 - skill)
            else:
                row["ref_mae_degC"] = None
                if _finite(crps) and _finite(skill):
                    # 1/(1-skill) amplifies rounding; near skill=1 the derivation is junk.
                    print(f"[warn] {label}/{key}: skill {skill:.6f} is too close to 1 to "
                          f"recover the ERA5 reference — it drops out of the comparability "
                          f"check rather than contributing a garbage value.")
            rows.append(row)
    return rows


def _get(rows, model, regime, mid):
    for r in rows:
        if r["model"] == model and r["regime"] == regime:
            return r.get(mid)
    return None


def _fmt(v, mid):
    if not _finite(v):
        return "—"
    return f"{100*v:.1f}%" if mid == "coverage_90" else f"{v:.3f}"


def _delta(cv, ho, mid):
    if not (_finite(cv) and _finite(ho)):
        return "—"
    return f"{100*(ho-cv):+.1f}pp" if mid == "coverage_90" else f"{ho-cv:+.3f}"


def _model_table(models) -> list[str]:
    """label -> run dir, so the short axis labels stay traceable to a run on disk."""
    L = ["## Models", "", "| Label | Run |", "|---|---|"]
    for label, mdir in models.items():
        run = Path(mdir).parent.name if Path(mdir).name == "tmax" else Path(mdir).name
        L.append(f"| {label} | `{run}` |")
    return L + [""]


def _baseline_note(baselines, skill_ok, tol) -> list[str]:
    """The skill-comparability blockquote — formatted from the check, never hardcoded."""
    if not skill_ok:
        bad = [(d, baselines[k]) for k, _dn, d in REGIMES
               if k in baselines and not baselines[k]["consistent"]]
        L = ["> ⚠️ **`Skill vs ERA5` is not comparable across these models on this run.** "
             "The ERA5 reference recovered per model (`crps_degC / (1 − skill_score)`) "
             "differs by more than the tolerance:"]
        for disp, b in bad:
            per = ", ".join(f"{m} {v:.4f}" if _finite(v) else f"{m} —"
                            for m, v in b["per_model"].items())
            L.append(f"> {disp}: spread {b['spread']:.4f} °C — {per}")
        L += ["> Rank on MAE/RMSE/CRPS (fixed MeteoSwiss truth) instead, and check "
              "`ERA5_MAX_TEMP_GLOB` in each `params.json` and "
              "`dists_grid.lat_bounds/lon_bounds` in each `manifest.json`.", ""]
        return L

    vals = ", ".join(
        f"{disp} **{baselines[k]['value']:.3f} °C** (spread {baselines[k]['spread']:.1e})"
        for k, _dn, disp in REGIMES
        if k in baselines and _finite(baselines[k]["value"]))
    L = ["> `Skill vs ERA5` = 1 − CRPS / CRPS_ref, where CRPS_ref is the MAE of the "
         "bilinear-ERA5 surface-tmax reference at the MeteoSwiss target points. That "
         "reference is **the same for every model here** — checked on this run, not "
         "assumed: each model's CRPS_ref is recovered from its own report as "
         "`crps_degC / (1 − skill_score)` and the models are required to agree to "
         f"{tol} °C. They do: {vals}. Skill is therefore a rescaling of CRPS by a shared "
         "constant, so it ranks the models exactly as CRPS does and is safe to compare "
         "across the surface and atmospheric runs.", ""]
    if any(b["unverified"] for b in baselines.values()):
        L += ["> (Only one model contributed to at least one regime, so comparability is "
              "unverified there rather than confirmed.)", ""]
    return L


def write_tables(rows, models, out: Path, baselines, skill_ok):
    cols = (["model", "regime", "eval_regime", "prediction_mode", "eval_days"]
            + [m[0] for m in METRICS] + ["ref_mae_degC"])
    with open(out / "metrics_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    (out / "metrics_comparison.json").write_text(json.dumps({
        "reference_baseline": {
            "definition": "crps_degC / (1 - skill_score)",
            "tolerance_degC": REF_TOL_DEGC,
            "consistent": skill_ok,
            "per_regime": baselines,
        },
        "rows": [{k: v for k, v in r.items() if k != "per_fold"} for r in rows],
    }, indent=2))

    L = ["# tmax model comparison — CV holdout vs 2024 holdout", ""]
    L += _model_table(models)
    L += [
        "**CV 2020-2023**: each of the 5 folds predicts only the contiguous block of days "
        "it was held out from during training, so every day is scored by the one model "
        "that never saw it. Not a fold ensemble.", "",
        "**2024 holdout**: a year no fold saw. All folds predict every day and are "
        "combined by Gaussian moment matching (μ = mean μₖ, σ² = mean σₖ² + var μₖ), with "
        "inputs normalised using the training-frozen statistics.", "",
        "The **Δ** column is 2024 − CV: the cost of moving from held-out days inside the "
        "training span to a genuinely unseen year.", "",
    ]
    L += _baseline_note(baselines, skill_ok, REF_TOL_DEGC)
    ref_line = "; ".join(
        f"{disp} {baselines[k]['value']:.3f} °C"
        for k, _dn, disp in REGIMES
        if k in baselines and _finite(baselines[k]["value"]))
    if ref_line:
        L += [f"ERA5 reference MAE (the skill denominator): {ref_line}.", ""]

    for mid, _sec, _k, disp, _good in METRICS:
        L += [f"## {disp}", "",
              "| Model | CV 2020-2023 | 2024 holdout | Δ |", "|---|---|---|---|"]
        star = " *" if (mid == "skill" and not skill_ok) else ""
        for label in models:
            cv, ho = _get(rows, label, "cv", mid), _get(rows, label, "2024", mid)
            L.append(f"| {label}{star} | {_fmt(cv, mid)} | {_fmt(ho, mid)} | "
                     f"{_delta(cv, ho, mid)} |")
        L.append("")
        if star:
            L += ["\\* reference differs between these models — see the warning above.", ""]

    # Per-fold CV spread — a diagnostic that only the CV regime can produce.
    L += ["## Per-fold CV spread (MAE °C)", "",
          "Each fold scores a different, disjoint sub-period, so spread here is as much "
          "about which months a fold held out as about the fold's model quality.", "",
          "| Model | " + " | ".join(f"fold {i}" for i in range(5)) + " | range |",
          "|---|" + "---|" * 6]
    for label in models:
        pf = next((r["per_fold"] for r in rows
                   if r["model"] == label and r["regime"] == "cv"), {})
        if not pf:
            continue
        vals = [pf.get(str(i), {}).get("mae_degC") for i in range(5)]
        got = [v for v in vals if v is not None]
        rng = f"{max(got)-min(got):.3f}" if got else "—"
        L.append(f"| {label} | " + " | ".join(_fmt(v, "mae") for v in vals) + f" | {rng} |")
    L.append("")
    (out / "metrics_comparison.md").write_text("\n".join(L))


def _ref_values(baselines) -> tuple:
    return tuple(baselines.get(k, {}).get("value") for k, _dn, _d in REGIMES)


def _regime_handles():
    """Legend for the bar charts: colour is the model, so the legend explains the hatch."""
    from matplotlib.patches import Patch
    return [Patch(facecolor="#b0b0b0", edgecolor="white",
                  hatch=REGIME_HATCH.get(k, ""), label=disp)
            for k, _dn, disp in REGIMES]


def _ref_rmse(models) -> tuple:
    """Reference RMSE per regime, computed from the first model whose prediction
    bundles are on disk. The reference series is bit-identical across models
    (checked for MAE by _baseline_check), so any bundle serves. Returns
    (cv, 2024), either entry None if no bundle is available."""
    caches = {"cv": "cv.npz", "2024": "holdout_2024.npz"}
    out = {}
    for mdir in models.values():
        for key, fname in caches.items():
            if key in out:
                continue
            path = Path(mdir) / "pred_cache" / fname
            if not path.exists():
                continue
            try:
                z = np.load(path, allow_pickle=True)
                r = z["era5_ref_degC"].astype(np.float64)
                t = z["truths_degC"].astype(np.float64)
                m = np.isfinite(r) & np.isfinite(t)
                out[key] = float(np.sqrt(((r - t)[m] ** 2).mean()))
            except Exception:
                continue
        if len(out) == len(caches):
            break
    return out.get("cv"), out.get("2024")


def fig_regime_bars(rows, models, out: Path, baselines, skill_ok):
    show = [m for m in METRICS if m[0] in ("mae", "rmse", "crps", "skill")]
    # 2x2, not 1x4: at report width a 4-across row leaves ~1.6in panels, where the rotated
    # model labels and the paired bars stop being readable.
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.6))
    flat = axes.ravel()
    labels = list(models)
    x = np.arange(len(labels))
    cv_ref, ho_ref = _ref_values(baselines)
    rmse_cv_ref, rmse_ho_ref = _ref_rmse(models)
    # The reference is deterministic, so its CRPS equals its MAE and the same
    # two values mark both panels; its RMSE is computed from the bundles.
    ref_lines = {"mae": (cv_ref, ho_ref), "crps": (cv_ref, ho_ref),
                 "rmse": (rmse_cv_ref, rmse_ho_ref)}
    for ax, (mid, _s, _k, disp, _g) in zip(flat, show):
        for i, (key, _d, rdisp) in enumerate(REGIMES):
            vals = [_get(rows, m, key, mid) for m in labels]
            vals = [v if _finite(v) else np.nan for v in vals]
            ax.bar(x + (i - 0.5) * 0.38, vals, width=0.38,
                   color=[model_color(m) for m in labels],
                   hatch=REGIME_HATCH.get(key, ""), edgecolor="white", lw=0.6)
        rcv, rho = ref_lines.get(mid, (None, None))
        if _finite(rcv) and _finite(rho):
            # Solid = CV, dashed = 2024, mirroring the hatch convention.
            ax.axhline(rcv, color="black", lw=1.1)
            ax.axhline(rho, color="black", lw=1.1, ls=(0, (4, 3)))
            ax.text(x[-1] + 0.55, min(rcv, rho) - 0.03,
                    f"CV {rcv:.2f} / 2024 {rho:.2f} °C", ha="right", va="top",
                    fontsize=7, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(disp, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if mid == "skill":
            if not skill_ok:
                # Skill is the only baseline-relative metric here (MAE/RMSE/CRPS
                # score against the fixed MeteoSwiss truth), so it is the only
                # panel that can lose comparability. Shade it when it does.
                ax.set_facecolor("#f6e4e4")
                ax.set_xlabel("ERA5 reference differs between models — not comparable",
                              fontsize=7, color="C3")
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    from matplotlib.lines import Line2D
    # The legend sits in the MAE panel, so it carries the MAE reference values;
    # the RMSE and CRPS panels annotate their own next to the lines.
    cv_lab = f"bilinear-ERA5 reference, CV — {cv_ref:.2f} °C" \
        if _finite(cv_ref) else "bilinear-ERA5 reference, CV"
    ho_lab = f"bilinear-ERA5 reference, 2024 — {ho_ref:.2f} °C" \
        if _finite(ho_ref) else "bilinear-ERA5 reference, 2024"
    handles = _regime_handles() + [
        Line2D([], [], color="black", lw=1.1, label=cv_lab),
        Line2D([], [], color="black", lw=1.1, ls=(0, (4, 3)), label=ho_lab),
    ]
    flat[0].legend(handles=handles, fontsize=7.5)
    fig.suptitle("tmax on the MeteoSwiss grid: CV holdout (2020-2023) vs genuine 2024 holdout",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    fig.savefig(out / "regime_comparison.png", dpi=130)
    plt.close(fig)


def collect_stations(models: dict[str, Path]) -> list[dict]:
    """One row per (model, regime) from station_analysis.py's summary.json."""
    rows = []
    for label, mdir in models.items():
        mdir = Path(mdir)
        for key, _dirname, disp in REGIMES:
            j = _read(mdir / f"station_analysis_{key}" / "summary.json")
            if not j:
                print(f"[warn] {label}: no station_analysis_{key}/summary.json under "
                      f"{mdir} — run `python station_analysis.py --model-dir {mdir}"
                      f"{'' if key == 'cv' else ' --eval-year ' + key}` (needs network "
                      f"for the MeteoSwiss NBCN download)")
                continue
            row = {"model": label, "regime": key, "regime_label": disp,
                   "eval_regime": j.get("eval_regime", "?"),
                   "prediction_mode": j.get("prediction_mode", "?"),
                   "n_stations_evaluated": j.get("n_stations_evaluated"),
                   "n_days_total": j.get("n_days_total"),
                   "excluded_stations": tuple(sorted(j.get("excluded_stations") or ())),
                   "years": j.get("years")}
            row.update({mid: j.get(k) for mid, k, _disp in STATION_METRICS})
            rows.append(row)
    return rows


def _station_cohort_check(srows) -> bool:
    """Do all models score the SAME stations over the SAME days, within each regime?

    Stronger than the baseline check: pooled station MAE averaged over different station
    sets moves by more than the model differences being compared, and nothing else in the
    pipeline would notice.
    """
    ok = True
    for rkey, _dn, disp in REGIMES:
        grp = [r for r in srows if r["regime"] == rkey]
        if len(grp) < 2:
            continue
        for field in ("n_stations_evaluated", "n_days_total", "excluded_stations"):
            seen = {r["model"]: r.get(field) for r in grp}
            if len(set(seen.values())) > 1:
                ok = False
                print(f"[WARN] station cohort differs across models in the {disp} regime: "
                      f"{field} = {seen}", file=sys.stderr)
    if not ok:
        print("[WARN] pooled station metrics are averages over different samples and are "
              "NOT comparable across models until the station sets agree.", file=sys.stderr)
    return ok


def _station_warnings(srows, st_baselines, cohort_ok) -> list[str]:
    ref = ", ".join(f"{disp} {st_baselines[k]['value']:.3f} °C"
                    for k, _dn, disp in REGIMES
                    if k in st_baselines and _finite(st_baselines[k]["value"]))
    spread = max((b["spread"] for b in st_baselines.values()
                  if _finite(b["spread"])), default=float("nan"))
    nst = next((r.get("n_stations_evaluated") for r in srows), "?")
    L = []
    if not cohort_ok:
        L += ["> ⚠️ **The models did not score the same stations or the same days.** "
              "Pooled numbers below are averages over different samples — fix the station "
              "cohort before reading any model-vs-model difference.", ""]
    L += [f"> **Station MAE is not the gridded MAE.** These score {nst} NBCN point "
          "observations; `metrics_comparison.md` scores the full 88 800-point MeteoSwiss "
          "gridded analysis. Point observations carry representativeness error the smoothed "
          "gridded product does not, so station MAE (~1.6 °C) sits well above gridded MAE "
          "(~1.2-1.45 °C) for every model. Compare models *within* this table; never "
          "compare a number here against one there.", "",
          "> **Pooled station skill is inflated.** The ERA5 baseline is raw bilinear "
          f"`t2m_max` with no lapse-rate correction, so its MAE is large ({ref}) — huge at "
          "alpine stations, far smaller in the lowlands. Pooled `skill_overall` therefore "
          "mostly measures *elevation* downscaling; `skill_lowland` (stations below 1000 m) "
          "and `skill_median` are the honest numbers.", ""]
    if any(b["unverified"] for b in st_baselines.values()):
        L += ["> (Only one model contributed to at least one regime, so the baseline being "
              "shared is unverified there rather than confirmed.)", ""]
    else:
        L[-2] = L[-2].rstrip() + (
            f" The baseline itself is identical for every model (checked: spread "
            f"{spread:.1e} °C), so the model *ranking* is fair even though the level is "
            f"flattered.")
    return L


def write_station_tables(srows, models, out: Path, st_baselines, cohort_ok):
    cols = (["model", "regime", "eval_regime", "prediction_mode",
             "n_stations_evaluated", "n_days_total"] + [m[0] for m in STATION_METRICS])
    with open(out / "station_metrics_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in srows:
            w.writerow({c: r.get(c) for c in cols})
    (out / "station_metrics_comparison.json").write_text(json.dumps({
        "station_baseline": {
            "definition": "mae_era5_overall (bilinear ERA5 t2m_max at station points)",
            "tolerance_degC": STATION_REF_TOL_DEGC,
            "cohort_consistent": cohort_ok,
            "per_regime": st_baselines,
        },
        "rows": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
                 for r in srows],
    }, indent=2))

    ref = next((r for r in srows if r["regime"] == "cv"), srows[0])
    excl = ", ".join(ref.get("excluded_stations") or ()) or "none"
    days = [(disp, next((r["n_days_total"] for r in srows if r["regime"] == k), None))
            for k, _dn, disp in REGIMES]
    cohort = (f"{ref.get('n_stations_evaluated')} NBCN stations evaluated "
              f"(excluded: {excl}); station-days "
              + ", ".join(f"{d} {v:,}" for d, v in days if v) + ".")

    L = ["# tmax model comparison at NBCN stations — CV holdout vs 2024 holdout", "",
         cohort, ""]
    L += _model_table(models)
    L += _station_warnings(srows, st_baselines, cohort_ok)
    for mid, _k, disp in STATION_METRICS:
        L += [f"## {disp}", "",
              "| Model | CV 2020-2023 | 2024 holdout | Δ |", "|---|---|---|---|"]
        for label in models:
            cv, ho = _get(srows, label, "cv", mid), _get(srows, label, "2024", mid)
            L.append(f"| {label} | {_fmt(cv, mid)} | {_fmt(ho, mid)} | "
                     f"{_delta(cv, ho, mid)} |")
        L.append("")
    (out / "station_metrics_comparison.md").write_text("\n".join(L))


def fig_station_regime_bars(srows, models, out: Path, st_baselines, cohort_ok):
    """Station-level mirror of fig_regime_bars — same order, colors, widths, dpi.

    Panel 3 carries pooled skill AND lowland skill rather than giving each its own panel:
    skill_overall = 1 - crps/mae_era5 with a baseline that has zero spread across models,
    so a pooled-skill panel would be a rank-duplicate of the CRPS panel. Overlaying the
    lowland value shows the lapse-rate inflation as a drop inside each bar.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.6))
    flat = axes.ravel()
    labels = list(models)
    x = np.arange(len(labels))

    def bars(ax, mid, title):
        top = 0.0
        for i, (key, _d, rdisp) in enumerate(REGIMES):
            vals = [_get(srows, m, key, mid) for m in labels]
            vals = [v if _finite(v) else np.nan for v in vals]
            ax.bar(x + (i - 0.5) * 0.38, vals, width=0.38,
                   color=[model_color(m) for m in labels],
                   hatch=REGIME_HATCH.get(key, ""), edgecolor="white", lw=0.6)
            top = max(top, np.nanmax(vals) if len(vals) else 0.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        # Bars start at 0, so there is no free interior space — make headroom for legends.
        if top > 0:
            ax.set_ylim(0, top * 1.30)

    bars(flat[0], "st_mae", "Station MAE (°C)")
    bars(flat[1], "st_crps", "Station CRPS (°C)")

    ax = flat[2]
    bars(ax, "st_skill", "Skill vs ERA5 — pooled, with lowland overlay")
    dash = None
    for i, (key, _d, _rd) in enumerate(REGIMES):
        low = [_get(srows, m, key, "st_skill_low") for m in labels]
        h, = ax.plot(x + (i - 0.5) * 0.38, [v if _finite(v) else np.nan for v in low],
                     marker="_", ms=13, mew=2, color="k", ls="none",
                     label="lowland <1000 m (the honest number)")
        dash = dash or h
    # Only the dash needs explaining here; the regime colors are in the first panel.
    ax.legend(handles=[dash], fontsize=7, loc="upper right")

    bars(flat[3], "st_abs_bias", "Mean |per-station bias| (°C)")

    for a in axes[0]:
        a.tick_params(labelbottom=False)
    flat[0].legend(handles=_regime_handles(), fontsize=8, loc="upper left")

    nst = next((r.get("n_stations_evaluated") for r in srows), "?")
    days = ", ".join(
        f"{disp} {v:,}" for k, _dn, disp in REGIMES
        for v in [next((r["n_days_total"] for r in srows if r["regime"] == k), None)] if v)
    ref = " / ".join(f"{st_baselines[k]['value']:.3f}" for k, _dn, _d in REGIMES
                     if k in st_baselines and _finite(st_baselines[k]["value"]))
    fig.suptitle(f"tmax at {nst} NBCN stations (point observations): "
                 f"CV holdout (2020-2023) vs genuine 2024 holdout", fontsize=12)
    note = (f"NOT comparable with regime_comparison.png: {nst} point observations "
            f"({days}) vs the 88 800-point gridded analysis.\n"
            f"Pooled skill is inflated — the ERA5 baseline (MAE {ref} °C) is raw bilinear "
            f"t2m_max with no lapse-rate correction; the black dashes are the honest number.")
    if not cohort_ok:
        note = "WARNING: models scored different station sets — see the table.\n" + note
    fig.text(0.5, 0.012, note, ha="center", fontsize=7,
             color="C3" if not cohort_ok else "0.4")
    fig.tight_layout(rect=(0, 0.065, 1, 0.96))
    fig.savefig(out / "station_regime_comparison.png", dpi=130)
    plt.close(fig)


def fig_perfold_spread(rows, models, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    plotted = False
    for ci, label in enumerate(models):
        pf = next((r["per_fold"] for r in rows
                   if r["model"] == label and r["regime"] == "cv"), {})
        if not pf:
            continue
        folds = sorted(int(f) for f in pf)
        ax.plot(folds, [pf[str(f)]["mae_degC"] for f in folds],
                marker="o", label=label, color=f"C{ci}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("CV fold (each holds out a different ~292-day block)")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Per-fold CV holdout MAE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "perfold_spread.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None,
                    help="label=path pairs (default: the 5 CLEAN tmax runs).")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output dir.")
    ap.add_argument("--years", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.years:
        ap.error("--years is gone: there are two regimes now, not a per-year sweep. "
                 "The 2020-2023 per-year evaluations were in-sample ensemble scores "
                 "and have been replaced by a single cross-validation holdout.")

    models = DEFAULT_MODELS
    if args.models:
        models = {}
        for spec in args.models:
            if "=" not in spec:
                print(f"[error] --models wants label=path, got {spec!r}", file=sys.stderr)
                return 2
            label, path = spec.split("=", 1)
            models[label] = path

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resolved = {k: (Path(v) if Path(v).is_absolute() else ROOT / v)
                for k, v in models.items()}
    rows = collect(resolved)
    if not rows:
        print("[error] no report_metrics.json found for any model/regime — run "
              "scripts/eval_all_clean_tmax.sh first.", file=sys.stderr)
        return 1

    baselines = _baseline_check(rows, "ref_mae_degC", REF_TOL_DEGC,
                                "bilinear-ERA5 skill reference")
    skill_ok = all(b["consistent"] for b in baselines.values())

    write_tables(rows, resolved, out, baselines, skill_ok)
    fig_regime_bars(rows, resolved, out, baselines, skill_ok)
    fig_perfold_spread(rows, resolved, out)
    print(f"wrote {out}/metrics_comparison.{{md,json,csv}}, regime_comparison.png, "
          f"perfold_spread.png  ({len(rows)} model-regime rows)")

    srows = collect_stations(resolved)
    if srows:
        st_baselines = _baseline_check(srows, "st_mae_era5", STATION_REF_TOL_DEGC,
                                       "station ERA5 baseline")
        cohort_ok = _station_cohort_check(srows)
        write_station_tables(srows, resolved, out, st_baselines, cohort_ok)
        fig_station_regime_bars(srows, resolved, out, st_baselines, cohort_ok)
        print(f"wrote {out}/station_metrics_comparison.{{md,json,csv}}, "
              f"station_regime_comparison.png  ({len(srows)} model-regime rows)")
    else:
        print("[warn] no station_analysis_*/summary.json found — skipping the "
              "station-level comparison (re-run station_analysis.py; it needs network "
              "access for the MeteoSwiss NBCN download, and see SKIP_STATIONS in "
              "scripts/eval_all_clean_tmax.sh)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
