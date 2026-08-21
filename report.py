#!/usr/bin/env python3
"""
Comprehensive spatial + calibration evaluation report for a trained model.

This is the script-based port of ``predictions.ipynb`` (cells 17-22): where
``infer.py`` writes a scalar metrics table (pooled MAE/RMSE/CRPS/NLL/...),
``report.py`` regenerates the full *visual* evaluation -- per-pixel error maps,
diagnostic correlations (uncertainty/altitude/mTPI/day-of-year vs error),
probabilistic skill maps, and uncertainty-calibration diagnostics (Q-Q/PIT and
reliability) -- as PNGs plus a machine-readable ``report_metrics.json`` and a
human-readable ``report.md`` index.

``render_evaluation`` is the shared renderer: it takes finished degC arrays for a
period and produces the figures + JSON + markdown. It is regime-agnostic on purpose,
so the 2020-2023 cross-validation holdout and a genuine holdout year come out looking
identical and stay directly comparable.

The cross-validation driver that used to live here (``generate_report``) has moved to
``evaluate.predict_cv``, which builds its context from ``manifest.json`` instead of
re-deriving normalization stats with ``train.DataBundle``, and shares the prediction
cache with the holdout-year path. Run ``python evaluate.py --model-dir <...>/tmax``.

Outputs (under the caller's output dir):
  error_maps.png                   per-pixel MAE / RMSE / bias
  crps_map.png                     per-pixel CRPS
  skill_map.png                    skill vs bilinear-ERA5 baseline
  uncertainty_vs_error.png         per-pixel sigma vs MAE (calibration in space)
  altitude_vs_error.png            terrain-dependent error (elevation)
  mtpi_vs_error.png                terrain-dependent error (mTPI)
  doy_vs_error.png                 seasonal error pattern
  qq_calibration.png               PIT-based Q-Q uncertainty calibration
  reliability_diagram.png          confidence vs observed coverage
  report_metrics.json              all scalar metrics
  report.md                        index + headline numbers

Example
-------
    python evaluate.py --model-dir CLEAN_trained_models/<trial>/tmax              # CV holdout
    python evaluate.py --model-dir CLEAN_trained_models/<trial>/tmax --eval-year 2024
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Headless plotting before visualization imports matplotlib.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics as metrics_mod
import visualization as vis


logger = logging.getLogger("report")


# ---------------------------------------------------------------------------
# Shared renderer: arrays -> per-pixel metrics + figures + report_metrics.json
# + report.md. Used by evaluate.py for both regimes, so the CV holdout and a
# holdout year produce identical-looking, directly comparable reports.
# ---------------------------------------------------------------------------
def render_evaluation(
    out_dir: Path,
    *,
    all_errors: np.ndarray,
    all_preds: np.ndarray,
    all_sigmas: np.ndarray,
    all_truths: np.ndarray,
    all_dates: np.ndarray,
    all_ref: np.ndarray,
    grid_shape: tuple[int, int],
    target_topo: torch.Tensor,
    valid_mask: np.ndarray,
    variable: str,
    encoder: str,
    grid_mode: str,
    n_folds_evaluated: int,
    model_dir: Path,
    period_label: str = "holdout",
    md_single_day: bool = False,
    md_fold_ids: list[int] | tuple[int, ...] = (),
    extra: dict | None = None,
    regime_note: str | None = None,
) -> dict:
    """Render one evaluation: per-pixel metrics, figures, report_metrics.json, report.md.

    ``extra`` is merged into the metrics dict *before* it is written, which is how the
    regime provenance (``eval_regime``, ``per_fold``, cache info) reaches disk. It used to
    be attached by the caller after this function returned, so it never did.

    ``regime_note`` is a sentence rendered under the heading of report.md saying exactly
    what was scored against what — the guard against a report labelling training-span
    days as "unseen".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_path(name: str) -> str:
        return str(out_dir / f"{name}.png")

    grid_N, grid_E = grid_shape
    total_days = all_errors.shape[0]
    n_points = all_truths.shape[1]

    # --- Per-pixel metrics ---
    m = metrics_mod.compute_perpixel_metrics(
        all_errors=all_errors,
        all_preds=all_preds,
        all_sigmas=all_sigmas,
        all_truths=all_truths,
        all_ref_preds=all_ref,
        grid_shape=grid_shape,
        valid_mask=valid_mask,
    )

    elev_np = target_topo.cpu().numpy()
    true_elev_grid = np.where(valid_mask, elev_np[:, 0].reshape(grid_N, grid_E), np.nan)
    mtpi_grid = np.where(valid_mask, elev_np[:, 2].reshape(grid_N, grid_E), np.nan)

    # --- Spatial maps ---
    logger.info("rendering spatial maps...")
    vis.plot_error_maps(
        mae_grid=m.mae_grid, rmse_grid=m.rmse_grid, bias_grid=m.bias_grid,
        title=f"Per-pixel error ({total_days} {period_label} days)",
        error_range=(0, 5), bias_range=(-3, 3),
        save_path=plot_path("error_maps"),
    )
    vis.plot_crps_map(
        crps_grid=m.crps_grid,
        title=f"Per-pixel CRPS ({total_days} days), overall {m.overall_crps:.3f} C",
        crps_range=(0, 3), save_path=plot_path("crps_map"),
    )
    vis.plot_skill_map(
        skill_grid=m.skill_grid, global_skill=m.global_skill,
        title=f"Skill vs bilinear ERA5 ({total_days} days), global {m.global_skill:.3f}",
        save_path=plot_path("skill_map"),
    )
    plt.close("all")

    # --- Diagnostic correlations ---
    logger.info("rendering diagnostic correlations...")
    _, unc_corr = vis.plot_perpixel_uncertainty_vs_error(
        mean_sigma_grid=m.mean_sigma_grid, mae_grid=m.mae_grid,
        title=f"Per-pixel uncertainty vs error ({total_days} days)",
        sigma_range=(0, float(np.nanmax(m.mean_sigma_grid))),
        error_range=(0, float(np.nanmax(m.mae_grid))),
        save_path=plot_path("uncertainty_vs_error"),
    )
    _, alt_corr = vis.plot_perpixel_correlation(
        x_grid=true_elev_grid, y_grid=m.mae_grid,
        x_label="Altitude (m)", y_label="MAE (C)",
        title=f"Altitude vs error ({total_days} days)",
        x_cmap="terrain", y_cmap="YlOrRd",
        save_path=plot_path("altitude_vs_error"),
    )
    _, mtpi_corr = vis.plot_perpixel_correlation(
        x_grid=mtpi_grid, y_grid=m.mae_grid,
        x_label="mTPI (m)", y_label="MAE (C)",
        title=f"mTPI vs error ({total_days} days)",
        x_cmap="RdBu_r", y_cmap="YlOrRd",
        save_path=plot_path("mtpi_vs_error"),
    )
    _, doy_corr = vis.plot_temporal_error_analysis(
        all_errors=all_errors, holdout_dates=all_dates,
        title=f"Day-of-year vs error ({total_days} days)",
        save_path=plot_path("doy_vs_error"),
    )
    plt.close("all")

    # --- Uncertainty calibration ---
    logger.info("rendering calibration diagnostics...")
    _, cal = vis.plot_qq_calibration(
        all_truths=all_truths, all_preds=all_preds, all_sigmas=all_sigmas,
        title=f"Q-Q calibration ({total_days} days)",
        save_path=plot_path("qq_calibration"),
    )
    _, rel = vis.plot_reliability_diagram(
        all_truths=all_truths, all_preds=all_preds, all_sigmas=all_sigmas,
        title=f"Reliability diagram ({total_days} days)",
        save_path=plot_path("reliability_diagram"),
    )
    plt.close("all")

    # --- Assemble machine-readable metrics ---
    result = {
        "model_dir": str(model_dir),
        "variable": variable,
        "encoder": encoder,
        "grid_mode": grid_mode,
        "n_folds_evaluated": n_folds_evaluated,
        "n_target_points": int(n_points),
        "holdout_days": int(total_days),
        "overall": {
            "mae_degC": float(np.nanmean(m.mae_grid)),
            "rmse_degC": float(np.nanmean(m.rmse_grid)),
            "bias_degC": float(np.nanmean(m.bias_grid)),
            "crps_degC": float(m.overall_crps),
            "skill_score": float(m.global_skill),
        },
        "correlations": {
            "uncertainty_error": float(unc_corr),
            "altitude_error": float(alt_corr),
            "mtpi_error": float(mtpi_corr),
            "doy_error": float(doy_corr),
        },
        "calibration": {
            "pit_mean": float(cal["pit_mean"]),
            "pit_std": float(cal["pit_std"]),
            "z_std": float(cal["z_std"]),
            "coverage_50": float(cal["coverage_50"]),
            "coverage_90": float(cal["coverage_90"]),
            "ks_p_value": float(cal["ks_p"]),
        },
        "reliability": {
            "mace": float(rel["mace"]),
            "rmsce": float(rel["rmsce"]),
            "coverage_at_50": float(rel["coverage_at_50"]),
            "coverage_at_90": float(rel["coverage_at_90"]),
            "coverage_at_95": float(rel["coverage_at_95"]),
        },
    }
    if extra:
        result.update(extra)
    (out_dir / "report_metrics.json").write_text(json.dumps(result, indent=2))
    _write_report_md(out_dir, result, md_single_day, list(md_fold_ids),
                     regime_note=regime_note)
    return result


# ---------------------------------------------------------------------------
# report.md index
# ---------------------------------------------------------------------------
def _write_report_md(out_dir: Path, r: dict, single_day: bool, fold_ids: list[int],
                     regime_note: str | None = None):
    o, c, cal, rel = r["overall"], r["correlations"], r["calibration"], r["reliability"]
    L = [
        f"# Evaluation report — {r['variable']} ({r['grid_mode']})",
        "",
    ]
    if regime_note:
        L += [f"> {regime_note}", ""]
    L += [
        f"- Model: `{r['model_dir']}`",
        f"- Encoder: `{r['encoder']}` | folds evaluated: {r['n_folds_evaluated']} | "
        f"target points: {r['n_target_points']:,} | holdout days: {r['holdout_days']}",
    ]
    if "eval_regime" in r:
        L.append(f"- Regime: `{r['eval_regime']}` | prediction mode: "
                 f"`{r.get('prediction_mode', '?')}` | valid points: "
                 f"{r.get('n_valid_points', 0):,}")
    L += [
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| MAE (°C) | {o['mae_degC']:.3f} |",
        f"| RMSE (°C) | {o['rmse_degC']:.3f} |",
        f"| Bias (°C) | {o['bias_degC']:.3f} |",
        f"| CRPS (°C) | {o['crps_degC']:.3f} |",
        f"| Skill vs bilinear ERA5 | {o['skill_score']:.3f} |",
        "",
        "## Error diagnostics (correlation of MAE with …)",
        "",
        "| Driver | Pearson r |",
        "|---|---|",
        f"| predicted σ (calibration in space) | {c['uncertainty_error']:.3f} |",
        f"| altitude | {c['altitude_error']:.3f} |",
        f"| mTPI (ridge/valley) | {c['mtpi_error']:.3f} |",
        f"| day of year (seasonality) | {c['doy_error']:.3f} |",
        "",
        "## Uncertainty calibration",
        "",
        "| Metric | Value | Ideal |",
        "|---|---|---|",
        f"| PIT mean | {cal['pit_mean']:.3f} | 0.5 |",
        f"| PIT std | {cal['pit_std']:.3f} | 0.289 |",
        f"| z std | {cal['z_std']:.3f} | 1.0 |",
        f"| coverage @50% | {cal['coverage_50']:.1%} | 50% |",
        f"| coverage @90% | {cal['coverage_90']:.1%} | 90% |",
        f"| KS p-value | {cal['ks_p_value']:.3g} | >0.05 |",
        f"| MACE (reliability) | {rel['mace']:.3f} | 0 |",
        f"| reliability @90% | {rel['coverage_at_90']:.1%} | 90% |",
        f"| reliability @95% | {rel['coverage_at_95']:.1%} | 95% |",
        "",
    ]
    per_fold = r.get("per_fold") or {}
    blocks = r.get("fold_blocks") or {}
    if per_fold:
        L += [
            "## Per-fold holdout blocks",
            "",
            "Each fold scores only the days it was held out from, so these are disjoint "
            "sub-periods, not repeated measurements of the same days.",
            "",
            "| Fold | Days | Period | Epoch | MAE (°C) | RMSE (°C) | CRPS (°C) | Skill |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for fold, fm in sorted(per_fold.items(), key=lambda kv: int(kv[0])):
            b = blocks.get(fold, {})
            period = (f"{b.get('first_date', '?')} … {b.get('last_date', '?')}"
                      if b else "—")
            L.append(
                f"| {fold} | {fm['n_days']} | {period} | {fm.get('checkpoint_epoch', '?')} | "
                f"{fm['mae_degC']:.3f} | {fm['rmse_degC']:.3f} | {fm['crps_degC']:.3f} | "
                f"{fm['skill_score']:.3f} |")
        L.append("")
    sig = r.get("sigma_decomposition")
    if sig:
        L += [
            "## Ensemble spread decomposition",
            "",
            "| Component | °C |",
            "|---|---|",
            f"| within-fold (mean σₖ) | {sig['mean_sigma_within_degC']:.3f} |",
            f"| between-fold (spread of μₖ) | {sig['mean_sigma_between_degC']:.3f} |",
            f"| total (moment-matched) | {sig['mean_sigma_total_degC']:.3f} |",
            "",
        ]
    L += [
        "## Figures",
        "",
    ]
    figs = [
        ("error_maps.png", "Per-pixel MAE / RMSE / bias"),
        ("crps_map.png", "Per-pixel CRPS"),
        ("skill_map.png", "Skill vs bilinear-ERA5 baseline"),
        ("uncertainty_vs_error.png", "Per-pixel uncertainty vs error"),
        ("altitude_vs_error.png", "Altitude vs error"),
        ("mtpi_vs_error.png", "mTPI vs error"),
        ("doy_vs_error.png", "Day-of-year vs error"),
        ("qq_calibration.png", "Q-Q / PIT calibration"),
        ("reliability_diagram.png", "Reliability diagram"),
    ]
    if single_day:
        for f in fold_ids:
            figs.insert(0, (f"fold{f}_prediction.png", f"Fold {f + 1} single-day comparison"))
    for fname, desc in figs:
        L.append(f"### {desc}")
        L.append("")
        L.append(f"![{desc}]({fname})")
        L.append("")
    (out_dir / "report.md").write_text("\n".join(L))


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(r: dict, out_dir: Path):
    o, c, cal, rel = r["overall"], r["correlations"], r["calibration"], r["reliability"]
    print()
    print(f"=== Report: {r['model_dir']} ===")
    print(f"variable={r['variable']} | encoder={r['encoder']} | grid={r['grid_mode']} | "
          f"folds={r['n_folds_evaluated']} | days={r['holdout_days']} | points={r['n_target_points']:,}")
    print()
    print(f"  MAE {o['mae_degC']:.3f} C | RMSE {o['rmse_degC']:.3f} C | "
          f"bias {o['bias_degC']:+.3f} C | CRPS {o['crps_degC']:.3f} C | "
          f"skill {o['skill_score']:.3f}")
    print(f"  corr(MAE, sigma) {c['uncertainty_error']:.3f} | "
          f"corr(MAE, alt) {c['altitude_error']:.3f} | "
          f"corr(MAE, mTPI) {c['mtpi_error']:.3f} | "
          f"corr(MAE, doy) {c['doy_error']:.3f}")
    print(f"  calib: PIT mean {cal['pit_mean']:.3f} (~0.5), z std {cal['z_std']:.3f} (~1.0), "
          f"MACE {rel['mace']:.3f}, cov@90 {cal['coverage_90']:.1%}")
    print()
    print(f"Figures + report.md + report_metrics.json written to: {out_dir}")


# ---------------------------------------------------------------------------
# CLI (retired)
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    print(
        "report.py is now a renderer library, not a driver.\n\n"
        "Its cross-validation path moved to evaluate.py, which shares one prediction\n"
        "cache with the holdout-year path and labels each regime explicitly:\n\n"
        "    python evaluate.py --model-dir <trial>/tmax                # CV holdout 2020-2023\n"
        "    python evaluate.py --model-dir <trial>/tmax --eval-year 2024\n",
        file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
