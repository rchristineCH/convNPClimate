#!/usr/bin/env python3
"""Compare the five CLEAN solo precip models.

  * ``NLL``     -- Bernoulli-Gamma NLL, atmospheric z/t/q inputs;
  * ``CRPS``    -- same inputs, trained from scratch on the CRPS loss;
  * ``WIND``    -- NLL loss, atmospheric inputs extended with u/v;
  * ``FT-CRPS`` -- the ``NLL`` run warm-started per fold and fine-tuned for 10
    further epochs on the CRPS loss at LR 5e-5.

Produces, under ``precip_processing_comparison/``:
  * training_comparison.png + training_comparison.md/json -- fold-averaged
    training curves overlaid (the CRPS diagnostic ``test CRPS`` and ``MAE`` are the
    fair common curves; ``test NLL`` is each model's own held-out training loss and
    for the CRPS runs is actually ``-CRPS``, so it is shown but not cross-compared);
  * physical_metrics_comparison.md/json + physical_metrics_comparison.png --
    the held-out physical metrics for every model x {CV holdout, 2024 holdout},
    including the threshold->probabilistic deltas.

All evaluation numbers are the FULLY PROBABILISTIC ones (no rho>=0.5 threshold);
every model is evaluated by the identical eval_precip pipeline.

CAVEAT on the training curves: ``FT-CRPS`` contributes only its 10 fine-tune
epochs, which start from an already-converged NLL model. Its epoch axis is NOT
comparable to the from-scratch runs -- read its curves as "epochs after the
warm start", and take the held-out physical metrics as the fair comparison.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# The tracked comparison dir lives under CLEAN_trained_models/ (moved there by the
# workspace cleanup); writing to a repo-root copy would leave the committed
# artifacts stale while an untracked duplicate silently accumulated.
OUT = ROOT / "CLEAN_trained_models/precip_processing_comparison"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "NLL": ROOT / "CLEAN_trained_models/clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8/precip",
    "CRPS": ROOT / "CLEAN_trained_models/clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8/precip",
    "WIND": ROOT / "CLEAN_trained_models/clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8/precip",
    "FT-CRPS": ROOT / "CLEAN_trained_models/clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps/precip",
    # Surface inputs (ERA5-Land t2m_max + tp) instead of pressure levels; 1460-day
    # axis (the tp series ends 2023-12-30) against 1461 for the four atmospheric runs.
    "SFC-TP": ROOT / "CLEAN_trained_models/clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8/precip",
}
COLORS = {"NLL": "C0", "CRPS": "C3", "WIND": "C2", "FT-CRPS": "C4", "SFC-TP": "C1"}
# Same three training-objective families physical_comparison() groups its model axis by
# (mirrored here rather than imported, since that function defines them locally): the
# three mixture-NLL runs differ from one another only in input, CRPS is a from-scratch
# alternate objective, and FT-CRPS is a warm start from NLL so belongs to neither pure
# objective. Any figure that groups models by column should use this same order/grouping
# so a reader does not have to re-learn a new model order per figure.
OBJECTIVE_GROUPS = [("mixture NLL", ["NLL", "WIND", "SFC-TP"]),
                    ("sampled CRPS", ["CRPS"]),
                    ("NLL, then CRPS fine-tune", ["FT-CRPS"])]
# Series whose epoch axis counts fine-tune epochs after a warm start, not
# from-scratch training epochs (see the module docstring caveat).
WARM_START = {"FT-CRPS"}


# --------------------------------------------------------------------------
# Training-curve comparison
# --------------------------------------------------------------------------
def _load_baseline_correlations() -> dict:
    """Baseline correlations from scripts/gen_precip_baseline_corr.py, keyed 'cv'/'2024'.

    A fallback only: once eval_precip has been re-run, the same numbers live in each
    metric file's ``baseline`` block and are preferred. Returns {} when absent.
    """
    f = OUT / "baseline_correlations.json"
    return json.loads(f.read_text()) if f.exists() else {}


def training_comparison():
    curves = {"test CRPS": "held-out CRPS (common diagnostic)",
              "Mean absolute error": "held-out MAE (mm)",
              "Pearson correlation": "Pearson", "Spearman correlation": "Spearman",
              "test NLL": "held-out train loss (CRPS run: -CRPS)"}
    stats = {}
    for name, d in MODELS.items():
        f = d / "stats.csv"
        if f.exists():
            stats[name] = pd.read_csv(f)
    if not stats:
        print("no stats.csv found; skipping training comparison")
        return

    keys = list(curves)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.2))
    for ax, key in zip(axes, keys):
        for name, df in stats.items():
            if key not in df.columns:
                continue
            avg = df.groupby("Epoch")[key].mean()
            warm = name in WARM_START
            ax.plot(avg.index, avg.values, color=COLORS[name], lw=1.6,
                    ls="--" if warm else "-",
                    label=(f"{name} (warm-start fine-tune epochs)" if warm
                           else f"{name}-trained"))
        ax.set_title(curves[key], fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Training curves, fold-averaged: NLL / CRPS / WIND / SFC-TP from scratch, "
                 "FT-CRPS fine-tuned from NLL (dashed — epoch axis not comparable)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "training_comparison.png", dpi=110)
    plt.close(fig)

    # Best-epoch summary (fold-averaged), on the common comparable curves.
    summary = {}
    for name, df in stats.items():
        avg = df.groupby("Epoch").mean(numeric_only=True)
        row = {}
        if "test CRPS" in avg:
            row["best_CRPS"] = float(avg["test CRPS"].min())
            row["best_CRPS_epoch"] = int(avg["test CRPS"].idxmin())
        if "Mean absolute error" in avg:
            row["best_MAE"] = float(avg["Mean absolute error"].min())
            row["best_MAE_epoch"] = int(avg["Mean absolute error"].idxmin())
        for c, k in [("Pearson correlation", "max_Pearson"),
                     ("Spearman correlation", "max_Spearman")]:
            if c in avg:
                row[k] = float(avg[c].max())
        summary[name] = row
    (OUT / "training_comparison.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Training comparison (fold-averaged best)", "",
             "| Model | best CRPS (epoch) | best MAE (epoch) | max Pearson | max Spearman |",
             "|---|---|---|---|---|"]
    for name, r in summary.items():
        lines.append(
            f"| {name}{' (fine-tune epochs)' if name in WARM_START else '-trained'} "
            f"| {r.get('best_CRPS', float('nan')):.4f} "
            f"({r.get('best_CRPS_epoch','-')}) | {r.get('best_MAE', float('nan')):.4f} "
            f"({r.get('best_MAE_epoch','-')}) | {r.get('max_Pearson', float('nan')):.4f} "
            f"| {r.get('max_Spearman', float('nan')):.4f} |")
    lines += ["", "Note: `test NLL` is each run's own training objective on the held-out fold "
              "(for the CRPS runs it is `-CRPS`, not a likelihood), so only `test CRPS`, MAE and "
              "the correlations are cross-comparable.",
              "", "Note: `FT-CRPS` reports only its 10 fine-tune epochs, which start from the "
              "already-converged `NLL` model — its epoch numbers count epochs after the warm "
              "start and are not comparable to the from-scratch runs. Compare it on the "
              "held-out physical metrics instead."]
    (OUT / "training_comparison.md").write_text("\n".join(lines) + "\n")
    print("wrote training_comparison.{png,md,json}")


# --------------------------------------------------------------------------
# Physical-metrics comparison
# --------------------------------------------------------------------------
def _load(model_dir, year=None):
    name = "eval_precip_metrics.json" if year is None else f"eval_precip_metrics_{year}.json"
    f = model_dir / name
    return json.loads(f.read_text()) if f.exists() else None


def physical_comparison():
    regimes = [(name, year, f"{name} / {'CV' if year is None else year}")
               for name in MODELS for year in (None, 2024)]
    data = {}
    for name, year, label in regimes:
        m = _load(MODELS[name], year)
        if m is not None:
            data[label] = m

    rows = [
        ("wetday_freq_obs", "Wet-day freq (obs)"),
        ("wetday_freq_pred", "Wet-day freq (pred)"),
        ("dryday_freq_obs", "Dry-day freq <0.1mm (obs)"),
        ("dryday_freq_pred", "Dry-day freq <0.1mm (pred)"),
        ("R01_rel_wetday_freq", "R01"),
        ("SDII_obs_mm", "SDII obs (mm)"),
        ("SDII_pred_mm", "SDII pred (mm)"),
        ("SDII_bias_mm", "SDII bias (mm)"),
        ("R10_freq_obs", "R10 obs"),
        ("R10_freq_pred", "R10 pred"),
        ("P98_obs_mm", "P98 obs (mm)"),
        ("P98_pred_mm", "P98 pred (mm)"),
        ("mae_mm", "MAE (mm)"),
        ("bias_mm", "Bias (mm)"),
        ("spearman_pooled", "Spearman"),
        ("pearson_pooled", "Pearson"),
    ]
    labels = list(data)
    header = "| Metric | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    lines = ["# Physical-metrics comparison (fully probabilistic; identical pipeline)",
             "", header, sep]
    for key, lab in rows:
        cells = []
        for lb in labels:
            v = data[lb]["overall"].get(key)
            cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "-")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    # PIT mean
    pit_cells = []
    for lb in labels:
        pit = data[lb].get("pit", {})
        pit_cells.append(f"{pit.get('mean_pit', float('nan')):.4f}" if pit.get("available") else "-")
    lines.append("| PIT mean (ideal 0.5) | " + " | ".join(pit_cells) + " |")
    (OUT / "physical_metrics_comparison.md").write_text("\n".join(lines) + "\n")
    (OUT / "physical_metrics_comparison.json").write_text(
        json.dumps({lb: data[lb]["overall"] for lb in labels}, indent=2))

    # Grouped-bar figures, restyled after compare_evaluations.fig_regime_bars (the tmax
    # "regime_comparison" figure) but split into two ISOLATED figures, one per regime,
    # rather than one figure with paired CV/2024 bars: the paired-bar + hatch + dashed-line
    # encoding crowded four things (bar, bar, observed, baseline) into every panel and the
    # observed/baseline lines stopped reading clearly against each other. One figure per
    # regime removes the need for hatching or dashed-vs-solid lines entirely -- each panel
    # now carries one bar per model and one line per reference -- and the two figures share
    # y-limits per metric and an identical legend so they scale together and can be read
    # side by side. The observed truth and the ERA5-Land baseline are both drawn as
    # full-width horizontal lines (not small per-bar ticks, which did not read against the
    # black baseline at this scale) in different colours, with a light fill between them so
    # the gap the models sit in is visible before reading either line individually.
    # baseline_spearman_pooled/baseline_wetday_freq/baseline_R10_freq/baseline_P98_mm are
    # written by scripts/gen_precip_baseline_corr.py (eval_precip only carries
    # baseline_mae_mm, baseline_SDII_mm, baseline_R01_rel_wetday_freq per model/regime).
    base_corr = _load_baseline_correlations()
    # Grouped by training objective rather than by the order of tab:single-precip-variants.
    # The three runs that minimise the mixture NLL differ from one another only in their
    # input, so read left to right they are the input comparison; CRPS and FT-CRPS differ
    # from NLL only in objective. Keeping the two families apart means a reader is never
    # comparing across both axes at once, which the old interleaved order forced.
    # Three objective families, labelled as in the Objective column of
    # tab:single-precip-variants. FT-CRPS is kept apart from CRPS rather than lumped in
    # with it: it is a warm start from the trained NLL model, so it belongs to neither
    # pure objective.
    OBJECTIVE_GROUPS = [("mixture NLL", ["NLL", "WIND", "SFC-TP"]),
                        ("sampled CRPS", ["CRPS"]),
                        ("NLL, then CRPS fine-tune", ["FT-CRPS"])]
    GROUP_SHADE = ["#eaf0f6", "#f7f0e8", "#ecf3ec"]
    GROUP_GAP = 0.9
    model_order = [m for _lab, members in OBJECTIVE_GROUPS for m in members]
    regime_of = {lb: ("2024" if lb.endswith("2024") else "cv") for lb in labels}
    # x carries the gap between the two families; group_spans drives both the background
    # shading and the figure-level legend that names them.
    x, group_spans, _pos = [], [], 0.0
    for _glab, _members in OBJECTIVE_GROUPS:
        _start = _pos
        for _m in _members:
            x.append(_pos)
            _pos += 1.0
        group_spans.append((_glab, _start - 0.5, _pos - 0.5))
        _pos += GROUP_GAP
    x = np.array(x)

    def _base(key, corr_key, lb):
        bv = data.get(lb, {}).get("baseline", {}).get(key)
        if bv is None and corr_key is not None:
            bv = base_corr.get(regime_of[lb], {}).get(corr_key)
        return bv

    def _resolve(entry, lb):
        """entry is None, a data-overall key, ('const', v), or ('base', basekey, corrkey)."""
        if entry is None:
            return None
        if isinstance(entry, tuple):
            if entry[0] == "const":
                return entry[1]
            if entry[0] == "base":
                return _base(entry[1], entry[2], lb)
            raise ValueError(entry)
        return data.get(lb, {}).get("overall", {}).get(entry)

    def plot_headline_pair(headline, fname_cv, fname_2024, suptitle_prefix):
        """headline: list of (title, valuekey, obs_entry, base_entry). obs_entry/base_entry
        are None, a data-overall key string, ('const', v) for a fixed reference (e.g. R01's
        ideal of 1), or ('base', basekey, corrkey) to read the ERA5-Land baseline."""
        # Both reference lines are black, distinguished by style rather than colour, at
        # equal weight -- observed/ideal solid, ERA5 baseline dotted -- so neither reads as
        # more important than the other, and a light fill still marks the gap between them.
        REF_LW = 1.8
        REF_FONTSIZE = 11.5
        FILL_COLOR = "#0e7c86"
        from matplotlib.lines import Line2D

        # Pass 1: gather both regimes' values per panel so the two figures can share one
        # y-limit per metric -- otherwise a taller 2024 bar would make the CV figure look
        # falsely better (or vice versa) when the two are compared side by side.
        per_regime = {}
        ylim = {}
        for regime, rtag in (("cv", "CV"), ("2024", "2024")):
            lbls = [f"{m} / {rtag}" for m in model_order]
            ref_lb = f"{model_order[0]} / {rtag}"
            vals = {title: [data.get(lb, {}).get("overall", {}).get(key, np.nan)
                             for lb in lbls]
                    for title, key, _obs, _base_e in headline}
            obsv = {title: _resolve(obs_e, ref_lb) for title, _key, obs_e, _base_e in headline}
            basev = {title: _resolve(base_e, ref_lb) for title, _key, _obs, base_e in headline}
            per_regime[regime] = (vals, obsv, basev)
        for title, *_ in headline:
            candidates = []
            for regime in ("cv", "2024"):
                vals, obsv, basev = per_regime[regime]
                candidates += [v for v in vals[title] if np.isfinite(v)]
                if obsv[title] is not None:
                    candidates.append(obsv[title])
                if basev[title] is not None:
                    candidates.append(basev[title])
            ylim[title] = (0, max(candidates) * 1.15) if candidates else None

        # One figure carrying both regimes, laid out like the tmax regime comparison:
        # hue is the model, and the regime is the hatch (CV solid, 2024 hatched). The
        # reference lines double up the same way -- CV solid/dotted, 2024 dashed/dash-dot
        # -- and their labels are combined as "CV / 2024" so each panel adds two
        # annotations rather than four. The observed-vs-baseline gap fill is gone: with
        # two regimes it no longer marks one unambiguous band, and the only shading left
        # is the objective-family bands.
        from matplotlib.patches import Patch
        BAR_W = 0.38

        def _fmt2(a, b):
            if a is None and b is None:
                return None
            if b is None:
                return f"{a:.3g}"
            if a is None:
                return f"{b:.3g}"
            return f"{a:.3g} / {b:.3g}"

        cv_vals, cv_obs, cv_base = per_regime["cv"]
        h_vals, h_obs, h_base = per_regime["2024"]
        colors = [COLORS.get(m, "C7") for m in model_order]
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.3))
        for ax_i, (ax, (title, key, obs_e, base_e)) in enumerate(
                zip(axes.ravel(), headline)):
            for (_glab, gs, ge), shade in zip(group_spans, GROUP_SHADE):
                ax.axvspan(gs, ge, color=shade, zorder=-1, lw=0)
            ax.set_xlim(group_spans[0][1], group_spans[-1][2])
            ax.bar(x - BAR_W / 2, cv_vals[title], width=BAR_W, color=colors,
                   edgecolor="white", lw=0.6, zorder=2)
            ax.bar(x + BAR_W / 2, h_vals[title], width=BAR_W, color=colors,
                   edgecolor="white", lw=0.6, hatch="///", zorder=2)
            label_transform = ax.get_yaxis_transform()
            panel_handles = []
            obs_label = "ideal" if isinstance(obs_e, tuple) and obs_e[0] == "const" \
                else "observed"
            ov_cv, ov_24 = cv_obs[title], h_obs[title]
            bv_cv, bv_24 = cv_base[title], h_base[title]
            for v, ls in ((ov_cv, "-"), (ov_24, "--")):
                if v is not None:
                    ax.axhline(v, color="black", ls=ls, lw=REF_LW, zorder=3)
            for v, ls in ((bv_cv, ":"), (bv_24, "-.")):
                if v is not None:
                    ax.axhline(v, color="#4d4d4d", ls=ls, lw=REF_LW, zorder=3)
            otxt, btxt = _fmt2(ov_cv, ov_24), _fmt2(bv_cv, bv_24)
            if otxt is not None:
                ax.text(0.02, max(v for v in (ov_cv, ov_24) if v is not None),
                        f"{obs_label} {otxt}", ha="left", va="bottom",
                        fontsize=REF_FONTSIZE, color="black", transform=label_transform)
                panel_handles += [
                    Line2D([], [], color="black", ls="-", lw=REF_LW,
                           label=f"{obs_label}, CV"),
                    Line2D([], [], color="black", ls="--", lw=REF_LW,
                           label=f"{obs_label}, 2024")]
            if btxt is not None:
                ax.text(0.98, max(v for v in (bv_cv, bv_24) if v is not None),
                        f"ERA5 {btxt}", ha="right", va="bottom",
                        fontsize=REF_FONTSIZE, color="#4d4d4d", transform=label_transform)
                panel_handles += [
                    Line2D([], [], color="#4d4d4d", ls=":", lw=REF_LW,
                           label="ERA5-Land, CV"),
                    Line2D([], [], color="#4d4d4d", ls="-.", lw=REF_LW,
                           label="ERA5-Land, 2024")]
            # The regime key belongs once, in the first panel; every panel still lists the
            # reference lines it actually draws.
            if ax_i == 0:
                panel_handles = [
                    Patch(facecolor="#9e9e9e", edgecolor="white", label="CV 2020-2023"),
                    Patch(facecolor="#9e9e9e", edgecolor="white", hatch="///",
                          label="2024 holdout")] + panel_handles
            if ylim[title]:
                ax.set_ylim(*ylim[title])
            ax.set_xticks(x)
            ax.set_xticklabels(model_order, rotation=30, ha="right", fontsize=11)
            ax.set_title(title, fontsize=13)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(axis="y", alpha=0.3)
            if panel_handles:
                ax.legend(handles=panel_handles, fontsize=8.5, loc="lower left",
                          framealpha=0.85, borderpad=0.35, labelspacing=0.3)
        # Second legend, at figure level: the per-panel one explains the reference lines
        # and the regime hatch, this one names the objective families the bands mark.
        fig.legend(handles=[Patch(facecolor=shade, edgecolor="#b0b0b0", label=glab)
                            for (glab, _gs, _ge), shade
                            in zip(group_spans, GROUP_SHADE)],
                   loc="lower center", ncol=3, fontsize=10.5, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f"{suptitle_prefix}: CV holdout (2020-2023) vs genuine 2024 holdout",
                     fontsize=14)
        fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=1.6, w_pad=1.4)
        fig.savefig(OUT / fname_cv, dpi=110)
        plt.close(fig)
        # The split-by-regime files this figure replaces would otherwise be collected and
        # indexed as live results.
        stale = OUT / fname_2024
        if stale.exists():
            stale.unlink()

    plot_headline_pair(
        headline=[
            ("MAE (mm)", "mae_mm", None, ("base", "baseline_mae_mm", "baseline_mae_mm")),
            # Not R01 itself -- R01 is the ratio pred/obs of this same quantity (Table
            # meth-precip-indices), so R01's own "observed" value would trivially be 1.0.
            # This panel plots the raw predicted frequency (a unitless fraction of days)
            # against the raw observed frequency; R01 is each bar divided by the obs.
            # line, never plotted directly, so the title must not read as "this panel is
            # R01".
            ("Wet-day frequency, pred.\n(fraction of days)", "wetday_freq_pred",
             "wetday_freq_obs", ("base", None, "baseline_wetday_freq")),
            ("SDII pred (mm)", "SDII_pred_mm", "SDII_obs_mm",
             ("base", "baseline_SDII_mm", None)),
            ("Spearman ρ (pooled, unitless)", "spearman_pooled", None,
             ("base", "baseline_spearman_pooled", "baseline_spearman_pooled")),
        ],
        fname_cv="physical_metrics_comparison.png",
        fname_2024="physical_metrics_comparison_2024.png",
        suptitle_prefix="precip on the MeteoSwiss grid",
    )

    # The four canonical physical precipitation indices (Table meth-precip-indices) in one
    # place, replacing the MAE/Spearman pairing above with the two indices it omitted
    # (R10, P98): occurrence (R01), typical intensity (SDII), heavy-day frequency (R10)
    # and the tail (P98). R01 has no "observed" value of its own (it IS the pred/obs
    # ratio), so its reference line is the ideal value 1 instead.
    plot_headline_pair(
        headline=[
            ("R01: wet-day frequency ratio\n(pred / obs)", "R01_rel_wetday_freq",
             ("const", 1.0), ("base", "baseline_R01_rel_wetday_freq",
                               "baseline_R01_rel_wetday_freq")),
            ("SDII: wet-day intensity (mm)", "SDII_pred_mm", "SDII_obs_mm",
             ("base", "baseline_SDII_mm", None)),
            ("R10: freq. $>10\\,$mm", "R10_freq_pred", "R10_freq_obs",
             ("base", None, "baseline_R10_freq")),
            ("P98: wet-day 98th pct. (mm)", "P98_pred_mm", "P98_obs_mm",
             ("base", None, "baseline_P98_mm")),
        ],
        fname_cv="precip_indices_comparison.png",
        fname_2024="precip_indices_comparison_2024.png",
        suptitle_prefix="the four physical precipitation indices",
    )

    # Threshold -> probabilistic deltas, collated across every model x regime.
    tvp = {}
    for lb in labels:
        tvp[lb] = data[lb].get("threshold_vs_probabilistic", {})
    (OUT / "threshold_vs_probabilistic_summary.json").write_text(json.dumps(tvp, indent=2))
    print("wrote physical_metrics_comparison.{md,json,png}, precip_indices_comparison.{png}, "
          "+ threshold_vs_probabilistic_summary.json")


def bss_category_comparison(regime="cv"):
    """Mean +- 1 std dev per-category Brier skill (vs ERA5-Land), all five models.

    ``regime``: ``"cv"`` (default) reads each model's ``eval_figures/``; ``"2024"`` reads
    ``eval_figures_2024/`` and suffixes every output filename with ``_2024``.

    Reads the per-point ``bss_cat`` array each model's ``eval_figures/precip_category_
    spatial.npz`` already caches (the same array precip_category_spatial.png maps
    spatially) and collapses it to the domain mean (diamond marker) with a +-1 standard
    deviation whisker across points. Median and min/max are still recorded in the JSON,
    just not plotted. No new inference.

    Within each category, the five markers are ordered and clustered by training-objective
    family (``OBJECTIVE_GROUPS``) with a small gap between families, the same grouping
    physical_comparison() uses for its model axis -- so a reader does not have to re-learn
    a new model order per figure.

    Each category's x-tick is also labelled with the pooled share of observed point-days
    that actually fall in it (from the same npz's ``counts``, which is the category's
    observed-event count pooled over every valid point and day; the categories are
    disjoint and exhaustive so ``counts`` sums to the total valid point-days) -- the BSS
    for a category built on 0.08% of the data is a very different kind of number from one
    built on 52%.
    """
    if regime not in ("cv", "2024"):
        raise ValueError(f"regime must be 'cv' or '2024', got {regime!r}")
    subdir = "eval_figures" if regime == "cv" else "eval_figures_2024"
    suffix = "" if regime == "cv" else "_2024"
    regime_label = "CV holdout" if regime == "cv" else "2024 holdout"

    model_order = [m for _lab, members in OBJECTIVE_GROUPS for m in members]
    # Small extra gap between objective-family sub-clusters, inside one category's span.
    n_models = len(model_order)
    n_gaps = len(OBJECTIVE_GROUPS) - 1
    bar_w = 0.8 / (n_models + n_gaps * 0.6)
    offsets = []
    pos = 0.0
    for gi, (_lab, members) in enumerate(OBJECTIVE_GROUPS):
        for _m in members:
            offsets.append(pos)
            pos += bar_w
        if gi < n_gaps:
            pos += bar_w * 0.6
    offsets = np.array(offsets) - pos / 2 + bar_w / 2   # centre the cluster on 0

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = None
    counts = None
    results = {}
    for name, off in zip(model_order, offsets):
        d = MODELS[name]
        npz = d / subdir / "precip_category_spatial.npz"
        if not npz.exists():
            continue
        z = np.load(npz, allow_pickle=True)
        cat_labels = [str(c) for c in z["cat_labels"]]
        if labels is None:
            labels = cat_labels
            counts = z["counts"].astype(np.float64)   # pooled obs-event count per category
        elif cat_labels != labels:
            raise SystemExit(f"{name}: category edges {cat_labels} differ from {labels}")
        bss = z["bss_cat"].astype(np.float64)                # (P, C)
        mean = np.nanmean(bss, axis=0)
        median = np.nanmedian(bss, axis=0)
        std = np.nanstd(bss, axis=0)
        lo = np.nanmin(bss, axis=0)
        hi = np.nanmax(bss, axis=0)
        results[name] = {"mean": mean.tolist(), "median": median.tolist(),
                         "std": std.tolist(), "min": lo.tolist(), "max": hi.tolist()}
        x = np.arange(len(labels)) + off
        ax.errorbar(x, mean, yerr=std, fmt="none",
                   ecolor="black", elinewidth=1.0, capsize=3, zorder=3)
        ax.scatter(x, mean, marker="D", s=55, facecolor=COLORS.get(name, "C7"),
                  edgecolor="black", linewidth=1.1, label=name, zorder=4)

    obs_pct = 100.0 * counts / counts.sum()
    xticklabels = [f"{lb} mm\n({p:.2g}% of obs)" for lb, p in zip(labels, obs_pct)]
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(labels)), xticklabels)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Brier skill score vs ERA5-Land")
    ax.set_xlabel("precipitation-intensity category")
    ax.legend(fontsize=8, ncols=n_models, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Mean (◆) ± 1 std dev across points, per-category Brier skill by model, "
                 f"{regime_label}", fontsize=11)
    fig.tight_layout()
    out_png = OUT / f"bss_category_comparison{suffix}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    out_json = OUT / f"bss_category_comparison{suffix}.json"
    out_json.write_text(json.dumps({"cat_labels": labels, "reference": "ERA5-Land",
                                    "regime": regime, "obs_pct_of_point_days": obs_pct.tolist(),
                                    "models": results}, indent=2))
    print(f"wrote {out_json}")

    # Markdown table matching the figure exactly: same mean +- std cells, same model
    # order (grouped by objective family), same category columns and obs% headers.
    header = ("| Model | " + " | ".join(f"{lb} mm ({p:.2g}% obs)"
              for lb, p in zip(labels, obs_pct)) + " |")
    sep = "|---|" + "---|" * len(labels)
    lines = [f"# Mean Brier skill by category (vs ERA5-Land, {regime_label})", "", header, sep]
    for glab, members in OBJECTIVE_GROUPS:
        for name in members:
            if name not in results:
                continue
            r = results[name]
            cells = [f"{m:+.3f} ± {s:.3f}" for m, s in zip(r["mean"], r["std"])]
            lines.append(f"| {name} ({glab}) | " + " | ".join(cells) + " |")
    out_md = OUT / f"bss_category_comparison{suffix}.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    training_comparison()
    physical_comparison()
    bss_category_comparison("cv")
    bss_category_comparison("2024")
    print(f"comparison artifacts in {OUT}")
