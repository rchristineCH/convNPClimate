#!/usr/bin/env python
"""Terrain-position regimes for tmax: classify mTPI, then show MAE against the classes.

``error_analysis`` plots MAE against mTPI as a scatter with a single binned mean and
reports a correlation. That correlation is about -0.01, which reads as "terrain position
does not matter" — but it is near zero because the effect is **symmetric**: valleys and
ridges are both worse than flat ground, so a linear coefficient cancels them against each
other. Splitting mTPI into regimes and colouring by them makes the structure visible, and
it is not small: at +/-30 m the gap between flat and valley is over 0.16 degC.

Two figures, written to ``CLEAN_trained_models/tmax_model_comparison``:

``mtpi_regime_classification.png``
    Where the class edges come from. The mTPI distribution over the 46,718 valid target
    points with the five bands marked and their populations, plus the per-class MAE of the
    reference model, so the choice of edges can be judged against what it separates.
    Terrain only — no model enters the left panel.

``mtpi_regime_mae.png``
    Per-class MAE for every CLEAN tmax model: does any of them handle terrain position
    better, or do they all degrade the same way off flat ground?

Reads the cached CV prediction bundles, so this costs no GPU. Usage:
    python scripts/mtpi_regime_figures.py [--eval-year 2024]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate as ev  # noqa: E402
from compare_evaluations import MODEL_STYLE  # noqa: E402

MODELS_ROOT = ROOT / "CLEAN_trained_models"
OUT = MODELS_ROOT / "tmax_model_comparison"

# (short label, run directory) in the order the report presents them.
MODELS = [
    ("atm", "clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8"),
    ("atm-clip60", "lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8"),
    ("atm+wind", "clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8"),
    ("atm+wind-clip60", "lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8"),
    ("atm+sfcanchors", "sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8"),
    ("sfc", "baseline__tmax_sfc_flat_y2020-2023_e30f5_b8"),
    ("sfc-nogeo", "baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo"),
]
# The models whose per-class MAE the classification figure shows. The histogram
# beside them is terrain only, so it does not depend on this choice.
REFERENCES = ("sfc", "atm-clip60", "atm+wind-clip60")

# Class edges in metres. Chosen from the measured distribution rather than round numbers:
# sd is 18.1 m and the median |mTPI| is 6.6 m, so +/-10 m separates genuinely flat ground
# (62% of points) from sloped, and +/-30 m isolates the pronounced tails (3.9% and 5.4%)
# where the terrain signal is strongest. Both tails are kept as their own class because
# the point of the figure is that they behave alike — which is what the correlation hides.
EDGES = (-np.inf, -30.0, -10.0, 10.0, 30.0, np.inf)
CLASS_NAMES = ("deep valley\n< -30 m", "valley\n-30 to -10 m", "flat\n-10 to +10 m",
               "ridge\n+10 to +30 m", "exposed ridge\n> +30 m")
# Shorter forms for the bar axes, where five full ranges collide.
CLASS_TICKS = ("deep valley\n<-30", "valley\n-30..-10", "flat\n-10..+10",
               "ridge\n+10..+30", "exposed\n>+30")
# Ranges alone for the narrow per-model panels: the bar colours already carry the class,
# and five worded labels cannot fit side by side at that width.
CLASS_TICKS_SHORT = ("< -30", "-30..-10", "-10..+10", "+10..+30", "> +30")
# Diverging, symmetric about flat: valleys and ridges are mirror cases, and the colours
# say so. Flat is neutral grey because it is the reference, not an extreme.
CLASS_COLORS = ("#2c5f8a", "#7fa8c9", "#9a9a9a", "#e0a267", "#b5561f")
OBS_INK, MUTED, LINE = "#333333", "#777777", "#dddddd"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#555555", labelsize=9)


def load(run: str, eval_year: int | None):
    """(per-point MAE, mTPI) for one model, from its cached bundle."""
    md = MODELS_ROOT / run / "tmax"
    path = ev.bundle_cache_path(md, eval_year)
    if not path.exists():
        raise SystemExit(f"missing {path} — run: python evaluate.py --model-dir {md}"
                         + ("" if eval_year is None else f" --eval-year {eval_year}"))
    B = ev.load_bundle(path, md)
    err = np.asarray(B.errors_c)
    day = B.day_mask if B.day_mask is not None else np.ones(err.shape[0], bool)
    mae = np.nanmean(np.abs(err[day]), axis=0)
    mtpi = np.asarray(B.target_topo)[:, 2]
    ok = np.isfinite(mae) & np.isfinite(mtpi)
    return mae[ok], mtpi[ok]


def classify(mtpi: np.ndarray) -> np.ndarray:
    """Class index per point, 0..4, from the fixed edges."""
    return np.clip(np.digitize(mtpi, EDGES[1:-1]), 0, len(CLASS_NAMES) - 1)


def classification_figure(refs: list[tuple[str, np.ndarray, np.ndarray]],
                          best: tuple[str, np.ndarray, np.ndarray],
                          regime: str) -> None:
    """The class edges, then per-class MAE for each reference model.

    The histogram is terrain only and identical for every model, so it is drawn once; the
    bar panels beside it show what those edges separate, one model each. The best model
    over all classes is overlaid on every panel, so each one is read against what is
    achievable rather than only against its own flat-ground baseline.
    """
    mtpi = refs[0][2]
    cls = classify(mtpi)
    fig, axes = plt.subplots(1, 1 + len(refs), figsize=(6.0 + 3.3 * len(refs), 5.0),
                             gridspec_kw={"width_ratios": [1.9] + [1] * len(refs)})
    ax0, bar_axes = axes[0], axes[1:]

    # --- left: the distribution and where the edges fall
    bins = np.linspace(-90, 90, 121)
    for k, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        sel = cls == k
        ax0.hist(np.clip(mtpi[sel], -90, 90), bins=bins, color=color, lw=0,
                 label=f"{name.replace(chr(10), '  ')}   {sel.mean():.1%}")
    for edge in EDGES[1:-1]:
        ax0.axvline(edge, color=OBS_INK, lw=1, ls="--", zorder=5)
    ax0.set_xlim(-90, 90)
    ax0.set_xlabel("mTPI (m)   — negative: sheltered / valley,  positive: exposed / ridge",
                   color=OBS_INK)
    ax0.set_ylabel("grid points", color=OBS_INK)
    ax0.set_title(f"mTPI distribution over {mtpi.size:,} target points\n"
                  f"sd {mtpi.std():.1f} m, median |mTPI| {np.median(np.abs(mtpi)):.1f} m; "
                  "outer bins clipped at ±90 m", fontsize=10.5, color=OBS_INK)
    ax0.legend(fontsize=9, frameon=False, labelcolor=OBS_INK, title="terrain class",
               title_fontsize=9, loc="upper left")
    style(ax0)

    # --- right: one panel per reference model, on a shared y so they compare directly
    best_means = [best[1][cls == k].mean() for k in range(len(CLASS_NAMES))]
    all_means = list(best_means)
    for label, mae, _ in refs:
        all_means += [mae[cls == k].mean() for k in range(len(CLASS_NAMES))]
    top = max(all_means) * 1.30

    xs = np.arange(len(CLASS_NAMES))
    for ax, (label, mae, _) in zip(bar_axes, refs):
        means = [mae[cls == k].mean() for k in range(len(CLASS_NAMES))]
        ses = [mae[cls == k].std() / np.sqrt(max((cls == k).sum(), 1))
               for k in range(len(CLASS_NAMES))]
        ax.bar(xs, means, yerr=ses, color=CLASS_COLORS, width=0.72, capsize=3,
               error_kw={"lw": 1, "ecolor": OBS_INK})
        flat_mean = means[2]
        ax.axhline(flat_mean, color=OBS_INK, ls=":", lw=1.2, zorder=4)
        for x, m in zip(xs, means):
            ax.text(x, m, f"{m:.3f}\n{m - flat_mean:+.3f}", ha="center", va="bottom",
                    fontsize=8, color=OBS_INK)
        ax.plot(xs, best_means, color="#111111", lw=1.6, ls="--", marker="o", ms=4,
                zorder=6, label=f"best: {best[0]}")
        if ax is bar_axes[0]:
            ax.legend(fontsize=8.5, frameon=False, loc="upper center", labelcolor=OBS_INK)
        ax.set_xticks(xs, CLASS_TICKS_SHORT, fontsize=8.5, rotation=30,
                      ha="right")
        ax.set_xlabel("mTPI class (m)", color=OBS_INK, fontsize=9)
        ax.set_ylim(0, top)
        ax.set_title(f"{label}\nflat {flat_mean:.3f} °C,  spread "
                     f"{max(means) - min(means):.3f} °C", fontsize=10.5,
                     color=MODEL_STYLE[label][0])
        if ax is bar_axes[0]:
            ax.set_ylabel("mean per-grid-point MAE (°C)", color=OBS_INK)
        else:
            ax.tick_params(axis="y", labelleft=False)
        style(ax)

    fig.suptitle(f"Terrain-position regimes: classifying mTPI — {regime}",
                 fontsize=13.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, OUT / f"mtpi_regime_classification{_suffix(regime)}.png")


def comparison_figure(per_model: list[tuple[str, np.ndarray, np.ndarray]],
                      regime: str) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    xs = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(per_model)
    for k, (label, mae, mtpi) in enumerate(per_model):
        cls = classify(mtpi)
        means = [mae[cls == c].mean() for c in range(len(CLASS_NAMES))]
        offset = (k - (len(per_model) - 1) / 2) * width
        ax.bar(xs + offset, means, width=width * 0.9, label=label,
               color=MODEL_STYLE[label][0], edgecolor="none")
    ax.set_xticks(xs, [n.replace("\n", "  ") for n in CLASS_TICKS], fontsize=10)
    ax.set_ylabel("mean per-grid-point MAE (°C)", color=OBS_INK)
    ax.set_title(f"MAE by terrain-position class — {len(per_model)} tmax models, {regime}",
                 fontsize=12)
    ax.legend(fontsize=8.5, frameon=False, ncol=len(per_model), loc="upper center",
              bbox_to_anchor=(0.5, -0.10))
    style(ax)
    fig.tight_layout()
    _save(fig, OUT / f"mtpi_regime_mae{_suffix(regime)}.png")


def _suffix(regime: str) -> str:
    return "" if regime.startswith("CV") else "_2024"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-year", type=int, default=None,
                    help="holdout year instead of the CV span")
    args = ap.parse_args()
    regime = "CV 2020-2023" if args.eval_year is None else f"{args.eval_year} holdout"

    per_model = []
    for label, run in MODELS:
        mae, mtpi = load(run, args.eval_year)
        per_model.append((label, mae, mtpi))
        print(f"  {label}: {mae.size:,} points, MAE {mae.mean():.3f} degC", flush=True)

    refs = [t for t in per_model if t[0] in REFERENCES]
    missing = set(REFERENCES) - {t[0] for t in refs}
    if missing:
        raise SystemExit(f"unknown reference model(s): {sorted(missing)}")
    refs.sort(key=lambda t: REFERENCES.index(t[0]))
    best = min(per_model, key=lambda t: t[1].mean())
    print(f"  best overall: {best[0]} ({best[1].mean():.3f} degC)", flush=True)
    classification_figure(refs, best, regime)
    comparison_figure(per_model, regime)


if __name__ == "__main__":
    main()
