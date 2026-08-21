#!/usr/bin/env python
"""One compact verification-at-stations figure for a short report chapter.

Left: the selected temperature model's per-station CRPS skill against the
lapse-corrected reference (the honest baseline), CV and 2024. Right: the two
selected precipitation runs' per-gauge MAE skill against bilinear ERA5-Land.
Every marker is one station; the bar is the median; pooled values are printed.

Usage:  python station_exploration/fig_summary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from fig_precip_stations import load as load_precip, per_gauge_rpss  # noqa: E402

FIGS = HERE / "figures"
AQUA = "#1baf7a"
FT, SFC = "#8e5bd0", "#c9932e"


def strip(ax, x, values, color, hatch=False):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(7)
    ax.scatter(x + rng.uniform(-0.13, 0.13, len(v)), v, s=16, color=color,
               alpha=0.65, edgecolors="none", zorder=3)
    med = np.median(v)
    ax.plot([x - 0.22, x + 0.22], [med, med], color="#222222", lw=2.0, zorder=4)
    return med


def main() -> int:
    sys.path.insert(0, str(HERE))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1.6]})

    # --- tmax: both cohorts (NBCN homogenised, SMN operational) ---
    ax = axes[0]
    SMN_TINT = "#0e7a54"
    cols = []
    for k, (cohort, color, label) in enumerate(
            [("nbcn", AQUA, "NBCN"), ("smn", SMN_TINT, "SMN")]):
        for j, reg in enumerate(("cv", "holdout_2024")):
            df = pd.read_csv(FIGS / f"tmax_per_station_{reg}.csv")
            df = df[df["cohort"] == cohort]
            x = k * 2 + j
            strip(ax, x, df["skill_corr"], color)
            cols.append((x, f"{label}\n{'CV' if reg == 'cv' else '2024'}"))
    ax.set_xticks([c[0] for c in cols], [c[1] for c in cols], fontsize=9)
    ax.set_ylim(-0.1, 0.9)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_ylabel("CRPS skill vs lapse-corrected ERA5-Land [–]", fontsize=9)
    ax.set_title("tmax — atm+wind-clip60\n28 NBCN (homogenised) · 147 SMN "
                 "(operational)", fontsize=10)

    # --- precip ---
    ax = axes[1]
    cols = []
    for k, (stem, color, label) in enumerate(
            [("ft-crps", FT, "FT-CRPS"), ("sfc-tp", SFC, "SFC-TP")]):
        for j, reg in enumerate(("cv", "2024")):
            b = load_precip(f"{stem}_{reg}")
            sk = per_gauge_rpss(b)
            x = k * 2 + j
            med = strip(ax, x, sk, color)
            cols.append((x, f"{label}\n{'CV' if reg == 'cv' else '2024'}", med))
    ax.set_xticks([c[0] for c in cols], [c[1] for c in cols], fontsize=9)
    ax.axhline(0, color="#999999", lw=0.9)
    ax.set_ylim(-0.65, 0.85)
    ax.set_ylabel("RPSS vs bilinear ERA5-Land (w0606) [–]\n"
                  "(five intensity categories)", fontsize=9)
    ax.set_title("precip — the two selected runs\n139 SMN rain gauges", fontsize=10)

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#555555", labelsize=9)

    fig.suptitle("Verification at the stations — one marker per station, bar = median",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "stations_summary.png", dpi=150, bbox_inches="tight")
    print("wrote", FIGS / "stations_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
