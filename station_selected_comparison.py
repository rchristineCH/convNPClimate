#!/usr/bin/env python3
"""Station verification of the selected temperature model, against its two baselines.

The chapter-level figure (station_regime_comparison.png) carries all seven temperature
configurations, which is more than the verification argument needs: the point is that the
selected model beats plain interpolation at real stations and keeps doing so on a year no
fold saw. This draws only the three configurations that carry that argument -- the surface
baseline it has to beat, the plain atmospheric run at the same input, and the selected model
itself -- under both evaluation regimes.

Reads the cached station metrics written by compare_evaluations.py; no inference.
Produces ``CLEAN_trained_models/tmax_model_comparison/station_selected_comparison.png``.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "CLEAN_trained_models/tmax_model_comparison"
SRC = OUT / "station_metrics_comparison.csv"
MODELS = ["sfc", "atm", "atm+wind-clip60"]
REGIMES = [("cv", "CV holdout", "#3b76af", False), ("2024", "2024 holdout", "#3b76af", True)]
BAR_W = 0.36
REF_COL = "#9a9a9a"


def main():
    df = pd.read_csv(SRC)
    df = df[df["model"].isin(MODELS)]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.1))
    x = np.arange(len(MODELS))

    ax = axes[0]
    for k, (reg, lab, col, hatch) in enumerate(REGIMES):
        v = [float(df[(df.model == m) & (df.regime == reg)]["st_mae"].iloc[0]) for m in MODELS]
        ax.bar(x + (k - 0.5) * BAR_W, v, BAR_W, color=col, edgecolor="white",
               hatch="///" if hatch else None, label=lab)
        for xi, vi in zip(x + (k - 0.5) * BAR_W, v):
            ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("station MAE [$^{\\circ}$C]")
    ax.set_ylim(0, 2.15)
    ax.set_title("Error at the stations", fontsize=11)

    # The skill panel on the right is a ratio and hides how large the errors actually are, so
    # the same comparison is drawn once in physical units: the CRPS each model reaches at the
    # stations next to the bilinear ERA5-Land reference it is scored against. The reference is
    # deterministic, so its CRPS is its MAE -- crps/(1-skill) reproduces st_mae_era5 exactly.
    ax = axes[1]
    xr = np.arange(len(MODELS) + 1)
    for k, (reg, lab, col, hatch) in enumerate(REGIMES):
        v = [float(df[(df.model == m) & (df.regime == reg)]["st_crps"].iloc[0]) for m in MODELS]
        v.append(float(df[(df.model == MODELS[0]) & (df.regime == reg)]["st_mae_era5"].iloc[0]))
        pos = xr + (k - 0.5) * BAR_W
        ax.bar(pos[:-1], v[:-1], BAR_W, color=col, edgecolor="white",
               hatch="///" if hatch else None, label=lab)
        ax.bar(pos[-1], v[-1], BAR_W, color=REF_COL, edgecolor="white",
               hatch="///" if hatch else None)
        for xi, vi in zip(pos, v):
            ax.text(xi, vi + 0.06, f"{vi:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("station CRPS [$^{\\circ}$C]")
    ax.set_ylim(0, 5.2)
    ax.set_title("Absolute error, models and reference", fontsize=11)
    ax.set_xticks(xr, MODELS + ["bilinear\nERA5-Land"], fontsize=9.5)

    ax = axes[2]
    for k, (reg, lab, col, hatch) in enumerate(REGIMES):
        v = [float(df[(df.model == m) & (df.regime == reg)]["st_skill"].iloc[0]) for m in MODELS]
        lo = [float(df[(df.model == m) & (df.regime == reg)]["st_skill_low"].iloc[0]) for m in MODELS]
        pos = x + (k - 0.5) * BAR_W
        ax.bar(pos, v, BAR_W, color=col, edgecolor="white",
               hatch="///" if hatch else None, label=lab)
        # the pooled skill is inflated by the unresolved station elevation; the lowland-only
        # value is the honest one and is marked on top of each bar rather than beside it.
        ax.scatter(pos, lo, marker="_", s=260, color="black", linewidths=1.8, zorder=4)
        for xi, vi in zip(pos, lo):
            ax.text(xi, vi - 0.035, f"{vi:.2f}", ha="center", va="top", fontsize=8)
    ax.set_ylabel("station skill vs bilinear ERA5-Land")
    ax.set_ylim(0, 0.95)
    ax.set_title("Skill against interpolation", fontsize=11)
    ax.scatter([], [], marker="_", s=260, color="black", linewidths=1.8, label="lowland stations only")

    for i, ax in enumerate(axes):
        if i != 1:  # the middle panel carries the extra reference tick
            ax.set_xticks(x, MODELS, fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    axes[1].legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    axes[2].legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    fig.suptitle("Verification at 28 NBCN stations, solid CV holdout and hatched 2024", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "station_selected_comparison.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT / 'station_selected_comparison.png'}")
    for m in MODELS:
        for reg, *_ in REGIMES:
            r = df[(df.model == m) & (df.regime == reg)].iloc[0]
            print("  %-16s %-5s MAE %.3f  skill %.3f  lowland %.3f  ERA5 MAE %.2f"
                  % (m, reg, r.st_mae, r.st_skill, r.st_skill_low, r.st_mae_era5))


if __name__ == "__main__":
    main()
