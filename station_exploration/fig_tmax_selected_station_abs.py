#!/usr/bin/env python3
"""Station verification of the selected temperature model alone -- no baselines.

The report chapter carries only the selected models and scores them against the same
reference as its gridded chapter, plain bilinear ERA5-Land with no elevation adjustment
(the lapse-corrected re-scoring lives in the appendix, fig_tmax_stations.py). This
produces the two chapter figures for atm+wind-clip60:

  tmax_selected_station_abs.png   absolute errors at the 28 NBCN stations, model
                                  beside the raw bilinear reference
  tmax_station_skill_map_raw.png  per-station raw-reference skill at the 147 SMN
                                  stations, NBCN sites ringed

Reads the per-station tables written by fig_tmax_stations.py (figures/tmax_per_station_*.csv);
pooled values are day-weighted over station-days, reproducing the pooled() convention there.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGS = Path(__file__).resolve().parent / "figures"
BLUE, GREY = "#3b76af", "#9a9a9a"
BAR_W = 0.36
REGIMES = [("cv", "CV holdout 2020-2023", False), ("holdout_2024", "2024 holdout", True)]


def load(regime: str) -> pd.DataFrame:
    return pd.read_csv(FIGS / f"tmax_per_station_{regime}.csv")


def pooled_nbcn(df: pd.DataFrame) -> dict:
    df = df[df.cohort == "nbcn"]
    w = df.n_days.to_numpy(float)

    def pool(col):
        return float(np.sum(df[col].to_numpy(float) * w) / np.sum(w))

    out = {k: pool(k) for k in ("mae", "crps", "mae_ref_raw")}
    out["skill_raw"] = 1.0 - out["crps"] / out["mae_ref_raw"]
    out["n_stations"] = len(df)
    return out


def fig_absolute(stats: dict):
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    keys = ["mae", "crps", "mae_ref_raw"]
    x = np.arange(len(keys))
    for k, (reg, lab, hatch) in enumerate(REGIMES):
        v = [stats[reg][key] for key in keys]
        pos = x + (k - 0.5) * BAR_W
        ax.bar(pos, v, BAR_W, color=[BLUE, BLUE, GREY], edgecolor="white",
               hatch="///" if hatch else None)
        for xi, vi in zip(pos, v):
            ax.text(xi, vi + 0.06, f"{vi:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("station error [$^{\\circ}$C]")
    ax.set_ylim(0, 5.0)
    ax.set_xticks(x, ["model\nMAE", "model\nCRPS", "bilinear ERA5-Land\n(MAE = CRPS)"],
                  fontsize=9.5)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=BLUE, label="model, solid CV and hatched 2024"),
               Patch(facecolor=GREY, label="bilinear ERA5-Land reference")]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    n = stats["cv"]["n_stations"]
    ax.set_title(f"atm+wind-clip60 at the {n} NBCN stations, absolute error against\n"
                 "the same bilinear ERA5-Land reference as the gridded evaluation", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "tmax_selected_station_abs.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGS / 'tmax_selected_station_abs.png'}")


def fig_skill_map_raw(frames: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.7), sharey=True)
    for ax, (reg, lab, _) in zip(axes, REGIMES):
        df = frames[reg]
        # plate carrée aspect: at Swiss latitude one degree of longitude spans
        # cos(46.8 deg) of a degree of latitude, else the map is stretched east-west
        ax.set_aspect(1.0 / np.cos(np.deg2rad(46.8)))
        sc = ax.scatter(df["lon"], df["lat"], c=df["skill_raw"], cmap="RdBu",
                        vmin=-0.9, vmax=0.9, s=34, zorder=3)
        nb = df[df.is_nbcn]
        ax.scatter(nb["lon"], nb["lat"], facecolors="none", edgecolors="black",
                   s=110, linewidths=0.9, zorder=4)
        share = float(np.mean(df["skill_raw"] > 0))
        ax.set_title(f"{lab}\nmedian {df['skill_raw'].median():+.2f} · stations "
                     f"above zero {share:.0%}", fontsize=10)
        ax.set_xlabel("longitude [$^{\\circ}$E]", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("latitude [$^{\\circ}$N]", fontsize=9)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("CRPS skill vs bilinear ERA5-Land [–]", fontsize=9)
    n = len(frames["cv"])
    fig.suptitle(f"atm+wind-clip60 at the {n} SMN stations (operational daily maximum) — "
                 "skill on the map\nblack rings mark the 28 NBCN sites", fontsize=11.5, y=1.14)
    fig.savefig(FIGS / "tmax_station_skill_map_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / 'tmax_station_skill_map_raw.png'}")


def main():
    dfs = {reg: load(reg) for reg, *_ in REGIMES}
    stats = {reg: pooled_nbcn(df) for reg, df in dfs.items()}
    fig_absolute(stats)
    fig_skill_map_raw({reg: df[df.cohort == "smn"] for reg, df in dfs.items()})
    for reg, lab, _ in REGIMES:
        s = stats[reg]
        smn = dfs[reg][dfs[reg].cohort == "smn"]
        w = smn.n_days.to_numpy(float)
        smn_mae = float(np.sum(smn.mae.to_numpy(float) * w) / np.sum(w))
        print("  %-20s NBCN: MAE %.3f  CRPS %.3f  ref %.2f  skill %.3f | "
              "SMN: MAE %.3f  median skill %+.2f  share>0 %.0f%%"
              % (lab, s["mae"], s["crps"], s["mae_ref_raw"], s["skill_raw"],
                 smn_mae, smn.skill_raw.median(), 100 * float((smn.skill_raw > 0).mean())))


if __name__ == "__main__":
    main()
