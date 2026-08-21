#!/usr/bin/env python
"""Per-pixel MAE against altitude and against elevation mismatch, overlaid across models.

Two panels sharing one y axis: the binned per-pixel MAE as a function of target altitude,
and as a function of the signed elevation difference between the target point and the
coarse orography the model is given. Both panels are built from the same prediction
bundles, the same days, the same grid points and the same binning rule, so the two views
differ only in the terrain variable on the x axis.

The default set is the three 30-epoch input families, which share a training budget --- so
the only thing that differs between the curves is what the model was allowed to see.

    python scripts/gen_tmax_mae_vs_terrain.py --out LATEX_REPORT/images/tmax_mae_vs_terrain.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "CLEAN_trained_models"

RUNS = [
    ("sfc", "baseline__tmax_sfc_flat_y2020-2023_e30f5_b8", "#eb6834"),
    ("atm", "clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8", "#2a78d6"),
    ("atm+sfcanchors", "sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8", "#4a3aa7"),
]

# Bin width in metres, and the smallest bin population we are willing to plot. Sparse tail
# bins are dropped rather than drawn with an uninformative error bar.
BIN_M = 200
MIN_COUNT = 50


def load(run_dir: str, regime: str):
    p = MODELS / run_dir / "tmax" / "pred_cache" / f"{regime}.npz"
    if not p.exists():
        raise SystemExit(f"missing prediction bundle: {p}")
    b = np.load(p, allow_pickle=True)
    mask = b["valid_mask"].ravel()
    mae = np.nanmean(np.abs(b["preds_degC"] - b["truths_degC"]), axis=0)
    return mae, b["target_topo"][mask], b["preds_degC"].shape[0]


def binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray):
    """Mean and standard error of y within each bin of x, dropping thin bins."""
    idx = np.digitize(x, edges) - 1
    centres, means, sems = [], [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        n = int(sel.sum())
        if n < MIN_COUNT:
            continue
        centres.append(0.5 * (edges[k] + edges[k + 1]))
        means.append(float(np.nanmean(y[sel])))
        sems.append(float(np.nanstd(y[sel]) / np.sqrt(n)))
    return np.array(centres), np.array(means), np.array(sems)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="LATEX_REPORT/images/tmax_mae_vs_terrain.png")
    ap.add_argument("--regime", default="cv", choices=["cv", "holdout_2024"])
    args = ap.parse_args()

    data, topo, ndays = {}, None, None
    for label, d, _ in RUNS:
        mae, t, ndays = load(d, args.regime)
        data[label] = mae
        if topo is None:
            topo = t
        elif not np.allclose(topo, t, equal_nan=True):
            raise SystemExit("models do not share a target grid; the overlay would be invalid")

    panels = [
        ("altitude", topo[:, 0], "target altitude (m)"),
        ("mismatch", topo[:, 1], "elevation difference to the coarse orography (m)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), sharey=True)
    for ax, (key, xv, xlabel) in zip(axes, panels):
        lo, hi = np.percentile(xv, [0.5, 99.5])
        edges = np.arange(np.floor(lo / BIN_M) * BIN_M, np.ceil(hi / BIN_M) * BIN_M + BIN_M, BIN_M)
        for label, _, colour in RUNS:
            c, m, s = binned(xv, data[label], edges)
            ax.plot(c, m, color=colour, lw=1.9, label=label, zorder=3)
            ax.fill_between(c, m - s, m + s, color=colour, alpha=0.20, lw=0, zorder=2)
        if key == "mismatch":
            ax.axvline(0, color="#999999", lw=0.9, ls=":", zorder=1)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(color="#ececec", lw=0.6)
        ax.tick_params(labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("per-pixel MAE ($^\\circ$C)", fontsize=10)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")

    period = "CV holdout" if args.regime == "cv" else "2024 holdout"
    fig.suptitle(f"Per-pixel MAE against terrain, {period} ({ndays} days, "
                 f"{BIN_M} m bins, shading is the standard error)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")
    for key, xv, _ in panels:
        lo, hi = np.percentile(xv, [0.5, 99.5])
        edges = np.arange(np.floor(lo / BIN_M) * BIN_M, np.ceil(hi / BIN_M) * BIN_M + BIN_M, BIN_M)
        print(f"  {key}:")
        for label, _, _c in RUNS:
            c, m, _s = binned(xv, data[label], edges)
            print(f"    {label:16s} first bin {m[0]:.2f} @ {c[0]:.0f} m, "
                  f"last bin {m[-1]:.2f} @ {c[-1]:.0f} m, span {m.max()-m.min():.2f}")
        c0, m0, _ = binned(xv, data["sfc"], edges)
        c1, m1, _ = binned(xv, data["atm"], edges)
        print(f"    sfc-atm gap: min {np.min(m0-m1):+.2f}, max {np.max(m0-m1):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
