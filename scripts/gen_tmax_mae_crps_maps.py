#!/usr/bin/env python
"""Per-pixel MAE and CRPS maps for three temperature models, on shared per-row scales.

Rows are the two metrics, columns the models, so a colour means the same thing across a
row and the three configurations can be compared directly. Read from the cached prediction
bundles written by ``evaluate.py``, so this costs no inference.

    python scripts/gen_tmax_mae_crps_maps.py --out LATEX_REPORT/images/tmax_mae_crps_maps.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import properscoring as ps

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "CLEAN_trained_models"

# Ordered as they are discussed: the two 30-epoch baselines, which differ only in input,
# then the two extended-budget retrains.
RUNS = [
    ("sfc", "baseline__tmax_sfc_flat_y2020-2023_e30f5_b8"),
    ("atm", "clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8"),
    ("atm-clip60", "lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8"),
    ("atm+wind-clip60", "lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8"),
]


def per_pixel(run_dir: str, regime: str):
    p = MODELS / run_dir / "tmax" / "pred_cache" / f"{regime}.npz"
    if not p.exists():
        raise SystemExit(f"missing prediction bundle: {p}")
    b = np.load(p, allow_pickle=True)
    preds, truths, sigmas = b["preds_degC"], b["truths_degC"], b["sigmas_degC"]
    mae = np.nanmean(np.abs(preds - truths), axis=0)
    crps = np.nanmean(ps.crps_gaussian(truths, mu=preds, sig=sigmas), axis=0)

    mask = b["valid_mask"]

    def grid(v):
        g = np.full(mask.size, np.nan)
        g[mask.ravel()] = v
        return g.reshape(mask.shape)

    return grid(mae), grid(crps), preds.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="LATEX_REPORT/images/tmax_mae_crps_maps.png")
    ap.add_argument("--regime", default="cv", choices=["cv", "holdout_2024"])
    args = ap.parse_args()

    data, ndays = {}, None
    for label, d in RUNS:
        mae, crps, ndays = per_pixel(d, args.regime)
        data[label] = (mae, crps)

    rows = [("MAE", 0), ("CRPS", 1)]
    # One scale per row, from the pooled distribution across the three models, so a colour
    # means the same in every column. 99th percentile rather than the max: a handful of
    # valley pixels would otherwise set the range and flatten everything else.
    limits = {}
    for name, i in rows:
        pooled = np.concatenate([data[l][i][np.isfinite(data[l][i])] for l, _ in RUNS])
        limits[name] = (float(np.percentile(pooled, 1)), float(np.percentile(pooled, 99)))

    fig, axes = plt.subplots(2, len(RUNS), figsize=(3.9 * len(RUNS) + 1.0, 6.4))
    for r, (name, i) in enumerate(rows):
        lo, hi = limits[name]
        for c, (label, _) in enumerate(RUNS):
            ax = axes[r, c]
            g = data[label][i]
            im = ax.imshow(g, cmap="YlOrRd", vmin=lo, vmax=hi, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(rf"$\mathtt{{{label}}}$", fontsize=11)
            ax.text(0.02, 0.03, f"mean {np.nanmean(g):.2f}$^\\circ$C", transform=ax.transAxes,
                    fontsize=8.5, va="bottom", ha="left",
                    bbox=dict(fc="white", ec="#999999", lw=0.5, pad=2.5))
            if c == 0:
                ax.set_ylabel(f"per-pixel {name}", fontsize=10)
        cb = fig.colorbar(im, ax=list(axes[r, :]), fraction=0.024, pad=0.012)
        cb.set_label(f"{name} ($^\\circ$C)", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    period = "CV holdout" if args.regime == "cv" else "2024 holdout"
    fig.suptitle(f"Per-pixel MAE and CRPS, {period} ({ndays} days)", fontsize=12)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")
    for name, i in rows:
        print(f"  {name}: scale {limits[name][0]:.2f}..{limits[name][1]:.2f}; means " +
              ", ".join(f"{l} {np.nanmean(data[l][i]):.3f}" for l, _ in RUNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
