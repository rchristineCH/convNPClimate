#!/usr/bin/env python
"""Per-pixel skill maps for the surface and atmospheric temperature models, plus their delta.

Three panels on the CV holdout (2020--2023): skill of ``sfc``, skill of ``atm``, and the
difference between them. The two skill panels are computed through ``metrics.py``, the same
code path that produced the per-model ``skill_map.png`` figures, so they reproduce those
maps exactly; the third panel is their pixelwise difference on a diverging scale centred
on zero.

Read from the cached prediction bundles written by ``evaluate.py``
(``<model_dir>/pred_cache/cv.npz``), so this costs no inference.

    python scripts/gen_tmax_skill_delta.py --out LATEX_REPORT/images/tmax_skill_delta.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import properscoring as ps

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import metrics  # noqa: E402

MODELS = REPO / "CLEAN_trained_models"
RUNS = {
    "sfc": "baseline__tmax_sfc_flat_y2020-2023_e30f5_b8",
    "atm": "clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8",
}


def skill_of(run: str, regime: str):
    """Per-pixel skill on the model grid, by the definition of ``metrics.py``:
    ``1 - mean_t CRPS_model / mean_t CRPS_ref``, with the deterministic reference
    contributing ``|ref - truth|``. Computed straight from the bundle rather than through
    ``compute_perpixel_metrics``, which expects arrays padded to the full grid --- the
    bundles store valid points only, and padding 1461 days of them costs gigabytes.
    """
    p = MODELS / RUNS[run] / "tmax" / "pred_cache" / f"{regime}.npz"
    if not p.exists():
        raise SystemExit(f"missing prediction bundle: {p}")
    b = np.load(p, allow_pickle=True)
    preds, truths, sigmas, ref = (b["preds_degC"], b["truths_degC"],
                                  b["sigmas_degC"], b["era5_ref_degC"])
    crps = ps.crps_gaussian(truths, mu=preds, sig=sigmas)
    crps_ref = np.abs(ref - truths)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm, cr = np.nanmean(crps, axis=0), np.nanmean(crps_ref, axis=0)
        skill_pt = np.where(cr > 0, 1.0 - cm / cr, np.nan)
    global_skill = metrics.skill_score(float(np.nanmean(crps)), float(np.nanmean(crps_ref)))

    mask = b["valid_mask"]
    grid = np.full(mask.size, np.nan)
    grid[mask.ravel()] = skill_pt
    return grid.reshape(mask.shape), float(global_skill), preds.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="LATEX_REPORT/images/tmax_skill_delta.png")
    ap.add_argument("--regime", default="cv", choices=["cv", "holdout_2024"])
    args = ap.parse_args()

    sk, gs, ndays = {}, {}, None
    for run in RUNS:
        sk[run], gs[run], ndays = skill_of(run, args.regime)
    delta = sk["atm"] - sk["sfc"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    # Skill spans about -0.08 to 0.90 over the domain, so the (-1, 1) range of the
    # per-model maps puts everything in the top third of the colourmap and hides the
    # difference this figure exists to show. This range clips nothing.
    panels = [
        ("sfc", sk["sfc"], "RdYlGn", (-0.1, 0.9), rf"$\mathtt{{sfc}}$ — global {gs['sfc']:.3f}"),
        ("atm", sk["atm"], "RdYlGn", (-0.1, 0.9), rf"$\mathtt{{atm}}$ — global {gs['atm']:.3f}"),
        ("delta", delta, "PuOr_r", (-0.3, 0.3),
         rf"$\mathtt{{atm}}-\mathtt{{sfc}}$ — global {gs['atm'] - gs['sfc']:+.3f}"),
    ]
    for ax, (key, grid, cmap, (lo, hi), title) in zip(axes, panels):
        im = ax.imshow(grid, cmap=cmap, vmin=lo, vmax=hi, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("skill difference" if key == "delta" else "skill vs bilinear ERA5",
                     fontsize=8)
        if key == "delta":
            # Over the valid domain only: NaN > 0 is False, so counting over the full
            # array would silently score every off-grid pixel as negative.
            fin = np.isfinite(grid)
            frac = float((grid[fin] > 0).mean() * 100)
            ax.text(0.02, 0.02, f"{frac:.0f}% of pixels positive", transform=ax.transAxes,
                    fontsize=8, va="bottom", ha="left",
                    bbox=dict(fc="white", ec="#999999", lw=0.5, pad=2.5))

    fig.suptitle(f"Per-pixel skill against bilinear ERA5, CV holdout ({ndays} days)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    med = {k: float(np.nanmedian(v)) for k, v in sk.items()}
    print(f"saved {out}")
    print(f"  global skill  sfc {gs['sfc']:.3f}  atm {gs['atm']:.3f}  delta {gs['atm']-gs['sfc']:+.3f}")
    print(f"  median pixel  sfc {med['sfc']:.3f}  atm {med['atm']:.3f}")
    print(f"  delta: median {np.nanmedian(delta):+.3f}, positive at "
          f"{(delta[np.isfinite(delta)] > 0).mean()*100:.1f}% of valid pixels, range {np.nanmin(delta):+.3f}..{np.nanmax(delta):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
