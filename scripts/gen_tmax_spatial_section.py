#!/usr/bin/env python
"""Spatial comparison of the downscaled temperature field, with a terrain cross-section.

Four maps for one day --- the coarse ERA5 reference the models are given, the surface
and atmospheric predictions, and the MeteoSwiss truth --- on one shared colour scale,
plus a south-to-north section through the Alps along a fixed meridian showing the same
four series against the terrain profile.

Everything is read from the cached prediction bundles written by ``evaluate.py``
(``<model_dir>/pred_cache/holdout_2024.npz``), so this costs no inference.

    python scripts/gen_tmax_spatial_section.py --out LATEX_REPORT/images/tmax_spatial_section.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "CLEAN_trained_models"

# Same labels and colours as every other cross-model temperature figure.
RUNS = {
    "sfc": "baseline__tmax_sfc_flat_y2020-2023_e30f5_b8",
    "atm": "clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8",
}
COLOR = {"sfc": "#eb6834", "atm": "#2a78d6", "truth": "#000000", "ERA5": "#8c8c8c"}

# Meridian for the section: 8 degE runs from Ticino across the main Alpine ridge to the
# Central Plateau, so the profile spans the full relief the downscaling has to resolve.
SECTION_LON = 8.0


def load(run: str) -> dict:
    p = MODELS / RUNS[run] / "tmax" / "pred_cache" / "holdout_2024.npz"
    if not p.exists():
        raise SystemExit(f"missing prediction bundle: {p}\nRun evaluate.py --eval-year 2024 first.")
    return np.load(p, allow_pickle=True)


def to_grid(flat_valid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scatter a (n_valid,) vector back onto the full (240, 370) grid, NaN off-grid."""
    out = np.full(mask.size, np.nan, dtype=float)
    out[mask.ravel()] = flat_valid
    return out.reshape(mask.shape)


def pick_day(bundles: dict, dates: pd.DatetimeIndex) -> int:
    """A day that is typical rather than favourable: the atmospheric model's domain-mean
    error must sit within 0.05 degC of its own annual median, and among those days we take
    the one whose observed field spans the widest range, so the terrain signal is visible.
    """
    err = {k: np.nanmean(np.abs(b["preds_degC"] - b["truths_degC"]), axis=1)
           for k, b in bundles.items()}
    truth = bundles["atm"]["truths_degC"]
    spread = np.nanmax(truth, axis=1) - np.nanmin(truth, axis=1)
    typical = np.where(np.abs(err["atm"] - np.median(err["atm"])) < 0.05)[0]
    return int(typical[np.argmax(spread[typical])])


def pick_warmest(bundles: dict) -> int:
    """The extreme warm day: the highest observed domain-mean daily maximum of the year."""
    return int(np.nanargmax(np.nanmean(bundles["atm"]["truths_degC"], axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="LATEX_REPORT/images/tmax_spatial_section.png")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD; default picks a typical day")
    ap.add_argument("--pick", choices=["typical", "warmest"], default="typical",
                    help="day selection when --date is not given: 'typical' (median-error "
                         "day with the widest observed range) or 'warmest' (highest "
                         "observed domain-mean maximum of the year)")
    args = ap.parse_args()

    bundles = {k: load(k) for k in RUNS}
    ref = bundles["atm"]
    mask = ref["valid_mask"]
    lat = ref["lat"].reshape(mask.shape)
    lon = ref["lon"].reshape(mask.shape)
    elev = to_grid(ref["target_topo"][mask.ravel()][:, 0], mask)
    dates = pd.to_datetime(ref["dates"])

    if not np.array_equal(pd.to_datetime(bundles["sfc"]["dates"]), dates):
        raise SystemExit("the two bundles cover different days; refusing to compare them")

    if args.date:
        d = int(np.where(dates == pd.Timestamp(args.date))[0][0])
    elif args.pick == "warmest":
        d = pick_warmest(bundles)
    else:
        d = pick_day(bundles, dates)
    day = dates[d]

    fields = {
        "ERA5": to_grid(ref["era5_ref_degC"][d], mask),
        "sfc": to_grid(bundles["sfc"]["preds_degC"][d], mask),
        "atm": to_grid(bundles["atm"]["preds_degC"][d], mask),
        "truth": to_grid(ref["truths_degC"][d], mask),
    }
    titles = {
        "ERA5": "ERA5 reference (input resolution)",
        "sfc": r"$\mathtt{sfc}$ prediction",
        "atm": r"$\mathtt{atm}$ prediction",
        "truth": "MeteoSwiss analysis (truth)",
    }
    order = ["ERA5", "sfc", "atm", "truth"]

    stack = np.concatenate([f[np.isfinite(f)] for f in fields.values()])
    vmin, vmax = np.percentile(stack, [0.5, 99.5])

    # Section column: the grid column whose mean longitude is closest to SECTION_LON.
    col = int(np.argmin(np.abs(np.nanmean(lon, axis=0) - SECTION_LON)))
    rows = np.where(np.isfinite(fields["truth"][:, col]))[0]

    fig = plt.figure(figsize=(13.5, 7.4))
    gs = GridSpec(2, 4, height_ratios=[1.0, 0.78], hspace=0.28, wspace=0.08,
                  left=0.04, right=0.98, top=0.90, bottom=0.09)

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    # Degrees of longitude shrink with latitude; aspect="auto" would stretch the
    # country east-west by ~1/cos(47°) ≈ 1.5.
    aspect = 1.0 / np.cos(np.deg2rad(np.nanmean(lat)))
    for i, key in enumerate(order):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(fields[key], origin="lower", extent=extent, aspect=aspect,
                       cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.plot([np.nanmean(lon[:, col])] * 2, [lat[rows[0], col], lat[rows[-1], col]],
                color="k", lw=1.2, ls="--")
        ax.set_title(titles[key], fontsize=10)
        ax.set_xticks([6, 7, 8, 9, 10])
        if i == 0:
            ax.set_ylabel("latitude ($^\\circ$N)", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=8)
        ax.set_xlabel("longitude ($^\\circ$E)", fontsize=9)

    cbar = fig.colorbar(im, ax=[fig.axes[i] for i in range(4)],
                        fraction=0.035, pad=0.012)
    cbar.set_label("daily maximum 2 m temperature ($^\\circ$C)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    axs = fig.add_subplot(gs[1, :])
    y = lat[rows, col]
    axt = axs.twinx()
    axt.fill_between(y, 0, elev[rows, col], color="#d9d9d9", zorder=0)
    axt.set_ylabel("elevation (m)", fontsize=9, color="#6e6e6e")
    axt.set_ylim(0, 4000)
    axt.tick_params(labelsize=8, colors="#6e6e6e")

    for key in order:
        axs.plot(y, fields[key][rows, col], color=COLOR[key], lw=1.8,
                 ls="--" if key == "ERA5" else "-",
                 label=titles[key].replace(" (input resolution)", "").replace(" (truth)", ""),
                 zorder=3)
    axs.set_zorder(axt.get_zorder() + 1)
    axs.patch.set_visible(False)
    axs.set_xlim(y.min(), y.max())
    axs.set_xlabel("latitude ($^\\circ$N) along the $8^\\circ$E section", fontsize=9)
    axs.set_ylabel("temperature ($^\\circ$C)", fontsize=9)
    axs.tick_params(labelsize=8)
    axs.grid(axis="y", color="#ececec", lw=0.6)
    axs.legend(fontsize=8, ncol=4, frameon=False, loc="upper center")
    for side in ("top",):
        axs.spines[side].set_visible(False)

    fig.suptitle(f"Downscaled daily maximum temperature, {day:%-d %B %Y}", fontsize=12)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}  (day {day:%Y-%m-%d}, section at {np.nanmean(lon[:, col]):.2f} degE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
