#!/usr/bin/env python
"""Combine the per-model altitude-band MAE into one figure per evaluation regime.

``error_analysis.fig_regions`` draws MAE by altitude band for a single tmax model. Read
across seven separate figures, the comparison a reader actually wants — does any model
close the gap in the Alps? — is impossible to make. This puts all of them on one axis, one
bar per model within each band, with the same standard error the per-model figure shows.

Reads ``error_analysis_{cv,2024}/error_analysis_summary.json``, so it costs nothing: the
bands are already computed there. Colour is the shared model palette
(``compare_evaluations.MODEL_STYLE``), so a model looks the same here as in every other
cross-model figure.

Precip has no counterpart — the altitude bands come from the tmax error analysis.

Usage:  python scripts/region_bands_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from compare_evaluations import MODEL_STYLE  # noqa: E402

MODELS_ROOT = REPO_ROOT / "CLEAN_trained_models"
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
REGIMES = [("error_analysis_cv", "CV 2020-2023", ""),
           ("error_analysis_2024", "2024 holdout", "_2024")]


def build(subdir: str, title: str, suffix: str) -> None:
    loaded = []
    for label, run in MODELS:
        path = MODELS_ROOT / run / "tmax" / subdir / "error_analysis_summary.json"
        if not path.exists():
            print(f"  missing {path.relative_to(MODELS_ROOT)}")
            continue
        loaded.append((label, json.loads(path.read_text())["bands"]))
    if not loaded:
        print(f"{subdir}: nothing to plot")
        return

    names = [b["name"] for b in loaded[0][1]]
    # Every model must share the band edges or the bars would not be comparable.
    for label, bands in loaded:
        if [b["name"] for b in bands] != names:
            raise SystemExit(f"{label} has different altitude bands: "
                             f"{[b['name'] for b in bands]} != {names}")

    x = np.arange(len(names))
    width = 0.8 / len(loaded)
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, (label, bands) in enumerate(loaded):
        offset = (k - (len(loaded) - 1) / 2) * width
        mae = [b["mae"] for b in bands]
        ax.bar(x + offset, mae, width=width * 0.9,
               color=MODEL_STYLE[label][0], edgecolor="none", label=label,
               yerr=[b["se"] for b in bands], capsize=2,
               error_kw={"lw": 0.7, "ecolor": "#555555"})
        for xi, v in zip(x + offset, mae):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6,
                    rotation=90, color="#333333")

    # The band populations are a property of the grid, not of any model, so they are
    # stated once on the axis rather than repeated per bar as the per-model figure does.
    # Headroom for the rotated value labels, which sit on top of the tallest bars.
    peak = max(b["mae"] + b["se"] for _, bands in loaded for b in bands)
    ax.set_ylim(0, peak * 1.16)
    counts = [b["n"] for b in loaded[0][1]]
    ax.set_xticks(x, [f"{nm}\n(n={n:,})" for nm, n in zip(names, counts)])
    ax.set_ylabel("mean per-grid-point MAE (°C)")
    ax.set_title(f"MAE by altitude band — {len(loaded)} tmax models, {title}")
    ax.grid(axis="y", color="#e9e9e9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=len(loaded), loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"region_bands_mae{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}  ({len(loaded)} models)")


def main() -> None:
    for subdir, title, suffix in REGIMES:
        build(subdir, title, suffix)


if __name__ == "__main__":
    main()
