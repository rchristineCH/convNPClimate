#!/usr/bin/env python
"""Combine the grouped feature importances of every CLEAN model into one figure.

``feature_importance._plot_group_importance`` draws the grouped view for a single model;
this is the cross-model version — one bar per model within each group, so the models can
be read against each other directly. Four panels: by variable, by level, by hour, and by
auxiliary channel.

The values are the stored group rows. For PFI that is the whole group permuted at once — a
measurement in its own right, never a sum over the per-channel rows. For LIME nothing is
permuted; its surrogate is linear in the presence mask, so a group value is the sum of its
channels' coefficients. The subtitle names whichever applies.

Either way the grouped level is where the feature-importance argument should rest: with 6
levels x 5 hours of correlated inputs, permuting a single channel leaves its information
reachable through its neighbours, so per-channel numbers systematically understate.

Colour follows the model palette the rest of the cross-model figures use
(``compare_evaluations.MODEL_STYLE`` for tmax, the precip comparison colours for precip),
not the importance colormap — here the reader needs to tell models apart, not magnitudes.
x-limits come from ``feature_importance.GROUP_XMAX``, the same fixed scales the per-model
figures use.

Reads ``feature_importance/{pfi,lime}.csv``; writes to the two cross-model directories.

Usage:  python scripts/fi_group_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

import feature_importance as fi  # noqa: E402
from compare_evaluations import MODEL_STYLE  # noqa: E402

MODELS_ROOT = REPO_ROOT / "CLEAN_trained_models"

# (short label, colour, run directory). tmax colours come from the shared style table so a
# model looks the same here as in every other cross-model figure; precip reuses the
# palette of the precip processing comparison.
TMAX_MODELS = [
    ("atm", MODEL_STYLE["atm"][0], "clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8"),
    ("atm-clip60", MODEL_STYLE["atm-clip60"][0],
     "lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8"),
    ("atm+wind", MODEL_STYLE["atm+wind"][0],
     "clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8"),
    ("atm+wind-clip60", MODEL_STYLE["atm+wind-clip60"][0],
     "lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8"),
    ("atm+sfcanchors", MODEL_STYLE["atm+sfcanchors"][0],
     "sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8"),
    ("sfc", MODEL_STYLE["sfc"][0], "baseline__tmax_sfc_flat_y2020-2023_e30f5_b8"),
    ("sfc-nogeo", MODEL_STYLE["sfc-nogeo"][0],
     "baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo"),
]
PRECIP_MODELS = [
    ("NLL", "#2a78d6", "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8"),
    ("CRPS", "#eb6834",
     "clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8"),
    ("WIND", "#1baf7a",
     "clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8"),
    ("FT-CRPS", "#8e5bd0",
     "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps"),
]
VARIABLES = {
    "tmax": (TMAX_MODELS, {"pfi": "mae", "lime": "pred_tmax_degC"},
             MODELS_ROOT / "tmax_model_comparison"),
    "precip": (PRECIP_MODELS, {"pfi": "pr_mae", "lime": "pred_precip_mm"},
               MODELS_ROOT / "precip_processing_comparison"),
}
PREFERRED_VARS = ("z", "t", "q", "u", "v", "t2m", "tp", "data", "scaffold")


def read_model(run: str, variable: str, method: str, metric: str,
               fi_dirname: str = "feature_importance"):
    """(group values, auxiliary channel values) for one model, or None if unusable."""
    csv = MODELS_ROOT / run / variable / fi_dirname / f"{method}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    groups = fi._group_values(df, method, metric)
    chan = df[(df["method"] == method) & (df["grouping"] == "channel")
              & (df["metric"] == metric)]
    if chan.empty or not groups["variable"]:
        return None
    parsed = {c: fi._parse_channel(c) for c in chan["feature"]}
    # A CSV predating the channel-parser fix measured t2m/tp inside 'variable=t', but
    # filed them under levels '2m'/'p' — which is their own group permutation by another
    # name. Read those back so the anchors appear independently, and mark the 't' row so
    # it is not compared against a clean atmospheric t.
    anchors, folded = fi._anchor_groups(parsed, groups)
    for name, value in anchors.items():
        groups["variable"][name] = value
    if anchors:
        # One 'surface' level: at this level what matters is that they are not on a
        # pressure surface, not which of them they are. It is the two groups added, not a
        # single permutation of both.
        groups["level"]["surface"] = sum(anchors.values())
    if folded:
        label = folded[0][0]
        joint = f"{'+'.join([label] + folded)} (joint)"
        groups["variable"][joint] = groups["variable"].pop(label)
        print(f"  {run}: {joint} was permuted as one group; "
              f"{', '.join(anchors)} recovered from their level rows")
    aux = {c: float(v) for c, v in zip(chan["feature"], chan["importance"])
           if parsed[c]["var"] in ("scaffold", "data")}
    return groups, aux


def panel_labels(name: str, per_model: list) -> list[str]:
    """Row labels for one panel: the union over models, in a stable order."""
    seen: set[str] = set()
    for groups, aux in per_model:
        seen |= set(aux if name == "auxiliary" else groups[name])
    if name == "variable":
        return ([v for v in PREFERRED_VARS if v in seen]
                + sorted(v for v in seen if v not in PREFERRED_VARS))
    if name == "level":
        return (sorted((lv for lv in seen if lv.isdigit()), key=int, reverse=True)
                + (["surface"] if "surface" in seen else []))
    if name == "auxiliary":
        return ([c for c in fi.AUX_ORDER if c in seen]
                + sorted(c for c in seen if c not in fi.AUX_ORDER))
    return sorted(seen)


def build(variable: str, method: str, year: int | None = None) -> None:
    """Cross-model grouped importance. ``year`` selects the holdout FI run
    (``feature_importance_<year>``); ``None`` is the CV run."""
    models, metrics, out_dir = VARIABLES[variable]
    fi_dirname = "feature_importance" if year is None else f"feature_importance_{year}"
    suffix = "" if year is None else f"_{year}"
    metric = metrics[method]
    loaded = []
    for label, color, run in models:
        got = read_model(run, variable, method, metric, fi_dirname)
        if got is not None:
            loaded.append((label, color, *got))
    if not loaded:
        print(f"{variable}/{method}: nothing to plot")
        return

    per_model = [(g, a) for _, _, g, a in loaded]
    panels = [(name, panel_labels(name, per_model)) for name in fi.GROUP_PANELS]
    panels = [(n, labs) for n, labs in panels if labs]
    xmax = fi.GROUP_XMAX.get((variable, method), {})

    rows = max(len(labs) for _, labs in panels)
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(3.4 * len(panels), 0.42 * rows + 1.9))
    height = 0.8 / len(loaded)
    for ax, (name, labels) in zip(axes[0], panels):
        base = np.arange(len(labels))
        lim = xmax.get(name)
        for k, (label, color, groups, aux) in enumerate(loaded):
            src = aux if name == "auxiliary" else groups[name]
            values = [float(src.get(lab, np.nan)) for lab in labels]
            offset = (k - (len(loaded) - 1) / 2) * height
            ax.barh(base + offset, values, height=height * 0.9, color=color,
                    edgecolor="none", label=label if name == panels[0][0] else None)
            # A bar past the fixed axis is otherwise silently clipped, which would read
            # as "at the limit" rather than several times beyond it.
            if lim:
                for y, v in zip(base + offset, values):
                    if np.isfinite(v) and abs(v) > lim:
                        ax.text(lim * 0.985, y, f"{v:.2f}", ha="right", va="center",
                                fontsize=6, color="white", fontweight="bold")
        if lim:
            ax.set_xlim(0, lim)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_yticks(base, labels, fontsize=8)
        ax.set_title(f"by {name}", fontsize=10)
        ax.grid(axis="x", color="#e9e9e9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.legend(*axes[0][0].get_legend_handles_labels(), fontsize=8, frameon=False,
               loc="lower center", ncol=min(len(loaded), 7), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{method.upper()} grouped importance across {len(loaded)} {variable} "
                 f"models — {metric} · {fi._group_subtitle(method)}", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fi_group_importance_{method}{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}  ({len(loaded)} models)")


def main() -> None:
    # CV always; the 2024 holdout wherever that FI run exists. Missing runs are
    # skipped by build() (no CSVs -> nothing loaded), so this stays a no-op for
    # variables whose holdout FI has not been produced yet.
    for variable in VARIABLES:
        for method in ("pfi", "lime"):
            for year in (None, 2024):
                tag = "CV" if year is None else str(year)
                print(f"{variable}/{method} [{tag}]:")
                build(variable, method, year)


if __name__ == "__main__":
    main()
