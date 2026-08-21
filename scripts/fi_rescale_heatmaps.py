#!/usr/bin/env python
"""Re-render every feature-importance heatmap on the fixed per-variable legend scale.

``feature_importance.HEATMAP_VMAX`` holds one hardcoded symmetric colour limit per
target variable, so any two heatmaps of the same variable can be compared by eye. A
limit derived per model — the old behaviour — made the same importance render deep red
in one figure and pale in another. This script applies the fixed limits to the models
that have already been scored, without re-running anything expensive.

It reads the existing ``feature_importance/{pfi,shap,lime}.csv``, so it takes seconds
rather than the hours the study itself took. Every model with a CSV gets a heatmap,
including the surface-input ones: the figure covers the auxiliary channels (scaffold
plus the surface ``data`` channel) as well as pressure levels and surface anchors, so
there is always something to draw.

``--dry-run`` reports the observed maxima against the fixed limits, naming any channel
that saturates, and writes nothing.

Usage:  python scripts/fi_rescale_heatmaps.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

import feature_importance as fi  # noqa: E402

MODELS_ROOT = REPO_ROOT / "CLEAN_trained_models"
METHODS = ("pfi", "shap", "lime")
# Per target variable: the reporting head whose metric naming the CSVs follow.
HEAD_FOR_VARIABLE = {
    "tmax": fi._Head("tmax", ".", fi.METRICS, "mae", fi.LIME_TARGET, "degC"),
    "precip": fi._Head("precip", ".", fi.PRECIP_METRICS, "pr_mae",
                       fi.LIME_TARGET_PRECIP, "mm"),
}


def channel_rows(df: pd.DataFrame, method: str, metric: str) -> pd.DataFrame:
    """The per-channel rows this method/metric contributes to its heatmap."""
    return df[(df["method"] == method) & (df["grouping"] == "channel")
              & (df["metric"] == metric)]


def discover() -> dict[str, list[tuple[str, Path, dict[str, pd.DataFrame]]]]:
    """{variable: [(model name, fi dir, {method: frame}), ...]} for models with CSVs."""
    found: dict[str, list[tuple[str, Path, dict[str, pd.DataFrame]]]] = {}
    for run_dir in sorted(MODELS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        for variable in HEAD_FOR_VARIABLE:
            fi_dir = run_dir / variable / "feature_importance"
            if not fi_dir.is_dir():
                continue
            frames = {m: pd.read_csv(fi_dir / f"{m}.csv")
                      for m in METHODS if (fi_dir / f"{m}.csv").exists()}
            if frames:
                found.setdefault(variable, []).append((run_dir.name, fi_dir, frames))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report maxima against the fixed limits, write nothing")
    args = ap.parse_args()

    written = 0
    for variable, entries in sorted(discover().items()):
        head = HEAD_FOR_VARIABLE[variable]
        vmax = fi.HEATMAP_VMAX[variable]
        note = f" · fixed {variable} scale ±{vmax:g}"
        print(f"\n=== {variable}: fixed scale ±{vmax:g}")
        for method in METHODS:
            metric = head.metric_for(method)
            for name, fi_dir, frames in entries:
                if method not in frames:
                    continue
                rows = channel_rows(frames[method], method, metric)
                if rows.empty:
                    continue
                peak = rows["importance"].abs().max()
                over = rows[rows["importance"].abs() > vmax]
                flag = ("  SATURATES: "
                        + ", ".join(f"{r.feature}={r.importance:+.2f}"
                                    for r in over.itertuples())) if len(over) else ""
                print(f"  {method:5s} {metric:16s} {name[:44]:46s} "
                      f"max {peak:7.3f}{flag}")
                if args.dry_run:
                    continue
                fi._plot_channel_heatmap(frames[method], fi_dir, method, metric,
                                         vmax=vmax, scale_note=note)
                fi._plot_group_importance(frames[method], fi_dir, method, metric, variable)
                written += 1
    print(f"\n{'dry run — nothing written' if args.dry_run else f'rewrote {written} heatmap(s)'}")


if __name__ == "__main__":
    main()
