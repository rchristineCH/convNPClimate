#!/usr/bin/env python
"""Collect every result PNG from CLEAN_trained_models into one flat folder.

Each file is copied (not moved) and renamed to

    <variable>__<original name>__<evaluation mode>__<model>.png

so a figure can be picked out without walking the per-model tree, and the folder
sorts into a tmax block and a precip block. Cross-model comparison figures keep the
same scheme with the model field set to ALL. The convention is written out in full
in the results folder's README.md.

The figures for the report live in the `doc` worktree:

    python scripts/collect_result_figures.py \
        --out /home/marc/convNPClimate-doc/LATEX_REPORT/images/results \
        --index RESULT_FIGURES_INDEX.md

`--index` writes a markdown table mapping every collected file back to the
path it came from.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO / "CLEAN_trained_models"

# Short, stable label per trained model. Keys are the run directory names.
MODEL_LABELS = {
    "baseline__tmax_sfc_flat_y2020-2023_e30f5_b8": "baseline",
    "baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo": "baseline_no_geo",
    "clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8": "clean_solo",
    "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8": "clean_solo_precip",
    "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps": "clean_solo_precip_ftcrps",
    "clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8": "clean_solo_precip_crps",
    "clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8": "clean_solo_precip_sfctp",
    "clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8": "clean_solo_precip_wind",
    "clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8": "clean_solo_wind",
    "lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8": "lbclip_atm",
    "lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8": "lbclip_wind",
    "sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8": "sfc_anchors",
}

# Source subdirectory -> evaluation-mode token. precip writes its evaluation
# figures to eval_figures*, tmax to eval_*; both are folded onto the same
# cv / 2024 tokens so the two variables read alike.
MODE_LABELS = {
    "eval_cv": "eval_cv",
    "eval_2024": "eval_2024",
    "eval_figures": "eval_cv",
    "eval_figures_2024": "eval_2024",
    "error_analysis_cv": "error_analysis_cv",
    "error_analysis_2024": "error_analysis_2024",
    "station_analysis_cv": "station_analysis_cv",
    "station_analysis_2024": "station_analysis_2024",
    "feature_importance": "feature_importance",
    "feature_importance_2024": "feature_importance_2024",
}
# PNGs sitting directly in the model directory (training curves, architecture).
ROOT_MODE = "training"

# One precip run (clean_solo_precip_crps) still carries an eval_2024/ directory
# from an earlier, partial eval_precip run; eval_figures_2024/ superseded it two
# minutes later. For precip the eval_figures* dirs are the canonical output, so
# the tmax-style names are ignored there.
LEGACY_FOR_PRECIP = {"eval_cv", "eval_2024"}

# (variable, figure stem) pairs whose per-model figure a cross-model figure replaces. The
# per-model original stays in the run directory; it just does not go to the report, where
# seven separate copies of the same axes cannot be compared.
#   region_bands -> tmax_model_comparison/region_bands_mae{,_2024}.png
SUPERSEDED = {("tmax", "region_bands")}

# Cross-model figure directories directly under CLEAN_trained_models -> variable.
COMPARISON_DIRS = {
    "tmax_model_comparison": "tmax",
    "precip_processing_comparison": "precip",
}

# The combined precip figures were briefly collected under a `caution_` prefix
# while their skill axis still came from the defective tp reference. They are now
# regenerated from the corrected, RhiresD-aligned field, so the marker is gone and
# they collect under the same scheme as everything else.


def collect(out_dir: Path) -> list[tuple[Path, str]]:
    """Return (source path, target file name) for every result PNG."""
    jobs: list[tuple[Path, str]] = []
    skipped: list[Path] = []
    legacy: list[Path] = []
    superseded: list[Path] = []

    for run_dir in sorted(MODELS_ROOT.iterdir()):
        if not run_dir.is_dir() or run_dir.name in COMPARISON_DIRS:
            continue
        label = MODEL_LABELS.get(run_dir.name)
        if label is None:
            raise SystemExit(
                f"unknown model directory {run_dir.name!r} — add it to MODEL_LABELS"
            )
        for png in sorted(run_dir.rglob("*.png")):
            rel = png.relative_to(run_dir)
            # rel is <variable>/[<group>/]<file>.png
            variable = rel.parts[0]
            if len(rel.parts) == 2:
                mode = ROOT_MODE
            elif len(rel.parts) == 3:
                if variable == "precip" and rel.parts[1] in LEGACY_FOR_PRECIP:
                    legacy.append(png)
                    continue
                mode = MODE_LABELS.get(rel.parts[1])
                if mode is None:
                    skipped.append(png)
                    continue
            else:
                skipped.append(png)
                continue
            if (variable, png.stem) in SUPERSEDED:
                superseded.append(png)
                continue
            jobs.append((png, f"{variable}__{png.stem}__{mode}__{label}.png"))

    for name, variable in COMPARISON_DIRS.items():
        comp_dir = MODELS_ROOT / name
        if not comp_dir.is_dir():
            continue
        for png in sorted(comp_dir.glob("*.png")):
            jobs.append((png, f"{variable}__{png.stem}__{name}__ALL.png"))

    if legacy:
        print(f"ignored {len(legacy)} superseded precip PNG(s):")
        for png in legacy:
            print(f"  {png.relative_to(MODELS_ROOT)}")

    if superseded:
        print(f"left out {len(superseded)} per-model PNG(s) a cross-model figure replaces:")
        for png in superseded:
            print(f"  {png.relative_to(MODELS_ROOT)}")

    if skipped:
        print(f"skipped {len(skipped)} PNG(s) in unrecognised locations:")
        for png in skipped:
            print(f"  {png.relative_to(MODELS_ROOT)}")

    clashes = {}
    for src, target in jobs:
        clashes.setdefault(target, []).append(src)
    for target, sources in clashes.items():
        if len(sources) > 1:
            raise SystemExit(
                f"name clash for {target}:\n  " + "\n  ".join(map(str, sources))
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop PNGs left over from an earlier collection (a renamed convention, or a
    # figure since deleted upstream) so the folder never mixes two schemes. Only
    # PNGs are touched — README.md and anything else in the folder stays.
    wanted = {target for _, target in jobs}
    stale = [p for p in out_dir.glob("*.png") if p.name not in wanted]
    for path in stale:
        path.unlink()
    if stale:
        print(f"removed {len(stale)} stale PNG(s) from {out_dir}")

    for src, target in jobs:
        shutil.copy2(src, out_dir / target)
    return jobs


def write_index(jobs: list[tuple[Path, str]], out_dir: Path, index_path: Path) -> None:
    """Write a markdown table: collected file name -> where it came from."""
    rows = sorted(jobs, key=lambda j: (j[1].split("__")[0], j[1].split("__")[3], j[1]))
    lines = [
        "# Result figure index",
        "",
        f"{len(rows)} figures collected into `{out_dir}` by",
        "`scripts/collect_result_figures.py`. Names follow",
        "`<variable>__<original name>__<evaluation mode>__<model>.png`; the source paths",
        "below are relative to `CLEAN_trained_models/`.",
        "",
        "| Collected file | Variable | Model | Evaluation mode | Source |",
        "|---|---|---|---|---|",
    ]
    for src, target in rows:
        variable, _, mode, model = target[: -len(".png")].split("__")
        rel = src.relative_to(MODELS_ROOT)
        lines.append(f"| `{target}` | {variable} | {model} | {mode} | `{rel}` |")
    lines.append("")
    index_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=REPO / "RESULT_FIGURES",
        type=Path,
        help="output directory (default: RESULT_FIGURES/)",
    )
    ap.add_argument(
        "--index",
        type=Path,
        help="write a markdown name -> source mapping to this path",
    )
    args = ap.parse_args()
    jobs = collect(args.out)
    print(f"copied {len(jobs)} figures to {args.out}")
    if args.index:
        write_index(jobs, args.out, args.index)
        print(f"wrote index for {len(jobs)} figures to {args.index}")


if __name__ == "__main__":
    main()
