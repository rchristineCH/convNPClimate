#!/usr/bin/env python
"""Put the PFI year studies side by side: does the evaluation year change the answer?

Reads every ``feature_importance[_<year>]/`` dir a model has and tabulates the
variable-level importances across years, both in absolute mm and normalised by that
year's baseline MAE.

Why normalise. PFI importance is reported in MAE units, so a wetter, harder year
inflates every channel together without the model's reliance having changed at all.
2024 has a baseline MAE ~46% above 2022 for the atmospheric runs, which on its own
moves the absolute numbers by about that much. The share-of-baseline column is what
should be compared across years; the absolute column is kept because it is what the
per-model summaries quote.

CAVEAT, printed with the tables: 2020-2023 are IN-SAMPLE (the models trained on
2020-2023, and every fold of the 5-fold ensemble saw ~80% of each of those years).
2024 is the only genuine holdout, so an apparent year effect can be an in-sample vs
holdout effect instead.

Usage:  python scripts/fi_year_comparison.py
        python scripts/fi_year_comparison.py --out CLEAN_trained_models/precip_processing_comparison
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "SFC-TP": "clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8",
    "WIND": "clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8",
}
DEFAULT_YEAR = 2022          # the unsuffixed feature_importance/ dir


def _studies(model_dir: Path) -> dict[int, Path]:
    """{year: fi_dir} for every PFI study present, including the unsuffixed one."""
    out = {}
    base = model_dir / "feature_importance"
    if (base / "pfi.csv").exists():
        out[DEFAULT_YEAR] = base
    for d in sorted(model_dir.glob("feature_importance_*")):
        m = re.fullmatch(r"feature_importance_(\d{4})", d.name)
        if m and (d / "pfi.csv").exists():
            out[int(m.group(1))] = d
    return dict(sorted(out.items()))


def _variable_table(fi_dir: Path) -> tuple[pd.Series, float]:
    df = pd.read_csv(fi_dir / "pfi.csv")
    sel = df[(df.metric == "pr_mae") & (df.grouping == "variable")]
    s = sel.set_index(sel.feature.str.replace("variable=", "", regex=False))["importance"]
    base = json.loads((fi_dir / "baseline.json").read_text())["pr_mae"]
    return s, float(base)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="Also write a markdown summary here.")
    args = ap.parse_args()

    lines = ["# PFI across evaluation years", ""]
    lines.append("Variable-level permutation importance (metric `pr_mae`), fold-ensembled.")
    lines.append("")
    lines.append("**Caveat:** 2020-2023 are in-sample (training span 2020-2023; every fold of")
    lines.append("the ensemble saw ~80% of each year). 2024 is the only genuine holdout, so a")
    lines.append("year-to-year difference may be an in-sample vs holdout difference.")
    lines.append("")
    lines.append("Compare the **share** columns across years: PFI is in MAE units, so a harder")
    lines.append("year inflates every channel together without reliance having changed.")
    lines.append("")

    for name, rel in MODELS.items():
        mdir = ROOT / "CLEAN_trained_models" / rel / "precip"
        studies = _studies(mdir)
        if not studies:
            print(f"{name}: no PFI studies found"); continue
        cols_abs, cols_share, bases = {}, {}, {}
        for year, d in studies.items():
            s, base = _variable_table(d)
            cols_abs[year] = s
            cols_share[year] = 100.0 * s / base
            bases[year] = base
        A = pd.DataFrame(cols_abs).sort_values(max(cols_abs), ascending=False)
        S = pd.DataFrame(cols_share).reindex(A.index)

        hdr = f"## {name}   ({len(studies)} years: {', '.join(map(str, studies))})"
        print("\n" + hdr)
        lines.append(hdr); lines.append("")
        brow = "baseline MAE (mm) | " + " | ".join(f"{bases[y]:.3f}" for y in A.columns)
        print("  " + brow)
        lines.append("| variable | " + " | ".join(f"{y} abs" for y in A.columns) +
                     " | " + " | ".join(f"{y} share%" for y in S.columns) + " |")
        lines.append("|---" * (1 + 2 * len(A.columns)) + "|")
        lines.append("| **baseline MAE (mm)** | " +
                     " | ".join(f"{bases[y]:.3f}" for y in A.columns) +
                     " | " + " | ".join("—" for _ in S.columns) + " |")
        print(f"  {'variable':10}" + "".join(f"{y:>10}" for y in A.columns)
              + "   |" + "".join(f"{y:>9}%" for y in S.columns))
        for v in A.index:
            print(f"  {v:10}" + "".join(f"{A.loc[v, y]:>10.2f}" for y in A.columns)
                  + "   |" + "".join(f"{S.loc[v, y]:>9.0f}%" for y in S.columns))
            lines.append(f"| {v} | " + " | ".join(f"{A.loc[v, y]:.2f}" for y in A.columns)
                         + " | " + " | ".join(f"{S.loc[v, y]:.0f}%" for y in S.columns) + " |")
        # Stability: spread of each variable's share across years.
        spread = (S.max(axis=1) - S.min(axis=1)).sort_values(ascending=False)
        top = spread.index[0]
        note = (f"share spread across years: max {spread.iloc[0]:.0f} pp on `{top}` "
                f"(range {S.loc[top].min():.0f}-{S.loc[top].max():.0f}%)")
        print("  " + note)
        lines.append(""); lines.append(note); lines.append("")

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "fi_year_comparison.md").write_text("\n".join(lines) + "\n")
        print(f"\nwrote {out / 'fi_year_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
