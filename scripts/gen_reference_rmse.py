#!/usr/bin/env python
"""Pooled MAE and RMSE of the bilinear ERA5-Land tmax reference, per regime.

The comparison CSV recovers the reference MAE from ``crps / (1 - skill)`` — an
identity that exists because a deterministic forecast's CRPS is its MAE. No such
identity exists for RMSE, so the RMSE panel of the regime-slope figure had no
reference line. This computes both, from the cached prediction bundle's own
``era5_ref_degC``/``truths_degC`` arrays — i.e. on exactly the mask and day set
``evaluate.py`` scores, which is what makes the MAE cross-check against the
CSV's ``ref_mae_degC`` a real gate rather than a coincidence.

The reference series is model-independent (verified bit-identical across the
CLEAN runs on CV, equal to 3e-05 °C on 2024 — see the skill-score note in the README),
so one bundle per regime suffices; ``--model-dir`` only picks whose bundle.

Writes/updates ``CLEAN_trained_models/tmax_model_comparison/reference_rmse.json``
with one entry per regime, which ``compare_evaluations.py`` reads into the
``ref_rmse_degC`` column of the comparison CSV.

Usage:
    python scripts/gen_reference_rmse.py --years 2020-2023   # CV regime
    python scripts/gen_reference_rmse.py --years 2024        # holdout year
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = (REPO / "CLEAN_trained_models/"
                 "lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8/tmax")
OUT_JSON = REPO / "CLEAN_trained_models/tmax_model_comparison/reference_rmse.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", required=True, choices=["2020-2023", "2024"],
                    help="'2020-2023' reads the CV bundle, '2024' the holdout bundle")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL,
                    help="whose pred_cache to read (the reference itself is "
                         "model-independent)")
    args = ap.parse_args(argv)

    regime = "cv" if args.years == "2020-2023" else "holdout_2024"
    bundle = args.model_dir / "pred_cache" / f"{regime}.npz"
    if not bundle.exists():
        sys.exit(f"no bundle: {bundle}")
    z = np.load(bundle, allow_pickle=True)
    ref, truth = z["era5_ref_degC"], z["truths_degC"]
    ok = np.isfinite(ref) & np.isfinite(truth)
    err = (ref - truth)[ok]
    result = {
        "regime": "cv" if regime == "cv" else "2024",
        "ref_mae_degC": float(np.mean(np.abs(err))),
        "ref_rmse_degC": float(np.sqrt(np.mean(err ** 2))),
        "ref_bias_degC": float(np.mean(err)),
        "n_point_days": int(ok.sum()),
        "source_bundle": str(bundle.relative_to(REPO)),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    all_regimes = json.load(open(OUT_JSON)) if OUT_JSON.exists() else {}
    all_regimes[result["regime"]] = result
    json.dump(all_regimes, open(OUT_JSON, "w"), indent=2)
    print(f"[{args.years}] reference MAE {result['ref_mae_degC']:.4f} °C | "
          f"RMSE {result['ref_rmse_degC']:.4f} °C | bias "
          f"{result['ref_bias_degC']:+.4f} °C over {result['n_point_days']:,} "
          f"point-days -> {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv=None))
