#!/usr/bin/env python
"""Rank and linear correlation of the bilinear ERA5-Land precipitation baseline.

``eval_precip`` records the baseline's MAE, bias, R01 and SDII but not its correlation with
the observations, so the Spearman panel of the model comparison has no reference line. The
baseline correlation involves no model term --- only the interpolated coarse field and the
MeteoSwiss truth --- so it can be computed without running inference, which matters because
no precipitation prediction bundles exist and re-running the evaluation would mean a full
inference pass for every run and regime.

The mask, the day set and the estimator are those of ``eval_precip.compute_baseline_skill``:
the shared finite mask, and ``_pooled_correlations`` with its fixed subsample seed, so the
result is directly comparable to the ``spearman_pooled`` the models report.

**Self-test.** The script prints its own baseline MAE next to the value already stored in
``eval_precip_metrics*.json``. They must agree; if they do not, the mask or day set differs
and the correlation is not on the same footing, so the run aborts rather than emit a number
that would silently mismatch the figure.

    python scripts/gen_precip_baseline_corr.py            # both regimes
    python scripts/gen_precip_baseline_corr.py --year 2024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import datasets as ds          # noqa: E402
import params as params_mod    # noqa: E402
import precip_baseline         # noqa: E402
from eval_precip import _pooled_correlations, WET_THRESHOLD_MM  # noqa: E402

MODELS = REPO / "CLEAN_trained_models"
# Any precipitation run serves: the baseline and the truth do not depend on the model, and
# every run shares the same target grid. This one is the reference NLL configuration.
REF_RUN = "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8"

# Agreement required between the recomputed baseline MAE and the stored one, in mm. The two
# should be bit-comparable; this leaves room only for float summation order.
MAE_TOL = 1e-6


def stored(run: str, year: int | None) -> dict:
    suffix = "" if year is None else f"_{year}"
    p = MODELS / run / "precip" / f"eval_precip_metrics{suffix}.json"
    return json.loads(p.read_text())["baseline"]


def build_truth(year: int | None):
    """Target coordinates, the MeteoSwiss truth and the day axis, without any model."""
    import train

    p = params_mod.Params.load_json(MODELS / REF_RUN / "precip" / "params.json")
    p.DEVICE = "cpu"
    if year is not None:
        p.DATA_YEAR_START = p.DATA_YEAR_END = year
    data = train.DataBundle(SimpleNamespace(base_params=p, device="cpu"))

    # data_var / convert_to_kelvin / normalize_targets live on the variable spec,
    # exactly as eval_precip reads them; the likelihood carries none of them.
    spec = train.VARIABLE_SPECS[p.VARIABLE]
    target_x, target_y, _topo = ds.prepare_meteoswiss_targets(
        getattr(p, spec.meteoswiss_glob_attr),
        normalization_stats=data.era5_metadata,
        data_var=spec.data_var,
        convert_to_kelvin=spec.convert_to_kelvin,
        normalize_targets=spec.normalize_targets,
        year_start=p.DATA_YEAR_START, year_end=p.DATA_YEAR_END,
        device="cpu",
    )
    target_y = ds.align_target_to_days(target_y, data.time_coords, "baseline correlation")
    m = data.era5_metadata
    lat = np.asarray(target_x.sel(coord="lat").values, float) * (m.lat_max - m.lat_min) + m.lat_min
    lon = np.asarray(target_x.sel(coord="lon").values, float) * (m.lon_max - m.lon_min) + m.lon_min
    return lat, lon, np.asarray(target_y.values, np.float32), np.asarray(data.time_coords)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=None,
                    help="holdout year; omit for the CV span")
    ap.add_argument("--baseline-glob", default="tp-202[0-4].nc")
    ap.add_argument("--out", default="CLEAN_trained_models/precip_processing_comparison/"
                                     "baseline_correlations.json")
    args = ap.parse_args()

    regimes = [args.year] if args.year is not None else [None, 2024]
    out: dict[str, dict] = {}
    for year in regimes:
        tag = "cv" if year is None else str(year)
        lat, lon, truth, dates = build_truth(year)
        base_year = year if year is not None else int(
            params_mod.Params.load_json(MODELS / REF_RUN / "precip" / "params.json").DATA_YEAR_START)
        base, source = precip_baseline.bilinear_era5_precip(
            lat, lon, dates, year=base_year, glob=args.baseline_glob)
        if base is None:
            print(f"{tag}: no baseline field; skipped")
            continue

        mask = np.isfinite(truth) & np.isfinite(base)
        o, b = truth[mask], base[mask]
        mae = float(np.mean(np.abs(b - o)))
        ref = stored(REF_RUN, year)
        delta = abs(mae - ref["baseline_mae_mm"])
        print(f"--- {tag} --- n={mask.sum():,}")
        print(f"  baseline MAE recomputed {mae:.6f}  stored {ref['baseline_mae_mm']:.6f}  "
              f"delta {delta:.2e}")
        if delta > MAE_TOL:
            print("  ABORT: mask or day set differs from the evaluator; correlation not "
                  "comparable.", file=sys.stderr)
            return 1

        sp, pr = _pooled_correlations(o, b)
        wet = b >= WET_THRESHOLD_MM
        r10 = float(np.mean(b > 10.0))
        p98 = float(np.percentile(b[wet], 98)) if wet.any() else float("nan")
        print(f"  baseline spearman {sp:.4f}  pearson {pr:.4f}  R10 {r10:.4f}  P98 {p98:.3f}")
        out[tag] = {"source": source, "n_obs": int(mask.sum()),
                    "baseline_mae_mm": mae,
                    "baseline_spearman_pooled": sp, "baseline_pearson_pooled": pr,
                    "baseline_wetday_freq": float(np.mean(wet)),
                    "baseline_R10_freq": r10, "baseline_P98_mm": p98}

    if out:
        p = Path(args.out)
        if not p.is_absolute():
            p = REPO / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
