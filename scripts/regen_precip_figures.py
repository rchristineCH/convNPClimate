#!/usr/bin/env python3
"""Regenerate the COMPLETE precip evaluation figure set for BOTH models.

Runbook:

    cd ~/work/convNPClimate
    python scripts/regen_precip_figures.py            # all models
    python scripts/regen_precip_figures.py FT-CRPS    # only the named model(s)

Do NOT override PYTHONPATH (the image's PYTHONPATH provides torch).

Thin orchestrator: per model {NLL baseline, CRPS, WIND, FT-CRPS} x regime {CV
holdout 2020-2023, 2024 holdout} it delegates to ``eval_precip_figures.main``, which
renders the full diagnostic set (see its module docstring: spatial/density/QQ/
PIT/boxplot/per-fold figures plus the threshold and intensity-category figures
with ERA5-Land-referenced Brier skill and RPSS, and
``precip_category_pooled.json``) into ``<model>/precip/eval_figures[_2024]/``.

Predictions come from ``eval_precip.load_or_predict``: the first pass per
(model, regime) runs GPU inference (~7 min) and persists the bundle under
``<model>/precip/pred_cache/`` (gitignored); warm passes are plot-only
(minutes). The threshold/category figures also persist as ``<name>.fig.pkl``
(local only, gitignored) + ``<name>.npz`` (committed editable arrays).

MeteoSwiss RhiresD is ground truth throughout; ERA5-Land (bilinear) is the
skill reference and degrades gracefully to climatology where tp-<year>.nc is
missing.

Afterwards commit the four eval_figures[_2024] dirs and push.
"""
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # relative datasets/ globs must resolve from the repo root

import eval_precip_figures as epf     # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)

MODELS = {
    "NLL": ROOT / "CLEAN_trained_models/clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8/precip",
    "CRPS": ROOT / "CLEAN_trained_models/clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8/precip",
    "WIND": ROOT / "CLEAN_trained_models/clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8/precip",
    "FT-CRPS": ROOT / "CLEAN_trained_models/clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps/precip",
    # Surface inputs (ERA5-Land t2m_max + tp) rather than pressure levels. Its day
    # axis is 1460 days, not 1461: the tp series ends 2023-12-30.
    "SFC-TP": ROOT / "CLEAN_trained_models/clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8/precip",
}

# Models whose holdout year cannot be produced from the data on disk. The mechanism
# is kept because it is the difference between "no figure" and a plausible-looking
# figure built on a corrupt input, which is far worse.
#
# SFC-TP was listed here while the only tp-2024.nc on disk was the day-shifted,
# ~67%-inflated one (see tp_hourly/README.md) -- it reads tp as a model INPUT,
# so that file would have been fed to the encoder. It is unblocked now:
# scripts/reeval_precip_w0606.sh scores its 2024 leg against the rebuilt w0606
# reference and drives it with the rebuilt next00 input via --precip-glob, and the
# resulting bundle is what these figures are served from.
NO_HOLDOUT_YEAR: dict[str, str] = {}


def main(argv=None) -> int:
    """Regenerate figures for every model, or only the named ones.

    ``python scripts/regen_precip_figures.py FT-CRPS`` restricts the run to that
    model -- useful because a model whose ``pred_cache`` is cold costs a full GPU
    inference pass per regime, so re-running the warm ones is pure waste.
    """
    names = list(argv if argv is not None else sys.argv[1:])
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"unknown model name(s) {unknown}; known: {list(MODELS)}", flush=True)
        return 2
    selected = {n: MODELS[n] for n in names} if names else MODELS

    rpss_summary = {}
    n_fail = 0
    for name, mdir in selected.items():
        for year in (None, 2024):
            if year is not None and name in NO_HOLDOUT_YEAR:
                print(f"=== {name} {year} SKIPPED: {NO_HOLDOUT_YEAR[name]} ===", flush=True)
                continue
            tag = "CV" if year is None else str(year)
            out = mdir / ("eval_figures" if year is None else f"eval_figures_{year}")
            print(f"=== {name} {tag} -> {out} ===", flush=True)
            argv = ["--model-dir", str(mdir), "--fig-label", f"{name} {tag}"]
            if year is not None:
                argv += ["--eval-year", str(year)]
            rc = epf.main(argv)
            if rc != 0:
                n_fail += 1
                print(f"### FAIL {name} {tag} (some figures did not render)", flush=True)
            pooled = out / "precip_category_pooled.json"
            if pooled.exists():
                rpss_summary[f"{name}_{tag}"] = json.loads(pooled.read_text())["rpss"]
            print(f"### DONE {name} {tag}", flush=True)
    print("### RPSS SUMMARY " + json.dumps(rpss_summary), flush=True)
    print("### ALL COMBINED DONE", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
