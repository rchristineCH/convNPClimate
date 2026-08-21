#!/usr/bin/env python3
"""Spatial intensity categories: best atmospheric run and surface run, one figure.

Replaces an earlier crop-and-paste of two independently rendered
precip_category_spatial.png files. Those are generated per model, and each picks
its own per-row frequency scale, so the >=80 mm row ended up on 0.020 for one run
and 0.015 for the other -- the two panels could not be compared by colour in
exactly the row where the runs differ most.

Here both runs are drawn in one figure from the cached per-point arrays, so

  * each category row shares one frequency scale across observed, ERA5-Land and
    both runs, and
  * both runs share the standard discrete Brier-skill scale,

and the rows align by construction rather than by matching two image heights.

Reads only the npz caches written by eval_precip_figures.py; no re-evaluation.
Produces ``CLEAN_trained_models/precip_processing_comparison/category_spatial_pair.png``.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from types import SimpleNamespace

from eval_precip_figures import (_freq_discrete, _nice_step, _bss_discrete,
                                 _bss_for_plot, _skill_range, _bss_ticklabels,
                                 _BSS_LABEL, _grid_map)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CLEAN_trained_models/precip_processing_comparison"
SRC = "precip/eval_figures/precip_category_spatial.npz"
RUNS = [
    ("precip-FT-CRPS", ROOT / "CLEAN_trained_models/clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps"),
    ("precip-SFC-TP", ROOT / "CLEAN_trained_models/clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8"),
]
SHARED_STEP = 0.1


def bundle_shim(run_dir, grid_shape):
    """Minimal stand-in for the prediction bundle that _grid_map needs.

    _grid_map uses only ``grid_shape`` and ``params`` (for the two MeteoSwiss
    globs, from which the exact Swiss silhouette mask is built), so the figure can
    be redrawn from the cached arrays without re-running inference.
    """
    pj = json.loads((run_dir / "precip/params.json").read_text())
    return SimpleNamespace(
        grid_shape=tuple(int(v) for v in grid_shape),
        params=SimpleNamespace(
            METEO_SWISS_PRECIP_GLOB=str(ROOT / pj["METEO_SWISS_PRECIP_GLOB"]),
            METEO_SWISS_MAX_TEMP_GLOB=str(ROOT / pj["METEO_SWISS_MAX_TEMP_GLOB"])))


def main():
    data, dirs = {}, {}
    for name, run in RUNS:
        f = run / SRC
        if not f.exists():
            raise SystemExit(f"missing {f}")
        data[name], dirs[name] = np.load(f, allow_pickle=True), run
    first = data[RUNS[0][0]]
    labels = [str(x) for x in first["cat_labels"]]
    ref = str(first["reference"])
    B = bundle_shim(RUNS[0][1], first["grid_shape"])
    bcmap, bnorm, bbounds = _bss_discrete()
    bticks = _bss_ticklabels(ref)

    C = len(labels)
    cols = 2 + 2 * len(RUNS)
    # Global scale for the busy rows and a finer own scale for the sparse ones --
    # the per-model figure's own rule, with both maxima taken across BOTH runs so
    # the two are comparable to each other as well as across categories.
    fields = ([np.asarray(first[k][:, c], float) for k in ("obs_cat", "era_cat")
               for c in range(C)]
              + [np.asarray(data[n]["model_cat"][:, c], float) for n, _ in RUNS
                 for c in range(C)])
    fmax = max([np.nanmax(a) for a in fields if np.isfinite(a).any()] + [0.1])
    shared = _freq_discrete(fmax, step=SHARED_STEP)

    fig, axes = plt.subplots(C, cols, figsize=(6 * cols, 5 * C), squeeze=False)
    for c in range(C):
        of = np.asarray(first["obs_cat"][:, c], float)
        ef = np.asarray(first["era_cat"][:, c], float)
        preds = {n: np.asarray(data[n]["model_cat"][:, c], float) for n, _ in RUNS}
        row_max = max([np.nanmax(a) for a in [of, ef, *preds.values()]
                       if np.isfinite(a).any()] + [0.0])
        if 0 < row_max <= 2 * SHARED_STEP:
            fcmap, fnorm, fbounds, fticks = _freq_discrete(row_max, step=_nice_step(row_max))
        else:
            fcmap, fnorm, fbounds, fticks = shared
        n_evt = int(first["counts"][c])
        j = 0
        _grid_map(axes[c][j], of, B, f"MeteoSwiss obs freq  (Y in {labels[c]} mm)\n"
                  f"N={n_evt:,} obs events", fcmap, label="", norm=fnorm,
                  boundaries=fbounds, extend="neither", ticks=fticks); j += 1
        _grid_map(axes[c][j], ef, B, f"{ref} freq  (Y in {labels[c]} mm)", fcmap,
                  label="", norm=fnorm, boundaries=fbounds, extend="neither",
                  ticks=fticks); j += 1
        for name, _ in RUNS:
            _grid_map(axes[c][j], preds[name], B,
                      f"{name} pred  (P(Y in {labels[c]} mm))", fcmap, label="",
                      norm=fnorm, boundaries=fbounds, extend="neither",
                      ticks=fticks); j += 1
            bs = np.asarray(data[name]["bss_cat"][:, c], float)
            _grid_map(axes[c][j], _bss_for_plot(bs, np.isfinite(of)), B,
                      f"{name} Brier skill vs {ref}  (Y in {labels[c]} mm)"
                      f"{_skill_range(bs)}", bcmap, label=_BSS_LABEL, norm=bnorm,
                      boundaries=bbounds, extend="min", tick_labels=bticks); j += 1
    # Pooled RPSS is quoted in the caption; the cached per-point array is not the
    # pooled value, so it is deliberately not summarised here.
    fig.suptitle("Spatial precipitation-intensity categories, CV holdout: MeteoSwiss / "
                 f"{ref} / prediction and Brier skill, per run", fontsize=13)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "category_spatial_pair.png", dpi=100)
    plt.close(fig)
    print(f"wrote {OUT / 'category_spatial_pair.png'}")
    for c, lab in enumerate(labels):
        rm = max([np.nanmax(np.asarray(first[k][:, c], float)) for k in ("obs_cat", "era_cat")]
                 + [float(np.nanmax(data[n]["model_cat"][:, c])) for n, _ in RUNS])
        print(f"  row {lab:>8} mm  row_max {rm:.4f} -> "
              + ("own fine scale" if 0 < rm <= 2 * SHARED_STEP else f"shared 0..{fmax:.1f}"))


if __name__ == "__main__":
    main()
