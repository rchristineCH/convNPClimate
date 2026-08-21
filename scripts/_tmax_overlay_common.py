"""Shared pieces of the cross-model tmax overlay figures.

Model definitions, styling and the pooled-distribution maths used by
``overlay_tmax_distributions.py`` (3-panel diagnostic) and
``overlay_tmax_amount_distribution.py`` (single-panel report figure).

Reads the gitignored prediction bundles at <model>/tmax/pred_cache/{cv,holdout_2024}.npz
-- populate them with `python evaluate.py --model-dir <model>/tmax [--eval-year 2024]`.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import evaluate as ev  # noqa: E402

from compare_evaluations import MODEL_STYLE  # noqa: E402  (shared colours/linestyles)


def M(label, descriptor, trial):
    """(name, descriptor, color, linestyle, trial), styled from the shared table."""
    color, ls, _hatch = MODEL_STYLE[label]
    return (label, descriptor, color, ls, trial)


# Hue = input family, linestyle = the variant within it — defined once in
# compare_evaluations.MODEL_STYLE so a model looks identical in every report figure.
ATM = M("atm", "z,t,q",
        "clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8")
ATM_CLIP = M("atm-clip60", "z,t,q | 60ep + grad clip",
             "lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8")
WIND = M("atm+wind", "z,t,q,u,v",
         "clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8")
WIND_CLIP = M("atm+wind-clip60", "z,t,q,u,v | 60ep + grad clip",
              "lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8")
ANCHORS = M("atm+sfcanchors", "z,t,q + t2m/tp anchors",
            "sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8")
SFC = M("sfc", "surface",
        "baseline__tmax_sfc_flat_y2020-2023_e30f5_b8")
SFC_NOGEO = M("sfc-nogeo", "surface, no geopotential",
              "baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo")

# Every CLEAN tmax model, in the order the report presents them.
ALL_MODELS = [ATM, ATM_CLIP, WIND, WIND_CLIP, ANCHORS, SFC, SFC_NOGEO]

OBS_INK = "#333333"
REGIMES = [(None, "CV 2020-2023", ""), (2024, "2024 holdout", "_2024")]
OUT = ROOT / "CLEAN_trained_models" / "tmax_model_comparison"
BIN_WIDTH = 0.5          # degC
TARGET_PAIRS = 3_000_000  # (day, point) pairs kept for the mixture curves
SQRT2PI = float(np.sqrt(2.0 * np.pi))


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", color="#dddddd", lw=0.6, zorder=0)
    ax.tick_params(colors="#555555", labelsize=9)


def mixture_curves(mu, sg, t):
    """Pooled Gaussian-mixture density and exceedance on the threshold grid `t`."""
    dens = np.zeros(t.size)
    exc = np.zeros(t.size)
    for i in range(0, mu.size, 100_000):
        m, s = mu[i:i + 100_000, None], sg[i:i + 100_000, None]
        u = (t[None, :] - m) / s
        dens += (np.exp(-0.5 * u * u) / (s * SQRT2PI)).sum(axis=0)
        exc += ndtr(-u).sum(axis=0)
    return dens / mu.size, exc / mu.size


def observed_curves(vals, edges, t):
    """Empirical density (binned) and exceedance from the full pooled truth."""
    dens, _ = np.histogram(vals, bins=edges, density=True)
    order = np.sort(vals)
    exc = 1.0 - np.searchsorted(order, t, side="left") / order.size
    return dens, exc


def load_or_die(trial, name, eval_year):
    md = ROOT / "CLEAN_trained_models" / trial / "tmax"
    path = ev.bundle_cache_path(md, eval_year)
    B = ev.load_bundle(path, md) if path.exists() else None
    if B is None:
        raise SystemExit(
            f"no usable prediction bundle for {name} ({path}). Run:\n"
            f"  python evaluate.py --model-dir {md}"
            + ("" if eval_year is None else f" --eval-year {eval_year}"))
    return B
