#!/usr/bin/env python
"""The warm tail of the selected temperature model on the unseen year.

One panel: the frequency with which the daily maximum reaches or exceeds a threshold,
observed against predicted, over all valid points and days of 2024. The observed curve is
the empirical exceedance of the MeteoSwiss analysis; the predicted curve is the analytic
exceedance of the pooled Gaussian mixture, mean_k Phi((mu_k - t) / sigma_k), which is the
same estimator the multi-model overlay uses (``scripts/_tmax_overlay_common.py``), so the
two figures agree where they overlap.

Reads the cached prediction bundle written by ``evaluate.py``
(``<model_dir>/pred_cache/holdout_2024.npz``), so it costs no inference.

    python scripts/gen_tmax_warm_tail.py --out LATEX_REPORT/images/tmax_warm_tail.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _tmax_overlay_common import (  # noqa: E402
    OBS_INK, WIND_CLIP, load_or_die, mixture_curves, style_ax,
)

MODEL_INK = "#1f6fb4"
ONSET = 30.0         # where the two curves start to separate
MARK = 34.0          # where the excess is largest before the observed record (35.7 degC)
T_LO, T_HI, T_STEP = 15.0, 36.0, 0.25
Y_FLOOR = 1e-6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    label, _note, trial = WIND_CLIP[0], WIND_CLIP[1], WIND_CLIP[4]
    B = load_or_die(trial, label, 2024)
    day = B.day_mask if B.day_mask is not None else np.ones(B.n_times, bool)

    truth = B.truths_c[day].reshape(-1)
    mu = B.preds_c[day].reshape(-1)
    sg = B.sigmas_c[day].reshape(-1)
    ok = np.isfinite(truth) & np.isfinite(mu) & np.isfinite(sg)
    truth, mu, sg = truth[ok], mu[ok], sg[ok]

    t = np.arange(T_LO, T_HI + T_STEP, T_STEP)
    order = np.sort(truth)
    obs_exc = 1.0 - np.searchsorted(order, t, side="left") / order.size
    _dens, mod_exc = mixture_curves(mu, sg, t)

    def at(x, curve):
        return float(np.interp(x, t, curve))

    for x in (ONSET, MARK):
        print(f"P(T>={x:.0f}) observed {at(x, obs_exc):.6f} "
              f"predicted {at(x, mod_exc):.6f}   ratio {at(x, mod_exc) / at(x, obs_exc):.2f}")
    print(f"n pairs {truth.size:,}   observed maximum {truth.max():.2f} degC")
    ratio = at(MARK, mod_exc) / at(MARK, obs_exc)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.semilogy(t, obs_exc, color=OBS_INK, lw=2.4, ls=":", label="Observed (MeteoSwiss)")
    ax.semilogy(t, mod_exc, color=MODEL_INK, lw=2.0, label=f"Predicted ({label})")

    ax.axvline(ONSET, color="0.6", lw=0.9, ls="--", zorder=0)
    ax.text(ONSET - 0.4, 0.45, f"{ONSET:.0f}$\\,^{{\\circ}}$C", fontsize=9,
            color="0.4", ha="right")
    ax.annotate(f"at {MARK:.0f}$\\,^{{\\circ}}$C the model predicts\n"
                f"{ratio:.1f}$\\times$ the observed frequency",
                xy=(MARK, at(MARK, mod_exc)), xytext=(24.0, 4e-6),
                fontsize=9, color=MODEL_INK,
                arrowprops=dict(arrowstyle="->", color=MODEL_INK, lw=1.0))

    ax.set_xlabel("Daily maximum temperature threshold $t$ ($^{\\circ}$C)")
    ax.set_ylabel("$P(T_{\\max} \\geq t)$")
    ax.set_xlim(T_LO, T_HI)
    ax.set_ylim(Y_FLOOR, 1.0)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    style_ax(ax)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
