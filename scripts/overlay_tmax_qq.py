"""Merged PIT/Q-Q calibration figure across all CLEAN tmax models.

The cross-model counterpart of the per-model ``eval_{cv,2024}/qq_calibration.png``,
using the same definition as ``visualization.plot_qq_calibration``: for a Gaussian
head the Probability Integral Transform is ``PIT = Phi((truth - mu) / sigma)``, and
a calibrated model makes it Uniform(0, 1).

Three panels, because the plain Q-Q alone cannot separate the models — every curve
hugs the diagonal at this sample size:
  (a) Q-Q: empirical PIT quantiles vs uniform, with the diagonal.
  (b) The same curves as a DEVIATION from the diagonal, which is where they separate.
      Above 0 = the model puts too much mass low; the sign flips across the median for
      a variance error and stays one-signed for a bias.
  (c) PIT histogram — the shape diagnosis. Flat = calibrated, U = overconfident
      (sigma too small), dome = underconfident, tilt = biased.

**Read the 2024 figure for any calibration claim about the delivered system.** CV
scores a single fold's sigma; only the holdout year carries the moment-matched
ensemble sigma (sigma^2 = mean sigma_k^2 + var mu_k) that the 5-fold system actually
produces.

Reads the gitignored bundles at <model>/tmax/pred_cache/{cv,holdout_2024}.npz —
populate them with `python evaluate.py --model-dir <model>/tmax [--eval-year 2024]`.
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import evaluate as ev  # noqa: E402
from compare_evaluations import DEFAULT_MODELS, MODEL_STYLE  # noqa: E402

REGIMES = [(None, "CV 2020-2023", ""), (2024, "2024 holdout", "_2024")]
OUT = ROOT / "CLEAN_trained_models" / "tmax_model_comparison"
PROBS = np.linspace(0.01, 0.99, 100)   # same levels as the per-model figure
HIST_BINS = np.linspace(0.0, 1.0, 41)
OBS_INK = "#333333"

# Quality bands on the Q-Q DEVIATION, in probability units: |empirical - theoretical|,
# whose maximum is exactly the KS distance. A deviation of 0.05 means the model's stated
# 80th percentile is really the 75th — directly interpretable, and bounded in [0,1].
#
# These cut points are rules of thumb, not a standard. They are deliberately NOT placed on
# the PIT-density axis: the density integrates to 1 over [0,1], so a bin above 1 forces
# another below it, and "bad at PIT=0.5" is not a statement that stands on its own.
# Significance testing is useless at this n — with 68M points any deviation rejects
# uniformity — so what is banded is effect size.
BANDS = [(0.02, "good", "#e8f2e8"), (0.05, "decent", "#fdf3e0"), (0.12, "poor", "#fbecec")]
DEV_LIM = 0.12          # shared across regimes so CV and 2024 are directly comparable
HIST_LIM = (0.35, 2.6)  # density as a ratio to the calibrated value of 1.0


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", color="#dddddd", lw=0.6, zorder=0)
    ax.tick_params(colors="#555555", labelsize=9)


def pit_stats(B):
    """PIT quantiles, histogram density and headline scalars for one bundle."""
    day = B.day_mask if B.day_mask is not None else np.ones(B.n_times, bool)
    t = B.truths_c[day].reshape(-1)
    m = B.preds_c[day].reshape(-1)
    s = B.sigmas_c[day].reshape(-1)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(s) & (s > 0)
    t, m, s = t[ok], m[ok], s[ok]
    z = (t - m) / s
    pit = ndtr(z)                       # = Phi((truth - mu)/sigma)
    dens, _ = np.histogram(pit, bins=HIST_BINS, density=True)
    return {
        "quantiles": np.percentile(pit, PROBS * 100),
        "hist": dens,
        "pit_mean": float(pit.mean()),
        "z_std": float(z.std()),
        "cov90": float(np.mean((pit >= 0.05) & (pit <= 0.95))),
        "n": int(t.size),
    }


for eval_year, title, suffix in REGIMES:
    stats = {}
    for label, mdir in DEFAULT_MODELS.items():
        md = ROOT / mdir
        path = ev.bundle_cache_path(md, eval_year)
        B = ev.load_bundle(path, md) if path.exists() else None
        if B is None:
            raise SystemExit(
                f"no usable prediction bundle for {label} ({path}). Run:\n"
                f"  python evaluate.py --model-dir {md}"
                + ("" if eval_year is None else f" --eval-year {eval_year}"))
        stats[label] = pit_stats(B)
        del B
        s = stats[label]
        print(f"  {label:17s} PIT mean {s['pit_mean']:.4f}  z-std {s['z_std']:.4f}  "
              f"cov90 {s['cov90']:.4f}  n {s['n']:,}", flush=True)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16.5, 5))
    centers = 0.5 * (HIST_BINS[:-1] + HIST_BINS[1:])

    ax0.plot([0, 1], [0, 1], ls=":", lw=2, color=OBS_INK, label="perfect calibration",
             zorder=5)
    for label in DEFAULT_MODELS:
        color, ls, _ = MODEL_STYLE[label]
        ax0.plot(PROBS, stats[label]["quantiles"], lw=2, color=color, ls=ls,
                 label=label, zorder=4)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_aspect("equal")
    ax0.set_xlabel("theoretical quantile (uniform)", color="#333333")
    ax0.set_ylabel("empirical PIT quantile", color="#333333")
    ax0.set_title("Q-Q: PIT vs uniform", fontsize=11, color="#222222")
    ax0.legend(fontsize=7.5, frameon=False, labelcolor="#333333", loc="upper left",
               handlelength=1.5, handletextpad=0.6, labelspacing=0.35, borderaxespad=0.3)

    prev = 0.0
    for edge, name, shade in BANDS:
        for sign in (1, -1):
            ax1.axhspan(sign * prev, sign * edge, color=shade, zorder=0)
        ax1.annotate(name, (0.995, edge), xycoords=("axes fraction", "data"),
                     xytext=(-2, -2), textcoords="offset points", ha="right", va="top",
                     fontsize=7.5, color="#888888")
        prev = edge
    ax1.axhline(0.0, ls=":", lw=2, color=OBS_INK, zorder=5)
    ax1.annotate("perfect calibration", (0.995, 0.0), xycoords=("axes fraction", "data"),
                 xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom",
                 fontsize=7.5, color="#666666")
    for label in DEFAULT_MODELS:
        color, ls, _ = MODEL_STYLE[label]
        dev = stats[label]["quantiles"] - PROBS
        stats[label]["ks"] = float(np.abs(dev).max())
        ax1.plot(PROBS, dev, lw=2, color=color, ls=ls,
                 label=f"{label}  {stats[label]['ks']:.3f}", zorder=4)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-DEV_LIM, DEV_LIM)
    ax1.set_xlabel("theoretical quantile (uniform)", color="#333333")
    ax1.set_ylabel("empirical - theoretical", color="#333333")
    ax1.set_title("Deviation from uniform (worst = KS distance)",
                  fontsize=11, color="#222222")
    ax1.legend(fontsize=7, frameon=False, labelcolor="#333333", loc="lower left",
               handlelength=1.5, handletextpad=0.6, labelspacing=0.3, borderaxespad=0.3)

    # Log scale centred on 1: a density of 1.25 and one of 0.8 are the same size of error,
    # and a linear axis hides that. Limits fixed so CV and 2024 can be read against
    # each other.
    ax2.axhline(1.0, ls=":", lw=2, color=OBS_INK, zorder=5)
    ax2.annotate("perfect calibration", (0.01, 1.0), xycoords=("axes fraction", "data"),
                 xytext=(2, 3), textcoords="offset points", ha="left", va="bottom",
                 fontsize=7.5, color="#666666")
    for label in DEFAULT_MODELS:
        color, ls, _ = MODEL_STYLE[label]
        ax2.plot(centers, stats[label]["hist"], lw=2, color=color, ls=ls, zorder=4)
    ax2.set_yscale("log")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(*HIST_LIM)
    ax2.set_yticks([0.4, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 2.5],
                   ["0.4x", "0.5x", "0.67x", "0.8x", "1x", "1.25x", "1.5x", "2x", "2.5x"])
    # The log locator would otherwise stamp a stray "6 x 10^-1" over the axis.
    ax2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax2.set_xlabel("PIT value", color="#333333")
    ax2.set_ylabel("density relative to calibrated", color="#333333")
    ax2.set_title("PIT histogram", fontsize=11, color="#222222")

    for ax in (ax0, ax1, ax2):
        style_ax(ax)

    sigma_note = ("single-fold sigma" if eval_year is None
                  else "moment-matched 5-fold ensemble sigma — the delivered system")
    fig.suptitle(f"tmax calibration (PIT) — {len(DEFAULT_MODELS)} models, {title}",
                 fontsize=12, color="#222222", y=0.97)
    fig.text(0.5, 0.015,
             f"PIT = Phi((truth - mu)/sigma); calibrated => Uniform(0,1), i.e. flat "
             f"histogram — U-shape = sigma too small (overconfident), dome = too large.  "
             f"{sigma_note}.  "
             f"n = {stats[next(iter(DEFAULT_MODELS))]['n']:,} (day, point) pairs per model.",
             ha="center", fontsize=7.5, color="0.4")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.13, wspace=0.26)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"tmax_qq_calibration_overlay{suffix}.png"
    fig.savefig(png, dpi=150)
    with open(png.with_suffix("").with_suffix(".fig.pkl"), "wb") as fh:
        pickle.dump(fig, fh)
    np.savez_compressed(
        png.with_suffix("").with_suffix(".npz"),
        probs=PROBS, hist_bins=HIST_BINS,
        **{f"{k}_quantiles": v["quantiles"] for k, v in stats.items()},
        **{f"{k}_hist": v["hist"] for k, v in stats.items()},
        **{f"{k}_scalars": np.array([v["pit_mean"], v["z_std"], v["cov90"], v["ks"],
                                     v["n"]]) for k, v in stats.items()})
    print("saved", png, flush=True)
    plt.close(fig)
