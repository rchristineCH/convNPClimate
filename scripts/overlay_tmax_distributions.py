"""Overlay of the predictive tmax distribution across the CLEAN tmax models.

The tmax counterpart of ``scripts/overlay_wetfreq_histograms.py``. One color per
model, one neutral dotted line for the MeteoSwiss observations.
Left panel:  pooled daily-tmax density (log y), observed vs each model.
Right panel: exceedance frequency P(Tmax >= t) vs threshold.

Both model curves are ANALYTIC: the pooled predictive law of a Gaussian head is the
equal-weight mixture over all (day, grid point) pairs, so the density is
mean_k N(t; mu_k, sigma_k) and the exceedance is mean_k Phi((mu_k - t)/sigma_k).
No sampling is needed (precip had to sample because its left panel mixed a point
mass at zero with a Gamma body).

Unlike the precip figure there is no broken x-axis: daily tmax is two-tailed and
not heavy-tailed, so one continuous axis with a log y-scale already shows both
tails. Observed curves use every valid (day, point); the mixture curves use a
spatial stride (the run prints the observed full-vs-strided deviation as a check).

Reads the gitignored prediction bundles at <model>/tmax/pred_cache/{cv,holdout_2024}.npz
-- populate them with `python evaluate.py --model-dir <model>/tmax [--eval-year 2024]`.
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _tmax_overlay_common import (  # noqa: E402
    ANCHORS, ATM, ATM_CLIP, BIN_WIDTH, OBS_INK, OUT, REGIMES, SFC, SFC_NOGEO,
    TARGET_PAIRS, WIND, WIND_CLIP, load_or_die, mixture_curves, observed_curves,
    style_ax,
)

# (file suffix, models, title note). The 5-model set keeps the original filenames.
MODEL_SETS = [
    ("", [ATM, WIND, ANCHORS, SFC, SFC_NOGEO], ""),
    ("_with_clip60", [ATM, ATM_CLIP, WIND, WIND_CLIP, ANCHORS, SFC, SFC_NOGEO],
     " incl. clipped 60-epoch retrains"),
]


# One job per (model set, regime) — flattened so the body stays a single loop level.
JOBS = [(ss, ms, sn, ey, t, rs)
        for ss, ms, sn in MODEL_SETS
        for ey, t, rs in REGIMES]

for set_suffix, MODELS, set_note, eval_year, title, suffix in JOBS:
    # One bundle at a time: expanded, each is ~2.6 GB.
    ref = load_or_die(MODELS[0][4], MODELS[0][0], eval_year)
    day = ref.day_mask if ref.day_mask is not None else np.ones(ref.n_times, bool)
    truth_flat = ref.truths_c[day].reshape(-1)
    truth_all = truth_flat[np.isfinite(truth_flat)]

    lo = np.floor(np.percentile(truth_all, 0.002) - 1)
    hi = np.ceil(np.percentile(truth_all, 99.998) + 1)
    edges = np.arange(lo, hi + BIN_WIDTH, BIN_WIDTH)
    centers = 0.5 * (edges[:-1] + edges[1:])
    obs_dens, obs_exc = observed_curves(truth_all, edges, centers)

    # Same (day, point) stride for every model, so the curves stay comparable.
    stride = max(1, truth_flat.size // TARGET_PAIRS)

    # Sanity check: the stride must not move the observed distribution.
    tr = truth_flat[::stride]
    tr = tr[np.isfinite(tr)]
    s_dens, s_exc = observed_curves(tr, edges, centers)
    m = obs_dens > 1e-4
    print(f"[{len(MODELS)} models] {title}: stride {stride}; strided-vs-full observed "
          f"max dev: density "
          f"{np.abs(s_dens[m] / obs_dens[m] - 1).max():.3%} rel, exceedance "
          f"{np.abs(s_exc - obs_exc).max():.4f} abs", flush=True)
    del truth_flat, truth_all, tr

    curves = {}
    for name, _, _, _, trial in MODELS:
        B = ref if name == MODELS[0][0] else load_or_die(trial, name, eval_year)
        mu = B.preds_c[day].reshape(-1)[::stride].copy()
        sg = B.sigmas_c[day].reshape(-1)[::stride].copy()
        del B
        if name == MODELS[0][0]:
            ref = None
        ok = np.isfinite(mu) & np.isfinite(sg) & (sg > 0)
        curves[name] = mixture_curves(mu[ok], sg[ok], centers)
        print(f"  {name}: {int(ok.sum()):,} pairs", flush=True)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16.5, 5))

    ax0.plot(centers, obs_dens, ls=":", lw=2, color=OBS_INK, label="observed (MeteoSwiss)",
             zorder=5)
    for name, _, color, ls, _ in MODELS:
        ax0.plot(centers, curves[name][0], lw=2, color=color, ls=ls, label=name, zorder=4)
    ax0.set_yscale("log")
    ax0.set_xlim(lo, hi)
    # Extra headroom above the peak so the (up to 8-entry) legend clears the curve.
    ax0.set_ylim(max(obs_dens[obs_dens > 0].min() * 0.5, 1e-7), obs_dens.max() * 5)
    ax0.set_xlabel("daily maximum temperature (degC)", color="#333333")
    ax0.set_ylabel("density (log)", color="#333333")
    ax0.set_title("Daily tmax distribution", fontsize=11, color="#222222")
    ax0.legend(fontsize=8, frameon=False, labelcolor="#333333", loc="upper left",
               handlelength=1.5, handletextpad=0.6, labelspacing=0.35, borderaxespad=0.3)

    # Zoomed to the warm tail: below ~5 degC every curve is pinned at 1 and says nothing,
    # while 25 degC (summer day) and 30 degC (hot day) are the thresholds that matter.
    ax1.plot(centers, obs_exc, ls=":", lw=2, color=OBS_INK,
             label="observed (MeteoSwiss)", zorder=5)
    for name, _, color, ls, _ in MODELS:
        ax1.plot(centers, curves[name][1], lw=2, color=color, ls=ls, label=name, zorder=4)
    ax1.set_yscale("log")
    ax1.set_xlim(5, hi)
    ax1.set_ylim(1e-5, 1.0)
    ax1.set_xlabel("threshold t (degC)", color="#333333")
    ax1.set_ylabel("exceedance freq  P(Tmax >= t)", color="#333333")
    ax1.set_title("Warm-tail exceedance frequency", fontsize=11, color="#222222")

    # The overlay curves sit on top of each other; the ratio is where the models differ.
    # Only where the observed density rests on enough samples to be a reference.
    keep = obs_dens >= 5e-5
    x_lo, x_hi = centers[keep].min(), centers[keep].max()
    ax2.axhline(1.0, color=OBS_INK, ls=":", lw=2, zorder=5)
    for name, _, color, ls, _ in MODELS:
        ax2.plot(centers[keep], curves[name][0][keep] / obs_dens[keep], lw=2,
                 color=color, ls=ls, label=name, zorder=4)
    ax2.set_yscale("log")
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(0.2, 5)
    ax2.set_yticks([0.25, 0.5, 1, 2, 4], ["0.25x", "0.5x", "1x", "2x", "4x"])
    ax2.set_xlabel("daily maximum temperature (degC)", color="#333333")
    ax2.set_ylabel("density ratio  model / observed", color="#333333")
    ax2.set_title("Distribution bias (ratio to observed)", fontsize=11, color="#222222")

    for ax in (ax0, ax1, ax2):
        style_ax(ax)

    fig.suptitle(f"Predictive tmax distribution — {len(MODELS)} models{set_note}, {title}",
                 fontsize=12, color="#222222", y=0.97)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.11, wspace=0.24)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"tmax_distribution_overlay{set_suffix}{suffix}.png"
    fig.savefig(png, dpi=150)
    with open(png.with_suffix("").with_suffix(".fig.pkl"), "wb") as fh:
        pickle.dump(fig, fh)
    np.savez_compressed(
        png.with_suffix("").with_suffix(".npz"),
        centers=centers, edges=edges, obs_density=obs_dens, obs_exc=obs_exc,
        stride=stride,
        **{f"{n}_density": curves[n][0] for n, _, _, _, _ in MODELS},
        **{f"{n}_exc": curves[n][1] for n, _, _, _, _ in MODELS})
    print("saved", png)
    plt.close(fig)
