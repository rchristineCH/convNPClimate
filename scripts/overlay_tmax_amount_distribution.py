"""Merged counterpart of the precip ``amount_distribution`` figure, for the tmax models.

``eval_precip_figures.fig_amount_distribution`` draws, for ONE precip model, the
pooled wet-day accumulation density: observed vs predicted, log y-axis, with the
R10 threshold and the observed P98 marked. This script is the tmax version of that
figure with every CLEAN tmax model merged into a single panel -- one coloured curve
per model against one neutral observed curve, so the report needs one figure instead
of seven.

Differences from the precip original, both forced by the variable:
  * the quantity is the daily maximum temperature itself, not a wet-day subset;
  * the model curves are ANALYTIC rather than sampled. The pooled predictive law of
    a Gaussian head is the equal-weight mixture over all (day, grid point) pairs, so
    the density is mean_k N(t; mu_k, sigma_k). Precip had to sample because its
    density mixes a point mass at zero with a Gamma body.
  * the marked thresholds are the observed P2/P98 tails and the 30 degC hot-day line,
    standing in for the wet-day P98 and R10.

Companion to ``overlay_tmax_distributions.py``, which shows the same pooled density
alongside warm-tail exceedance and the density ratio; that figure is the diagnostic,
this one is the single-panel report figure. Both share their model table, styling and
maths via ``_tmax_overlay_common``.

Usage:  python scripts/overlay_tmax_amount_distribution.py [--quick]
"""
import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _tmax_overlay_common import (  # noqa: E402
    ALL_MODELS, BIN_WIDTH, OBS_INK, OUT, REGIMES, TARGET_PAIRS,
    load_or_die, mixture_curves, observed_curves, style_ax,
)

HOT_DAY_DEGC = 30.0  # the tmax stand-in for precip's R10 marker
TAIL_INK = "#c23b22"


def build(models, eval_year, title, suffix, target_pairs):
    # One bundle at a time: expanded, each is ~2.6 GB.
    ref = load_or_die(models[0][4], models[0][0], eval_year)
    day = ref.day_mask if ref.day_mask is not None else np.ones(ref.n_times, bool)
    truth_flat = ref.truths_c[day].reshape(-1)
    truth_all = truth_flat[np.isfinite(truth_flat)]

    lo = np.floor(np.percentile(truth_all, 0.002) - 1)
    hi = np.ceil(np.percentile(truth_all, 99.998) + 1)
    edges = np.arange(lo, hi + BIN_WIDTH, BIN_WIDTH)
    centers = 0.5 * (edges[:-1] + edges[1:])
    obs_dens, _ = observed_curves(truth_all, edges, centers)
    p02, p98 = np.percentile(truth_all, [2, 98])

    # Same (day, point) stride for every model, so the curves stay comparable.
    stride = max(1, truth_flat.size // target_pairs)
    tr = truth_flat[::stride]
    tr = tr[np.isfinite(tr)]
    s_dens, _ = observed_curves(tr, edges, centers)
    m = obs_dens > 1e-4
    print(f"[{len(models)} models] {title}: stride {stride}; strided-vs-full observed "
          f"max density dev {np.abs(s_dens[m] / obs_dens[m] - 1).max():.3%} rel",
          flush=True)
    del truth_flat, truth_all, tr

    curves = {}
    for name, _, _, _, trial in models:
        B = ref if name == models[0][0] else load_or_die(trial, name, eval_year)
        mu = B.preds_c[day].reshape(-1)[::stride].copy()
        sg = B.sigmas_c[day].reshape(-1)[::stride].copy()
        del B
        if name == models[0][0]:
            ref = None
        ok = np.isfinite(mu) & np.isfinite(sg) & (sg > 0)
        curves[name] = mixture_curves(mu[ok], sg[ok], centers)[0]
        print(f"  {name}: {int(ok.sum()):,} pairs", flush=True)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    # Observed as a step histogram (the precip figure's idiom), models as lines.
    ax.step(centers, obs_dens, where="mid", lw=2, color=OBS_INK,
            label="observed (MeteoSwiss)", zorder=6)
    for name, _, color, ls, _ in models:
        ax.plot(centers, curves[name], lw=1.8, color=color, ls=ls, label=name, zorder=4)

    ax.axvline(HOT_DAY_DEGC, color="grey", ls=":", lw=1, zorder=2,
               label=f"hot day ({HOT_DAY_DEGC:.0f} degC)")
    for x in (p02, p98):
        ax.axvline(x, color=TAIL_INK, ls="--", lw=1, zorder=2)
    ax.plot([], [], color=TAIL_INK, ls="--", lw=1,
            label=f"obs P2={p02:.1f} / P98={p98:.1f} degC")

    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    # Headroom above the peak so the 10-entry legend clears the curves.
    ax.set_ylim(max(obs_dens[obs_dens > 0].min() * 0.5, 1e-7), obs_dens.max() * 12)
    ax.set_xlabel("daily maximum temperature (degC)", color="#333333")
    ax.set_ylabel("density (log)", color="#333333")
    ax.set_title(f"Daily tmax distribution: observed vs {len(models)} models — {title}",
                 fontsize=11, color="#222222")
    ax.legend(fontsize=8, frameon=False, labelcolor="#333333", loc="upper left",
              ncol=2, handlelength=1.6, handletextpad=0.6, labelspacing=0.35,
              borderaxespad=0.3)
    style_ax(ax)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"tmax_amount_distribution_overlay{suffix}.png"
    fig.savefig(png, dpi=150)
    with open(png.with_suffix("").with_suffix(".fig.pkl"), "wb") as fh:
        pickle.dump(fig, fh)
    np.savez_compressed(
        png.with_suffix("").with_suffix(".npz"),
        centers=centers, edges=edges, obs_density=obs_dens, stride=stride,
        obs_p02=p02, obs_p98=p98,
        **{f"{n}_density": curves[n] for n, _, _, _, _ in models})
    print("saved", png, flush=True)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 models, 2024 holdout only, coarser stride")
    args = ap.parse_args()

    models = ALL_MODELS[:2] if args.quick else ALL_MODELS
    regimes = REGIMES[1:] if args.quick else REGIMES
    target = 200_000 if args.quick else TARGET_PAIRS
    for eval_year, title, suffix in regimes:
        build(models, eval_year, title, "_quick" if args.quick else suffix, target)


if __name__ == "__main__":
    main()
