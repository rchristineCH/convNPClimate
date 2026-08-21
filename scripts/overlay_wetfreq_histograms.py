"""Overlay of the wet-freq threshold histogram figure across the CLEAN precip models.

One color per precip model (solid = model prediction), one neutral dotted
line for the observations (identical across models — verified byte-equal).
Left panel: pooled daily-accumulation histograms (log density, sampled preds).
Right panel: exceedance frequency P(accum >= t) vs threshold (analytic BG).
Each panel has a broken x-axis: the left 2/3 zooms on [0, p98] of the observed
daily accumulation (p98 = threshold where obs exceedance drops to 0.02), the
right 1/3 shows the compressed tail on a gray background.
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# (short name, legend descriptor "atmos inputs | loss", color, model dir)
MODELS = [
    ("NLL", "atm z,t,q | BG-NLL loss", "#2a78d6",
     "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8"),
    ("CRPS", "atm z,t,q | BG-CRPS loss", "#eb6834",
     "clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8"),
    ("WIND", "atm z,t,q,u,v | BG-NLL loss", "#1baf7a",
     "clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8"),
    ("FT-CRPS", "atm z,t,q | BG-NLL then BG-CRPS fine-tune", "#8e5bd0",
     "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps"),
    ("SFC-TP", "sfc t2m_max,tp | BG-NLL loss", "#d4a017",
     "clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8"),
]
OBS_INK = "#333333"
REGIMES = [("eval_figures", "CV 2020-2023", ""), ("eval_figures_2024", "2024 holdout", "_2024")]
OUT = Path("CLEAN_trained_models/precip_processing_comparison")


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", color="#dddddd", lw=0.6, zorder=0)
    ax.tick_params(colors="#555555", labelsize=9)


def split_panel(fig, slot, p98, xmax):
    """Broken x-axis pair: [0, p98] on the left 2/3, (p98, xmax] tail on the right 1/3."""
    inner = slot.subgridspec(1, 2, width_ratios=(2, 1), wspace=0.07)
    ax_l = fig.add_subplot(inner[0])
    ax_r = fig.add_subplot(inner[1], sharey=ax_l)
    ax_l.set_xlim(0, p98)
    ticks = [t for t in ax_l.get_xticks() if 0 <= t <= p98 * 0.9]
    ax_l.set_xticks(ticks + [p98], [f"{t:g}" for t in ticks] + [f"p98={p98:.1f}"])
    ax_r.set_xlim(p98, xmax)
    ax_r.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="lower"))
    for ax in (ax_l, ax_r):
        style_ax(ax)
        ax.set_yscale("log")
    ax_l.get_xticklabels()[-1].set_color("#d62728")
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(left=False, labelleft=False)
    ax_r.set_facecolor("#f7f7f7")
    ax_r.set_xlabel("tail", color="#888888")
    # slanted break marks at the cut, top and bottom
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=9, ls="none",
              color="#555555", mec="#555555", mew=1, clip_on=False, zorder=10)
    ax_l.plot([1, 1], [0, 1], transform=ax_l.transAxes, **kw)
    ax_r.plot([0, 0], [0, 1], transform=ax_r.transAxes, **kw)
    return ax_l, ax_r


for regime, title, suffix in REGIMES:
    # A model need not have every regime: SFC-TP has no 2024 holdout, because it
    # takes ERA5-Land tp as an input and tp-2024.nc is defective (see
    # tp_hourly/README.md). Draw whoever is present rather than failing the figure,
    # and say in the caption who is missing so the panel is not read as complete.
    present, missing = [], []
    for name, desc, colour, d in MODELS:
        f = Path(f"CLEAN_trained_models/{d}/precip/{regime}/wetfreq_threshold_histograms.npz")
        (present if f.exists() else missing).append((name, desc, colour, d))
    if missing:
        print(f"[{regime}] skipping {[m[0] for m in missing]} (no {regime} artifacts)")
    if not present:
        print(f"[{regime}] no models have artifacts; skipping this regime")
        continue
    models_here = present
    data = {name: np.load(f"CLEAN_trained_models/{d}/precip/{regime}/wetfreq_threshold_histograms.npz")
            for name, _, _, d in models_here}
    z0 = data[models_here[0][0]]
    grid, bins, ts = z0["grid"], z0["hist_bins"], z0["thresholds"]
    centers = 0.5 * (bins[:-1] + bins[1:])

    exc = z0["obs_exc"]
    p98 = float(grid[np.argmax(exc <= 0.02)]) if (exc <= 0.02).any() else float(grid.max()) / 2
    xmax = float(grid.max())

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 2, wspace=0.22)
    ax0l, ax0r = split_panel(fig, gs[0], p98, xmax)
    ax1l, ax1r = split_panel(fig, gs[1], p98, xmax)

    # (a) accumulation histograms: obs dotted, per-model sampled solid.
    for ax in (ax0l, ax0r):
        ax.plot(centers, z0["obs_hist_density"], ls=":", lw=2, color=OBS_INK,
                marker="o", ms=3, label="observed (binned)", zorder=5)
        for name, desc, color, _ in models_here:
            ax.plot(centers, data[name]["samp_hist_density"], lw=2, color=color,
                    marker="o", ms=3, label=f"{name} ({desc}; sampled)", zorder=4)
    ax0l.set_xlabel("daily accumulation (mm)", color="#333333")
    ax0l.set_ylabel("density (log)", color="#333333")
    ax0l.set_title("Daily accumulation histograms", fontsize=11, color="#222222", x=0.78)

    # (b) exceedance frequency vs threshold: obs dotted, per-model analytic solid.
    for ax in (ax1l, ax1r):
        ax.plot(grid, z0["obs_exc"], ls=":", lw=2, color=OBS_INK,
                marker="o", ms=2.5, label="observed (empirical, per grid t)", zorder=5)
        for name, desc, color, _ in models_here:
            ax.plot(grid, data[name]["ana_exc"], lw=2, color=color,
                    label=f"{name} ({desc}; analytic)", zorder=4)
    ax1l.set_xlabel("threshold t (mm)", color="#333333")
    ax1l.set_ylabel("exceedance freq  P(accum >= t)", color="#333333")
    ax1l.set_title("Wet-day frequency vs threshold", fontsize=11, color="#222222", x=0.78)
    for ax in (ax0l, ax1l):
        ax.legend(fontsize=7.5, frameon=False, labelcolor="#333333", loc="lower left",
                  handlelength=1.5, handletextpad=0.6, labelspacing=0.35, borderaxespad=0.3)

    note = f"  (no {', '.join(m[0] for m in missing)})" if missing else ""
    fig.suptitle(f"Wet-frequency threshold histograms — {len(models_here)} precip models, "
                 f"{title}{note}", fontsize=12, color="#222222", y=0.97)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.85, bottom=0.11)

    png = OUT / f"wetfreq_threshold_histograms_overlay{suffix}.png"
    fig.savefig(png, dpi=150)
    with open(png.with_suffix("").with_suffix(".fig.pkl"), "wb") as fh:
        pickle.dump(fig, fh)
    print("saved", png)
    plt.close(fig)
