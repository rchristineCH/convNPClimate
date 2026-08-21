#!/usr/bin/env python
"""Exploration figures: the two selected precip runs at the 139 SMN rain gauges.

Reads the station bundles written by precip_station_inference.py. Six figures
into station_exploration/figures/:

  precip_absolute_values.png   MAE, CRPS and RPSS at the gauges in physical units,
                               with the bilinear ERA5-Land reference as its own bar
  precip_gauge_skill_map.png   MAE skill vs bilinear w0606 ERA5-Land, per gauge
  precip_wetday_scatter.png    observed vs predicted wet-day frequency per gauge
  precip_category_bss.png      Brier skill by intensity category at the gauges
                               (same categories and reference as the gridded study)
  precip_seasonal_cycle.png    monthly wet-day frequency and mean accumulation
  precip_pit.png               randomized PIT pooled over the gauges

Everything is exceedance-based -- no fixed rho cut anywhere -- and every
convention (1 mm wet threshold, category edges, deterministic 0/1 reference
probabilities, dry-atom randomized PIT) matches eval_precip/eval_precip_figures,
so a gauge number can be read against its gridded counterpart directly.

Usage:  python station_exploration/fig_precip_stations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import special as sspecial
from scipy import stats as sstats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from eval_precip import (WET_THRESHOLD_MM, DRY_OBS_THRESHOLD_MM,  # noqa: E402
                         _bg_exceedance_prob)

FIGS = HERE / "figures"
BUNDLES = HERE / "bundles"
MODELS = [("FT-CRPS", "#8e5bd0", "ft-crps"), ("SFC-TP", "#c9932e", "sfc-tp")]
OBS_INK = "#333333"
BASE_COLOR = "#d95f02"      # ERA5-Land's colour in the skill timeseries figure
REGIMES = [("cv", "CV holdout 2020-2023"), ("2024", "2024 holdout")]
CAT_EDGES = [0.0, 0.1, 10.0, 40.0, 80.0, np.inf]
CAT_LABELS = ["0-0.1", "0.1-10", "10-40", "40-80", ">=80"]
PIT_BINS = 20


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#dddddd", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#555555", labelsize=9)


def load(stem: str):
    z = np.load(BUNDLES / f"{stem}.npz", allow_pickle=True)
    dm = z["day_mask"]
    return {
        "params": z["params"][dm], "obs": z["truth_mm"][dm], "base": z["base_mm"][dm],
        "dates": np.asarray(z["dates"])[dm], "stn": z["stn"], "lat": z["lat"],
        "lon": z["lon"], "elev": z["elev_m"], "nbcn": z["is_nbcn"],
    }


def mean_pred(P):
    return P[..., 0] * P[..., 1] / P[..., 2]


def per_gauge_rpss(b):
    """(n_gauges,) ranked probability skill vs the interpolated reference.

    Same construction as the gridded study (eval_precip_figures): the RPS is the
    squared CDF distance accumulated over the internal category boundaries
    (0.1, 10, 40, 80 mm), the reference CDF is the deterministic 0/1 step of the
    bilinear ERA5-Land value, and the skill is 1 - RPS_model / RPS_ref per gauge.
    Categorical by construction: it scores which intensity class the day lands
    in, not the millimetre error of a point value.
    """
    P, obs, base = b["params"], b["obs"], b["base"]
    rho, alpha, beta = P[..., 0], P[..., 1], P[..., 2]
    valid = np.isfinite(obs) & np.isfinite(base) & np.isfinite(rho)
    w = valid.astype(float)
    rps_m = np.zeros(obs.shape[1])
    rps_r = np.zeros(obs.shape[1])
    for edge in CAT_EDGES[1:-1]:
        Fk = 1.0 - rho * (1.0 - sstats.gamma.cdf(edge, a=alpha, scale=1.0 / beta))
        Ok = (obs < edge).astype(np.float64)
        Bk = (base < edge).astype(np.float64)
        rps_m += (w * np.nan_to_num((Fk - Ok) ** 2)).sum(axis=0)
        rps_r += (w * np.nan_to_num((Bk - Ok) ** 2)).sum(axis=0)
    denom = w.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((denom >= 30) & (rps_r > 0), 1.0 - rps_m / rps_r, np.nan)


def fig_skill_map(data):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), sharex=True, sharey=True)
    vmax = 0.6
    sc = None
    for r, (label, _color, stem) in enumerate(MODELS):
        for c, (reg, reg_label) in enumerate(REGIMES):
            ax = axes[r][c]
            b = data[f"{stem}_{reg}"]
            sk = per_gauge_rpss(b)
            sc = ax.scatter(b["lon"], b["lat"], c=sk, cmap="RdBu", vmin=-vmax,
                            vmax=vmax, s=34, edgecolors="white", lw=0.5, zorder=3)
            ring = b["nbcn"].astype(bool)
            ax.scatter(b["lon"][ring], b["lat"][ring], facecolors="none",
                       edgecolors="#333333", s=90, lw=0.8, zorder=4)
            share = np.mean(sk[np.isfinite(sk)] > 0)
            ax.set_title(f"{label} — {reg_label}\n"
                         f"median {np.nanmedian(sk):+.2f} · positive skill "
                         f"at {share:.0%} of gauges", fontsize=9.5)
            ax.set_aspect(1.4)   # ~cos(46.8°): degrees to comparable km axes
            style_ax(ax)
    for ax in axes[1]:
        ax.set_xlabel("longitude [°E]", fontsize=9)
    for row in axes:
        row[0].set_ylabel("latitude [°N]", fontsize=9)
    fig.colorbar(sc, ax=axes, shrink=0.8,
                 label="RPSS vs bilinear ERA5-Land (w0606) [–]")
    fig.suptitle("Precipitation at the SMN rain gauges — ranked probability skill over "
                 "the five intensity categories\nblack rings mark the 28 NBCN sites of "
                 "the temperature chapter", fontsize=11)
    fig.savefig(FIGS / "precip_gauge_skill_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_wetday_scatter(data):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0), sharey=True)
    for ax, (reg, reg_label) in zip(axes, REGIMES):
        drew_base = False
        for label, color, stem in MODELS:
            b = data[f"{stem}_{reg}"]
            P, obs, base = b["params"], b["obs"], b["base"]
            n = obs.shape[1]
            wf_obs = np.full(n, np.nan)
            wf_prd = np.full(n, np.nan)
            wf_bas = np.full(n, np.nan)
            for i in range(n):
                ok = np.isfinite(obs[:, i]) & np.isfinite(P[:, i, 0])
                if ok.sum() < 30:
                    continue
                wf_obs[i] = np.mean(obs[ok, i] >= WET_THRESHOLD_MM)
                wf_prd[i] = np.mean(_bg_exceedance_prob(P[ok, i], WET_THRESHOLD_MM))
                okb = ok & np.isfinite(base[:, i])
                wf_bas[i] = np.mean(base[okb, i] >= WET_THRESHOLD_MM)
            if not drew_base:
                ax.scatter(wf_obs, wf_bas, s=22, marker="x", color=BASE_COLOR,
                           lw=1.0, zorder=2, label="bilinear ERA5-Land")
                drew_base = True
            ax.scatter(wf_obs, wf_prd, s=26, color=color, zorder=3, label=label,
                       edgecolors="white", lw=0.4)
        ax.plot([0.2, 0.65], [0.2, 0.65], color="#999999", lw=0.8, ls=":")
        ax.set_xlim(0.2, 0.65)
        ax.set_ylim(0.2, 0.65)
        ax.set_xlabel("observed wet-day frequency (>= 1 mm) [fraction of days]",
                      fontsize=9)
        ax.set_title(reg_label, fontsize=10)
        style_ax(ax)
    axes[0].set_ylabel("predicted wet-day frequency [fraction of days]\n"
                       "P(Y >= 1 mm), no rho cut", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Wet-day frequency at each gauge — the models inherit far less of "
                 "ERA5-Land's drizzle", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "precip_wetday_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def gauge_category_bss(b):
    """(n_gauges, n_cat) Brier skill vs the deterministic ERA5-Land reference."""
    P, obs, base = b["params"], b["obs"], b["base"]
    rho, alpha, beta = P[..., 0], P[..., 1], P[..., 2]

    def Pge(e):
        if e <= 0:
            return np.ones_like(rho)
        return rho * (1.0 - sstats.gamma.cdf(e, a=alpha, scale=1.0 / beta))

    n, C = obs.shape[1], len(CAT_LABELS)
    bss = np.full((n, C), np.nan)
    obs_share = np.zeros(C)
    total = 0
    prev = Pge(0.0)
    valid = np.isfinite(obs) & np.isfinite(base) & np.isfinite(rho)
    for c in range(C):
        upper = CAT_EDGES[c + 1]
        Pge_up = np.zeros_like(rho) if np.isinf(upper) else Pge(upper)
        Pc = prev - Pge_up
        lo = CAT_EDGES[c]
        o_in = (obs >= lo) & (obs < upper)
        b_in = (base >= lo) & (base < upper)
        for i in range(n):
            ok = valid[:, i]
            if ok.sum() < 30:
                continue
            bs_m = np.mean((Pc[ok, i] - o_in[ok, i]) ** 2)
            bs_r = np.mean((b_in[ok, i].astype(float) - o_in[ok, i]) ** 2)
            if bs_r > 0:
                bss[i, c] = 1.0 - bs_m / bs_r
        obs_share[c] = o_in[valid].sum()
        total += 0  # obs_share normalised below
        prev = Pge_up
    obs_share = obs_share / valid.sum()
    return bss, obs_share


def fig_category_bss(data):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    W = 0.32
    for ax, (reg, reg_label) in zip(axes, REGIMES):
        obs_share = None
        for k, (label, color, stem) in enumerate(MODELS):
            b = data[f"{stem}_{reg}"]
            bss, share = gauge_category_bss(b)
            obs_share = share
            x = np.arange(len(CAT_LABELS)) + (k - 0.5) * W
            # mean and +-1 SD over gauges, the same summary the gridded
            # category figure draws over grid points
            mean = np.nanmean(bss, axis=0)
            sd = np.nanstd(bss, axis=0)
            ax.bar(x, mean, width=W * 0.92, color=color, zorder=3,
                   label=label if reg == "cv" else None)
            ax.errorbar(x, mean, yerr=sd, fmt="none",
                        ecolor="#555555", lw=0.9, capsize=2, zorder=4)
            print("  category BSS %-8s %-4s " % (label, reg)
                  + "  ".join(f"{m:+.3f}±{s:.2f}" for m, s in zip(mean, sd)))
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xticks(np.arange(len(CAT_LABELS)),
                      [f"{l} mm\n({s:.1%} obs)" for l, s in
                       zip(CAT_LABELS, obs_share)], fontsize=8)
        ax.set_title(reg_label, fontsize=10)
        style_ax(ax)
    axes[0].set_ylabel("Brier skill vs ERA5-Land [–]\n(mean over gauges, ±1 SD whiskers)",
                       fontsize=9)
    fig.legend(fontsize=9, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Intensity-category Brier skill at the rain gauges — same categories "
                 "and reference as the gridded study", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(FIGS / "precip_category_bss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def bg_crps(P, y):
    """Analytic CRPS [mm] of the Bernoulli-Gamma mixture against observations y [mm].

    F(x) = (1-rho)*1{x>=0} + rho*Gamma_cdf(x; alpha, rate beta), scored with the
    energy form CRPS = E|X-y| - 0.5*E|X-X'|. Both expectations are closed form:

        E|X-y|  = (1-rho)*y + rho*[ y*(2*F_alpha(y)-1) + mu*(1-2*F_{alpha+1}(y)) ]
        E|X-X'| = 2*rho*(1-rho)*mu + rho^2 * 2/(beta*B(1/2, alpha))

    with mu = alpha/beta. The Gamma part is the Scheuerer-Moeller closed form, so
    no Monte-Carlo noise enters the numbers plotted here (checked against a
    4e6-draw simulation, agreement to 1e-3 mm). Setting rho = 1 recovers the
    plain Gamma CRPS; rho = 0 gives |y|, the score of a certain dry forecast.
    """
    rho, alpha, beta = P[..., 0], P[..., 1], P[..., 2]
    mu = alpha / beta
    Fa = sstats.gamma.cdf(y, a=alpha, scale=1.0 / beta)
    Fa1 = sstats.gamma.cdf(y, a=alpha + 1.0, scale=1.0 / beta)
    e_xy = (1.0 - rho) * y + rho * (y * (2.0 * Fa - 1.0) + mu * (1.0 - 2.0 * Fa1))
    e_xx = 2.0 * rho * (1.0 - rho) * mu + rho ** 2 * 2.0 / (beta * sspecial.beta(0.5, alpha))
    return e_xy - 0.5 * e_xx


def _common_days(bundles):
    """Restrict a list of bundles to the dates all of them cover, in date order.

    The two runs do not span identical days: sfc-tp takes ERA5-Land tp as an
    input channel and so cannot predict 2023-12-31, the one day the tp files
    leave uncovered. An absolute comparison has to be made over the same days
    for both models and for the reference, so the intersection is taken rather
    than each run being scored on its own span.
    """
    keep = set(bundles[0]["dates"])
    for b in bundles[1:]:
        keep &= set(b["dates"])
    out = []
    for b in bundles:
        sel = np.array([d in keep for d in b["dates"]])
        out.append({**b, "params": b["params"][sel], "obs": b["obs"][sel],
                    "base": b["base"][sel], "dates": b["dates"][sel]})
    return out


def fig_absolute_values(data):
    """Millimetres before ratios: the same comparison the skill figures make,
    but in physical units and with the reference drawn as a bar of its own."""
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3))
    labels = [m[0] for m in MODELS]
    xs = np.arange(len(MODELS) + 1)
    W = 0.36
    rows = {}
    for k, (reg, reg_label) in enumerate(REGIMES):
        bs = _common_days([data[f"{stem}_{reg}"] for _l, _c, stem in MODELS])
        hatch = "///" if reg != "cv" else None
        mae, crps, rpss = [], [], []
        for b in bs:
            ok = np.isfinite(b["obs"]) & np.isfinite(b["params"][..., 0])
            mae.append(np.mean(np.abs(mean_pred(b["params"])[ok] - b["obs"][ok])))
            crps.append(np.mean(bg_crps(b["params"][ok], b["obs"][ok])))
            rpss.append(per_gauge_rpss(b))
        # the reference is deterministic, so its CRPS is its MAE; both models see
        # the same bilinear field over the same days, so one bar covers both.
        okb = np.isfinite(bs[0]["obs"]) & np.isfinite(bs[0]["base"])
        ref = np.mean(np.abs(bs[0]["base"][okb] - bs[0]["obs"][okb]))
        rows[reg] = (mae, crps, ref, [np.nanmedian(r) for r in rpss])

        for ax, vals in ((axes[0], mae + [ref]), (axes[1], crps + [ref])):
            pos = xs + (k - 0.5) * W
            for i, (p, v) in enumerate(zip(pos, vals)):
                col = BASE_COLOR if i == len(MODELS) else MODELS[i][1]
                ax.bar(p, v, W, color=col, edgecolor="white", hatch=hatch, zorder=3)
                ax.text(p, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        ax = axes[2]
        pos = np.arange(len(MODELS)) + (k - 0.5) * W
        for i, (p, r) in enumerate(zip(pos, rpss)):
            ax.bar(p, np.nanmedian(r), W, color=MODELS[i][1], edgecolor="white",
                   hatch=hatch, zorder=3)
            ax.errorbar(p, np.nanmedian(r),
                        yerr=[[np.nanmedian(r) - np.nanpercentile(r, 25)],
                              [np.nanpercentile(r, 75) - np.nanmedian(r)]],
                        fmt="none", ecolor="#555555", lw=0.9, capsize=2, zorder=4)
            ax.text(p, np.nanpercentile(r, 75) + 0.012, f"{np.nanmedian(r):+.2f}",
                    ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("MAE of the mean prediction [mm/day]", fontsize=9)
    axes[0].set_title("Millimetre error, models and reference", fontsize=10.5)
    axes[1].set_ylabel("CRPS [mm/day]", fontsize=9)
    axes[1].set_title("Probabilistic error in the same units", fontsize=10.5)
    axes[2].set_ylabel("RPSS vs bilinear ERA5-Land [–]\n"
                       "(median over gauges, IQR whiskers)", fontsize=9)
    axes[2].set_title("Categorical skill against interpolation", fontsize=10.5)
    for ax in axes[:2]:
        ax.set_xticks(xs, labels + ["bilinear\nERA5-Land"], fontsize=9)
        ax.set_ylim(0, 4.4)
    axes[2].set_xticks(np.arange(len(MODELS)), labels, fontsize=9)
    axes[2].set_ylim(0, 0.72)
    # the bar colour already carries the model, so the regime legend is drawn in
    # neutral grey rather than borrowing whichever model happens to come first.
    handles = [mpatches.Patch(facecolor="#bbbbbb", edgecolor="white",
                              hatch="///" if reg != "cv" else None, label=lab)
               for reg, lab in REGIMES]
    for ax in axes:
        style_ax(ax)
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Precipitation at 139 SMN rain gauges — absolute scores beside the "
                 "skill ratio, solid CV holdout and hatched 2024", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "precip_absolute_values.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for reg, (mae, crps, ref, med) in rows.items():
        for i, lab in enumerate(labels):
            print("  %-8s %-5s MAE %.3f  CRPS %.3f  RPSS %+.3f  (ref MAE %.3f)"
                  % (lab, reg, mae[i], crps[i], med[i], ref))


def fig_seasonal(data):
    import pandas as pd
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0), sharex=True)
    months = np.arange(1, 13)
    for c, (reg, reg_label) in enumerate(REGIMES):
        drew = False
        for label, color, stem in MODELS:
            b = data[f"{stem}_{reg}"]
            dates = pd.DatetimeIndex(pd.to_datetime(b["dates"]))
            P, obs, base = b["params"], b["obs"], b["base"]
            exc = _bg_exceedance_prob(P, WET_THRESHOLD_MM)
            mp = mean_pred(P)
            wf_o, wf_m, wf_b, am_o, am_m, am_b = ([] for _ in range(6))
            for m in months:
                sel = dates.month == m
                o, e, v, ba = obs[sel], exc[sel], mp[sel], base[sel]
                ok = np.isfinite(o) & np.isfinite(e)
                okb = np.isfinite(o) & np.isfinite(ba)
                wf_o.append(np.mean(o[ok] >= WET_THRESHOLD_MM))
                wf_m.append(np.mean(e[ok]))
                wf_b.append(np.mean(ba[okb] >= WET_THRESHOLD_MM))
                am_o.append(np.mean(o[ok]))
                am_m.append(np.mean(v[ok]))
                am_b.append(np.mean(ba[okb]))
            if not drew:
                axes[0][c].plot(months, wf_o, color=OBS_INK, lw=2.2, label="observed")
                axes[0][c].plot(months, wf_b, color=BASE_COLOR, lw=1.6, ls=":",
                                label="bilinear ERA5-Land")
                axes[1][c].plot(months, am_o, color=OBS_INK, lw=2.2)
                axes[1][c].plot(months, am_b, color=BASE_COLOR, lw=1.6, ls=":")
                drew = True
            axes[0][c].plot(months, wf_m, color=color, lw=1.8, label=label)
            axes[1][c].plot(months, am_m, color=color, lw=1.8)
        axes[0][c].set_title(reg_label, fontsize=10)
        for r in (0, 1):
            axes[r][c].set_xticks(months, ["J", "F", "M", "A", "M", "J", "J", "A",
                                           "S", "O", "N", "D"])
            style_ax(axes[r][c])
    axes[0][0].set_ylabel("wet-day frequency (>= 1 mm) [fraction of days]", fontsize=9)
    axes[1][0].set_ylabel("mean daily precip [mm/day]", fontsize=9)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.suptitle("Seasonal cycle at the rain gauges — occurrence and amount, pooled "
                 "over 139 gauges", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "precip_seasonal_cycle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_pit(data):
    rng_seed = 42
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), sharey=True)
    for ax, (reg, reg_label) in zip(axes, REGIMES):
        for label, color, stem in MODELS:
            b = data[f"{stem}_{reg}"]
            P, obs = b["params"], b["obs"]
            flat = P.reshape(-1, 3)
            o = obs.reshape(-1)
            ok = np.isfinite(o) & np.isfinite(flat[:, 0])
            flat, o = flat[ok], o[ok]
            rng = np.random.default_rng(rng_seed)
            rho, alpha, beta = flat[:, 0], flat[:, 1], flat[:, 2]
            pit = np.empty(o.shape[0])
            dry = o <= DRY_OBS_THRESHOLD_MM
            pit[dry] = rng.uniform(size=int(dry.sum())) * (1.0 - rho[dry])
            wet = ~dry
            cdf = sstats.gamma.cdf(o[wet], a=alpha[wet], scale=1.0 / beta[wet])
            pit[wet] = (1.0 - rho[wet]) + rho[wet] * cdf
            dens, edges = np.histogram(pit, bins=PIT_BINS, range=(0, 1), density=True)
            mid = (edges[:-1] + edges[1:]) / 2
            ax.step(mid, dens, where="mid", color=color, lw=1.8,
                    label=label if reg == "cv" else None)
        ax.axhline(1.0, color="#999999", lw=0.9, ls="--")
        ax.set_xlabel("randomized PIT [–]", fontsize=9)
        ax.set_title(reg_label, fontsize=10)
        style_ax(ax)
    axes[0].set_ylabel("density [–]", fontsize=9)
    fig.legend(fontsize=9, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Randomized PIT at the rain gauges — dry atom mapped to U(0, 1−ρ), "
                 "as in the gridded evaluation", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(FIGS / "precip_pit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGS.mkdir(exist_ok=True)
    data = {f"{stem}_{reg}": load(f"{stem}_{'cv' if reg == 'cv' else '2024'}")
            for _l, _c, stem in MODELS for reg, _ in REGIMES}
    fig_absolute_values(data)
    fig_skill_map(data)
    fig_wetday_scatter(data)
    fig_category_bss(data)
    fig_seasonal(data)
    fig_pit(data)
    print(f"wrote 6 figures to {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
