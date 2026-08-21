#!/usr/bin/env python
"""What each output head emits, and how a precip forecast becomes a metric.

**16_tmax_distribution** — a plain Gaussian with its mu and sigma marked: what the tmax
head emits for one point on one day, a distribution rather than a number. The two values
are real model output (the median-sigma point of 15 July 2024, from the holdout bundle),
not round numbers chosen for the drawing.

**17_precip_wet_dry_day** — four forecasts from the same head, one per dial: rho for
occurrence, beta for intensity, alpha for the shape of a wet day. Parameters are
illustrative but anchored on the fitted climatological body; per-day values would need the
gitignored precip bundle.

**18_precip_frequency_flow** — the flow chart for the draw itself: uniform, one decision
on rho, zero or a Gamma amount, repeated over every point and day to give the frequency.
The measured outcome sits at the foot: 0.362 this way against an observed 0.356, versus
0.571 if the draw is skipped and rho > 0.5 is thresholded instead.

Writes 16_tmax_distribution, 17_precip_wet_dry_day and 18_precip_frequency_flow
({png,svg}) under docs/diagrams. Run:
    python scripts/gen_output_distribution_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import evaluate as ev  # noqa: E402
METRICS = (ROOT / "CLEAN_trained_models"
           / "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8"
           / "precip" / "eval_precip_metrics.json")
TMAX_RUN = "clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8"
PARAM_PERCENTILES = (Path(__file__).resolve().parent.parent / "docs"
                     / "diagrams" / "precip_param_percentiles.json")
DIAGRAMS = ROOT / "docs" / "diagrams"
OUT_TMAX = DIAGRAMS / "16_tmax_distribution"
OUT_PRECIP_DAY = DIAGRAMS / "17_precip_wet_dry_day"
OUT_PRECIP_FLOW = DIAGRAMS / "18_precip_frequency_flow"

BG_INK, GAMMA_INK, GAUSS_INK = "#2a78d6", "#eb6834", "#8e5bd0"
OBS_INK, MUTED = "#333333", "#777777"
# Schematic palette, shared with the other docs/diagrams figures.
INK, DATA, RED = "#131c24", "#2560a6", "#c0392b"
PANEL, LINE, LINE_DARK = "#f6f8fa", "#c7d0d8", "#9aa7b2"


def fit_gamma(mean: float, q98: float) -> tuple[float, float]:
    """Gamma (shape, scale) reproducing a wet-day mean and 98th percentile.

    Two constraints, two parameters: scale follows from the mean once the shape is known,
    so only the shape has to be solved for.
    """
    def residual(shape):
        return stats.gamma.ppf(0.98, shape, scale=mean / shape) - q98

    # Bracketed from 0.2, not from ~0: with the mean held fixed the scale blows up as the
    # shape falls, and the 98th percentile turns over rather than growing monotonically,
    # so a bracket reaching to zero can hold the same sign at both ends.
    shape = optimize.brentq(residual, 0.2, 50.0)
    return shape, mean / shape




def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e9e9e9", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#555555", labelsize=9)


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .svg")


def tmax_figure(target_day: str = "2024-07-15") -> None:
    """A plain Gaussian with mu and sigma: what the tmax head emits per point per day.

    The object the head produces is a distribution, not a number, and that is the whole
    claim — so the figure is one curve with its two parameters marked and nothing else.
    The values are real model output rather than round numbers: the median-sigma point of
    the chosen day, read from the 2024 holdout bundle.
    """
    md = ROOT / "CLEAN_trained_models" / TMAX_RUN / "tmax"
    path = ev.bundle_cache_path(md, 2024)
    if not path.exists():
        raise SystemExit(f"missing {path} — run: python evaluate.py --model-dir {md} "
                         "--eval-year 2024")
    B = ev.load_bundle(path, md)
    dates = pd.DatetimeIndex(np.asarray(B.dates))
    d = int(np.argmin(np.abs(dates - pd.Timestamp(target_day))))

    mu, sd, truth = B.preds_c[d].reshape(-1), B.sigmas_c[d].reshape(-1), B.truths_c[d].reshape(-1)
    ok = np.isfinite(mu) & np.isfinite(sd) & np.isfinite(truth) & (sd > 0)
    mu, sd, truth = mu[ok], sd[ok], truth[ok]
    # One point: the median predicted sigma for that day, so mu and sigma are real
    # model output rather than round numbers chosen for the drawing.
    i = int(np.argsort(sd)[sd.size // 2])
    m, sg = float(mu[i]), float(sd[i])

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.linspace(m - 4 * sg, m + 4 * sg, 800)
    peak = float(stats.norm.pdf(m, m, sg))
    ax.plot(x, stats.norm.pdf(x, m, sg), color=GAUSS_INK, lw=2.2, zorder=4)
    ax.fill_between(x, stats.norm.pdf(x, m, sg), color=GAUSS_INK, alpha=0.12, lw=0,
                    zorder=2)

    # mu: a line at the centre. sigma: the distance from the centre to the inflection.
    ax.plot([m, m], [0, peak], color=GAUSS_INK, lw=1.4, zorder=5)
    ax.text(m, peak * 1.04, f"$\\mu$ = {m:.1f} °C", ha="center", fontsize=11,
            color=GAUSS_INK)
    y_sig = float(stats.norm.pdf(m + sg, m, sg))
    ax.annotate("", xy=(m + sg, y_sig), xytext=(m, y_sig),
                arrowprops=dict(arrowstyle="<->", color=OBS_INK, lw=1.2))
    ax.text(m + sg * 0.55, y_sig * 1.08, f"$\\sigma$ = {sg:.2f} °C", ha="left",
            va="bottom", fontsize=11, color=OBS_INK)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, peak * 1.18)
    ax.set_xlabel("daily maximum temperature (°C)", color=OBS_INK)
    ax.set_ylabel("probability density  (per °C)", color=OBS_INK)
    style(ax)

    fig.suptitle("The Gaussian head: one mean and one spread, per point per day",
                 fontsize=12.5, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, OUT_TMAX)


def precip_day_figure() -> None:
    """How a Bernoulli-Gamma forecast behaves, one dial at a time.

    Plotted as the exceedance curve P(Y >= t) rather than a density, which is what makes
    the whole law legible in one frame. The density is unbounded at zero whenever
    alpha < 1 — the usual case here — so a density plot is a spike at the origin and needs
    a log axis plus a second axis for the atom, and still shows little. The exceedance
    curve is bounded in [0, 1], needs no log, and carries both parts of the mixture at
    once: it starts at rho, so the drop from 1 down to the curve *is* the dry-day
    probability, and its decay is the wet body.

    One panel per parameter, each holding the other two fixed, so the effect of a dial is
    visible within a panel instead of by comparing across panels. Values are measured
    percentiles of the model's own output (``scripts/sample_precip_params.py``).
    """
    pct = json.loads(PARAM_PERCENTILES.read_text())
    qs = pct["percentiles"]

    def q(name, want):
        return float(pct[name]["pct"][qs.index(want)])

    a_med = q("alpha", 50)
    rho_mid = q("rho", 50)
    shades = ("#9ec6ea", "#3d85c6", "#0b3d66")

    # (title, what is held fixed, [(label, rho, alpha, wet-day mean)]). The held-fixed
    # values are spelled out rather than described, since a reader cannot check "same body"
    # against anything.
    MEAN = 5.0                       # wet-day mean, mm, where it is the thing held fixed
    panels = [
        (r"$\rho$ — does it rain at all",
         rf"fixed:  $\alpha$ = {a_med:.2f},  $\beta$ = {a_med / MEAN:.3f}"
         rf"  (wet-day mean {MEAN:g} mm)",
         [(rf"$\rho$ = {q('rho', p):.2f}  (p{p})", q("rho", p), a_med, MEAN)
          for p in (25, 50, 75)]),
        (r"$\beta$ — how much, if it does",
         rf"fixed:  $\rho$ = {rho_mid:.2f},  $\alpha$ = {a_med:.2f}",
         [(rf"$\beta$ = {a_med / mm:.3f}  (mean {mm:g} mm)", rho_mid, a_med, mm)
          for mm in (1.0, 5.0, 20.0)]),
        (r"$\alpha$ — the shape of a wet day",
         rf"fixed:  $\rho$ = {rho_mid:.2f};  $\beta$ follows, holding the mean at "
         rf"{MEAN:g} mm",
         [(rf"$\alpha$ = {a:.2f},  $\beta$ = {a / MEAN:.3f}", rho_mid, a, MEAN)
          for a in (q("alpha", 25), 1.0, q("alpha", 99))]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), sharey=True)
    t = np.linspace(0, 30, 900)
    for ax, (title, note, curves) in zip(axes, panels):
        for (label, rho, alpha, mean), color in zip(curves, shades):
            scale = mean / alpha
            exc = rho * stats.gamma.sf(t, alpha, scale=scale)
            ax.plot(t, exc, color=color, lw=2.4, zorder=4, label=label)
            # The atom: everything between 1 and rho is the probability of exactly zero.
            ax.plot([0], [rho], marker="o", ms=7, color=color, zorder=6)
            ax.plot([0, 0], [rho, 1.0], color=color, lw=1.4, ls=":", alpha=0.8, zorder=3)

        ax.axhline(1.0, color=OBS_INK, lw=1, ls="-", alpha=0.35, zorder=2)
        ax.axvline(1.0, color=MUTED, lw=1, ls="--", zorder=2)
        ax.set_xlim(-1.4, 30)
        ax.set_ylim(0, 1.06)
        ax.set_xlabel("threshold  t  (mm)", color=OBS_INK)
        ax.set_title(title, fontsize=12, color=INK, pad=16)
        ax.text(0.5, 1.012, note, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9.5, color=MUTED)
        ax.legend(fontsize=9.5, frameon=False, loc="upper right", labelcolor=OBS_INK)
        style(ax)

    axes[0].set_ylabel(r"$P(Y \geq t)$", color=OBS_INK, fontsize=12)
    # Said once, on the panel where the dry mass varies most.
    axes[0].annotate("the gap up to 1 is\nthe dry-day mass\n" r"$P(Y=0) = 1-\rho$",
                     xy=(0, 0.60), xytext=(4.2, 0.60), fontsize=9.5, color=OBS_INK,
                     va="center",
                     arrowprops=dict(arrowstyle="->", color=OBS_INK, lw=0.9))
    axes[1].text(2.0, 0.86, "wet-day threshold,\n1 mm", fontsize=9, color=MUTED)

    fig.suptitle("Bernoulli-Gamma mixture — probability of at least t mm precipitation",
                 fontsize=14, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, OUT_PRECIP_DAY)


def precip_flow_figure() -> None:
    """The two routes from the head's three numbers to a wet-day frequency, side by side.

    Left, the probabilistic route the evaluation uses: every point contributes its own
    exceedance probability, and a draw from the head is that same law sampled. Right, the
    deterministic one: compare rho against a fixed 0.5, call the point wet or dry outright,
    and take the conditional mean as the amount. Same head, same data, two frequencies —
    0.362 against 0.571, with the observed 0.356 underneath.

    Both definitions are reproduced from ``eval_precip.compare_threshold_vs_probabilistic``
    rather than paraphrased, and the numbers are read from the metrics it wrote.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

    payload = json.loads(METRICS.read_text())
    m, tvp = payload["overall"], payload["threshold_vs_probabilistic"]
    shape, scale = fit_gamma(m["SDII_obs_mm"], m["P98_obs_mm"])
    rho = 1.0 - m["dryday_freq_obs"]
    prob = tvp["wetday_freq_pred"]["probabilistic"]
    thr = tvp["wetday_freq_pred"]["threshold"]
    obs = tvp["_obs_reference"]["wetday_freq_obs"]

    fig = plt.figure(figsize=(13.2, 8.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(cx, cy, w, h, text, fc="#eef4fb", ec=DATA, fs=11, tc=INK):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.004,rounding_size=0.010",
                                    fc=fc, ec=ec, lw=1.3, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)

    def arrow(x0, y0, x1, y1, color=LINE_DARK):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12, color=color, lw=1.2, zorder=1))

    def diamond(cx, cy, w, h, text, ec):
        ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
                              (cx - w / 2, cy)], closed=True, fc="#fff6e8", ec=ec,
                             lw=1.3, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=12, color=INK, zorder=3)

    LEFT, RIGHT = 0.255, 0.745

    ax.text(0.5, 0.968, "Two routes to a wet-day frequency, from the same three numbers",
            ha="center", fontsize=15.5, fontweight="bold", color=INK)

    # 1. the head, shared by both routes
    box(0.5, 0.888, 0.60, 0.088,
        "Bernoulli-Gamma head — per grid point per day\n"
        r"$f(y)\;=\;(1-\rho)\,\delta_0(y)\;+\;\rho\;\mathrm{Gamma}(y;\alpha,\beta)$"
        "\n"
        rf"here  $\rho={rho:.2f}$,  $\alpha={shape:.2f}$,  $\beta={1 / scale:.3f}$"
        r"  mm$^{-1}$", fs=11)

    # One tint per route, same geometry: neither is the default, so neither should look
    # like the annotation on the other. Both stop above the shared footer.
    for x0, fc, ec in ((0.018, "#f4f8fd", "#c3d3e2"), (0.512, "#fdf5f4", "#e2c3bf")):
        ax.add_patch(FancyBboxPatch((x0, 0.268), 0.470, 0.552,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    fc=fc, ec=ec, lw=1.2, zorder=0))
    arrow(0.44, 0.843, LEFT + 0.03, 0.808)
    arrow(0.56, 0.843, RIGHT - 0.03, 0.808)

    ax.text(LEFT, 0.790, "probabilistic — use the whole distribution", ha="center",
            fontsize=12.5, fontweight="bold", color=DATA)
    ax.text(RIGHT, 0.790, r"deterministic — threshold $\rho$ at 0.5", ha="center",
            fontsize=12.5, fontweight="bold", color=RED)

    # 2. each route: an entry step, one decision, two outcomes
    for centre, entry, question, ec, dry_label, wet_text, wet_note, wet_fc, wet_ec in (
        (LEFT, r"draw  $u \sim \mathcal{U}(0,\,1)$", r"$u < \rho$ ?", "#b9791c",
         r"$P(y=0)=1-\rho$", r"$y \sim \mathrm{Gamma}(\alpha,\beta)$",
         r"$P(y>0)=\rho$", "#eef4fb", DATA),
        (RIGHT, "no draw at all", r"$\rho \geq 0.5$ ?", RED,
         "counted dry", r"$\hat{y} = \alpha/\beta$", "counted wet", "#fdeceb", RED),
    ):
        box(centre, 0.735, 0.27, 0.056, entry, fc=PANEL, ec=LINE_DARK, fs=11)
        arrow(centre, 0.707, centre, 0.678)
        diamond(centre, 0.632, 0.185, 0.088, question, ec)

        arrow(centre - 0.098, 0.632, centre - 0.088, 0.632)
        box(centre - 0.145, 0.632, 0.110, 0.054,
            r"$y = 0$" if centre == LEFT else r"$\hat{y} = 0$",
            fc="#f2f2f2", ec=MUTED, fs=12)
        ax.text(centre - 0.145, 0.592, dry_label, ha="center", va="top", fontsize=9.5,
                color=MUTED)

        arrow(centre + 0.098, 0.632, centre + 0.088, 0.632)
        box(centre + 0.145, 0.632, 0.125, 0.054, wet_text, fc=wet_fc, ec=wet_ec, fs=10.5)
        ax.text(centre + 0.145, 0.592, wet_note, ha="center", va="top", fontsize=9.5,
                color=wet_ec if wet_ec != MUTED else MUTED)

        for side in (-0.145, 0.145):
            arrow(centre + side, 0.558, centre + side, 0.520)
        ax.add_patch(FancyArrowPatch((centre - 0.145, 0.520), (centre + 0.145, 0.520),
                                     arrowstyle="-", color=LINE_DARK, lw=1.2, zorder=1))
        arrow(centre, 0.520, centre, 0.487)

    # 3. what each route computes
    box(LEFT, 0.425, 0.40, 0.100,
        "every point contributes its own\nexceedance probability\n"
        r"$\hat{f}_{\mathrm{wet}} = \mathrm{mean}\;P(y \geq t)$" "\n"
        r"$P(y \geq t) = \rho\,\left[\,1 - F_{\Gamma}(t;\alpha,\beta)\,\right]$", fs=10.5)
    box(RIGHT, 0.425, 0.40, 0.100,
        "every point counts as a whole day,\nwet or dry, nothing in between\n"
        r"$\hat{f}_{\mathrm{wet}} = \mathrm{mean}\;\mathbf{1}\left[\,\rho \geq 0.5\,\right]$"
        "\n" r"(no threshold $t$ enters at all)", fc="#fdeceb", ec=RED, fs=10.5)

    # 4. the two answers
    for cx, value, color in ((LEFT, prob, DATA), (RIGHT, thr, RED)):
        arrow(cx, 0.373, cx, 0.345, color=color)
        ax.text(cx, 0.305, f"{value:.3f}", ha="center", fontsize=27, color=color,
                family="monospace")
        ax.text(cx, 0.283, f"{value - obs:+.3f} vs observed", ha="center", fontsize=10.5,
                color=MUTED)

    ax.plot([0.14, 0.86], [0.238, 0.238], color=LINE, lw=1)
    ax.text(0.5, 0.198, f"observed (MeteoSwiss):  {obs:.3f}", ha="center", fontsize=13,
            color=INK, family="monospace")
    ax.text(0.5, 0.118,
            r"A point at $\rho = 0.6$ is dry two days in five. The left route charges it"
            " 0.6 of a wet day; the right one counts it\nwet every time, and the error "
            r"does not cancel because $\rho$ is not symmetric about 0.5 over the domain."
            "\nThe same shortcut drags SDII the other way — "
            rf"{tvp['SDII_pred_mm']['threshold']:.2f} mm against "
            rf"{tvp['SDII_pred_mm']['probabilistic']:.2f} mm, observed "
            rf"{tvp['_obs_reference']['SDII_obs_mm']:.2f} mm.",
            ha="center", fontsize=10.5, color=OBS_INK)

    save(fig, OUT_PRECIP_FLOW)


if __name__ == "__main__":
    precip_day_figure()
    precip_flow_figure()
    tmax_figure()
