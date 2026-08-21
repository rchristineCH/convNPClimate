#!/usr/bin/env python3
"""Visualize the model input channels: surface (left) vs atmospheric (right).

Left  — the 6 surface-config channels, each with a small drawn icon.
Right — an emagram (temperature vs log-pressure) marking the 6 pressure levels,
        with the per-level variables (z, t, q) sampled at 5 hours = 90 channels,
        plus the 5 scaffold channels (lat, lon, cos_time, sin_time, elevation) = 95.

Outputs docs/diagrams/12_input_channels.{png,svg}. Run:
    python scripts/gen_input_channels_figure.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
from matplotlib.ticker import NullLocator

OUT = Path(__file__).resolve().parent.parent / "docs" / "diagrams" / "12_input_channels"

INK, MUTED = "#131c24", "#57697a"
DATA, MODEL, TEAL, AMBER = "#2560a6", "#2c8b58", "#0d7a84", "#b9791c"
RED, BLUE = "#c0392b", "#2f79c0"
PANEL, LINE = "#f6f8fa", "#c7d0d8"

# ---------------------------------------------------------------- icons
def ic_thermo(ax, x, y, s=1.0, c=RED):
    w = 0.010 * s
    ax.add_patch(Rectangle((x - w, y - 0.006), 2 * w, 0.055 * s, color=c, zorder=5))
    ax.add_patch(Circle((x, y - 0.006), 0.018 * s, color=c, zorder=5))
    ax.add_patch(Circle((x, y - 0.006), 0.008 * s, color="white", zorder=6))

def ic_grid(ax, x, y, s=1.0, c=DATA):
    r = 0.024 * s
    ax.add_patch(Circle((x, y), r, fill=False, ec=c, lw=1.6, zorder=5))
    for dy in (-0.011 * s, 0.011 * s):
        ax.plot([x - r * 0.93, x + r * 0.93], [y + dy, y + dy], color=c, lw=1.1, zorder=5)
    ax.plot([x, x], [y - r, y + r], color=c, lw=1.1, zorder=5)

def ic_sun(ax, x, y, s=1.0, c=AMBER):
    ax.add_patch(Circle((x, y), 0.014 * s, color=c, zorder=5))
    for a in range(0, 360, 45):
        a = np.radians(a)
        ax.plot([x + 0.020 * s * np.cos(a), x + 0.030 * s * np.cos(a)],
                [y + 0.020 * s * np.sin(a), y + 0.030 * s * np.sin(a)],
                color=c, lw=1.4, zorder=5, solid_capstyle="round")

def ic_mountain(ax, x, y, s=1.0, c=MODEL):
    p = [(x - 0.028 * s, y - 0.022 * s), (x, y + 0.026 * s), (x + 0.028 * s, y - 0.022 * s)]
    ax.add_patch(Polygon(p, closed=True, color=c, zorder=5))
    cap = [(x - 0.009 * s, y + 0.006 * s), (x, y + 0.026 * s), (x + 0.009 * s, y + 0.006 * s)]
    ax.add_patch(Polygon(cap, closed=True, color="white", zorder=6))

def ic_drop(ax, x, y, s=1.0, c=BLUE):
    ax.add_patch(Circle((x, y - 0.006 * s), 0.016 * s, color=c, zorder=5))
    tip = [(x - 0.012 * s, y + 0.002 * s), (x, y + 0.030 * s), (x + 0.012 * s, y + 0.002 * s)]
    ax.add_patch(Polygon(tip, closed=True, color=c, zorder=5))

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(13.5, 7.4), dpi=150)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.14,
                      left=0.04, right=0.97, top=0.99, bottom=0.06)

# ===== LEFT: surface channels =====
axL = fig.add_subplot(gs[0, 0]); axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
axL.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0.01,rounding_size=0.02",
              fc=PANEL, ec=DATA, lw=1.6, zorder=0))
axL.text(0.07, 0.925, "Surface input", fontsize=15, fontweight="bold", color=INK)
axL.text(0.07, 0.885, "--use-surface  ·  6 channels", fontsize=10.5, color=MUTED, family="monospace")

rows = [
    (ic_thermo,   "data",      "ERA5-Land t2m_max  (z-scored)"),
    (ic_grid,     "lat",       "normalized latitude  [0,1]"),
    (ic_grid,     "lon",       "normalized longitude  [0,1]"),
    (ic_sun,      "cos_time",  "cos(2π · (doy − 1) / 365)"),
    (ic_sun,      "sin_time",  "sin(2π · (doy − 1) / 365)"),
    (ic_mountain, "elevation", "geopotential / g  (z-scored)"),
]
y = 0.80
for icon, name, desc in rows:
    icon(axL, 0.13, y, s=1.0)
    axL.text(0.24, y + 0.012, name, fontsize=12.5, fontweight="bold", family="monospace", color=INK, va="center")
    axL.text(0.24, y - 0.028, desc, fontsize=10, color=MUTED, va="center")
    y -= 0.125
axL.text(0.24, 0.055 + 0.012, "dropped with --no-geopotential (→ 5 channels)",
         fontsize=9, color=AMBER, style="italic", family="monospace")

# ===== RIGHT: TRUE atmospheric emagram from ERA5 (2023 mean, 12 UTC) =====
import glob, xarray as xr
GRAV = 9.80665

def _mean_profile(var):
    """Domain- and time-mean vertical profile for one pressure-level variable."""
    f = sorted(glob.glob(f"datasets/ERA5_PressureLevels/{var}/{var}_pl-2023-12.nc"))[0]
    ds = xr.open_dataset(f)
    dv = list(ds.data_vars)[0]
    lv = next(c for c in ds.coords if c.lower() in
              ("pressure_level", "level", "plev", "isobaricinhpa"))
    prof = ds[dv].mean(dim=[d for d in ds[dv].dims if d != lv]).sortby(lv, ascending=False)
    return prof[lv].values.astype(float), prof.values.astype(float)

pl, t_prof = _mean_profile("t"); t_prof = t_prof - 273.15    # K -> °C
_,  q_prof = _mean_profile("q"); q_prof = q_prof * 1000.0     # kg/kg -> g/kg
_,  z_prof = _mean_profile("z"); z_prof = z_prof / GRAV       # m²/s² -> geopotential height (m)
levels = [int(round(p)) for p in pl]

# atmospheric input CARD — same rounded-panel style as the surface card, emagram embedded
pos = gs[0, 1].get_position(fig)
axCard = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height])
axCard.set_xlim(0, 1); axCard.set_ylim(0, 1); axCard.axis("off")
axCard.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.94,
                 boxstyle="round,pad=0.01,rounding_size=0.02", fc=PANEL, ec=DATA, lw=1.6, zorder=0))
axCard.text(0.07, 0.925, "Atmospheric input", fontsize=15, fontweight="bold", color=INK)
axCard.text(0.07, 0.885, "--use-atmospheric  ·  95 channels", fontsize=10.5, color=MUTED, family="monospace")
axCard.text(0.07, 0.845, "z·t·q × 6 levels × 5 hours = 90  +  5 scaffold  =  95   ·   ERA5 2023 mean, 12 UTC",
            fontsize=8, color=MUTED, family="monospace")

# scaffold row: same icons as the surface panel, so the two cards read together
axCard.text(0.07, 0.800, "scaffold:", fontsize=8, color=MUTED, family="monospace", va="center")
_sx = 0.19
for _icon, _name in ((ic_grid, "lat"), (ic_grid, "lon"), (ic_sun, "cos_time"),
                     (ic_sun, "sin_time"), (ic_mountain, "elevation")):
    _icon(axCard, _sx, 0.800, s=0.65)
    axCard.text(_sx + 0.026, 0.800, _name, fontsize=8, color=MUTED,
                family="monospace", va="center")
    _sx += 0.026 + 0.0100 * len(_name) + 0.022 + 0.0195

_pl, _pr, _pt, _pb = 0.15, 0.14, 0.33, 0.12
axR = fig.add_axes([pos.x0 + _pl * pos.width, pos.y0 + _pb * pos.height,
                    (1 - _pl - _pr) * pos.width, (1 - _pt - _pb) * pos.height])
axR.set_facecolor(PANEL)
axR.set_yscale("log"); axR.set_ylim(1050, 280)
axR.set_yticks(levels); axR.set_yticklabels([str(l) for l in levels], fontsize=10)
axR.yaxis.set_minor_locator(NullLocator())
axR.set_ylabel("pressure  (hPa)", fontsize=11, color=INK)
axR.set_xlabel("temperature  t  (°C)", fontsize=11.5, color=RED, fontweight="bold")
axR.set_xlim(-55, 20)
axR.tick_params(axis="x", colors=RED); axR.tick_params(axis="y", colors=MUTED)
for s in axR.spines.values():
    s.set_color(LINE)
for p in levels:
    axR.axhline(p, color=LINE, lw=0.7, ls=(0, (3, 3)), zorder=0)

# temperature line (bottom x-axis)
lt, = axR.plot(t_prof, levels, "-o", color=RED, lw=2.3, ms=9, mec="white", mew=1.3,
               zorder=4, label="t — temperature")

# specific humidity on a top x-axis
axQ = axR.twiny()
axQ.set_xlim(0, max(q_prof) * 1.2 + 0.3)
axQ.set_xlabel("specific humidity  q  (g/kg)", fontsize=11.5, color=BLUE, fontweight="bold")
axQ.tick_params(axis="x", colors=BLUE)
for s in axQ.spines.values():
    s.set_color(LINE)
lq, = axQ.plot(q_prof, levels, "-s", color=BLUE, lw=2.1, ms=8, mec="white", mew=1.2,
               zorder=4, label="q — specific humidity")

# geopotential height on a right-hand secondary y-axis
axZ = axR.twinx()
axZ.set_yscale("log"); axZ.set_ylim(axR.get_ylim())
axZ.set_yticks(levels); axZ.set_yticklabels([f"{h:,.0f}" for h in z_prof], fontsize=9.5, color=MODEL)
axZ.yaxis.set_minor_locator(NullLocator())
axZ.set_ylabel("geopotential height  z  (m)", fontsize=11.5, color=MODEL, fontweight="bold")
axZ.tick_params(axis="y", colors=MODEL)
for s in axZ.spines.values():
    s.set_color(LINE)

# two-line legend + channel arithmetic
axR.legend(handles=[lt, lq], loc="upper right", fontsize=10, frameon=True,
           framealpha=0.96, edgecolor=LINE)
# (channel arithmetic + period now live in the card header above the plot)

for ext in ("png", "svg"):
    fig.savefig(f"{OUT}.{ext}", facecolor="white", bbox_inches="tight")
    print(f"  wrote {OUT.name}.{ext}")
print("done")
