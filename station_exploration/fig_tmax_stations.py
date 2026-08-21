#!/usr/bin/env python
"""Exploration figures: the selected tmax model at the stations, both cohorts.

Two station sets, same model (``atm+wind-clip60``), same treatment:

  * NBCN — 28 stations, homogenised ``ths200dx``; bundles in the model's
    ``pred_cache/stations_{cv,holdout_2024}.npz`` (written by station_analysis.py).
  * SMN — 147 stations, operational ``tre200dx``; bundles written by
    tmax_smn_inference.py to station_exploration/bundles/. The 28 NBCN sites are
    a subset; per-station MAE agrees to a median of 0.000 °C across the overlap,
    so the two obs sources can share figures.

Figures into station_exploration/figures/:

  tmax_honest_skill.png      per-NBCN-station CRPS skill against the raw bilinear
                             ERA5-Land reference vs a lapse-corrected one
                             (ref - 6.5 K/km x elev_diff) -- the honest number
  tmax_station_skill_map.png the corrected skill on the map, full SMN cohort,
                             NBCN sites ringed (counterpart to the precip map)
  tmax_seasonal_cycle.png    monthly station MAE and bias, both regimes (NBCN)
  tmax_warm_tail.png         error and 90% coverage on each station's hottest
                             decile of days vs all days, both cohorts
  tmax_calibration_split.png z-std per station vs elevation, both cohorts

Usage:  python station_exploration/fig_tmax_stations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import properscoring as ps
from scipy import stats as sstats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

FIGS = HERE / "figures"
MODEL_DIR = (REPO / "CLEAN_trained_models/"
             "lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8/tmax")
NBCN_META = REPO / "station_exploration/cache/station_analysis/_nbcn_stations.csv"
T2M_GLOB = REPO / "datasets/ERA5_Land/max_temperature/t2m_max-2020.nc"
AQUA = "#1baf7a"          # atm+wind-clip60's hue in every cross-model figure
GREY = "#7a7a7a"
LAPSE_K_PER_M = 0.0065    # standard atmosphere moist lapse rate
REGIMES = [("cv", "CV holdout 2020-2023"), ("holdout_2024", "2024 holdout")]


_CELL_ELEV_FIELD = None


def reference_elevation_at_stations(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Elevation the bilinear t2m_max reference effectively 'lives at', per station.

    NOT the repo's ERA5 geopotential file: that field is a heavily smoothed
    orography (it tops out at 2010 m over the whole Alps and reads 648 m at the
    Jungfraujoch cell), fine as the model's input channel — training and
    inference use it consistently — but off by kilometres as a physical height.
    A lapse correction built on it makes the reference WORSE (4.38 -> 5.97 °C
    pooled MAE), which is how the defect was noticed.

    Instead: aggregate the 25 m DEM to cell means on the ERA5-Land t2m grid
    (the grid the reference is interpolated from), then interpolate those cell
    means bilinearly to the stations — the same operation the reference itself
    undergoes, applied to elevation.
    """
    global _CELL_ELEV_FIELD
    import datasets as dsmod
    import xarray as xr

    if _CELL_ELEV_FIELD is None:
        t2m = xr.open_dataset(T2M_GLOB)
        glats, glons = t2m["latitude"].values, t2m["longitude"].values
        dem_da, _ = dsmod.load_high_res_topography(
            str(REPO / "datasets/topo_subset.zarr"))
        dem = dem_da.load()

        dlat = float(np.abs(np.diff(glats)).mean()) / 2
        dlon = float(np.abs(np.diff(glons)).mean()) / 2
        cell_elev = np.full((len(glats), len(glons)), np.nan, dtype=np.float64)
        for i, la in enumerate(glats):
            for j, lo in enumerate(glons):
                corners_lon = np.array([lo - dlon, lo + dlon, lo - dlon, lo + dlon])
                corners_lat = np.array([la - dlat, la - dlat, la + dlat, la + dlat])
                xs, ys = dsmod.wgs84_to_lv95(corners_lon, corners_lat)
                box = dem.sel(x=slice(xs.min(), xs.max()),
                              y=slice(ys.max(), ys.min()))
                if box.size == 0:
                    box = dem.sel(x=slice(xs.min(), xs.max()),
                                  y=slice(ys.min(), ys.max()))
                if box.size:
                    cell_elev[i, j] = float(box.mean())
        _CELL_ELEV_FIELD = xr.DataArray(
            cell_elev, coords={"latitude": glats, "longitude": glons},
            dims=("latitude", "longitude"))
    at_stn = _CELL_ELEV_FIELD.interp(latitude=xr.DataArray(lats, dims="p"),
                                     longitude=xr.DataArray(lons, dims="p"))
    return np.asarray(at_stn.values, dtype=np.float64)


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#dddddd", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#555555", labelsize=9)


def load(regime: str):
    z = np.load(MODEL_DIR / "pred_cache" / f"stations_{regime}.npz",
                allow_pickle=True)
    meta = pd.read_csv(NBCN_META)
    return z, meta


def load_smn(regime: str):
    """SMN bundle + a meta frame in the same column convention as the NBCN CSV."""
    stem = "cv" if regime == "cv" else "2024"
    z = np.load(HERE / "bundles" / f"tmax-smn_{stem}.npz", allow_pickle=True)
    meta = pd.DataFrame({"stn_abbr": z["stn"], "name": z["name"], "lat": z["lat"],
                         "lon": z["lon"], "elev_m": z["elev_m"],
                         "is_nbcn": z["is_nbcn"]})
    return z, meta


def per_station_frame(z, meta, ref_elev, cohort: str = "nbcn") -> pd.DataFrame:
    """One row per station: MAE, bias, CRPS, both skills, z-std, coverage, tail stats."""
    preds, sig, truth = z["preds_C"], z["sigmas_C"], z["truth_C"]
    era5, topo = z["era5_C"], z["topo"]
    dm = z["day_mask"]
    preds, sig, truth, era5 = preds[dm], sig[dm], truth[dm], era5[dm]
    # Temperature falls with height, so the fair reference cools the cell value
    # down from the elevation the reference field lives at to the station's.
    elev_diff = topo[:, 0] - ref_elev
    ref_corr = era5 - LAPSE_K_PER_M * elev_diff[None, :]

    rows = []
    for i in range(preds.shape[1]):
        ok = np.isfinite(truth[:, i]) & np.isfinite(preds[:, i])
        if ok.sum() < 10:
            continue
        t, p, s = truth[ok, i], preds[ok, i], sig[ok, i]
        e5, ec = era5[ok, i], ref_corr[ok, i]
        crps = ps.crps_gaussian(t, mu=p, sig=s)
        z_ = (t - p) / s
        pit = sstats.norm.cdf(z_)
        hot = t >= np.quantile(t, 0.9)
        rows.append({
            "stn": meta["stn_abbr"].iloc[i], "name": meta["name"].iloc[i],
            "cohort": cohort,
            "lat": float(meta["lat"].iloc[i]), "lon": float(meta["lon"].iloc[i]),
            "is_nbcn": bool(meta["is_nbcn"].iloc[i]) if "is_nbcn" in meta else True,
            "elev": float(topo[i, 0]), "elev_diff": float(elev_diff[i]),
            "mae": float(np.mean(np.abs(p - t))),
            "bias": float(np.mean(p - t)),
            "crps": float(np.mean(crps)),
            "mae_ref_raw": float(np.mean(np.abs(e5 - t))),
            "mae_ref_corr": float(np.mean(np.abs(ec - t))),
            "skill_raw": 1.0 - float(np.mean(crps)) / float(np.mean(np.abs(e5 - t))),
            "skill_corr": 1.0 - float(np.mean(crps)) / float(np.mean(np.abs(ec - t))),
            "z_std": float(np.std(z_)),
            "cov90": float(np.mean(np.abs(z_) <= 1.645)),
            "pit_mean": float(np.mean(pit)),
            "pit_std": float(np.std(pit)),
            "mae_tail": float(np.mean(np.abs(p[hot] - t[hot]))),
            "bias_tail": float(np.mean(p[hot] - t[hot])),
            "cov90_tail": float(np.mean(np.abs((t[hot] - p[hot]) / s[hot]) <= 1.645)),
            "n_days": int(ok.sum()),
        })
    return pd.DataFrame(rows)


def pooled(z, ref_elev):
    """Pooled skill (raw + corrected reference) over all station-days."""
    dm = z["day_mask"]
    preds, sig, truth = z["preds_C"][dm], z["sigmas_C"][dm], z["truth_C"][dm]
    era5 = z["era5_C"][dm]
    ref_corr = era5 - LAPSE_K_PER_M * (z["topo"][:, 0] - ref_elev)[None, :]
    ok = np.isfinite(truth) & np.isfinite(preds)
    crps = float(np.mean(ps.crps_gaussian(truth[ok], mu=preds[ok], sig=sig[ok])))
    return {
        "crps": crps,
        "mae": float(np.mean(np.abs(preds[ok] - truth[ok]))),
        "skill_raw": 1.0 - crps / float(np.mean(np.abs(era5[ok] - truth[ok]))),
        "skill_corr": 1.0 - crps / float(np.mean(np.abs(ref_corr[ok] - truth[ok]))),
        "mae_ref_raw": float(np.mean(np.abs(era5[ok] - truth[ok]))),
        "mae_ref_corr": float(np.mean(np.abs(ref_corr[ok] - truth[ok]))),
    }


def fig_honest_skill(frames, pooleds):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    for ax, (regime, title) in zip(axes, REGIMES):
        df = frames[regime].sort_values("elev").reset_index(drop=True)
        y = np.arange(len(df))
        for yy, raw, corr in zip(y, df["skill_raw"], df["skill_corr"]):
            ax.plot([raw, corr], [yy, yy], color="#cccccc", lw=1.2, zorder=1)
        ax.scatter(df["skill_raw"], y, s=26, color=GREY, zorder=2,
                   label="vs raw bilinear ERA5-Land")
        ax.scatter(df["skill_corr"], y, s=30, color=AQUA, zorder=3,
                   label="vs lapse-corrected reference")
        pool = pooleds[regime]
        ax.axvline(pool["skill_raw"], color=GREY, lw=1.0, ls=":")
        ax.axvline(pool["skill_corr"], color=AQUA, lw=1.0, ls=":")
        ax.text(pool["skill_raw"], len(df) - 0.2, f" pooled {pool['skill_raw']:.2f}",
                fontsize=8, color=GREY, ha="left", va="top")
        ax.text(pool["skill_corr"], -0.4, f"pooled {pool['skill_corr']:.2f} ",
                fontsize=8, color=AQUA, ha="right", va="bottom")
        ax.set_yticks(y, [f"{s} {int(e)} m" for s, e in zip(df["stn"], df["elev"])],
                      fontsize=7)
        ax.set_xlabel("CRPS skill (1 − CRPS / MAE of reference) [–]", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.axvline(0, color="#999999", lw=0.8)
        style_ax(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[0].set_ylabel("stations, sorted by elevation", fontsize=9)
    fig.suptitle("atm+wind-clip60 at the NBCN stations — the reference decides the "
                 "skill\nraw bilinear ERA5-Land carries no lapse correction to "
                 "station elevation; correcting it is what an honest baseline does",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "tmax_honest_skill.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_skill_map(smn_frames):
    """Corrected skill on the map, full SMN cohort, NBCN sites ringed."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), sharex=True, sharey=True)
    sc = None
    for ax, (regime, title) in zip(axes, REGIMES):
        df = smn_frames[regime]
        sc = ax.scatter(df["lon"], df["lat"], c=df["skill_corr"], cmap="RdBu",
                        vmin=-0.8, vmax=0.8, s=34, edgecolors="white", lw=0.5,
                        zorder=3)
        ring = df["is_nbcn"].astype(bool)
        ax.scatter(df["lon"][ring], df["lat"][ring], facecolors="none",
                   edgecolors="#333333", s=90, lw=0.8, zorder=4)
        share = float(np.mean(df["skill_corr"] > 0))
        ax.set_title(f"{title}\nmedian {df['skill_corr'].median():+.2f} · stations "
                     f"above zero {share:.0%}", fontsize=9.5)
        ax.set_aspect(1.4)
        ax.set_xlabel("longitude [°E]", fontsize=9)
        style_ax(ax)
    axes[0].set_ylabel("latitude [°N]", fontsize=9)
    fig.colorbar(sc, ax=axes, shrink=0.85,
                 label="CRPS skill vs lapse-corrected ERA5-Land [–]")
    fig.suptitle("atm+wind-clip60 at the 147 SMN stations (operational tre200dx) — "
                 "corrected-reference skill on the map\nblack rings mark the 28 "
                 "NBCN sites", fontsize=11)
    fig.savefig(FIGS / "tmax_station_skill_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_seasonal(zs):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    for k, stat in enumerate(("mae", "bias")):
        ax = axes[k]
        for regime, title in REGIMES:
            z = zs[regime]
            dm = z["day_mask"]
            dates = pd.DatetimeIndex(pd.to_datetime(z["dates"][dm]))
            preds, truth = z["preds_C"][dm], z["truth_C"][dm]
            err = preds - truth
            months = np.arange(1, 13)
            vals = []
            for m in months:
                sel = dates.month == m
                e = err[sel]
                e = e[np.isfinite(e)]
                vals.append(np.mean(np.abs(e)) if stat == "mae" else np.mean(e))
            ls = "-" if regime == "cv" else "--"
            ax.plot(months, vals, ls, color=AQUA, lw=2,
                    label=title if k == 0 else None)
            for m, v in zip(months, vals):
                pass
        ax.set_xticks(np.arange(1, 13),
                      ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_title("station MAE by month [°C]" if stat == "mae"
                     else "station bias by month [°C]", fontsize=10)
        if stat == "bias":
            ax.axhline(0, color="#999999", lw=0.8)
        style_ax(ax)
    fig.legend(fontsize=9, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("atm+wind-clip60 at the NBCN stations — seasonal error structure",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(FIGS / "tmax_seasonal_cycle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_warm_tail(frames, smn_frames):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for ax, (regime, title) in zip(axes, REGIMES):
        df, sf = frames[regime], smn_frames[regime]
        sf_only = sf[~sf["is_nbcn"].astype(bool)]
        ax.scatter(sf_only["mae"], sf_only["mae_tail"], s=16, color="#b8b8b8",
                   zorder=2, label="SMN (operational)")
        ax.scatter(df["mae"], df["mae_tail"], s=30, color=AQUA, zorder=3,
                   label="NBCN (homogenised)")
        lim = max(sf["mae"].max(), sf["mae_tail"].max(),
                  df["mae"].max(), df["mae_tail"].max()) * 1.12
        ax.plot([0, lim], [0, lim], color="#999999", lw=0.8, ls=":")
        for _, r in df.iterrows():
            if r["mae_tail"] > 1.35 * r["mae"] or r["mae_tail"] > 2.2:
                ax.annotate(r["stn"], (r["mae"], r["mae_tail"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points",
                            color="#555555")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("station MAE, all days [°C]", fontsize=9)
        ax.set_ylabel("station MAE, hottest decile [°C]", fontsize=9)
        r_nb = (df["mae_tail"] / df["mae"]).mean()
        r_sm = (sf["mae_tail"] / sf["mae"]).mean()
        ax.set_title(f"{title}\ntail/all MAE ratio NBCN {r_nb:.2f} · SMN {r_sm:.2f} · "
                     f"mean 90% tail coverage {sf['cov90_tail'].mean():.2f}",
                     fontsize=9.5)
        style_ax(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("atm+wind-clip60 at the stations — error on each station's "
                 "hottest decile of days", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "tmax_warm_tail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_station_pit(zs, zsm, smn_frames):
    """Station-level PIT: pooled histogram over the SMN cohort per regime, plus the
    per-station PIT mean against elevation. PIT = Phi((y - mu)/sigma), so this is
    the station counterpart of the gridded qq_calibration figure — a flat
    histogram means the predictive distributions are right at real measurement
    points, mass piled below 0.5 means the model predicts too warm there."""
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    for c, (regime, title) in enumerate(REGIMES):
        # --- pooled histogram, both cohorts ---
        ax = axes[0][c]
        peak = 1.0
        for z, label, color, lw in ((zsm[regime], "SMN (147, operational)", "#0e7a54", 1.9),
                                    (zs[regime], "NBCN (28, homogenised)", AQUA, 1.6)):
            dm = z["day_mask"]
            t, p, s = z["truth_C"][dm], z["preds_C"][dm], z["sigmas_C"][dm]
            ok = np.isfinite(t) & np.isfinite(p) & (s > 0)
            pit = sstats.norm.cdf((t[ok] - p[ok]) / s[ok])
            dens, edges = np.histogram(pit, bins=20, range=(0, 1), density=True)
            mid = (edges[:-1] + edges[1:]) / 2
            peak = max(peak, dens.max())
            ax.step(mid, dens, where="mid", color=color, lw=lw,
                    label=f"{label} · mean {np.mean(pit):.3f}" if c == 0
                          else f"mean {np.mean(pit):.3f}")
        ax.axhline(1.0, color="#999999", lw=0.9, ls="--")
        ax.set_ylim(0, peak * 1.08)
        ax.set_xlabel("PIT [–]", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="lower center")
        style_ax(ax)
        # --- per-station PIT mean vs elevation ---
        ax = axes[1][c]
        sf = smn_frames[regime]
        sf_only = sf[~sf["is_nbcn"].astype(bool)]
        nb = sf[sf["is_nbcn"].astype(bool)]
        ax.scatter(sf_only["elev"], sf_only["pit_mean"], s=16, color="#b8b8b8",
                   zorder=2, label="SMN (operational)")
        ax.scatter(nb["elev"], nb["pit_mean"], s=30, color=AQUA, zorder=3,
                   label="NBCN sites")
        ax.axhline(0.5, color="#999999", lw=0.9, ls="--")
        ax.text(sf["elev"].max(), 0.5, " unbiased", fontsize=8, color="#777777",
                va="bottom", ha="right")
        for _, r in sf.iterrows():
            # Label only the extremes; a 0.06 cut drowned the lowland cluster.
            if abs(r["pit_mean"] - 0.5) > 0.16 or (r["is_nbcn"]
                                                   and abs(r["pit_mean"] - 0.5) > 0.12):
                ax.annotate(r["stn"], (r["elev"], r["pit_mean"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points",
                            color="#555555")
        ax.set_xlabel("station elevation [m]", fontsize=9)
        style_ax(ax)
    axes[0][0].set_ylabel("density [–]", fontsize=9)
    axes[1][0].set_ylabel("per-station PIT mean [–]", fontsize=9)
    axes[1][0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("atm+wind-clip60 at the stations — PIT calibration\n"
                 "top: pooled PIT histogram per regime; bottom: per-station PIT mean "
                 "(< 0.5 = predicted too warm at that station)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "tmax_station_pit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_calibration_split(frames, smn_frames):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    for ax, (regime, title) in zip(axes, REGIMES):
        df, sf = frames[regime], smn_frames[regime]
        sf_only = sf[~sf["is_nbcn"].astype(bool)]
        ax.scatter(sf_only["elev"], sf_only["z_std"], s=16, color="#b8b8b8",
                   zorder=2, label="SMN (operational)")
        ax.scatter(df["elev"], df["z_std"], s=30, color=AQUA, zorder=3,
                   label="NBCN (homogenised)")
        ax.axhline(1.0, color="#999999", lw=0.9, ls="--")
        ax.text(df["elev"].max(), 1.0, " calibrated", fontsize=8, color="#777777",
                va="bottom", ha="right")
        for _, r in df.iterrows():
            if r["z_std"] > 1.12 or r["z_std"] < 0.62:
                ax.annotate(r["stn"], (r["elev"], r["z_std"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points",
                            color="#555555")
        ax.set_xlabel("station elevation [m]", fontsize=9)
        ax.set_title(title, fontsize=10)
        style_ax(ax)
    axes[0].set_ylabel("z-std (errors / predicted σ) [–]", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("atm+wind-clip60 at the stations — spread calibration by "
                 "elevation\nz-std > 1: over-confident (σ too small); < 1: "
                 "under-confident", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGS / "tmax_calibration_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGS.mkdir(exist_ok=True)
    zs, frames, pooleds, smn_frames = {}, {}, {}, {}
    meta = pd.read_csv(NBCN_META)
    ref_elev = reference_elevation_at_stations(meta["lat"].to_numpy(),
                                               meta["lon"].to_numpy())
    for regime, _ in REGIMES:
        z, meta = load(regime)
        zs[regime] = z
        frames[regime] = per_station_frame(z, meta, ref_elev)
        pooleds[regime] = pooled(z, ref_elev)
        p = pooleds[regime]
        print(f"[nbcn {regime}] pooled MAE {p['mae']:.3f} °C | CRPS {p['crps']:.3f} | "
              f"skill raw {p['skill_raw']:.3f} (ref MAE {p['mae_ref_raw']:.2f}) | "
              f"skill corrected {p['skill_corr']:.3f} (ref MAE {p['mae_ref_corr']:.2f})")

    _z0, smn_meta = load_smn("cv")
    smn_ref_elev = reference_elevation_at_stations(smn_meta["lat"].to_numpy(),
                                                   smn_meta["lon"].to_numpy())
    zsm = {}
    for regime, _ in REGIMES:
        z, smn_meta = load_smn(regime)
        zsm[regime] = z
        smn_frames[regime] = per_station_frame(z, smn_meta, smn_ref_elev,
                                               cohort="smn")
        p = pooled(z, smn_ref_elev)
        print(f"[smn  {regime}] pooled MAE {p['mae']:.3f} °C | CRPS {p['crps']:.3f} | "
              f"skill raw {p['skill_raw']:.3f} (ref MAE {p['mae_ref_raw']:.2f}) | "
              f"skill corrected {p['skill_corr']:.3f} (ref MAE {p['mae_ref_corr']:.2f})")

    fig_honest_skill(frames, pooleds)
    fig_skill_map(smn_frames)
    fig_seasonal(zs)
    fig_warm_tail(frames, smn_frames)
    fig_calibration_split(frames, smn_frames)
    fig_station_pit(zs, zsm, smn_frames)
    for regime, _ in REGIMES:
        both = pd.concat([frames[regime], smn_frames[regime]], ignore_index=True)
        both.to_csv(FIGS / f"tmax_per_station_{regime}.csv", index=False)
    print(f"wrote 6 figures + 2 CSVs (cohort column: nbcn/smn) to {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
