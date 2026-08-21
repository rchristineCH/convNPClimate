# Station exploration — the three selected models at real measurement points

Explorative study on branch `station_explore`: what station-level verification can say
about the report's three selected models — `atm+wind-clip60` (tmax, 28 NBCN stations +
147 SMN stations) and `precip-FT-CRPS` / `precip-SFC-TP` (first-ever precip station
verification, 139 SMN rain gauges). Four alternative chapter drafts, written in the
report's voice, to pick from (or combine):

| Draft | Angle | Figures used |
|---|---|---|
| `draft_A_verification_at_stations.md` | One short cross-variable chapter: both verdicts + the honest reference, in a page | `stations_summary` |
| `draft_B_honest_reference.md` | The reference decides the skill: lapse-corrected tmax baseline, precip skill geography | `tmax_honest_skill`, `precip_gauge_skill_map` |
| `draft_C_structure_in_time.md` | When errors happen: seasonal cycle, warm tail, calibration split, PIT | `tmax_seasonal_cycle`, `precip_seasonal_cycle` |
| `draft_D_stations_and_data.md` | Shortest possible: the station sets, what they provide, how they are considered | none (one table) |

## Key numbers (all reproducible from the scripts here)

tmax — SMN cohort (`atm+wind-clip60`, 147 stations, operational `tre200dx`; day-label
lag 0 verified vs TmaxD, median r 0.999):
- MAE 1.406 °C (CV) / 1.385 °C (2024); corrected-reference skill pooled **0.52 / 0.54**,
  median per station +0.52 / +0.56, **99 % of stations above zero** in both regimes.
- Overlap regression vs the NBCN bundles at the 28 shared sites: median |ΔMAE| 0.000 °C,
  max 0.08 °C (SMA) — operational vs homogenised is a non-issue over 2020–2024.
- Tail/all MAE ratio 0.85 / 0.76; tail 90 % coverage 0.74 / 0.85; median z-std 1.20 → 1.04.
- Station-level PIT (`tmax_station_pit.png`; per-station `pit_mean` in the CSVs): pooled
  PIT mean 0.476 (CV) / 0.445 (2024) — the warm drift, stronger at points than on the
  grid (0.45 vs the gridded 0.45 on 2024); the CV histogram is U-shaped (single-fold σ
  over-confident in the tails), 2024 nearly flat with a mild downslope. Per station the
  PIT mean falls with elevation: lowland stations sit above 0.5 (predicted too cold),
  high-alpine ones at 0.25–0.35 (predicted too warm) — an elevation-dependent bias the
  pooled numbers average away.

tmax — NBCN cohort (`atm+wind-clip60`, 28 stations, homogenised `ths200dx`):
- MAE 1.381 °C (CV) / 1.352 °C (2024); CRPS 1.015 / 0.969.
- Raw bilinear ERA5-Land reference: MAE 4.38 °C → pooled skill 0.77/0.78 (**inflated**).
- Lapse-corrected reference (6.5 K/km over DEM-derived cell elevations): MAE 2.30 °C →
  honest pooled skill **0.56 / 0.58**, positive at all 28 stations in both regimes
  (min +0.12 SAE CV, +0.22 CHM 2024).
- Seasonal: summer MAE floor ~1.15 °C, December 1.8 °C with bias +0.8/+0.9 °C.
- Hottest decile per station: MAE *lower* than all-day (ratio 0.83 CV / 0.73 2024);
  tail 90 % coverage 0.77 CV → 0.89 2024.
- Calibration: CV single-fold σ over-confident everywhere (MER 2.15, SIO/ALT 1.9);
  2024 ensemble mostly 0.8–1.35; SAE stays over-confident (1.80 → 1.61).

precip (139 SMN gauges, `rre150d0` = 06–06 UTC totals; RhiresD's own window; day-label
lag 0 verified, median r 0.982; reference = w0606 rebuild, bilinear):
- Categorical skill (per-gauge RPSS over the five intensity categories, the map/summary
  metric): positive at **100 % of gauges**, both models, both regimes — median SFC-TP
  **+0.50 / +0.45**, FT-CRPS **+0.47 / +0.43**.
- Millimetre MAE skill (the honest counterpoint, pooled): SFC-TP +0.050 / +0.042
  (53 % of gauges); FT-CRPS −0.023 / −0.063 (39 % / 35 %).
- Geography: the RPSS margin is thinnest along the north-eastern pre-Alps (where the
  millimetre MAE falls behind interpolation) and widest on the Plateau + south; stable
  across regimes.
- Wet-day frequency: observed 0.327/0.369; bilinear ERA5-Land 0.495/0.554 (drizzle);
  FT-CRPS 0.350/0.410; SFC-TP 0.371/0.419.
- Category Brier skill (median over gauges, vs ERA5-Land): positive through 40–80 mm
  (SFC-TP 0.27 CV / 0.19 2024; FT-CRPS 0.17 / 0.14); ≥80 mm ≈ 0.
- PIT: gentle downslope both runs; FT-CRPS last-bin spike (under-dispersed tail).

## Caveats every use of these numbers must carry

1. The SMN gauges feed RhiresD: independent of the models, **not** of the gridded truth.
2. SMN daily totals are operational, not homogenised; 19/158 stations have no gauge;
   139 kept at ≥95 % coverage 2020–2024.
3. Use `rre150d0` (06–06 UTC), never `rka150d0` (00–00).
4. The repo's ERA5 geopotential file is a smoothed orography (Jungfraujoch cell 648 m,
   Alpine max 2010 m) — fine as the model input channel, unusable for physical lapse
   corrections. Use DEM cell means (see `fig_tmax_stations.reference_elevation_at_stations`).

## Reproduce

```bash
python station_exploration/download_smn_precip.py       # network; rre150d0 + tre200dx
python station_exploration/check_alignment.py           # lag verdicts (precip + tmax)
python station_analysis.py --model-dir CLEAN_trained_models/lbclip_wind__*/tmax \
    --cache-dir station_exploration/cache/station_analysis            # + --eval-year 2024
python station_exploration/tmax_smn_inference.py        # GPU, ~30 s/regime, bundles/
python station_exploration/precip_station_inference.py  # GPU, ~2 min/model, bundles/
python station_exploration/fig_tmax_stations.py
python station_exploration/fig_precip_stations.py
python station_exploration/fig_summary.py
```
