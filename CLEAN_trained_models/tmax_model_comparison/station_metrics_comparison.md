# tmax model comparison at NBCN stations — CV holdout vs 2024 holdout

28 NBCN stations evaluated (excluded: ANT); station-days CV 2020-2023 40,908, 2024 holdout 10,248.

## Models

| Label | Run |
|---|---|
| atm | `clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8` |
| atm-clip60 | `lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8` |
| atm+wind | `clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8` |
| atm+wind-clip60 | `lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8` |
| atm+sfcanchors | `sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8` |
| sfc | `baseline__tmax_sfc_flat_y2020-2023_e30f5_b8` |
| sfc-nogeo | `baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo` |

> **Station MAE is not the gridded MAE.** These score 28 NBCN point observations; `metrics_comparison.md` scores the full 88 800-point MeteoSwiss gridded analysis. Point observations carry representativeness error the smoothed gridded product does not, so station MAE (~1.6 °C) sits well above gridded MAE (~1.2-1.45 °C) for every model. Compare models *within* this table; never compare a number here against one there.

> **Pooled station skill is inflated.** The ERA5 baseline is raw bilinear `t2m_max` with no lapse-rate correction, so its MAE is large (CV 2020-2023 4.378 °C, 2024 holdout 4.348 °C) — huge at alpine stations, far smaller in the lowlands. Pooled `skill_overall` therefore mostly measures *elevation* downscaling; `skill_lowland` (stations below 1000 m) and `skill_median` are the honest numbers. The baseline itself is identical for every model (checked: spread 0.0e+00 °C), so the model *ranking* is fair even though the level is flattered.

## Station MAE (°C)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 1.595 | 1.605 | +0.010 |
| atm-clip60 | 1.437 | 1.382 | -0.055 |
| atm+wind | 1.671 | 1.655 | -0.016 |
| atm+wind-clip60 | 1.384 | 1.353 | -0.031 |
| atm+sfcanchors | 1.583 | 1.576 | -0.006 |
| sfc | 1.731 | 1.783 | +0.053 |
| sfc-nogeo | 1.741 | 1.773 | +0.032 |

## Station CRPS (°C)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 1.172 | 1.183 | +0.011 |
| atm-clip60 | 1.051 | 0.992 | -0.059 |
| atm+wind | 1.225 | 1.213 | -0.012 |
| atm+wind-clip60 | 1.016 | 0.970 | -0.046 |
| atm+sfcanchors | 1.169 | 1.167 | -0.003 |
| sfc | 1.252 | 1.293 | +0.041 |
| sfc-nogeo | 1.255 | 1.278 | +0.023 |

## Skill vs ERA5 (pooled)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 0.732 | 0.728 | -0.004 |
| atm-clip60 | 0.760 | 0.772 | +0.012 |
| atm+wind | 0.720 | 0.721 | +0.001 |
| atm+wind-clip60 | 0.768 | 0.777 | +0.009 |
| atm+sfcanchors | 0.733 | 0.732 | -0.001 |
| sfc | 0.714 | 0.703 | -0.011 |
| sfc-nogeo | 0.713 | 0.706 | -0.007 |

## Skill, lowland <1000 m

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 0.654 | 0.641 | -0.013 |
| atm-clip60 | 0.703 | 0.719 | +0.016 |
| atm+wind | 0.631 | 0.629 | -0.001 |
| atm+wind-clip60 | 0.713 | 0.726 | +0.013 |
| atm+sfcanchors | 0.660 | 0.650 | -0.011 |
| sfc | 0.605 | 0.575 | -0.030 |
| sfc-nogeo | 0.608 | 0.584 | -0.023 |

## Skill, per-station median

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 0.728 | 0.719 | -0.009 |
| atm-clip60 | 0.758 | 0.764 | +0.007 |
| atm+wind | 0.692 | 0.699 | +0.006 |
| atm+wind-clip60 | 0.761 | 0.764 | +0.004 |
| atm+sfcanchors | 0.728 | 0.730 | +0.002 |
| sfc | 0.724 | 0.711 | -0.014 |
| sfc-nogeo | 0.705 | 0.702 | -0.003 |

## Mean bias (°C)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 0.173 | 0.319 | +0.145 |
| atm-clip60 | -0.077 | 0.073 | +0.149 |
| atm+wind | 0.025 | 0.220 | +0.195 |
| atm+wind-clip60 | 0.147 | 0.300 | +0.153 |
| atm+sfcanchors | 0.021 | 0.197 | +0.176 |
| sfc | -0.025 | 0.175 | +0.199 |
| sfc-nogeo | -0.016 | 0.114 | +0.130 |

## Mean |per-station bias| (°C)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 0.469 | 0.515 | +0.046 |
| atm-clip60 | 0.468 | 0.418 | -0.051 |
| atm+wind | 0.514 | 0.490 | -0.024 |
| atm+wind-clip60 | 0.389 | 0.424 | +0.035 |
| atm+sfcanchors | 0.464 | 0.486 | +0.022 |
| sfc | 0.533 | 0.567 | +0.034 |
| sfc-nogeo | 0.612 | 0.579 | -0.032 |

## Systematic share of MAE

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 0.294 | 0.321 | +0.027 |
| atm-clip60 | 0.326 | 0.302 | -0.024 |
| atm+wind | 0.308 | 0.296 | -0.011 |
| atm+wind-clip60 | 0.281 | 0.313 | +0.032 |
| atm+sfcanchors | 0.293 | 0.308 | +0.015 |
| sfc | 0.308 | 0.318 | +0.010 |
| sfc-nogeo | 0.351 | 0.327 | -0.025 |

## ERA5 baseline MAE (°C)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 4.378 | 4.348 | -0.030 |
| atm-clip60 | 4.378 | 4.348 | -0.030 |
| atm+wind | 4.378 | 4.348 | -0.030 |
| atm+wind-clip60 | 4.378 | 4.348 | -0.030 |
| atm+sfcanchors | 4.378 | 4.348 | -0.030 |
| sfc | 4.378 | 4.348 | -0.030 |
| sfc-nogeo | 4.378 | 4.348 | -0.030 |
