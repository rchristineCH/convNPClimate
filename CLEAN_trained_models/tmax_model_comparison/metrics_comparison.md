# tmax model comparison — CV holdout vs 2024 holdout

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

**CV 2020-2023**: each of the 5 folds predicts only the contiguous block of days it was held out from during training, so every day is scored by the one model that never saw it. Not a fold ensemble.

**2024 holdout**: a year no fold saw. All folds predict every day and are combined by Gaussian moment matching (μ = mean μₖ, σ² = mean σₖ² + var μₖ), with inputs normalised using the training-frozen statistics.

The **Δ** column is 2024 − CV: the cost of moving from held-out days inside the training span to a genuinely unseen year.

> `Skill vs ERA5` = 1 − CRPS / CRPS_ref, where CRPS_ref is the MAE of the bilinear-ERA5 surface-tmax reference at the MeteoSwiss target points. That reference is **the same for every model here** — checked on this run, not assumed: each model's CRPS_ref is recovered from its own report as `crps_degC / (1 − skill_score)` and the models are required to agree to 0.001 °C. They do: CV 2020-2023 **2.560 °C** (spread 8.9e-16), 2024 holdout **2.543 °C** (spread 8.9e-16). Skill is therefore a rescaling of CRPS by a shared constant, so it ranks the models exactly as CRPS does and is safe to compare across the surface and atmospheric runs.

ERA5 reference MAE (the skill denominator): CV 2020-2023 2.560 °C; 2024 holdout 2.543 °C.

## MAE (°C)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 1.239 | 1.206 | -0.033 |
| atm-clip60 | 1.088 | 1.047 | -0.041 |
| atm+wind | 1.308 | 1.252 | -0.056 |
| atm+wind-clip60 | 1.055 | 1.029 | -0.026 |
| atm+sfcanchors | 1.203 | 1.174 | -0.029 |
| sfc | 1.415 | 1.475 | +0.060 |
| sfc-nogeo | 1.449 | 1.493 | +0.045 |

## RMSE (°C)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 1.652 | 1.664 | +0.011 |
| atm-clip60 | 1.462 | 1.419 | -0.043 |
| atm+wind | 1.734 | 1.711 | -0.022 |
| atm+wind-clip60 | 1.414 | 1.388 | -0.026 |
| atm+sfcanchors | 1.614 | 1.620 | +0.006 |
| sfc | 1.866 | 1.974 | +0.109 |
| sfc-nogeo | 1.899 | 1.983 | +0.083 |

## Bias (°C)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 0.003 | 0.119 | +0.116 |
| atm-clip60 | -0.140 | -0.021 | +0.119 |
| atm+wind | -0.097 | 0.069 | +0.165 |
| atm+wind-clip60 | 0.079 | 0.231 | +0.152 |
| atm+sfcanchors | -0.107 | 0.046 | +0.153 |
| sfc | -0.033 | 0.133 | +0.166 |
| sfc-nogeo | 0.010 | 0.115 | +0.105 |

## CRPS (°C)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 0.919 | 0.909 | -0.010 |
| atm-clip60 | 0.787 | 0.757 | -0.030 |
| atm+wind | 0.960 | 0.935 | -0.025 |
| atm+wind-clip60 | 0.762 | 0.741 | -0.021 |
| atm+sfcanchors | 0.897 | 0.889 | -0.008 |
| sfc | 1.031 | 1.077 | +0.046 |
| sfc-nogeo | 1.056 | 1.090 | +0.035 |

## Skill vs ERA5

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 0.641 | 0.642 | +0.001 |
| atm-clip60 | 0.693 | 0.702 | +0.010 |
| atm+wind | 0.625 | 0.632 | +0.007 |
| atm+wind-clip60 | 0.702 | 0.708 | +0.006 |
| atm+sfcanchors | 0.650 | 0.650 | +0.001 |
| sfc | 0.597 | 0.576 | -0.021 |
| sfc-nogeo | 0.588 | 0.571 | -0.016 |

## z-std (→1)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 0.926 | 0.838 | -0.088 |
| atm-clip60 | 1.038 | 0.887 | -0.151 |
| atm+wind | 1.006 | 0.860 | -0.146 |
| atm+wind-clip60 | 1.078 | 0.899 | -0.179 |
| atm+sfcanchors | 0.975 | 0.864 | -0.111 |
| sfc | 0.991 | 0.963 | -0.027 |
| sfc-nogeo | 0.947 | 0.911 | -0.036 |

## Coverage@90 (→0.9)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 92.5% | 94.4% | +1.9pp |
| atm-clip60 | 89.8% | 93.2% | +3.5pp |
| atm+wind | 90.8% | 94.0% | +3.2pp |
| atm+wind-clip60 | 88.6% | 93.0% | +4.4pp |
| atm+sfcanchors | 91.3% | 93.6% | +2.4pp |
| sfc | 90.5% | 91.0% | +0.5pp |
| sfc-nogeo | 91.9% | 92.3% | +0.5pp |

## MACE (→0)

| Model | CV 2020-2023 | 2024 holdout | Δ |
|---|---|---|---|
| atm | 0.067 | 0.101 | +0.035 |
| atm-clip60 | 0.021 | 0.058 | +0.037 |
| atm+wind | 0.042 | 0.090 | +0.048 |
| atm+wind-clip60 | 0.012 | 0.050 | +0.039 |
| atm+sfcanchors | 0.060 | 0.096 | +0.035 |
| sfc | 0.035 | 0.045 | +0.010 |
| sfc-nogeo | 0.048 | 0.057 | +0.010 |

## Per-fold CV spread (MAE °C)

Each fold scores a different, disjoint sub-period, so spread here is as much about which months a fold held out as about the fold's model quality.

| Model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | range |
|---|---|---|---|---|---|---|
| atm | 1.301 | 1.239 | 1.310 | 1.180 | 1.164 | 0.146 |
| atm-clip60 | 1.042 | 1.105 | 1.157 | 1.079 | 1.056 | 0.115 |
| atm+wind | 1.288 | 1.366 | 1.309 | 1.445 | 1.131 | 0.314 |
| atm+wind-clip60 | 1.044 | 1.020 | 1.178 | 1.059 | 0.974 | 0.204 |
| atm+sfcanchors | 1.299 | 1.150 | 1.283 | 1.063 | 1.220 | 0.236 |
| sfc | 1.355 | 1.504 | 1.489 | 1.361 | 1.369 | 0.149 |
| sfc-nogeo | 1.376 | 1.521 | 1.470 | 1.417 | 1.461 | 0.145 |
