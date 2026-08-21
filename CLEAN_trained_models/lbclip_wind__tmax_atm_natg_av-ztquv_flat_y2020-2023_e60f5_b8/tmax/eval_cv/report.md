# Evaluation report — tmax (native coarse atmospheric)

> Cross-validation holdout over the **training span 2020-2023**. Each of the 5 folds predicts **only** the contiguous block it was held out from during training; the blocks tile the 1461-day axis, so **every day is predicted exactly once, by the one model that never saw it**. This is not a fold ensemble — per-fold numbers are in `per_fold`. These are training-span days scored out-of-sample, *not* an unseen year.

- Model: `CLEAN_trained_models/lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8/tmax`
- Encoder: `flat` | folds evaluated: 5 | target points: 88,800 | holdout days: 1461
- Regime: `cv_holdout` | prediction mode: `fold_holdout` | valid points: 46,718

## Headline metrics

| Metric | Value |
|---|---|
| MAE (°C) | 1.055 |
| RMSE (°C) | 1.384 |
| Bias (°C) | 0.079 |
| CRPS (°C) | 0.762 |
| Skill vs bilinear ERA5 | 0.702 |

## Error diagnostics (correlation of MAE with …)

| Driver | Pearson r |
|---|---|
| predicted σ (calibration in space) | 0.548 |
| altitude | 0.319 |
| mTPI (ridge/valley) | -0.005 |
| day of year (seasonality) | -0.001 |

## Uncertainty calibration

| Metric | Value | Ideal |
|---|---|---|
| PIT mean | 0.484 | 0.5 |
| PIT std | 0.287 | 0.289 |
| z std | 1.078 | 1.0 |
| coverage @50% | 51.7% | 50% |
| coverage @90% | 88.6% | 90% |
| KS p-value | 0 | >0.05 |
| MACE (reliability) | 0.012 | 0 |
| reliability @90% | 87.8% | 90% |
| reliability @95% | 92.3% | 95% |

## Per-fold holdout blocks

Each fold scores only the days it was held out from, so these are disjoint sub-periods, not repeated measurements of the same days.

| Fold | Days | Period | Epoch | MAE (°C) | RMSE (°C) | CRPS (°C) | Skill |
|---|---|---|---|---|---|---|---|
| 0 | 292 | 2020-01-01 … 2020-10-18 | 56 | 1.044 | 1.374 | 0.755 | 0.716 |
| 1 | 292 | 2020-10-19 … 2021-08-06 | 54 | 1.020 | 1.324 | 0.732 | 0.711 |
| 2 | 292 | 2021-08-07 … 2022-05-25 | 52 | 1.178 | 1.547 | 0.860 | 0.673 |
| 3 | 292 | 2022-05-26 … 2023-03-13 | 59 | 1.059 | 1.374 | 0.765 | 0.687 |
| 4 | 293 | 2023-03-14 … 2023-12-31 | 58 | 0.974 | 1.258 | 0.700 | 0.724 |

## Figures

### Per-pixel MAE / RMSE / bias

![Per-pixel MAE / RMSE / bias](error_maps.png)

### Per-pixel CRPS

![Per-pixel CRPS](crps_map.png)

### Skill vs bilinear-ERA5 baseline

![Skill vs bilinear-ERA5 baseline](skill_map.png)

### Per-pixel uncertainty vs error

![Per-pixel uncertainty vs error](uncertainty_vs_error.png)

### Altitude vs error

![Altitude vs error](altitude_vs_error.png)

### mTPI vs error

![mTPI vs error](mtpi_vs_error.png)

### Day-of-year vs error

![Day-of-year vs error](doy_vs_error.png)

### Q-Q / PIT calibration

![Q-Q / PIT calibration](qq_calibration.png)

### Reliability diagram

![Reliability diagram](reliability_diagram.png)
