# Evaluation report — tmax (native coarse atmospheric)

> Cross-validation holdout over the **training span 2020-2023**. Each of the 5 folds predicts **only** the contiguous block it was held out from during training; the blocks tile the 1461-day axis, so **every day is predicted exactly once, by the one model that never saw it**. This is not a fold ensemble — per-fold numbers are in `per_fold`. These are training-span days scored out-of-sample, *not* an unseen year.

- Model: `CLEAN_trained_models/lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8/tmax`
- Encoder: `flat` | folds evaluated: 5 | target points: 88,800 | holdout days: 1461
- Regime: `cv_holdout` | prediction mode: `fold_holdout` | valid points: 46,718

## Headline metrics

| Metric | Value |
|---|---|
| MAE (°C) | 1.088 |
| RMSE (°C) | 1.426 |
| Bias (°C) | -0.140 |
| CRPS (°C) | 0.787 |
| Skill vs bilinear ERA5 | 0.693 |

## Error diagnostics (correlation of MAE with …)

| Driver | Pearson r |
|---|---|
| predicted σ (calibration in space) | 0.546 |
| altitude | 0.305 |
| mTPI (ridge/valley) | -0.002 |
| day of year (seasonality) | 0.013 |

## Uncertainty calibration

| Metric | Value | Ideal |
|---|---|---|
| PIT mean | 0.529 | 0.5 |
| PIT std | 0.280 | 0.289 |
| z std | 1.038 | 1.0 |
| coverage @50% | 53.5% | 50% |
| coverage @90% | 89.8% | 90% |
| KS p-value | 0 | >0.05 |
| MACE (reliability) | 0.021 | 0 |
| reliability @90% | 89.0% | 90% |
| reliability @95% | 93.1% | 95% |

## Per-fold holdout blocks

Each fold scores only the days it was held out from, so these are disjoint sub-periods, not repeated measurements of the same days.

| Fold | Days | Period | Epoch | MAE (°C) | RMSE (°C) | CRPS (°C) | Skill |
|---|---|---|---|---|---|---|---|
| 0 | 292 | 2020-01-01 … 2020-10-18 | 51 | 1.042 | 1.380 | 0.759 | 0.715 |
| 1 | 292 | 2020-10-19 … 2021-08-06 | 59 | 1.105 | 1.434 | 0.802 | 0.683 |
| 2 | 292 | 2021-08-07 … 2022-05-25 | 59 | 1.157 | 1.522 | 0.843 | 0.679 |
| 3 | 292 | 2022-05-26 … 2023-03-13 | 45 | 1.079 | 1.391 | 0.769 | 0.685 |
| 4 | 293 | 2023-03-14 … 2023-12-31 | 58 | 1.056 | 1.364 | 0.762 | 0.700 |

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
