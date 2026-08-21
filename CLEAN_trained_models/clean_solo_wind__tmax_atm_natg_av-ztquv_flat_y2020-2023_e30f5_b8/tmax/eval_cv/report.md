# Evaluation report — tmax (native coarse atmospheric)

> Cross-validation holdout over the **training span 2020-2023**. Each of the 5 folds predicts **only** the contiguous block it was held out from during training; the blocks tile the 1461-day axis, so **every day is predicted exactly once, by the one model that never saw it**. This is not a fold ensemble — per-fold numbers are in `per_fold`. These are training-span days scored out-of-sample, *not* an unseen year.

- Model: `CLEAN_trained_models/clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8/tmax`
- Encoder: `flat` | folds evaluated: 5 | target points: 88,800 | holdout days: 1461
- Regime: `cv_holdout` | prediction mode: `fold_holdout` | valid points: 46,718

## Headline metrics

| Metric | Value |
|---|---|
| MAE (°C) | 1.308 |
| RMSE (°C) | 1.697 |
| Bias (°C) | -0.097 |
| CRPS (°C) | 0.960 |
| Skill vs bilinear ERA5 | 0.625 |

## Error diagnostics (correlation of MAE with …)

| Driver | Pearson r |
|---|---|
| predicted σ (calibration in space) | -0.290 |
| altitude | 0.156 |
| mTPI (ridge/valley) | 0.005 |
| day of year (seasonality) | -0.018 |

## Uncertainty calibration

| Metric | Value | Ideal |
|---|---|---|
| PIT mean | 0.515 | 0.5 |
| PIT std | 0.271 | 0.289 |
| z std | 1.006 | 1.0 |
| coverage @50% | 56.9% | 50% |
| coverage @90% | 90.8% | 90% |
| KS p-value | 0 | >0.05 |
| MACE (reliability) | 0.042 | 0 |
| reliability @90% | 90.2% | 90% |
| reliability @95% | 93.8% | 95% |

## Per-fold holdout blocks

Each fold scores only the days it was held out from, so these are disjoint sub-periods, not repeated measurements of the same days.

| Fold | Days | Period | Epoch | MAE (°C) | RMSE (°C) | CRPS (°C) | Skill |
|---|---|---|---|---|---|---|---|
| 0 | 292 | 2020-01-01 … 2020-10-18 | 11 | 1.288 | 1.673 | 0.948 | 0.644 |
| 1 | 292 | 2020-10-19 … 2021-08-06 | 9 | 1.366 | 1.719 | 0.994 | 0.607 |
| 2 | 292 | 2021-08-07 … 2022-05-25 | 27 | 1.309 | 1.716 | 0.956 | 0.636 |
| 3 | 292 | 2022-05-26 … 2023-03-13 | 3 | 1.445 | 1.838 | 1.068 | 0.563 |
| 4 | 293 | 2023-03-14 … 2023-12-31 | 26 | 1.131 | 1.462 | 0.833 | 0.671 |

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
