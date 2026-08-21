# Evaluation report — tmax (native coarse atmospheric)

> Genuine **holdout year 2024**: no fold saw any 2024 day. Inputs are normalised with the **training-frozen** statistics in `manifest.json` — never re-fit on 2024. All 5 folds predict all 366 days and are combined as a Gaussian-mixture moment match (mu = mean mu_k; sigma^2 = mean sigma_k^2 + var mu_k), so sigma carries both the within-fold and the between-fold uncertainty (see `sigma_decomposition`).

- Model: `CLEAN_trained_models/clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8/tmax`
- Encoder: `flat` | folds evaluated: 5 | target points: 88,800 | holdout days: 366
- Regime: `holdout_year_2024` | prediction mode: `fold_ensemble` | valid points: 46,718

## Headline metrics

| Metric | Value |
|---|---|
| MAE (°C) | 1.206 |
| RMSE (°C) | 1.614 |
| Bias (°C) | 0.119 |
| CRPS (°C) | 0.909 |
| Skill vs bilinear ERA5 | 0.642 |

## Error diagnostics (correlation of MAE with …)

| Driver | Pearson r |
|---|---|
| predicted σ (calibration in space) | -0.071 |
| altitude | 0.171 |
| mTPI (ridge/valley) | 0.003 |
| day of year (seasonality) | 0.176 |

## Uncertainty calibration

| Metric | Value | Ideal |
|---|---|---|
| PIT mean | 0.484 | 0.5 |
| PIT std | 0.242 | 0.289 |
| z std | 0.838 | 1.0 |
| coverage @50% | 65.8% | 50% |
| coverage @90% | 94.4% | 90% |
| KS p-value | 0 | >0.05 |
| MACE (reliability) | 0.101 | 0 |
| reliability @90% | 93.9% | 90% |
| reliability @95% | 96.3% | 95% |

## Ensemble spread decomposition

| Component | °C |
|---|---|
| within-fold (mean σₖ) | 1.946 |
| between-fold (spread of μₖ) | 0.542 |
| total (moment-matched) | 2.044 |

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
