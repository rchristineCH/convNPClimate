# Feature importance — precip (2023)

- Target points: 98050  |  days scored: 365  |  folds: 5
- Baseline (all channels): PR_MAE 2.467 | PR_NLL 1.622  [mm]

## Top channels (|importance|)

| rank | PFI (pr_mae) |
|---|---|
| 1 | q700_18 (+0.249) |
| 2 | q500_18 (+0.152) |
| 3 | q850_18 (+0.115) |
| 4 | t500_18 (+0.106) |
| 5 | t700_18 (+0.071) |
| 6 | q700_06 (+0.067) |
| 7 | cos_time (+0.066) |
| 8 | t1000_18 (+0.062) |

## By variable

- **PFI** (pr_mae): q (+11.053), t (+5.366), z (+1.412), u (+0.642), v (+0.517), scaffold (+0.076)

## By hour

- **PFI** (pr_mae): 18 (+2.095), 06 (+1.393), 15 (+1.028), 00 (+0.959), 12 (+0.432)

## By level

- **PFI** (pr_mae): 500 (+1.083), 300 (+0.622), 700 (+0.606), 925 (+0.527), 850 (+0.456), 1000 (+0.430)

## Figures

- `pfi_channel.png`
- `pfi_variable.png`
- `pfi_level.png`
- `pfi_hour.png`
- `pfi_heatmap_*.png` (level × hour)