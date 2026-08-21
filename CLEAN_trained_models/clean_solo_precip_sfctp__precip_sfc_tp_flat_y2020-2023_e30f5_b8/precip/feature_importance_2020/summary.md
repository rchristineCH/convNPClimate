# Feature importance — precip (2020)

- Target points: 98050  |  days scored: 366  |  folds: 5
- Baseline (all channels): PR_MAE 2.177 | PR_NLL 1.431  [mm]

## Top channels (|importance|)

| rank | PFI (pr_mae) |
|---|---|
| 1 | tp (+3.512) |
| 2 | data (+0.254) |
| 3 | cos_time (+0.136) |
| 4 | sin_time (+0.042) |
| 5 | lat (+0.000) |
| 6 | lon (+0.000) |
| 7 | elevation (+0.000) |
| 8 |  |

## By variable

- **PFI** (pr_mae): tp (+3.591), data (+0.261), scaffold (+0.107)

## By hour

- **PFI** (pr_mae): - (+3.648)

## By level

- **PFI** (pr_mae): 

## Figures

- `pfi_channel.png`
- `pfi_variable.png`
- `pfi_level.png`
- `pfi_hour.png`
- `pfi_heatmap_*.png` (level × hour)