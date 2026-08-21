# Feature importance — precip (2021)

- Target points: 98050  |  days scored: 365  |  folds: 5
- Baseline (all channels): PR_MAE 2.322 | PR_NLL 1.555  [mm]

## Top channels (|importance|)

| rank | PFI (pr_mae) |
|---|---|
| 1 | tp (+3.870) |
| 2 | data (+0.251) |
| 3 | cos_time (+0.218) |
| 4 | sin_time (+0.035) |
| 5 | lat (+0.000) |
| 6 | lon (+0.000) |
| 7 | elevation (+0.000) |
| 8 |  |

## By variable

- **PFI** (pr_mae): tp (+3.664), scaffold (+0.253), data (+0.228)

## By hour

- **PFI** (pr_mae): - (+3.737)

## By level

- **PFI** (pr_mae): 

## Figures

- `pfi_channel.png`
- `pfi_variable.png`
- `pfi_level.png`
- `pfi_hour.png`
- `pfi_heatmap_*.png` (level × hour)