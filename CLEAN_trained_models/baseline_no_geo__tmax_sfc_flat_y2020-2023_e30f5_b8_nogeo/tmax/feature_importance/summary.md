# Feature importance — tmax (2022)

- Target points: 88800  |  days scored: 365  |  folds: 5
- Baseline (all channels): MAE 1.356 | NLL 2.012 | CRPS 0.995  [degC]

## Top channels (|importance|)

| rank | PFI (mae) | LIME (pred) |
|---|---|---|
| 1 | surface temp (+7.945) | surface temp (+0.397) |
| 2 | cos_time (+0.456) | cos_time (+0.279) |
| 3 | sin_time (+0.087) | sin_time (+0.227) |
| 4 | lon (+0.000) | lat (+0.007) |
| 5 | lat (+0.000) | lon (-0.005) |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

## By variable

- **PFI** (mae): surface temp (+8.039), scaffold (+0.489)
- **LIME** (pred_tmax_degC): scaffold (+0.508), surface temp (+0.397)

## By hour

- **PFI** (mae): 
- **LIME** (pred_tmax_degC): 

## By level

- **PFI** (mae): 
- **LIME** (pred_tmax_degC): 

## Figures

- `pfi_channel.png`
- `pfi_variable.png`
- `pfi_level.png`
- `pfi_hour.png`
- `pfi_heatmap_*.png` (level × hour)
- `lime_channel.png`
- `lime_variable.png`
- `lime_level.png`
- `lime_hour.png`
- `lime_heatmap_*.png` (level × hour)