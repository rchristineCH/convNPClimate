# Feature importance — tmax (2024)

- Target points: 88800  |  days scored: 366  |  folds: 5
- Baseline (all channels): MAE 1.475 | NLL 2.076 | CRPS 1.077  [degC]

## Top channels (|importance|)

| rank | PFI (mae) | LIME (pred) |
|---|---|---|
| 1 | surface temp (+7.245) | cos_time (+0.668) |
| 2 | cos_time (+0.257) | sin_time (+0.141) |
| 3 | sin_time (+0.081) | surface temp (-0.106) |
| 4 | lat (+0.000) | lon (-0.015) |
| 5 | lon (+0.000) | lat (-0.013) |
| 6 | elevation (+0.000) | elevation (+0.011) |
| 7 |  |  |
| 8 |  |  |

## By variable

- **PFI** (mae): surface temp (+7.278), scaffold (+0.287)
- **LIME** (pred_tmax_degC): scaffold (+0.792), surface temp (-0.106)

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