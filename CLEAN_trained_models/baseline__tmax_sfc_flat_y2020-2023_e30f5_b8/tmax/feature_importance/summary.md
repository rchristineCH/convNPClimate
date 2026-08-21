# Feature importance — tmax (2022)

- Target points: 88800  |  days scored: 365  |  folds: 5
- Baseline (all channels): MAE 1.328 | NLL 1.988 | CRPS 0.972  [degC]

## Top channels (|importance|)

| rank | PFI (mae) | LIME (pred) |
|---|---|---|
| 1 | surface temp (+8.255) | cos_time (+0.309) |
| 2 | cos_time (+0.309) | surface temp (+0.222) |
| 3 | sin_time (+0.110) | sin_time (+0.137) |
| 4 | lat (+0.000) | elevation (+0.007) |
| 5 | lon (+0.000) | lat (-0.007) |
| 6 | elevation (+0.000) | lon (-0.007) |
| 7 |  |  |
| 8 |  |  |

## By variable

- **PFI** (mae): surface temp (+8.115), scaffold (+0.360)
- **LIME** (pred_tmax_degC): scaffold (+0.439), surface temp (+0.222)

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