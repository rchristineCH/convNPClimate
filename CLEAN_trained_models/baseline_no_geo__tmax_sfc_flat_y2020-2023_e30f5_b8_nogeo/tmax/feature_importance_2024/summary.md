# Feature importance — tmax (2024)

- Target points: 88800  |  days scored: 366  |  folds: 5
- Baseline (all channels): MAE 1.493 | NLL 2.087 | CRPS 1.090  [degC]

## Top channels (|importance|)

| rank | PFI (mae) | LIME (pred) |
|---|---|---|
| 1 | surface temp (+6.959) | cos_time (+0.768) |
| 2 | cos_time (+0.393) | sin_time (+0.238) |
| 3 | sin_time (+0.055) | surface temp (+0.215) |
| 4 | lon (+0.000) | lat (+0.012) |
| 5 | lat (+0.000) | lon (-0.009) |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

## By variable

- **PFI** (mae): surface temp (+7.000), scaffold (+0.404)
- **LIME** (pred_tmax_degC): scaffold (+1.010), surface temp (+0.215)

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