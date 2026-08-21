# Feature importance — precip (2024)

- Target points: 98050  |  days scored: 366  |  folds: 5
- Baseline (all channels): PR_MAE 2.707 | PR_NLL 1.760  [mm]

## Top channels (|importance|)

| rank | PFI (pr_mae) | LIME (pred) |
|---|---|---|
| 1 | tp (+3.569) | tp (-2.538) |
| 2 | cos_time (+0.212) | sin_time (+0.042) |
| 3 | data (+0.189) | data (+0.030) |
| 4 | sin_time (+0.021) | cos_time (+0.024) |
| 5 | lat (+0.000) | lon (+0.005) |
| 6 | lon (+0.000) | elevation (+0.002) |
| 7 | elevation (+0.000) | lat (-0.001) |
| 8 |  |  |

## By variable

- **PFI** (pr_mae): tp (+3.516), data (+0.216), scaffold (+0.215)
- **LIME** (pred_precip_mm): tp (-2.538), scaffold (+0.072), data (+0.030)

## By hour

- **PFI** (pr_mae): - (+3.558)
- **LIME** (pred_precip_mm): - (-2.538)

## By level

- **PFI** (pr_mae): 
- **LIME** (pred_precip_mm): 

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