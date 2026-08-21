# Feature importance — precip (2022)

- Target points: 98050  |  days scored: 365  |  folds: 5
- Baseline (all channels): PR_MAE 2.032 | PR_NLL 1.443  [mm]

## Top channels (|importance|)

| rank | LIME (pred) |
|---|---|
| 1 | tp (-1.405) |
| 2 | t2m_max (+0.130) |
| 3 | sin_time (-0.110) |
| 4 | cos_time (-0.061) |
| 5 | lon (+0.002) |
| 6 | lat (-0.002) |
| 7 | elevation (-0.001) |
| 8 |  |

## By variable

- **LIME** (pred_precip_mm): tp (-1.405), scaffold (-0.172), t2m_max (+0.130)

## By hour

- **LIME** (pred_precip_mm): - (-1.405)

## By level

- **LIME** (pred_precip_mm): 

## Figures

- `lime_channel.png`
- `lime_variable.png`
- `lime_level.png`
- `lime_hour.png`
- `lime_heatmap_*.png` (level × hour)