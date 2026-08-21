# Feature importance — tmax (2024)

- Target points: 88800  |  days scored: 366  |  folds: 5
- Baseline (all channels): MAE 1.174 | NLL 1.922 | CRPS 0.889  [degC]

## Top channels (|importance|)

| rank | PFI (mae) | LIME (pred) |
|---|---|---|
| 1 | t1000_12 (+0.421) | cos_time (-0.193) |
| 2 | t2m_12 (+0.357) | t2m_18 (-0.080) |
| 3 | t925_12 (+0.348) | q1000_15 (+0.069) |
| 4 | t2m_15 (+0.315) | q1000_12 (+0.068) |
| 5 | t1000_15 (+0.274) | t925_12 (+0.064) |
| 6 | t925_15 (+0.223) | t850_12 (+0.061) |
| 7 | t1000_00 (+0.184) | q700_12 (+0.060) |
| 8 | t1000_18 (+0.156) | t2m_15 (-0.060) |

## By variable

- **PFI** (mae): t (+5.034), t2m (+1.273), z (+0.300), q (+0.226), scaffold (+0.159), tp (+0.029)
- **LIME** (pred_tmax_degC): q (+0.400), t (+0.282), scaffold (-0.190), t2m (-0.168), z (+0.022), tp (+0.004)

## By hour

- **PFI** (mae): 12 (+1.931), 15 (+1.593), 18 (+0.894), 00 (+0.769), 06 (+0.303)
- **LIME** (pred_tmax_degC): 06 (+0.311), 12 (+0.267), 18 (-0.028), 00 (-0.019), 15 (+0.008)

## By level

- **PFI** (mae): 1000 (+1.806), 925 (+1.653), 850 (+0.559), 700 (+0.121), 300 (+0.093), 500 (+0.089)
- **LIME** (pred_tmax_degC): 925 (+0.361), 1000 (+0.269), 700 (+0.246), 300 (-0.163), 500 (-0.078), 850 (+0.068)

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