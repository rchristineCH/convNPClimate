# Feature importance — tmax (2024)

- Target points: 88800  |  days scored: 366  |  folds: 5
- Baseline (all channels): MAE 1.206 | NLL 1.943 | CRPS 0.909  [degC]

## Top channels (|importance|)

| rank | PFI (mae) | LIME (pred) |
|---|---|---|
| 1 | t1000_12 (+0.585) | cos_time (-0.325) |
| 2 | t925_12 (+0.455) | t925_12 (+0.130) |
| 3 | t1000_15 (+0.367) | t850_12 (+0.105) |
| 4 | t925_15 (+0.300) | q700_06 (+0.087) |
| 5 | cos_time (+0.240) | t300_15 (-0.086) |
| 6 | t1000_00 (+0.207) | q700_15 (+0.078) |
| 7 | t1000_18 (+0.195) | q850_15 (-0.073) |
| 8 | z1000_12 (+0.167) | q700_12 (+0.070) |

## By variable

- **PFI** (mae): t (+6.197), z (+0.511), scaffold (+0.269), q (+0.197)
- **LIME** (pred_tmax_degC): t (+0.572), scaffold (-0.343), z (+0.124), q (-0.033)

## By hour

- **PFI** (mae): 12 (+1.691), 15 (+1.305), 18 (+0.999), 00 (+0.638), 06 (+0.311)
- **LIME** (pred_tmax_degC): 12 (+0.437), 06 (+0.344), 00 (-0.181), 15 (+0.059), 18 (+0.003)

## By level

- **PFI** (mae): 1000 (+2.796), 925 (+2.266), 850 (+0.534), 700 (+0.160), 500 (+0.111), 300 (+0.110)
- **LIME** (pred_tmax_degC): 925 (+0.438), 700 (+0.380), 300 (-0.270), 850 (+0.266), 500 (-0.118), 1000 (-0.033)

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