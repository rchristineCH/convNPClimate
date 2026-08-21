# Feature importance — tmax (2024)

- Target points: 88800  |  days scored: 366  |  folds: 5
- Baseline (all channels): MAE 1.047 | NLL 1.700 | CRPS 0.757  [degC]

## Top channels (|importance|)

| rank | PFI (mae) | LIME (pred) |
|---|---|---|
| 1 | t1000_12 (+1.005) | cos_time (-0.355) |
| 2 | t925_12 (+0.739) | t925_12 (+0.185) |
| 3 | t1000_15 (+0.513) | t850_12 (+0.150) |
| 4 | t925_15 (+0.349) | q850_12 (+0.106) |
| 5 | cos_time (+0.309) | q1000_18 (-0.104) |
| 6 | t1000_00 (+0.223) | q925_18 (-0.096) |
| 7 | z1000_12 (+0.203) | t925_15 (+0.082) |
| 8 | t1000_18 (+0.192) | t925_06 (+0.080) |

## By variable

- **PFI** (mae): t (+6.526), z (+0.539), scaffold (+0.336), q (+0.283)
- **LIME** (pred_tmax_degC): t (+0.545), scaffold (-0.375), z (+0.040), q (+0.035)

## By hour

- **PFI** (mae): 12 (+2.567), 15 (+1.557), 00 (+1.067), 18 (+0.630), 06 (+0.400)
- **LIME** (pred_tmax_degC): 12 (+0.721), 06 (+0.394), 18 (-0.316), 00 (-0.200), 15 (+0.022)

## By level

- **PFI** (mae): 1000 (+3.148), 925 (+2.509), 850 (+0.672), 700 (+0.195), 500 (+0.123), 300 (+0.093)
- **LIME** (pred_tmax_degC): 850 (+0.527), 925 (+0.377), 700 (+0.177), 300 (-0.170), 1000 (-0.150), 500 (-0.140)

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