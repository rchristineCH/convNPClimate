# Feature importance — tmax (2022)

- Target points: 88800  |  days scored: 365  |  folds: 5
- Baseline (all channels): MAE 0.875 | NLL 1.557 | CRPS 0.640  [degC]

## Top channels (|importance|)

| rank | PFI (mae) | LIME (pred) |
|---|---|---|
| 1 | t1000_12 (+0.899) | cos_time (-0.380) |
| 2 | t925_12 (+0.690) | t925_12 (+0.265) |
| 3 | t1000_15 (+0.537) | t1000_12 (+0.217) |
| 4 | t925_15 (+0.413) | t925_15 (+0.184) |
| 5 | cos_time (+0.297) | t1000_15 (+0.170) |
| 6 | t1000_18 (+0.248) | t925_18 (+0.118) |
| 7 | t1000_00 (+0.179) | t1000_18 (+0.117) |
| 8 | t925_18 (+0.165) | z1000_12 (-0.115) |

## By variable

- **PFI** (mae): t (+7.136), z (+0.672), scaffold (+0.305), q (+0.283), v (+0.130), u (+0.119)
- **LIME** (pred_tmax_degC): t (+1.375), q (-0.433), scaffold (-0.374), z (+0.295), u (-0.252), v (-0.126)

## By hour

- **PFI** (mae): 12 (+2.339), 15 (+1.720), 18 (+1.242), 00 (+0.765), 06 (+0.311)
- **LIME** (pred_tmax_degC): 12 (+0.514), 18 (+0.352), 15 (+0.213), 00 (-0.167), 06 (-0.053)

## By level

- **PFI** (mae): 1000 (+3.324), 925 (+2.591), 850 (+0.585), 700 (+0.307), 500 (+0.147), 300 (+0.079)
- **LIME** (pred_tmax_degC): 925 (+0.508), 850 (+0.190), 1000 (+0.104), 500 (+0.091), 300 (-0.027), 700 (-0.008)

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