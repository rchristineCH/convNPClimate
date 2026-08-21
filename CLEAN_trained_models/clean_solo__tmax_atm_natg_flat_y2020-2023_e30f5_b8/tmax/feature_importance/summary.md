# Feature importance — tmax (2022)

- Target points: 88800  |  days scored: 365  |  folds: 5
- Baseline (all channels): MAE 1.107 | NLL 1.884 | CRPS 0.843  [degC]

## Top channels (|importance|)

| rank | PFI (mae) | LIME (pred) |
|---|---|---|
| 1 | t1000_12 (+0.698) | cos_time (-0.412) |
| 2 | t925_12 (+0.528) | t925_12 (+0.242) |
| 3 | t1000_15 (+0.444) | t1000_12 (+0.207) |
| 4 | t925_15 (+0.358) | t925_15 (+0.183) |
| 5 | cos_time (+0.268) | t1000_15 (+0.178) |
| 6 | t1000_18 (+0.237) | t1000_18 (+0.150) |
| 7 | t1000_00 (+0.205) | z1000_12 (-0.143) |
| 8 | t925_18 (+0.179) | t925_18 (+0.142) |

## By variable

- **PFI** (mae): t (+6.851), z (+0.530), scaffold (+0.299), q (+0.223)
- **LIME** (pred_tmax_degC): t (+1.413), q (-0.523), scaffold (-0.405), z (+0.268)

## By hour

- **PFI** (mae): 12 (+1.863), 15 (+1.522), 18 (+1.132), 00 (+0.675), 06 (+0.374)
- **LIME** (pred_tmax_degC): 12 (+0.498), 18 (+0.347), 15 (+0.220), 06 (+0.119), 00 (-0.025)

## By level

- **PFI** (mae): 1000 (+3.279), 925 (+2.552), 850 (+0.535), 700 (+0.240), 500 (+0.099), 300 (+0.088)
- **LIME** (pred_tmax_degC): 925 (+0.792), 1000 (+0.300), 700 (-0.075), 500 (+0.064), 850 (+0.062), 300 (+0.015)

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