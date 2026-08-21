# Feature importance — tmax (2024)

- Target points: 88800  |  days scored: 366  |  folds: 5
- Baseline (all channels): MAE 1.252 | NLL 1.965 | CRPS 0.935  [degC]

## Top channels (|importance|)

| rank | PFI (mae) | LIME (pred) |
|---|---|---|
| 1 | t1000_12 (+0.330) | cos_time (-0.218) |
| 2 | t1000_15 (+0.238) | t925_12 (+0.091) |
| 3 | t925_12 (+0.232) | t925_06 (+0.071) |
| 4 | t925_15 (+0.207) | t925_00 (+0.069) |
| 5 | t1000_18 (+0.135) | q850_15 (-0.059) |
| 6 | t1000_00 (+0.103) | t850_12 (+0.058) |
| 7 | cos_time (+0.102) | u1000_15 (-0.057) |
| 8 | t925_18 (+0.098) | t925_15 (+0.057) |

## By variable

- **PFI** (mae): t (+6.298), z (+0.466), q (+0.160), scaffold (+0.131), u (+0.083), v (+0.063)
- **LIME** (pred_tmax_degC): t (+0.501), scaffold (-0.222), z (-0.073), v (-0.073), u (-0.044), q (+0.016)

## By hour

- **PFI** (mae): 18 (+1.310), 15 (+1.237), 12 (+1.104), 00 (+0.558), 06 (+0.273)
- **LIME** (pred_tmax_degC): 06 (+0.274), 12 (+0.253), 00 (-0.107), 18 (-0.089), 15 (-0.004)

## By level

- **PFI** (mae): 1000 (+2.701), 925 (+1.864), 850 (+0.706), 700 (+0.097), 500 (+0.062), 300 (+0.055)
- **LIME** (pred_tmax_degC): 925 (+0.239), 700 (+0.232), 300 (-0.153), 500 (-0.113), 850 (+0.105), 1000 (+0.017)

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