# Feature importance — precip (2024)

- Target points: 98050  |  days scored: 366  |  folds: 5
- Baseline (all channels): PR_MAE 3.133 | PR_NLL 2.437  [mm]

## Top channels (|importance|)

| rank | PFI (pr_mae) | LIME (pred) |
|---|---|---|
| 1 | t500_18 (+0.324) | t300_15 (-0.178) |
| 2 | q700_18 (+0.240) | z850_15 (+0.175) |
| 3 | t700_18 (+0.222) | t925_06 (+0.159) |
| 4 | q850_18 (+0.142) | t1000_06 (+0.147) |
| 5 | t500_15 (+0.138) | t1000_00 (+0.114) |
| 6 | z300_18 (+0.116) | z925_18 (+0.111) |
| 7 | t500_12 (+0.108) | t300_12 (-0.111) |
| 8 | z1000_00 (+0.105) | q300_18 (+0.109) |

## By variable

- **PFI** (pr_mae): q (+18.256), t (+8.671), z (+2.039), scaffold (+0.057)
- **LIME** (pred_precip_mm): z (+0.499), q (+0.284), t (-0.121), scaffold (+0.029)

## By hour

- **PFI** (pr_mae): 06 (+1.507), 18 (+1.343), 15 (+1.192), 00 (+0.505), 12 (+0.497)
- **LIME** (pred_precip_mm): 12 (-0.518), 18 (+0.471), 06 (+0.268), 00 (+0.259), 15 (+0.181)

## By level

- **PFI** (pr_mae): 500 (+1.826), 700 (+0.617), 1000 (+0.468), 300 (+0.264), 850 (+0.151), 925 (+0.149)
- **LIME** (pred_precip_mm): 300 (-0.341), 700 (+0.309), 925 (+0.222), 850 (+0.175), 1000 (+0.159), 500 (+0.138)

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