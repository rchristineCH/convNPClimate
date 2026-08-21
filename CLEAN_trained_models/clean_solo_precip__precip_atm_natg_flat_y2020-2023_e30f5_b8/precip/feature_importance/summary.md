# Feature importance — precip (2022)

- Target points: 98050  |  days scored: 365  |  folds: 5
- Baseline (all channels): PR_MAE 2.292 | PR_NLL 1.479  [mm]

## Top channels (|importance|)

| rank | PFI (pr_mae) | LIME (pred) |
|---|---|---|
| 1 | t500_18 (+0.395) | q700_06 (+0.412) |
| 2 | q700_18 (+0.353) | q700_18 (+0.382) |
| 3 | z1000_18 (+0.349) | q1000_06 (+0.288) |
| 4 | t700_18 (+0.322) | q300_18 (+0.229) |
| 5 | t500_15 (+0.173) | t500_18 (-0.227) |
| 6 | q850_18 (+0.158) | q1000_12 (+0.215) |
| 7 | z850_06 (+0.155) | q700_12 (+0.206) |
| 8 | z300_18 (+0.144) | q925_06 (+0.205) |

## By variable

- **PFI** (pr_mae): q (+25.454), t (+9.818), z (+2.974), scaffold (+0.060)
- **LIME** (pred_precip_mm): q (+3.130), z (-0.947), t (-0.789), scaffold (-0.150)

## By hour

- **PFI** (pr_mae): 06 (+4.411), 18 (+1.706), 15 (+1.597), 00 (+0.745), 12 (+0.440)
- **LIME** (pred_precip_mm): 06 (+1.429), 00 (+0.292), 15 (-0.222), 18 (-0.146), 12 (+0.042)

## By level

- **PFI** (pr_mae): 500 (+2.405), 1000 (+1.245), 700 (+0.390), 925 (+0.310), 300 (+0.306), 850 (+0.292)
- **LIME** (pred_precip_mm): 500 (-0.767), 1000 (+0.644), 700 (+0.530), 925 (+0.505), 850 (+0.270), 300 (+0.212)

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