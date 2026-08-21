# Feature importance — precip (2024)

- Target points: 98050  |  days scored: 366  |  folds: 5
- Baseline (all channels): PR_MAE 3.346 | PR_NLL 1.853  [mm]

## Top channels (|importance|)

| rank | PFI (pr_mae) | LIME (pred) |
|---|---|---|
| 1 | t500_18 (+0.801) | z1000_18 (-0.301) |
| 2 | z1000_18 (+0.698) | t925_06 (+0.294) |
| 3 | t700_18 (+0.540) | z925_18 (+0.234) |
| 4 | q700_18 (+0.436) | t1000_06 (+0.232) |
| 5 | z300_18 (+0.341) | z850_15 (+0.231) |
| 6 | t500_15 (+0.328) | t300_15 (-0.197) |
| 7 | q850_18 (+0.275) | t850_06 (+0.195) |
| 8 | z500_18 (+0.212) | t1000_18 (-0.190) |

## By variable

- **PFI** (pr_mae): q (+36.061), t (+14.240), z (+4.654), scaffold (+0.037)
- **LIME** (pred_precip_mm): z (+0.533), q (+0.178), t (-0.126), scaffold (-0.075)

## By hour

- **PFI** (pr_mae): 06 (+5.233), 15 (+2.799), 18 (+2.433), 00 (+1.022), 12 (+0.678)
- **LIME** (pred_precip_mm): 12 (-0.518), 06 (+0.430), 15 (+0.278), 00 (+0.204), 18 (+0.193)

## By level

- **PFI** (pr_mae): 500 (+4.130), 1000 (+2.105), 700 (+0.622), 925 (+0.439), 300 (+0.301), 850 (+0.274)
- **LIME** (pred_precip_mm): 700 (+0.490), 300 (-0.490), 925 (+0.317), 850 (+0.291), 1000 (-0.084), 500 (+0.061)

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