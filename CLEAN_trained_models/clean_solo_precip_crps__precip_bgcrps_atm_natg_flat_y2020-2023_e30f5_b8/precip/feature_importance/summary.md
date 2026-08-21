# Feature importance — precip (2022)

- Target points: 98050  |  days scored: 365  |  folds: 5
- Baseline (all channels): PR_MAE 2.232 | PR_NLL 2.376  [mm]

## Top channels (|importance|)

| rank | PFI (pr_mae) | LIME (pred) |
|---|---|---|
| 1 | q700_18 (+0.244) | q700_18 (+0.274) |
| 2 | q850_18 (+0.116) | q700_06 (+0.194) |
| 3 | q500_18 (+0.116) | q300_18 (+0.183) |
| 4 | t500_18 (+0.104) | q700_12 (+0.168) |
| 5 | t700_18 (+0.089) | q1000_06 (+0.159) |
| 6 | z1000_00 (+0.073) | t500_18 (-0.149) |
| 7 | q700_15 (+0.064) | q850_18 (+0.146) |
| 8 | q700_12 (+0.058) | q1000_12 (+0.119) |

## By variable

- **PFI** (pr_mae): q (+13.678), t (+6.305), z (+1.218), scaffold (+0.047)
- **LIME** (pred_precip_mm): q (+2.082), t (-0.680), z (-0.553), scaffold (-0.102)

## By hour

- **PFI** (pr_mae): 06 (+1.324), 18 (+0.930), 15 (+0.655), 00 (+0.467), 12 (+0.258)
- **LIME** (pred_precip_mm): 06 (+0.822), 00 (+0.298), 15 (-0.126), 18 (-0.110), 12 (-0.035)

## By level

- **PFI** (pr_mae): 500 (+0.958), 700 (+0.385), 1000 (+0.358), 300 (+0.224), 850 (+0.141), 925 (+0.136)
- **LIME** (pred_precip_mm): 500 (-0.665), 1000 (+0.432), 700 (+0.365), 925 (+0.299), 850 (+0.224), 300 (+0.193)

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