# Feature importance — precip (2020)

- Target points: 98050  |  days scored: 366  |  folds: 5
- Baseline (all channels): PR_MAE 2.104 | PR_NLL 1.428  [mm]

## Top channels (|importance|)

| rank | PFI (pr_mae) |
|---|---|
| 1 | q700_18 (+0.172) |
| 2 | q500_18 (+0.104) |
| 3 | q850_18 (+0.098) |
| 4 | cos_time (+0.068) |
| 5 | q700_06 (+0.057) |
| 6 | t500_18 (+0.056) |
| 7 | q925_18 (+0.055) |
| 8 | v500_18 (+0.052) |

## By variable

- **PFI** (pr_mae): q (+7.843), t (+3.651), z (+0.993), u (+0.452), v (+0.410), scaffold (+0.082)

## By hour

- **PFI** (pr_mae): 18 (+1.873), 06 (+1.068), 15 (+0.779), 00 (+0.655), 12 (+0.320)

## By level

- **PFI** (pr_mae): 500 (+0.922), 700 (+0.456), 300 (+0.446), 925 (+0.408), 850 (+0.320), 1000 (+0.311)

## Figures

- `pfi_channel.png`
- `pfi_variable.png`
- `pfi_level.png`
- `pfi_hour.png`
- `pfi_heatmap_*.png` (level × hour)