# Feature importance — precip (2021)

- Target points: 98050  |  days scored: 365  |  folds: 5
- Baseline (all channels): PR_MAE 2.276 | PR_NLL 1.534  [mm]

## Top channels (|importance|)

| rank | PFI (pr_mae) |
|---|---|
| 1 | q700_18 (+0.191) |
| 2 | t700_18 (+0.112) |
| 3 | t500_18 (+0.110) |
| 4 | t1000_18 (+0.089) |
| 5 | q500_18 (+0.081) |
| 6 | cos_time (+0.079) |
| 7 | t925_18 (+0.069) |
| 8 | t850_18 (+0.060) |

## By variable

- **PFI** (pr_mae): q (+10.055), t (+5.037), z (+1.253), v (+0.591), u (+0.552), scaffold (+0.081)

## By hour

- **PFI** (pr_mae): 18 (+2.031), 06 (+1.192), 15 (+0.964), 00 (+0.763), 12 (+0.408)

## By level

- **PFI** (pr_mae): 500 (+0.929), 700 (+0.643), 300 (+0.615), 925 (+0.409), 850 (+0.405), 1000 (+0.369)

## Figures

- `pfi_channel.png`
- `pfi_variable.png`
- `pfi_level.png`
- `pfi_hour.png`
- `pfi_heatmap_*.png` (level × hour)