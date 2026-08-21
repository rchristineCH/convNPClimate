# Feature importance — precip (2022)

- Target points: 98050  |  days scored: 365  |  folds: 5
- Baseline (all channels): PR_MAE 2.018 | PR_NLL 1.422  [mm]

## Top channels (|importance|)

| rank | PFI (pr_mae) | LIME (pred) |
|---|---|---|
| 1 | q700_18 (+0.135) | q700_06 (+0.142) |
| 2 | t500_18 (+0.065) | q700_18 (+0.139) |
| 3 | q500_18 (+0.062) | q850_06 (+0.100) |
| 4 | t1000_18 (+0.051) | q925_06 (+0.093) |
| 5 | q850_18 (+0.047) | q1000_06 (+0.086) |
| 6 | t700_18 (+0.047) | q300_18 (+0.083) |
| 7 | cos_time (+0.042) | q700_12 (+0.067) |
| 8 | t925_18 (+0.042) | t925_18 (-0.060) |

## By variable

- **PFI** (pr_mae): q (+8.206), t (+3.859), z (+0.901), u (+0.421), v (+0.253), scaffold (+0.053)
- **LIME** (pred_precip_mm): q (+1.252), t (-0.483), z (-0.307), u (-0.170), v (+0.135), scaffold (+0.028)

## By hour

- **PFI** (pr_mae): 18 (+1.470), 06 (+0.951), 00 (+0.672), 15 (+0.626), 12 (+0.268)
- **LIME** (pred_precip_mm): 06 (+0.523), 15 (-0.238), 00 (+0.208), 18 (-0.071), 12 (+0.005)

## By level

- **PFI** (pr_mae): 500 (+0.619), 300 (+0.390), 700 (+0.385), 925 (+0.354), 1000 (+0.271), 850 (+0.241)
- **LIME** (pred_precip_mm): 700 (+0.225), 1000 (+0.164), 925 (+0.138), 500 (-0.133), 850 (+0.037), 300 (-0.004)

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