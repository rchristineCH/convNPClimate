# Training run: lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8

- Generated: 2026-07-28 23:02:07
- Git commit: `8b54461`
- Host: rchristine-a063e8e3af7a-0 (Linux-5.15.0-1116-azure-x86_64-with-glibc2.39)
- Python: 3.11.15 | torch: 2.6.0+cu124 | CUDA: 12.4
- Device: `cuda`
- GPUs (1): NVIDIA A100 80GB PCIe MIG 2g.20gb
- DataParallel: False
- Peak GPU memory: 2648 MB
- CPU cores: 24 | torch threads: 24
- Total wall time: 7h 27m 17s

## Configuration

| Param | Value |
| --- | --- |
| VARIABLE | tmax |
| DISTRIBUTION | None |
| DATA_YEAR_START | 2020 |
| DATA_YEAR_END | 2023 |
| SEASONAL_FEATURES | True |
| SEASONAL_FEATURES_IN_MLP | True |
| USE_ELEVATION | True |
| USE_MTPI | True |
| USE_ELEVATION_CHANNEL | True |
| USE_SURFACE | False |
| USE_ATMOSPHERIC | True |
| ATMOS_VARIABLES | ['z', 't', 'q'] |
| ATMOS_LEVELS | [1000, 925, 850, 700, 500, 300] |
| ATMOS_HOURS | ['00', '06', '12', '15', '18'] |
| USE_SFC_ATMOS | False |
| ATMOS_SFC_VARIABLES | ['t2m', 'tp'] |
| ATMOS_NATIVE_GRID | True |
| ENCODER | flat |
| CHANNEL_GROUP_BY | variable |
| N_CHANNELS | 128 |
| N_BLOCKS | 6 |
| KERNEL_SIZE | 5 |
| LENGTH_SCALE | 0.1 |
| IN_CHANNELS | 95 |
| N_EPOCHS | 60 |
| CRPS_N_SAMPLES | 25 |
| BATCH_SIZE | 8 |
| LR | 0.0005 |
| PATIENCE | 20 |
| N_FOLDS | 5 |
| SEED | 42 |
| TRIAL_NAME | lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8 |
| RUN_TYPE | cloud |
| DEVICE | cuda |

## Data

- Grid mode: **native coarse atmospheric**
- Context tensor (time, channel, lat, lon): `(1461, 95, 11, 23)`
- Input channels: 95
- Time steps: 1461 (2020-01-01 -> 2023-12-31)
- Normalization: data_mean=285.5405, data_std=9.0359, lat=[45.400, 48.200], lon=[5.000, 11.000]
- Encoder: **flat**

### Channel groups (by physical variable)

| Group | #channels | Members |
| --- | --- | --- |
| `z` | 30 | z1000_00, z925_00, z850_00, z700_00, z500_00, z300_00, z1000_06, z925_06, z850_06, z700... |
| `t` | 30 | t1000_00, t925_00, t850_00, t700_00, t500_00, t300_00, t1000_06, t925_06, t850_06, t700... |
| `q` | 30 | q1000_00, q925_00, q850_00, q700_00, q500_00, q300_00, q1000_06, q925_06, q850_06, q700... |
| `scaffold` | 5 | lat, lon, cos_time, sin_time, elevation |

## Results

### tmax (gaussian)

- Model trainable params: 307,403
- MeteoSwiss target points: 88,800
- Gradient clip: 1.0
- Fold wall times: 1h 29m 49s, 1h 29m 15s, 1h 29m 7s, 1h 29m 0s, 1h 29m 22s  (total 7h 26m 35s)

| Fold | best epoch | epochs | test NLL | train NLL | CRPS | MAE | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 51 | 60 | -0.4762 | -0.4778 | nan | 0.1086 | 0.9846 | 0.9852 |
| 1 | 59 | 60 | -0.4062 | -0.4946 | nan | 0.1139 | 0.9856 | 0.9845 |
| 2 | 59 | 60 | -0.3298 | -0.4280 | nan | 0.1179 | 0.9817 | 0.9803 |
| 3 | 45 | 60 | -0.4821 | -0.5370 | nan | 0.1130 | 0.9903 | 0.9904 |
| 4 | 58 | 60 | -0.4607 | -0.4878 | nan | 0.1087 | 0.9884 | 0.9881 |
| **mean±std** | | | -0.4310±0.0573 | | nan±nan | 0.1124±0.0035 | 0.9861±0.0030 | 0.9857±0.0034 |

**Artifacts:** `stats.csv`, `params.json`, `metadata.json`, `trainingstats_all.png` (in `CLEAN_trained_models/lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8/tmax`)

