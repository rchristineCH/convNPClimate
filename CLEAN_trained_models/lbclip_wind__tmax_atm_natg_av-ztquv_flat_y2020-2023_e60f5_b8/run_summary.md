# Training run: lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8

- Generated: 2026-07-28 23:02:43
- Git commit: `8b54461`
- Host: rchristine-a063e8e3af7a-0 (Linux-5.15.0-1116-azure-x86_64-with-glibc2.39)
- Python: 3.11.15 | torch: 2.6.0+cu124 | CUDA: 12.4
- Device: `cuda`
- GPUs (1): NVIDIA A100 80GB PCIe MIG 2g.20gb
- DataParallel: False
- Peak GPU memory: 2821 MB
- CPU cores: 24 | torch threads: 24
- Total wall time: 7h 27m 53s

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
| ATMOS_VARIABLES | ['z', 't', 'q', 'u', 'v'] |
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
| IN_CHANNELS | 155 |
| N_EPOCHS | 60 |
| CRPS_N_SAMPLES | 25 |
| BATCH_SIZE | 8 |
| LR | 0.0005 |
| PATIENCE | 20 |
| N_FOLDS | 5 |
| SEED | 42 |
| TRIAL_NAME | lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8 |
| RUN_TYPE | cloud |
| DEVICE | cuda |

## Data

- Grid mode: **native coarse atmospheric**
- Context tensor (time, channel, lat, lon): `(1461, 155, 11, 23)`
- Input channels: 155
- Time steps: 1461 (2020-01-01 -> 2023-12-31)
- Normalization: data_mean=285.5405, data_std=9.0359, lat=[45.400, 48.200], lon=[5.000, 11.000]
- Encoder: **flat**

### Channel groups (by physical variable)

| Group | #channels | Members |
| --- | --- | --- |
| `z` | 30 | z1000_00, z925_00, z850_00, z700_00, z500_00, z300_00, z1000_06, z925_06, z850_06, z700... |
| `t` | 30 | t1000_00, t925_00, t850_00, t700_00, t500_00, t300_00, t1000_06, t925_06, t850_06, t700... |
| `q` | 30 | q1000_00, q925_00, q850_00, q700_00, q500_00, q300_00, q1000_06, q925_06, q850_06, q700... |
| `u` | 30 | u1000_00, u925_00, u850_00, u700_00, u500_00, u300_00, u1000_06, u925_06, u850_06, u700... |
| `v` | 30 | v1000_00, v925_00, v850_00, v700_00, v500_00, v300_00, v1000_06, v925_06, v850_06, v700... |
| `scaffold` | 5 | lat, lon, cos_time, sin_time, elevation |

## Results

### tmax (gaussian)

- Model trainable params: 324,383
- MeteoSwiss target points: 88,800
- Gradient clip: 1.0
- Fold wall times: 1h 30m 9s, 1h 29m 16s, 1h 29m 8s, 1h 29m 4s, 1h 29m 4s  (total 7h 26m 42s)

| Fold | best epoch | epochs | test NLL | train NLL | CRPS | MAE | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 56 | 60 | -0.4750 | -0.4102 | nan | 0.1070 | 0.9851 | 0.9857 |
| 1 | 54 | 60 | -0.5285 | -0.7642 | nan | 0.1074 | 0.9876 | 0.9868 |
| 2 | 52 | 60 | -0.3139 | -0.3826 | nan | 0.1218 | 0.9811 | 0.9803 |
| 3 | 59 | 60 | -0.4353 | -0.4945 | nan | 0.1106 | 0.9901 | 0.9901 |
| 4 | 58 | 60 | -0.5489 | -0.6232 | nan | 0.1021 | 0.9895 | 0.9891 |
| **mean±std** | | | -0.4603±0.0834 | | nan±nan | 0.1098±0.0066 | 0.9867±0.0033 | 0.9864±0.0034 |

**Artifacts:** `stats.csv`, `params.json`, `metadata.json`, `trainingstats_all.png` (in `CLEAN_trained_models/lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8/tmax`)

