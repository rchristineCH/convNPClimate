# Training run: clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8

- Generated: 2026-07-24 15:33:06
- Git commit: `b909829`
- Host: DESKTOP-6COJL4O (Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35)
- Python: 3.11.15 | torch: 2.6.0+cu124 | CUDA: 12.4
- Device: `cuda`
- GPUs (1): NVIDIA GeForce RTX 3080
- DataParallel: False
- Peak GPU memory: 2818 MB
- CPU cores: 16 | torch threads: 16
- Total wall time: 2h 39m 26s

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
| N_EPOCHS | 30 |
| CRPS_N_SAMPLES | 25 |
| BATCH_SIZE | 8 |
| LR | 0.0005 |
| PATIENCE | 10 |
| N_FOLDS | 5 |
| SEED | 42 |
| TRIAL_NAME | clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8 |
| RUN_TYPE | local |
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
- Gradient clip: None
- Fold wall times: 29m 45s, 27m 15s, 40m 46s, 19m 41s, 40m 36s  (total 2h 38m 5s)

| Fold | best epoch | epochs | test NLL | train NLL | CRPS | MAE | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 11 | 22 | -0.2183 | -0.1704 | nan | 0.1364 | 0.9802 | 0.9826 |
| 1 | 9 | 20 | -0.1714 | -0.1908 | nan | 0.1458 | 0.9801 | 0.9786 |
| 2 | 27 | 30 | -0.1726 | -0.4951 | nan | 0.1341 | 0.9768 | 0.9758 |
| 3 | 3 | 14 | -0.0628 | -0.1085 | nan | 0.1524 | 0.9851 | 0.9855 |
| 4 | 26 | 30 | -0.3443 | -0.2161 | nan | 0.1173 | 0.9864 | 0.9863 |
| **mean±std** | | | -0.1939±0.0910 | | nan±nan | 0.1372±0.0119 | 0.9817±0.0036 | 0.9818±0.0040 |

**Artifacts:** `stats.csv`, `params.json`, `metadata.json`, `trainingstats_all.png` (in `CLEAN_trained_models/clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8/tmax`)

