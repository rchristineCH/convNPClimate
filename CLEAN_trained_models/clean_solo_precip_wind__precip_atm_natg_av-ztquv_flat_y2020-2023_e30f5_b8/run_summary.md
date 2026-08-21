# Training run: clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8

- Generated: 2026-07-24 21:43:18
- Git commit: `c9aae04`
- Host: rchristine-e3de6a0e9d2c-0 (Linux-5.15.0-1114-azure-x86_64-with-glibc2.39)
- Python: 3.11.15 | torch: 2.6.0+cu124 | CUDA: 12.4
- Device: `cuda`
- GPUs (1): NVIDIA A100 80GB PCIe MIG 2g.20gb
- DataParallel: False
- Peak GPU memory: 9504 MB
- CPU cores: 24 | torch threads: 24
- Total wall time: 4h 3m 14s

## Configuration

| Param | Value |
| --- | --- |
| VARIABLE | precip |
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
| TRIAL_NAME | clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8 |
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

### precip (bernoulli_gamma)

- Model trainable params: 324,578
- MeteoSwiss target points: 98,050
- Gradient clip: 1.0
- Fold wall times: 55m 3s, 44m 21s, 42m 20s, 54m 59s, 45m 16s  (total 4h 2m 0s)

| Fold | best epoch | epochs | test NLL | train NLL | CRPS | MAE | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 22 | 30 | 1.4985 | 1.1614 | 1.6414 | 2.2691 | 0.7595 | 0.8428 |
| 1 | 13 | 24 | 1.6931 | 1.6147 | 1.8796 | 2.6191 | 0.7446 | 0.8334 |
| 2 | 12 | 23 | 1.3033 | 1.5361 | 1.2433 | 1.7039 | 0.7247 | 0.7918 |
| 3 | 21 | 30 | 1.5712 | 1.3648 | 1.6640 | 2.3557 | 0.6836 | 0.8073 |
| 4 | 14 | 25 | 1.8217 | 1.7394 | 2.1360 | 3.2170 | 0.7541 | 0.8217 |
| **mean±std** | | | 1.5775±0.1758 | | 1.7129±0.2949 | 2.4330±0.4927 | 0.7333±0.0275 | 0.8194±0.0182 |

**Artifacts:** `stats.csv`, `params.json`, `metadata.json`, `trainingstats_all.png` (in `CLEAN_trained_models/clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8/precip`)

