# Training run: baseline__tmax_sfc_flat_y2020-2023_e30f5_b8

- Generated: 2026-07-24 15:29:26
- Git commit: `8dac547`
- Host: rchristine-864fa536af94-0 (Linux-5.15.0-1114-azure-x86_64-with-glibc2.39)
- Python: 3.11.15 | torch: 2.6.0+cu124 | CUDA: 12.4
- Device: `cuda`
- GPUs (1): NVIDIA A100 80GB PCIe MIG 2g.20gb
- DataParallel: False
- Peak GPU memory: 3106 MB
- CPU cores: 24 | torch threads: 24
- Total wall time: 4h 2m 37s

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
| USE_SURFACE | True |
| USE_ATMOSPHERIC | False |
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
| IN_CHANNELS | 6 |
| N_EPOCHS | 30 |
| CRPS_N_SAMPLES | 25 |
| BATCH_SIZE | 8 |
| LR | 0.0005 |
| PATIENCE | 10 |
| N_FOLDS | 5 |
| SEED | 42 |
| TRIAL_NAME | baseline__tmax_sfc_flat_y2020-2023_e30f5_b8 |
| RUN_TYPE | cloud |
| DEVICE | cuda |

## Data

- Grid mode: **fine surface**
- Context tensor (time, channel, lat, lon): `(1461, 6, 29, 61)`
- Input channels: 6
- Time steps: 1461 (2020-01-01 -> 2023-12-31)
- Normalization: data_mean=285.5405, data_std=9.0359, lat=[45.400, 48.200], lon=[5.000, 11.000]
- Encoder: **flat**

### Channel groups (by physical variable)

| Group | #channels | Members |
| --- | --- | --- |
| `scaffold` | 6 | data, lat, lon, cos_time, sin_time, elevation |

## Results

### tmax (gaussian)

- Model trainable params: 282,216
- MeteoSwiss target points: 88,800
- Gradient clip: None
- Fold wall times: 50m 27s, 39m 27s, 51m 1s, 51m 9s, 50m 4s  (total 4h 2m 9s)

| Fold | best epoch | epochs | test NLL | train NLL | CRPS | MAE | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 22 | 30 | -0.2053 | -0.1725 | nan | 0.1446 | 0.9751 | 0.9783 |
| 1 | 12 | 23 | -0.0568 | -0.0548 | nan | 0.1582 | 0.9739 | 0.9737 |
| 2 | 26 | 30 | -0.1061 | -0.1897 | nan | 0.1604 | 0.9676 | 0.9648 |
| 3 | 29 | 30 | -0.1998 | -0.2575 | nan | 0.1447 | 0.9835 | 0.9839 |
| 4 | 24 | 30 | -0.1972 | -0.1324 | nan | 0.1448 | 0.9795 | 0.9794 |
| **mean±std** | | | -0.1530±0.0606 | | nan±nan | 0.1505±0.0072 | 0.9759±0.0054 | 0.9760±0.0065 |

**Artifacts:** `stats.csv`, `params.json`, `metadata.json`, `trainingstats_all.png` (in `CLEAN_trained_models/baseline__tmax_sfc_flat_y2020-2023_e30f5_b8/tmax`)

