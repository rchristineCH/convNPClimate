# Training run: baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo

- Generated: 2026-07-24 19:47:42
- Git commit: `8dac547`
- Host: rchristine-864fa536af94-0 (Linux-5.15.0-1114-azure-x86_64-with-glibc2.39)
- Python: 3.11.15 | torch: 2.6.0+cu124 | CUDA: 12.4
- Device: `cuda`
- GPUs (1): NVIDIA A100 80GB PCIe MIG 2g.20gb
- DataParallel: False
- Peak GPU memory: 3085 MB
- CPU cores: 24 | torch threads: 24
- Total wall time: 4h 3m 30s

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
| USE_ELEVATION_CHANNEL | False |
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
| IN_CHANNELS | 5 |
| N_EPOCHS | 30 |
| CRPS_N_SAMPLES | 25 |
| BATCH_SIZE | 8 |
| LR | 0.0005 |
| PATIENCE | 10 |
| N_FOLDS | 5 |
| SEED | 42 |
| TRIAL_NAME | baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo |
| RUN_TYPE | cloud |
| DEVICE | cuda |

## Data

- Grid mode: **fine surface**
- Context tensor (time, channel, lat, lon): `(1461, 5, 29, 61)`
- Input channels: 5
- Time steps: 1461 (2020-01-01 -> 2023-12-31)
- Normalization: data_mean=285.5405, data_std=9.0359, lat=[45.400, 48.200], lon=[5.000, 11.000]
- Encoder: **flat**

### Channel groups (by physical variable)

| Group | #channels | Members |
| --- | --- | --- |
| `scaffold` | 5 | data, lat, lon, cos_time, sin_time |

## Results

### tmax (gaussian)

- Model trainable params: 281,933
- MeteoSwiss target points: 88,800
- Gradient clip: None
- Fold wall times: 51m 1s, 50m 21s, 51m 0s, 50m 26s, 40m 18s  (total 4h 3m 7s)

| Fold | best epoch | epochs | test NLL | train NLL | CRPS | MAE | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 29 | 30 | -0.1931 | -0.0970 | nan | 0.1447 | 0.9754 | 0.9783 |
| 1 | 20 | 30 | -0.0939 | -0.2364 | nan | 0.1642 | 0.9733 | 0.9730 |
| 2 | 29 | 30 | -0.0939 | -0.2602 | nan | 0.1581 | 0.9681 | 0.9656 |
| 3 | 29 | 30 | -0.1626 | -0.1920 | nan | 0.1501 | 0.9829 | 0.9835 |
| 4 | 13 | 24 | -0.0942 | -0.0587 | nan | 0.1526 | 0.9783 | 0.9786 |
| **mean±std** | | | -0.1275±0.0422 | | nan±nan | 0.1539±0.0067 | 0.9756±0.0049 | 0.9758±0.0061 |

**Artifacts:** `stats.csv`, `params.json`, `metadata.json`, `trainingstats_all.png` (in `CLEAN_trained_models/baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo/tmax`)

