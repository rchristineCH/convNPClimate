# Training run: sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8

- Generated: 2026-07-23 17:42:38
- Git commit: `cb99a7f`
- Host: rchristine-864fa536af94-0 (Linux-5.15.0-1114-azure-x86_64-with-glibc2.39)
- Python: 3.11.15 | torch: 2.6.0+cu124 | CUDA: 12.4
- Device: `cuda`
- GPUs (1): NVIDIA A100 80GB PCIe MIG 2g.20gb
- DataParallel: False
- Peak GPU memory: 2676 MB
- CPU cores: 24 | torch threads: 24
- Total wall time: 3h 6m 23s

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
| USE_SFC_ATMOS | True |
| ATMOS_SFC_VARIABLES | ['t2m', 'tp'] |
| ATMOS_NATIVE_GRID | True |
| ENCODER | flat |
| CHANNEL_GROUP_BY | variable |
| N_CHANNELS | 128 |
| N_BLOCKS | 6 |
| KERNEL_SIZE | 5 |
| LENGTH_SCALE | 0.1 |
| IN_CHANNELS | 105 |
| N_EPOCHS | 30 |
| CRPS_N_SAMPLES | 25 |
| BATCH_SIZE | 8 |
| LR | 0.0005 |
| PATIENCE | 10 |
| N_FOLDS | 5 |
| SEED | 42 |
| TRIAL_NAME | sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8 |
| RUN_TYPE | cloud |
| DEVICE | cuda |

## Data

- Grid mode: **native coarse atmospheric**
- Context tensor (time, channel, lat, lon): `(1461, 105, 11, 23)`
- Input channels: 105
- Time steps: 1461 (2020-01-01 -> 2023-12-31)
- Normalization: data_mean=285.5405, data_std=9.0359, lat=[45.400, 48.200], lon=[5.000, 11.000]
- Encoder: **flat**

### Channel groups (by physical variable)

| Group | #channels | Members |
| --- | --- | --- |
| `z` | 30 | z1000_00, z925_00, z850_00, z700_00, z500_00, z300_00, z1000_06, z925_06, z850_06, z700... |
| `t` | 35 | t1000_00, t925_00, t850_00, t700_00, t500_00, t300_00, t1000_06, t925_06, t850_06, t700... |
| `q` | 30 | q1000_00, q925_00, q850_00, q700_00, q500_00, q300_00, q1000_06, q925_06, q850_06, q700... |
| `scaffold` | 5 | lat, lon, cos_time, sin_time, elevation |
| `tp` | 5 | tp_00, tp_06, tp_12, tp_15, tp_18 |

## Results

### tmax (gaussian)

- Model trainable params: 310,233
- MeteoSwiss target points: 88,800
- Gradient clip: None
- Fold wall times: 30m 1s, 44m 30s, 37m 23s, 41m 42s, 30m 28s  (total 3h 4m 6s)

| Fold | best epoch | epochs | test NLL | train NLL | CRPS | MAE | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9 | 20 | -0.1844 | -0.2465 | nan | 0.1357 | 0.9809 | 0.9827 |
| 1 | 28 | 30 | -0.3402 | -0.3413 | nan | 0.1175 | 0.9848 | 0.9838 |
| 2 | 15 | 26 | -0.2015 | -0.2374 | nan | 0.1328 | 0.9785 | 0.9770 |
| 3 | 23 | 30 | -0.3334 | -0.4025 | nan | 0.1099 | 0.9903 | 0.9900 |
| 4 | 11 | 22 | -0.2075 | -0.2012 | nan | 0.1262 | 0.9850 | 0.9855 |
| **mean±std** | | | -0.2534±0.0686 | | nan±nan | 0.1244±0.0096 | 0.9839±0.0040 | 0.9838±0.0042 |

**Artifacts:** `stats.csv`, `params.json`, `metadata.json`, `trainingstats_all.png` (in `CLEAN_trained_models/sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8/tmax`)

