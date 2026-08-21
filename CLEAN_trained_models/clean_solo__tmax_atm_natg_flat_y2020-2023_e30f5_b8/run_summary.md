# Training run: clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8

- Generated: 2026-07-23 12:42:10
- Git commit: `a1bfb03`
- Host: rchristine-864fa536af94-0 (Linux-5.15.0-1114-azure-x86_64-with-glibc2.39)
- Python: 3.11.15 | torch: 2.6.0+cu124 | CUDA: 12.4
- Device: `cuda`
- GPUs (1): NVIDIA A100 80GB PCIe MIG 2g.20gb
- DataParallel: False
- Peak GPU memory: 2648 MB
- CPU cores: 24 | torch threads: 24
- Total wall time: 3h 16m 43s

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
| N_EPOCHS | 30 |
| BATCH_SIZE | 8 |
| LR | 0.0005 |
| PATIENCE | 10 |
| N_FOLDS | 5 |
| SEED | 42 |
| TRIAL_NAME | clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8 |
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
- Gradient clip: None
- Fold wall times: 26m 10s, 43m 5s, 42m 5s, 41m 49s, 42m 26s  (total 3h 15m 38s)

| Fold | best epoch | epochs | test NLL | train NLL | MAE | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 7 | 18 | -0.1364 | -0.0285 | 0.1351 | 0.9796 | 0.9821 |
| 1 | 25 | 30 | -0.2544 | -0.3729 | 0.1293 | 0.9826 | 0.9812 |
| 2 | 19 | 30 | -0.1603 | -0.3278 | 0.1374 | 0.9783 | 0.9771 |
| 3 | 28 | 30 | -0.3291 | -0.2853 | 0.1241 | 0.9879 | 0.9890 |
| 4 | 29 | 30 | -0.3364 | -0.3460 | 0.1220 | 0.9863 | 0.9858 |
| **mean±std** | | | -0.2433±0.0830 | | 0.1296±0.0060 | 0.9830±0.0037 | 0.9830±0.0041 |

**Artifacts:** `stats.csv`, `params.json`, `metadata.json`, `trainingstats_all.png` (in `CLEAN_trained_models/clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8/tmax`)

