# Training run: clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8

- Generated: 2026-08-01 02:47:11
- Git commit: `085e48b`
- Host: rchristine-e3de6a0e9d2c-0 (Linux-5.15.0-1116-azure-x86_64-with-glibc2.39)
- Python: 3.11.15 | torch: 2.6.0+cu124 | CUDA: 12.4
- Device: `cuda`
- GPUs (1): NVIDIA A100 80GB PCIe MIG 2g.20gb
- DataParallel: False
- Peak GPU memory: 9746 MB
- CPU cores: 24 | torch threads: 24
- Total wall time: 10h 15m 52s

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
| USE_SURFACE | True |
| USE_ATMOSPHERIC | False |
| USE_SURFACE_PRECIP | True |
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
| IN_CHANNELS | 7 |
| N_EPOCHS | 30 |
| CRPS_N_SAMPLES | 25 |
| BATCH_SIZE | 8 |
| LR | 0.0005 |
| PATIENCE | 10 |
| N_FOLDS | 5 |
| SEED | 42 |
| TRIAL_NAME | clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8 |
| RUN_TYPE | cloud |
| DEVICE | cuda |

## Data

- Grid mode: **fine surface + ERA5-Land precip**
- Context tensor (time, channel, lat, lon): `(1460, 7, 29, 61)`
- Input channels: 7
- Time steps: 1460 (2020-01-01 -> 2023-12-30)
- Normalization: data_mean=285.5405, data_std=9.0359, lat=[45.400, 48.200], lon=[5.000, 11.000]
- Encoder: **flat**

### Channel groups (by physical variable)

| Group | #channels | Members |
| --- | --- | --- |
| `scaffold` | 6 | data, lat, lon, cos_time, sin_time, elevation |
| `tp` | 1 | tp |

## Results

### precip (bernoulli_gamma)

- Model trainable params: 282,694
- MeteoSwiss target points: 98,050
- Gradient clip: 1.0
- Fold wall times: 2h 15m 8s, 2h 0m 33s, 1h 59m 51s, 1h 59m 3s, 1h 58m 30s  (total 10h 13m 7s)

| Fold | best epoch | epochs | test NLL | train NLL | CRPS | MAE | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 8 | 1.5236 | 1.4631 | 1.6584 | 2.2563 | 0.7793 | 0.8169 |
| 1 | 27 | 30 | 1.6174 | 1.5845 | 1.7434 | 2.5077 | 0.7939 | 0.8515 |
| 2 | 28 | 30 | 1.2844 | 1.5072 | 1.1459 | 1.7063 | 0.7911 | 0.7988 |
| 3 | 28 | 30 | 1.4849 | 1.6618 | 1.5119 | 2.0072 | 0.7547 | 0.8289 |
| 4 | 25 | 30 | 1.7893 | 1.5253 | 2.0259 | 2.9130 | 0.7958 | 0.8205 |
| **mean±std** | | | 1.5399±0.1654 | | 1.6171±0.2891 | 2.2781±0.4139 | 0.7829±0.0152 | 0.8233±0.0172 |

**Artifacts:** `stats.csv`, `params.json`, `metadata.json`, `trainingstats_all.png` (in `CLEAN_trained_models/clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8/precip`)

