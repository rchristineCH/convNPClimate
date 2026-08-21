# Model architecture graphs — CLEAN solo runs (2020-2023)

Rendered by `visualize_architecture.py` from each run's `params.json` on
`CLEAN_SOLO_RUNS_2020-2023`; one file per model (short name); collected here for the report.

| Short name | Full run | Model | Loss | Params |
|---|---|---|---|---|
| `tmax-atm` | clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8 | Gaussian tmax, atm z/t/q | gll | 307,403 |
| `tmax-atm-anchors` | sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8 | Gaussian tmax, atm + surface anchors | gll | 310,233 |
| `tmax-atm-wind` | clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8 | Gaussian tmax, atm z/t/q/u/v | gll | 324,383 |
| `tmax-surface` | baseline__tmax_sfc_flat_y2020-2023_e30f5_b8 | Gaussian tmax, surface | gll | 282,216 |
| `tmax-surface-nogeo` | baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo | Gaussian tmax, surface w/o geopotential | gll | 281,933 |
| `precip-nll` | clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8 | Bernoulli-Gamma, atm z/t/q | NLL | 307,598 |
| `precip-crps` | clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8 | Bernoulli-Gamma, atm z/t/q | CRPS | 307,598 |
| `precip-wind` | clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8 | Bernoulli-Gamma, atm z/t/q/u/v | NLL | 324,578 |
| `precip-ft-crps` | clean_solo_precip__…_e30f5_b8__ft-crps | Bernoulli-Gamma, atm z/t/q (NLL warm-start) | CRPS fine-tune | 307,598 |
