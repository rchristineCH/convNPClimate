# convNPClimate — ConvCNP downscaling of daily weather fields over Switzerland

Statistical downscaling of daily ERA5 / ERA5-Land fields (~11 km and coarser) to the
~1 km MeteoSwiss analysis grids over Switzerland using Convolutional Conditional
Neural Processes (ConvCNPs), with built-in uncertainty quantification. Two target
variables are modelled:

- **tmax** — daily maximum 2 m temperature, with a **Gaussian** output distribution
  (targets: MeteoSwiss TmaxD).
- **precip** — daily precipitation, with a **Bernoulli–Gamma** output distribution
  (targets: MeteoSwiss RhiresD).

The full write-up lives in `LATEX_REPORT/` (compiled hand-in copy:
`LATEX_REPORT/REPORT_CAPSTONE_CHRISTINE_ROTH_V0.pdf`).

## Repository layout

| Path | Contents |
|------|----------|
| `convCNP/` | The model package (encoder, CNN decoder, RBF final layers, elevation correction). Install with `pip install -e convCNP/`. |
| `train.py`, `finetune_precip.py` | Training entry points (k-fold CV; optional CRPS fine-tune stage for precip) |
| `evaluate.py`, `report.py` | tmax evaluation (CV holdout + holdout year) and report rendering |
| `eval_precip.py`, `eval_precip_figures.py`, `precip_baseline.py` | precip evaluation: physical metrics (wet-day frequency, SDII, R10, R01), BSS/RPSS, figures |
| `predict.py`, `inference.py`, `infer.py` | Inference over date ranges from a trained model directory |
| `datasets.py`, `params.py`, `model_factory.py` | Data loading, configuration dataclass, model construction |
| `feature_importance.py` | PFI / SHAP / LIME channel-attribution study for the atmospheric models |
| `error_analysis.py` | Per-grid-point error structure (terrain, season, spatial maps) from the prediction bundles |
| `station_analysis.py`, `station_exploration/` | Verification at independent MeteoSwiss stations (see `station_exploration/README.md`) |
| `compare_evaluations.py`, `compare_precip_models.py` | Cross-run comparison tables and figures |
| `scripts/` | Run drivers (`clean_run_*.sh`), feature-importance drivers (`fi_*`), and the report-figure generators (`gen_*.py`, `overlay_*.py`, `collect_result_figures.py`) |
| `CLEAN_trained_models/` | The twelve runs behind the report: configs, manifests, training stats, evaluation metrics and figures (see below) |
| `datasets/` | Download scripts and instructions for all input data (`datasets/README.md`); the data itself is not in git |
| `tp_hourly/` | Rebuild of daily ERA5-Land precipitation from hourly data, and the audit that motivated it (`tp_hourly/README.md`) |
| `docs/` | Data-processing notes, feature-importance methods, figure-naming conventions, background papers |
| `LATEX_REPORT/` | LaTeX sources, figures and compiled PDF of the final report |
| `RESULT_FIGURES_INDEX.md` | Generated index mapping every collected result figure back to the run and file it came from |

## Getting started

```bash
pip install -r requirements.txt
pip install -e convCNP/
```

Then fetch the input data following `datasets/README.md` (CDS API for ERA5,
MeteoSwiss grid products for the targets). All paths are resolved relative to
`datasets/` by `datasets.build_data_paths()`.

## Training

```bash
# tmax, surface ERA5-Land input, fast smoke test
python train.py --variables tmax --use-surface --data-year-start 2023 --n-epochs 5 --n-folds 2 --batch-size 8

# tmax with atmospheric pressure-level inputs on the native coarse grid
python train.py --variables tmax --use-atmospheric --atmos-native-grid --data-year-start 2020 --data-year-end 2023

# tmax with wind components added to the atmospheric input set
python train.py --variables tmax --use-atmospheric --atmos-native-grid \
  --atmos-variables z t q u v --data-year-start 2020 --data-year-end 2023

# precip with the default Bernoulli-Gamma distribution
python train.py --variables precip --use-atmospheric --atmos-native-grid --data-year-start 2020 --data-year-end 2023

# precip on the surface input set (ERA5-Land tp + t2m_max); --use-surface alone
# carries no precipitation channel, so --use-surface-precip is required
python train.py --variables precip --use-surface --use-surface-precip --data-year-start 2020 --data-year-end 2023
```

The full training-plus-evaluation drivers used for the report runs are
`scripts/clean_run_*.sh`; `scripts/run_precip_sfctp_nll_then_crps_ft.sh` chains
NLL training, CRPS fine-tuning and evaluation for the surface-tp precip model.

All hyperparameters live in `params.py` (`Params` dataclass); run directories are
auto-named from the configuration.

## Evaluation — two regimes, never mixed

- **CV holdout** (`evaluate.py --model-dir <dir>`): scores the training span
  2020–2023, each fold predicting only its own held-out block, so every day is
  predicted exactly once by a model that never saw it.
- **Holdout year** (`evaluate.py --model-dir <dir> --eval-year 2024`): scores an
  unseen year with all folds ensembled and training-frozen normalisation.

The same split applies to precip via `eval_precip.py`. Scoring the full ensemble
over the training years would be an in-sample number and is deliberately not
supported.

```bash
python evaluate.py    --model-dir CLEAN_trained_models/<run>/tmax               # CV holdout
python evaluate.py    --model-dir CLEAN_trained_models/<run>/tmax --eval-year 2024
python eval_precip.py --model-dir CLEAN_trained_models/<run>/precip
python predict.py     --model-dir <run>/tmax --date-start 2024-01-01 --date-end 2024-12-31 --output preds.npz
python feature_importance.py --method all --model-dir <run>/tmax --year 2022
```

## Trained runs and report figures

`CLEAN_trained_models/` holds the twelve runs the report is built on
(tmax: atm, atm-clip60, atm+wind, atm+wind-clip60, atm+sfcanchors, sfc, sfc-nogeo;
precip: NLL, CRPS, WIND, FT-CRPS, SFC-TP). Each run directory carries its
`params.json`, `manifest.json`, normalisation metadata, per-fold training stats,
evaluation metrics (JSON) and evaluation figures for both regimes, plus the
feature-importance outputs where the study was run.

To keep the repository lean, model checkpoints and prediction caches are **not**
in git; reproducing a run end-to-end means retraining with the recorded
`params.json` via the matching `scripts/clean_run_*.sh` driver. The report's
figures were collected from these directories by
`scripts/collect_result_figures.py`; `RESULT_FIGURES_INDEX.md` maps each figure
name back to its source file. The bespoke report figures are generated by
`scripts/gen_*.py` and `scripts/overlay_*.py`.

## Gotchas worth knowing

- **Skill scores are baseline-relative.** `skill_score = 1 − CRPS/CRPS_ref`
  against a bilinear-ERA5 reference; two runs are only skill-comparable when the
  reference matches. MAE/RMSE/CRPS against the fixed MeteoSwiss truth are always
  comparable. `compare_evaluations.py` re-derives each run's reference and warns
  on divergence.
- **Gaussian ensembling is a moment match, not a parameter average**:
  `mu = mean(mu_k)`, `sigma = sqrt(mean(sigma_k²) + var(mu_k))`. Precip averages
  its `(rho, alpha, beta)` elementwise.
- **Temperature is handled in Kelvin internally**; multiply sigma by the stored
  `data_std` for spread in °C. Precip targets are raw mm — no z-scoring, which
  would break the Gamma log-probability at zero.
- **ERA5-Land `tp` files are labelled by accumulation window, not calendar
  year**, and the CDS `tp-2024.nc` download is day-shifted and wet-biased; see
  `tp_hourly/README.md` for the audit and the corrected rebuild. Alignment is
  floor-to-day with no ±1 shift, implemented once in
  `datasets.load_era5_precip_aligned`.
- **`params.IN_CHANNELS` defaults to 0 (deliberately invalid)** — set it from the
  loaded context tensor via `params.with_in_channels(...)` before
  `model_factory.build_model()`.
- **Zarr versions**: if `datasets/topo_subset.zarr/zarr.json` exists, rename it
  away — the v3 metadata file breaks v2 loading.

## Origins

The ConvCNP implementation started from the code accompanying Vaughan et al.
(2022), *Convolutional conditional neural processes for local climate
downscaling* (GMD 15, 251–268); see `convCNP/README.md` for its attribution.
