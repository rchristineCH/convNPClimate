# Naming convention for collected result figures

Every result PNG produced under `CLEAN_trained_models/` is copied into one flat folder for
the report, at `LATEX_REPORT/images/results/` on the `doc` worktree. This file defines the
names in that folder. `scripts/collect_result_figures.py` implements it; nothing there
should ever be renamed by hand.

## The scheme

```
<variable>__<original name>__<evaluation mode>__<model>.png
```

Fields are separated by a double underscore, so each field may contain single underscores
of its own. Read it as: *what was modelled, which figure, scored how, by which run.*

```
tmax__mae_vs_altitude__error_analysis_cv__lbclip_atm.png
precip__amount_distribution__eval_2024__clean_solo_precip_crps.png
tmax__pfi_heatmap_mae__feature_importance__clean_solo_wind.png
precip__skill_timeseries__precip_processing_comparison__ALL.png
```

The variable comes first so the folder sorts into a tmax block and a precip block; within
each, figures of the same kind sit together across models, which is what makes them easy to
pull into a chapter.

### `<variable>`

`tmax` or `precip` — the model's target, taken from the variable subdirectory of the run
(`<run>/tmax/…`, `<run>/precip/…`). Cross-model figures take it from their comparison
directory: `tmax_model_comparison` → `tmax`, `precip_processing_comparison` → `precip`.

### `<original name>`

The file's stem exactly as the evaluation wrote it — `crps_map`, `spatial_wetday_freq`,
`trainingstats_fold2`. Never shortened or reworded: it is the link back to the code that
drew the figure.

### `<evaluation mode>`

The source subdirectory, which is what stops the collection from collapsing distinct
figures — `eval_cv`, `error_analysis_cv` and `station_analysis_cv` all contain a
`mae_vs_altitude.png`.

| Mode | Meaning |
|---|---|
| `eval_cv` | CV holdout over 2020-2023; each fold scores only its own held-out block |
| `eval_2024` | 2024 holdout year; all folds ensembled, training-frozen normalisation |
| `error_analysis_cv` / `error_analysis_2024` | `error_analysis.py` for that regime |
| `station_analysis_cv` / `station_analysis_2024` | `station_analysis.py` for that regime |
| `feature_importance` | PFI / SHAP / LIME study (regime-independent) |
| `training` | PNGs at the top of the run directory — training curves, architecture graph |
| `tmax_model_comparison` / `precip_processing_comparison` | cross-model figures |

Precip writes its evaluation figures to `eval_figures/` and `eval_figures_2024/`; these are
mapped onto `eval_cv` and `eval_2024` so both variables read alike. The two regimes must
never be mixed in one comparison — see the evaluation-regime note in the README.

### `<model>`

The run's short label, from `MODEL_LABELS` in the collector — the prefix each run directory
carries before its config slug:

| Label | Variable | Run |
|---|---|---|
| `baseline` | tmax | surface inputs |
| `baseline_no_geo` | tmax | surface inputs, no geopotential |
| `clean_solo` | tmax | atmospheric z,t,q |
| `clean_solo_wind` | tmax | atmospheric z,t,q,u,v |
| `sfc_anchors` | tmax | atmospheric + t2m/tp surface anchors |
| `lbclip_atm` | tmax | z,t,q, 60 epochs + gradient clipping |
| `lbclip_wind` | tmax | z,t,q,u,v, 60 epochs + gradient clipping |
| `clean_solo_precip` | precip | Bernoulli-Gamma NLL |
| `clean_solo_precip_crps` | precip | Bernoulli-Gamma CRPS |
| `clean_solo_precip_ftcrps` | precip | NLL then CRPS fine-tune |
| `clean_solo_precip_wind` | precip | Bernoulli-Gamma NLL, z,t,q,u,v |
| `ALL` | either | cross-model figure, no single owning run |

## Regenerating

```bash
python scripts/collect_result_figures.py \
    --out /home/marc/convNPClimate-doc/LATEX_REPORT/images/results \
    --index RESULT_FIGURES_INDEX.md
```

Copies, never moves; the originals under `CLEAN_trained_models/` stay untouched. PNGs in the
destination that the run did not produce are deleted first, so the folder can never hold two
naming schemes at once — anything else there, including `README.md`, is left alone. The
tracked `RESULT_FIGURES_INDEX.md` maps every collected name back to the path it came from.

The script stops rather than guess:

- **a name clash** — two sources competing for one target name;
- **an unknown run directory** — a new model needs a `MODEL_LABELS` entry first.

## Adding to it

- **New model:** add the run directory to `MODEL_LABELS` with a short label. Keep it distinct
  from the other labels after the `__` split, and keep using single underscores.
- **New figure in an existing directory:** nothing to do, it is picked up by name.
- **New output directory:** add it to `MODE_LABELS` (per-model) or `COMPARISON_DIRS`
  (cross-model, with its variable). Until then its figures are reported as skipped rather
  than silently dropped.

## Known wrinkles

- The 15 `ALL` figures duplicate `images/tmax_model_comparison/` and
  `images/precip_processing_comparison/`, which predate this folder and have their own
  READMEs.
- `clean_solo_precip_crps/precip/eval_2024/` is skipped on purpose: a partial `eval_precip`
  run superseded two minutes later by `eval_figures_2024/`, which is the one collected.
