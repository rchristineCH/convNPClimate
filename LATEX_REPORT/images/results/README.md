# Collected result figures

Every result PNG from `CLEAN_trained_models/`, flattened into this folder and renamed

```
<variable>__<original name>__<evaluation mode>__<model>.png
```

for example `tmax__mae_vs_altitude__error_analysis_cv__lbclip_atm.png`. Include them as
`images/results/<name>.png`.

The full convention — the allowed value of every field, and what to do when a model or an
output directory is added — is `docs/result_figure_naming.md` on the run branch
(`CLEAN_SOLO_RUNS_2020-2023`).

Generated, never edited by hand. To refresh after an evaluation:

```bash
python scripts/collect_result_figures.py \
    --out /home/marc/convNPClimate-doc/LATEX_REPORT/images/results \
    --index RESULT_FIGURES_INDEX.md
```

That run rewrites the folder in full: PNGs it did not produce are deleted, so the names here
always match one scheme. `RESULT_FIGURES_INDEX.md`, on the run branch, maps every name back
to the path it came from.

The 15 `ALL` figures are cross-model comparisons and duplicate `../tmax_model_comparison/`
and `../precip_processing_comparison/`, which predate this folder.
