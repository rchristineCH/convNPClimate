#!/usr/bin/env bash
#
# Feature importance for every CLEAN tmax model, on the 2024 HOLDOUT year.
#
# Generalizes scripts/fi_pfi_all_tmax_2024.sh (which this replaces) from PFI-only to
# any single method, plus an optional cross-method combine stage.
#
# WHY not YEAR=2024 bash scripts/fi_run_local.sh per model:
#   - fi_run_local.sh writes into <model_dir>/feature_importance, which already holds the
#     committed 2022 study (pfi.csv, lime.csv, REPORT.md, group figures) for all 7 models.
#     Overwriting it would destroy that study. Everything here goes to
#     <model_dir>/feature_importance_2024 instead.
#   - It always runs PFI *and* LIME; the two are staged separately here so the ~12 h PFI
#     pass and the ~1 h LIME pass can be launched, monitored and resumed independently.
#
# 2024 is a genuine holdout for every model (all trained on 2020-2023), and the context is
# normalized with the TRAINING manifest — _build_backbone goes through
# predict.build_{atmospheric,surface}_context(manifest, ...), never re-fit on 2024. So these
# importances are out-of-sample, unlike the in-sample 2022 study.
#
# NOTE on summary.md: feature_importance.py writes a METHOD-LOCAL summary.md, so a later
# method overwrites the earlier one (this is why the committed 2022 summaries are LIME-only
# and its PFI numbers live only in pfi.csv). METHOD=combine fixes that — scripts/fi_combine.py
# needs only two of the three CSVs, so PFI+LIME yields a real cross-method summary.md plus
# REPORT.md without SHAP.
#
# Measured on the 2g.20gb MIG slice of the A100 (--day-batch 64, 366 days, 5 folds,
# 5 permutation repeats): PFI ~15.5 s/eval and the eval count scales with IN_CHANNELS, so
# ~11 min for the two surface models and hours for the atmospheric ones. LIME is far cheaper
# (4 representative days x 1000 masks).
#
# Usage:
#   METHOD=pfi     nohup bash scripts/fi_all_tmax_2024.sh > fi_pfi_2024.log  2>&1 &
#   METHOD=lime    nohup bash scripts/fi_all_tmax_2024.sh > fi_lime_2024.log 2>&1 &
#   METHOD=combine bash scripts/fi_all_tmax_2024.sh          # cross-method summary + REPORT.md
# Env:
#   METHOD=pfi     pfi | lime | shap | combine
#   MODELS="a b"   run only these model dir names (default: all 7)
#   YEAR=2024      evaluation year
#   DAY_BATCH=64   days per forward batch
#   NO_GIT=1       do not commit between models
#   DRY_RUN=1      print the commands only
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

METHOD="${METHOD:-pfi}"
YEAR="${YEAR:-2024}"
DAY_BATCH="${DAY_BATCH:-64}"
NO_GIT="${NO_GIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
ROOT=CLEAN_trained_models

case "$METHOD" in
  pfi|lime|shap|combine) ;;
  *) echo "ERROR: METHOD must be pfi|lime|shap|combine, got '$METHOD'" >&2; exit 1 ;;
esac

# Cheapest first, so a mistake in the setup surfaces in minutes rather than hours.
DEFAULT_MODELS="
baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo
baseline__tmax_sfc_flat_y2020-2023_e30f5_b8
clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8
lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8
sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8
clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8
lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8
"
MODELS="${MODELS:-$DEFAULT_MODELS}"

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$NO_GIT" != "1" && ( "$branch" == "master" || "$branch" == "main" ) ]]; then
  echo "ERROR: on '$branch' — branch first before auto-committing (working agreement)." >&2
  exit 1
fi

echo "=== ${METHOD} on the ${YEAR} holdout — $(echo $MODELS | wc -w) tmax models"
echo "    day-batch $DAY_BATCH | branch $branch | started $(date -Is)"

failed=()
for m in $MODELS; do
  md="${ROOT}/${m}/tmax"
  out="${md}/feature_importance_${YEAR}"
  if [[ ! -d "$md" ]]; then
    echo "!!! SKIP ${m}: no such model dir ${md}"; failed+=("$m(missing)"); continue
  fi

  if [[ "$METHOD" == "combine" ]]; then
    # fi_evaluate.sh reads FI_DIR/YEAR/MODEL_DIR from the environment and does its own
    # commit; it needs >=2 method CSVs, which PFI+LIME satisfies.
    echo
    echo "=== [$(date +%H:%M:%S)] ${m} (combine)"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "+ MODEL_DIR=$md YEAR=$YEAR FI_DIR=$out bash scripts/fi_evaluate.sh"; continue
    fi
    if MODEL_DIR="$md" YEAR="$YEAR" FI_DIR="$out" NO_GIT="$NO_GIT" \
         bash scripts/fi_evaluate.sh; then
      echo "=== [$(date +%H:%M:%S)] ${m} OK"
    else
      echo "!!! [$(date +%H:%M:%S)] ${m} FAILED — continuing"; failed+=("$m")
    fi
    continue
  fi

  # Already complete from an earlier (possibly interrupted) invocation -> skip, so the
  # script is resumable: rerunning it costs nothing for the models already done.
  if [[ -f "${out}/${METHOD}.csv" ]]; then
    echo "=== SKIP ${m}: ${out}/${METHOD}.csv already exists"; continue
  fi

  echo
  echo "=== [$(date +%H:%M:%S)] ${m}"
  mkdir -p "$out"
  cmd=(python feature_importance.py --method "$METHOD" --model-dir "$md" --year "$YEAR"
       --day-batch "$DAY_BATCH" --output-dir "$out")
  echo "+ ${cmd[*]}"
  if [[ "$DRY_RUN" == "1" ]]; then continue; fi

  if "${cmd[@]}" 2>&1 | tee "${out}/fi_${METHOD}.log"; then
    echo "=== [$(date +%H:%M:%S)] ${m} OK"
  else
    echo "!!! [$(date +%H:%M:%S)] ${m} FAILED — continuing with the rest"
    failed+=("$m")
    continue
  fi

  if [[ "$NO_GIT" != "1" ]]; then
    git add "$out"
    if ! git diff --cached --quiet -- "$out"; then
      git commit -m "feature-importance: ${METHOD} on the ${YEAR} holdout — ${m}" \
        && git push && echo "committed + pushed: ${m}"
    fi
  fi
done

echo
echo "=== finished $(date -Is)"
if (( ${#failed[@]} )); then
  echo "FAILED: ${failed[*]}"; exit 1
fi
echo "all models OK"
