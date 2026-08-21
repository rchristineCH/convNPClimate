#!/usr/bin/env bash
#
# Feature importance on the 2024 holdout year for the five CLEAN precip runs.
#
# Generalizes the PFI-only fi_pfi_2024_precip.sh (which this replaces) to any single
# method plus a combine stage, exactly as scripts/fi_all_tmax_2024.sh does for tmax.
#
# WHY 2024: the existing studies were run on 2022, an IN-SAMPLE year (training
# spans 2020-2023). Re-running on the holdout puts feature importance on the same
# footing as every other 2024 number the corrected w0606 re-evaluation produced.
#
# Writes to <model>/precip/feature_importance_2024/ so the committed 2022 study is
# preserved side by side rather than overwritten -- the two are answering different
# questions (in-sample vs holdout) and both are worth keeping.
#
# The sfctp run reads tp as an INPUT channel, and datasets/ERA5_Land/precipitation/
# tp-2024.nc is the defective file (day-shifted, ~67% wet-biased). It is driven from
# the rebuilt next00 field via --precip-glob instead, matching what
# scripts/reeval_precip_w0606.sh does for that run's 2024 evaluation. Without this
# the encoder would be fed a corrupt channel and the importances would be garbage.
#
# METHOD=combine: feature_importance.py writes a METHOD-LOCAL summary.md, so LIME
# overwrites PFI's. scripts/fi_evaluate.sh (via scripts/fi_combine.py) needs only two
# of the three method CSVs, so PFI+LIME yields a real cross-method summary.md plus
# REPORT.md without waiting on SHAP. Run it after the second method or the first
# method's summary is lost.
#
# Cost (A100 MIG slice, --day-batch 64): PFI scales with the channel count (~13-14 h
# for all five: 95/155 channels atmospheric, 7 for sfctp); LIME does not (4
# representative days x 1000 masks), ~20-60 min per model.
#
# Resume-safe: a model whose <method>.csv already exists is skipped, and a failure
# does not abort the remaining models.
#
# Usage:  METHOD=lime    bash scripts/fi_all_precip_2024.sh
#         METHOD=pfi     bash scripts/fi_all_precip_2024.sh
#         METHOD=combine bash scripts/fi_all_precip_2024.sh
#         DRY_RUN=1 METHOD=lime bash scripts/fi_all_precip_2024.sh
set -uo pipefail
cd "$(dirname "$0")/.."

METHOD="${METHOD:-pfi}"
YEAR="${YEAR:-2024}"
TP_DIR="${TP_DIR:-datasets/ERA5_Land/tp_hourly}"
SFCTP_INPUT_GLOB="${TP_DIR}/tp-*.nc"        # 00-00 UTC next00, continuous with training
DEVICE="${DEVICE:-cuda}"
DAY_BATCH="${DAY_BATCH:-64}"
LOG_DIR="reeval_logs"
DRY_RUN="${DRY_RUN:-0}"
NO_GIT="${NO_GIT:-1}"
VENV="${VENV:-/home/renku/work/.venv}"

case "$METHOD" in
  pfi|lime|shap|combine) ;;
  *) echo "FATAL: METHOD must be pfi|lime|shap|combine, got '$METHOD'" >&2; exit 1 ;;
esac

if ! python -c "import torch" 2>/dev/null; then
  if [[ -z "${VIRTUAL_ENV:-}" && -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
  fi
fi
python -c "import torch" 2>/dev/null || { echo "FATAL: torch not importable" >&2; exit 1; }

MODELS=(
  "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8"
  "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps"
  "clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8"
  "clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8"
  "clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8"
)
SFCTP="clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8"

mkdir -p "$LOG_DIR"
echo "=== ${METHOD} on the ${YEAR} holdout — ${#MODELS[@]} precip models"
echo "    day-batch ${DAY_BATCH} | device ${DEVICE} | started $(date -Is)"
failed=()
for m in "${MODELS[@]}"; do
  dir="CLEAN_trained_models/${m}/precip"
  out="${dir}/feature_importance_${YEAR}"

  if [ "$METHOD" = "combine" ]; then
    # The combine stage runs its own baseline forward pass, so sfctp needs the same tp
    # input override the method runs used — otherwise the report is normalised against
    # the defective tp-2024.nc (PR_MAE 4.673) while its importances came from the
    # rebuilt field (2.707).
    fi_glob=""
    [ "$m" = "$SFCTP" ] && fi_glob="$SFCTP_INPUT_GLOB"
    echo "[run ] ${m} (combine) -> ${out}"
    if [ "$DRY_RUN" = "1" ]; then
      echo "       MODEL_DIR=$dir YEAR=$YEAR FI_DIR=$out FI_PRECIP_GLOB=$fi_glob bash scripts/fi_evaluate.sh"
      continue
    fi
    if MODEL_DIR="$dir" YEAR="$YEAR" FI_DIR="$out" NO_GIT="$NO_GIT" FI_PRECIP_GLOB="$fi_glob" \
         bash scripts/fi_evaluate.sh > "${LOG_DIR}/fi_combine_${YEAR}__${m}.log" 2>&1; then
      echo "[done] ${m}"
    else
      echo "[FAIL] ${m} -- see ${LOG_DIR}/fi_combine_${YEAR}__${m}.log" >&2
      failed+=("$m")
    fi
    continue
  fi

  if [ -f "${out}/${METHOD}.csv" ]; then
    echo "[skip] ${out}/${METHOD}.csv already exists"
    continue
  fi
  args=(--method "$METHOD" --model-dir "$dir" --year "$YEAR" --output-dir "$out"
        --device "$DEVICE" --day-batch "$DAY_BATCH")
  if [ "$m" = "$SFCTP" ]; then
    args+=(--precip-glob "$SFCTP_INPUT_GLOB")
  fi
  echo "[run ] ${m} -> ${out}"
  if [ "$DRY_RUN" = "1" ]; then
    echo "       python feature_importance.py ${args[*]}"
    continue
  fi
  if python feature_importance.py "${args[@]}" \
       > "${LOG_DIR}/fi_${METHOD}_${YEAR}__${m}.log" 2>&1; then
    echo "[done] $(date +%H:%M:%S) ${m}"
  else
    echo "[FAIL] ${m} -- see ${LOG_DIR}/fi_${METHOD}_${YEAR}__${m}.log" >&2
    failed+=("$m")
  fi
done

echo
if [ "${#failed[@]}" -gt 0 ]; then
  echo "FAILED (${#failed[@]}):" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "[all done] ${METHOD} ${YEAR} complete for ${#MODELS[@]} precip runs."
