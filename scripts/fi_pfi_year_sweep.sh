#!/usr/bin/env bash
#
# PFI year sweep for the two best precip runs (WIND and SFC-TP): does the choice of
# evaluation year materially change the attributed importances?
#
# Fills in 2020, 2021 and 2023. 2022 (feature_importance/) and 2024
# (feature_importance_2024/) already exist at full resolution, so together this gives
# a five-year series per model.
#
# STRIDE 1 DELIBERATELY. The existing 2022/2024 studies used every day; subsampling
# the new years would make the series incomparable, which is the one thing this
# sweep exists to measure. That is what makes it expensive.
#
# FOLD ENSEMBLE: feature_importance.py loads every model_fold_* checkpoint and
# ensembles them, so this is already "on the ensemble" -- there is no separate mode.
#
# READ THE RESULT WITH THIS CAVEAT: 2020-2023 are IN-SAMPLE (the models trained on
# 2020-2023, and the 5-fold ensemble means ~80% of each of those years was seen in
# training by any given fold). 2024 is the only genuine holdout. A year-to-year
# difference can therefore reflect in-sample vs holdout as much as the year itself.
#
# sfctp reads tp as an INPUT, and is driven from the rebuilt next00 field for EVERY
# year here. next00 is bit-identical to the legacy files on 2020-2023 and additionally
# contains 2023-12-31, which the legacy set lacks -- so this keeps all five years on
# one input convention with full coverage, rather than 2023 silently scoring 364 days.
#
# Resume-safe: a year whose pfi.csv exists is skipped.
#
# Usage:  bash scripts/fi_pfi_year_sweep.sh
#         DRY_RUN=1 bash scripts/fi_pfi_year_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."

YEARS=(2020 2021 2023)
TP_DIR="${TP_DIR:-datasets/ERA5_Land/tp_hourly}"
SFCTP_INPUT_GLOB="${TP_DIR}/tp-*.nc"
DEVICE="${DEVICE:-cuda}"
DAY_BATCH="${DAY_BATCH:-64}"
LOG_DIR="reeval_logs"
DRY_RUN="${DRY_RUN:-0}"
VENV="${VENV:-/home/renku/work/.venv}"

if ! python -c "import torch" 2>/dev/null; then
  if [[ -z "${VIRTUAL_ENV:-}" && -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
  fi
fi
python -c "import torch" 2>/dev/null || { echo "FATAL: torch not importable" >&2; exit 1; }

SFCTP="clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8"
WIND="clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8"
# Cheap model first: a fast full pass surfaces any wiring problem before the
# 155-channel run commits four hours to it.
MODELS=("$SFCTP" "$WIND")

mkdir -p "$LOG_DIR"
failed=()
for m in "${MODELS[@]}"; do
  for y in "${YEARS[@]}"; do
    dir="CLEAN_trained_models/${m}/precip"
    out="${dir}/feature_importance_${y}"
    if [ -f "${out}/pfi.csv" ]; then
      echo "[skip] ${out}/pfi.csv exists"
      continue
    fi
    args=(--method pfi --model-dir "$dir" --year "$y" --output-dir "$out"
          --device "$DEVICE" --day-batch "$DAY_BATCH")
    [ "$m" = "$SFCTP" ] && args+=(--precip-glob "$SFCTP_INPUT_GLOB")
    echo "[run ] ${m%%__*} ${y} -> ${out}"
    if [ "$DRY_RUN" = "1" ]; then
      echo "       python feature_importance.py ${args[*]}"
      continue
    fi
    if python feature_importance.py "${args[@]}" > "${LOG_DIR}/fi_pfi_${y}__${m}.log" 2>&1; then
      echo "[done] ${m%%__*} ${y}"
    else
      echo "[FAIL] ${m%%__*} ${y} -- see ${LOG_DIR}/fi_pfi_${y}__${m}.log" >&2
      failed+=("$m/$y")
    fi
  done
done

echo
if [ "${#failed[@]}" -gt 0 ]; then
  echo "FAILED (${#failed[@]}):" >&2; printf '  %s\n' "${failed[@]}" >&2; exit 1
fi
echo "[all done] PFI year sweep complete. Summarise with:"
echo "  python scripts/fi_year_comparison.py"
