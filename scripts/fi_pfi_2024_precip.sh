#!/usr/bin/env bash
#
# PFI only, on the 2024 holdout year, for the five CLEAN precip runs.
#
# WHY 2024: the existing studies were run on 2022, an IN-SAMPLE year (training
# spans 2020-2023). Re-running on the holdout puts feature importance on the same
# footing as every other 2024 number the corrected w0606 re-evaluation produced.
#
# Writes to <model>/precip/feature_importance_2024/ so the committed 2022 study is
# preserved side by side rather than overwritten -- the two are answering different
# questions (in-sample vs holdout) and both are worth keeping.
#
# LIME and SHAP are deliberately not run: PFI is the method whose cost scales with
# the channel count and whose 2022 numbers are the ones quoted in the write-up.
#
# The sfctp run reads tp as an INPUT channel, and datasets/ERA5_Land/precipitation/
# tp-2024.nc is the defective file (day-shifted, ~67% wet-biased). It is driven from
# the rebuilt next00 field via --precip-glob instead, matching what
# scripts/reeval_precip_w0606.sh does for that run's 2024 evaluation. Without this
# the encoder would be fed a corrupt channel and the importances would be garbage.
#
# Resume-safe: a model whose pfi.csv already exists is skipped.
#
# Usage:  bash scripts/fi_pfi_2024_precip.sh
#         DRY_RUN=1 bash scripts/fi_pfi_2024_precip.sh
set -uo pipefail
cd "$(dirname "$0")/.."

YEAR=2024
TP_DIR="${TP_DIR:-datasets/ERA5_Land/tp_hourly}"
SFCTP_INPUT_GLOB="${TP_DIR}/tp-*.nc"        # 00-00 UTC next00, continuous with training
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

MODELS=(
  "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8"
  "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps"
  "clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8"
  "clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8"
  "clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8"
)
SFCTP="clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8"

mkdir -p "$LOG_DIR"
failed=()
for m in "${MODELS[@]}"; do
  dir="CLEAN_trained_models/${m}/precip"
  out="${dir}/feature_importance_${YEAR}"
  if [ -f "${out}/pfi.csv" ]; then
    echo "[skip] ${out}/pfi.csv already exists"
    continue
  fi
  args=(--method pfi --model-dir "$dir" --year "$YEAR" --output-dir "$out"
        --device "$DEVICE" --day-batch "$DAY_BATCH")
  if [ "$m" = "$SFCTP" ]; then
    args+=(--precip-glob "$SFCTP_INPUT_GLOB")
  fi
  echo "[run ] ${m} -> ${out}"
  if [ "$DRY_RUN" = "1" ]; then
    echo "       python feature_importance.py ${args[*]}"
    continue
  fi
  if python feature_importance.py "${args[@]}" > "${LOG_DIR}/fi_pfi_${YEAR}__${m}.log" 2>&1; then
    echo "[done] ${m}"
  else
    echo "[FAIL] ${m} -- see ${LOG_DIR}/fi_pfi_${YEAR}__${m}.log" >&2
    failed+=("$m")
  fi
done

echo
if [ "${#failed[@]}" -gt 0 ]; then
  echo "FAILED (${#failed[@]}):" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "[all done] PFI ${YEAR} complete for ${#MODELS[@]} precip runs."
