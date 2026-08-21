#!/usr/bin/env bash
#
# Two-phase precip training routine: NLL pre-train -> short low-LR CRPS fine-tune.
#
# Phase 1 (NLL) gives the well-calibrated base (best standard-event skill);
# phase 2 warm-starts every fold from the phase-1 checkpoints and fine-tunes
# briefly on the Bernoulli-Gamma CRPS loss (finetune_precip.py), which the
# from-scratch comparison (precip_processing_comparison/) shows is the better
# objective for heavy-precip events. Finally both dirs are evaluated with
# eval_precip.py so the standard-vs-heavy trade-off is directly comparable
# (add the pure-CRPS reference run for a three-way comparison).
#
# Usage:
#   bash scripts/run_precip_nll_then_crps_ft.sh                # full: train + finetune + eval
#   NLL_DIR=CLEAN_trained_models/clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8/precip \
#     bash scripts/run_precip_nll_then_crps_ft.sh              # reuse an existing NLL run (skips phase 1)
#   DRY_RUN=1   bash scripts/run_precip_nll_then_crps_ft.sh    # print commands only
#   SKIP_EVAL=1 bash scripts/run_precip_nll_then_crps_ft.sh    # no eval_precip at the end
#
# Optional env overrides:
#   NLL_DIR=...        existing phase-1 run dir (skips training)
#   FT_LR=...          fine-tune LR (default: phase-1 LR / 10, decided in finetune_precip.py)
#   FT_EPOCHS=10       max fine-tune epochs per fold
#   FT_PATIENCE=3      early-stopping patience on held-out CRPS
#   DEVICE=cuda        torch device
#   NO_KEEPALIVE=1     do not start gpu_keepalive.py

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODELS_SUBDIR="CLEAN_trained_models"
TRIAL_PREFIX="clean_solo_precip"   # phase-1 slug matches the existing NLL clean run
VENV="${VENV:-/home/renku/work/.venv}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
NO_KEEPALIVE="${NO_KEEPALIVE:-0}"
FT_EPOCHS="${FT_EPOCHS:-10}"
FT_PATIENCE="${FT_PATIENCE:-3}"
NLL_DIR="${NLL_DIR:-}"
KEEPALIVE_PID=""

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (DRY_RUN=1 — not executed)"
    return 0
  fi
  "$@"
}

# --- venv preflight (same contract as scripts/_clean_run_common.sh) ----------
if ! python -c "import torch" 2>/dev/null; then
  if [[ -z "${VIRTUAL_ENV:-}" && -f "${VENV}/bin/activate" ]]; then
    echo "torch not importable — activating venv ${VENV}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
  fi
fi
if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: PyTorch is not available; activate ${VENV} and retry." >&2
  exit 1
fi
echo "python env OK: $(command -v python)"

# --- keepalive so the Renku session isn't culled during long phases ----------
if [[ "$NO_KEEPALIVE" != "1" && "$DRY_RUN" != "1" ]] \
    && command -v nvidia-smi > /dev/null 2>&1 && [[ -f gpu_keepalive.py ]]; then
  nohup python gpu_keepalive.py --busy 2 --idle 20 > gpu_keepalive.log 2>&1 &
  KEEPALIVE_PID=$!
  echo "started gpu_keepalive.py (pid ${KEEPALIVE_PID})"
  trap '[[ -n "$KEEPALIVE_PID" ]] && kill "$KEEPALIVE_PID" 2>/dev/null || true' EXIT
fi

# --- Phase 1: NLL pre-training (skipped when NLL_DIR is given) ---------------
if [[ -z "$NLL_DIR" ]]; then
  echo "=== Phase 1: NLL pre-training (bernoulli_gamma) ==="
  run python train.py \
    --variables precip \
    --use-atmospheric --atmos-native-grid \
    --data-year-start 2020 --data-year-end 2023 \
    --models-subdir "$MODELS_SUBDIR" \
    --trial-name "$TRIAL_PREFIX" \
    --device "$DEVICE"
  NLL_DIR="${MODELS_SUBDIR}/${TRIAL_PREFIX}__precip_atm_natg_flat_y2020-2023_e30f5_b8/precip"
else
  echo "=== Phase 1 skipped: reusing NLL run at ${NLL_DIR} ==="
fi
[[ "$DRY_RUN" == "1" || -f "${NLL_DIR}/params.json" ]] \
  || { echo "ERROR: ${NLL_DIR}/params.json not found" >&2; exit 1; }

# --- Phase 2: CRPS fine-tune -------------------------------------------------
echo "=== Phase 2: CRPS fine-tune (warm start, low LR) ==="
FT_ARGS=(--model-dir "$NLL_DIR" --n-epochs "$FT_EPOCHS" --patience "$FT_PATIENCE" --device "$DEVICE")
[[ -n "${FT_LR:-}" ]] && FT_ARGS+=(--lr "$FT_LR")
run python finetune_precip.py "${FT_ARGS[@]}"
FT_DIR="$(dirname "$(dirname "$NLL_DIR")")/$(basename "$(dirname "$NLL_DIR")")__ft-crps/precip"

# --- Phase 3: evaluate both endpoints ---------------------------------------
if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "SKIP_EVAL=1 — done."
  exit 0
fi
echo "=== Phase 3: eval_precip on NLL base and fine-tuned model ==="
run python eval_precip.py --model-dir "$NLL_DIR" --device "$DEVICE"
run python eval_precip.py --model-dir "$FT_DIR" --device "$DEVICE"
echo "Done. Compare:"
echo "  NLL base:   ${NLL_DIR}/eval_precip_metrics.json"
echo "  fine-tuned: ${FT_DIR}/eval_precip_metrics.json"
echo "  (pure-CRPS reference: ${MODELS_SUBDIR}/clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8/precip)"
