#!/usr/bin/env bash
#
# Surface-precip CLEAN run: ERA5-Land tp + t2m_max context, BG-NLL pre-train ->
# short low-LR CRPS fine-tune -> CV evaluation -> PFI + LIME feature importance.
#
# This is the surface counterpart of run_precip_nll_then_crps_ft.sh (which trains
# the atmospheric model). It is the first configuration to use ERA5-Land daily
# precipitation as a model INPUT rather than only as an evaluation reference, via
# --use-surface-precip. Every other hyperparameter is a train.py default, matching
# the four existing CLEAN precip runs, so the input set is the only variable.
#
# The run's day axis is 2020-01-01..2023-12-30 (1460 days), not 1461: the ERA5-Land
# precip files are labelled by accumulation window and the 2020-2023 set ends
# 2023-12-30. DataBundle trims every source to the days tp actually covers rather
# than filling, so nothing is invented. Fold blocks stay [0,292)...[1168,1460), i.e.
# the first four are identical day-sets to the 1461-day runs.
#
# The 2024 holdout leg is DELIBERATELY not run here: tp-2024.nc is day-shifted and
# ~67% wet-biased (see tp_hourly/README.md) and must be re-downloaded
# and gated first. Set EVAL_YEAR=2024 once a corrected file has been promoted.
#
# Usage:
#   bash scripts/run_precip_sfctp_nll_then_crps_ft.sh              # train + finetune + CV eval
#   NLL_DIR=... bash scripts/run_precip_sfctp_nll_then_crps_ft.sh  # reuse an existing phase-1 run
#   DRY_RUN=1   bash scripts/run_precip_sfctp_nll_then_crps_ft.sh  # print commands only
#   SKIP_EVAL=1 bash scripts/run_precip_sfctp_nll_then_crps_ft.sh  # stop after the fine-tune
#   SKIP_FT=1   bash scripts/run_precip_sfctp_nll_then_crps_ft.sh  # NLL only, no CRPS fine-tune
#   RUN_FI=0    bash scripts/run_precip_sfctp_nll_then_crps_ft.sh  # skip feature importance
#
# Optional env overrides:
#   NLL_DIR=...        existing phase-1 run dir (skips training)
#   EVAL_YEAR=2024     additionally evaluate a holdout year (needs a gated tp-<year>.nc)
#   FT_LR=...          fine-tune LR (default: phase-1 LR / 10, decided in finetune_precip.py)
#   FT_EPOCHS=10       max fine-tune epochs per fold
#   FT_PATIENCE=3      early-stopping patience on held-out CRPS
#   DEVICE=cuda        torch device
#   RUN_FI=1           run PFI + LIME feature importance after eval (default on)
#   FI_YEAR=2022       year for the feature-importance study
#   NO_KEEPALIVE=1     do not start gpu_keepalive.py

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODELS_SUBDIR="CLEAN_trained_models"
TRIAL_PREFIX="clean_solo_precip_sfctp"
VENV="${VENV:-/home/renku/work/.venv}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_FT="${SKIP_FT:-0}"
RUN_FI="${RUN_FI:-1}"
FI_YEAR="${FI_YEAR:-2022}"
NO_KEEPALIVE="${NO_KEEPALIVE:-0}"
FT_EPOCHS="${FT_EPOCHS:-10}"
FT_PATIENCE="${FT_PATIENCE:-3}"
EVAL_YEAR="${EVAL_YEAR:-}"
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

# Resolve the phase-1 dir by GLOB rather than hardcoding the slug: the sibling
# script hardcodes it, so any flag that touches _config_slug silently breaks it.
resolve_nll_dir() {
  local matches=("${MODELS_SUBDIR}/${TRIAL_PREFIX}__"*/precip)
  if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
    echo "ERROR: expected exactly one ${MODELS_SUBDIR}/${TRIAL_PREFIX}__*/precip, got: ${matches[*]}" >&2
    return 1
  fi
  printf '%s' "${matches[0]}"
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

# --- Phase 1: BG-NLL pre-training on the surface tp + t2m_max context --------
if [[ -z "$NLL_DIR" ]]; then
  echo "=== Phase 1: BG-NLL pre-training (surface: t2m_max + ERA5-Land tp) ==="
  run python train.py \
    --variables precip \
    --use-surface --use-surface-precip \
    --data-year-start 2020 --data-year-end 2023 \
    --models-subdir "$MODELS_SUBDIR" \
    --trial-name "$TRIAL_PREFIX" \
    --device "$DEVICE"
  [[ "$DRY_RUN" == "1" ]] || NLL_DIR="$(resolve_nll_dir)"
else
  echo "=== Phase 1 skipped: reusing NLL run at ${NLL_DIR} ==="
fi
[[ "$DRY_RUN" == "1" || -f "${NLL_DIR}/params.json" ]] \
  || { echo "ERROR: ${NLL_DIR}/params.json not found" >&2; exit 1; }
echo "phase-1 dir: ${NLL_DIR:-<dry-run>}"

# --- Phase 2: CRPS fine-tune (optional) --------------------------------------
FT_DIR="$(dirname "$(dirname "$NLL_DIR")")/$(basename "$(dirname "$NLL_DIR")")__ft-crps/precip"
if [[ "$SKIP_FT" == "1" ]]; then
  echo "=== Phase 2 skipped (SKIP_FT=1): NLL only ==="
  FT_DIR=""
else
  echo "=== Phase 2: CRPS fine-tune (warm start, low LR) ==="
  FT_ARGS=(--model-dir "$NLL_DIR" --n-epochs "$FT_EPOCHS" --patience "$FT_PATIENCE" --device "$DEVICE")
  [[ -n "${FT_LR:-}" ]] && FT_ARGS+=(--lr "$FT_LR")
  run python finetune_precip.py "${FT_ARGS[@]}"
fi

# --- Phase 3: evaluate both endpoints ---------------------------------------
if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "=== Phase 3 skipped (SKIP_EVAL=1) ==="
else
echo "=== Phase 3: eval_precip (CV holdout 2020-2023) ==="
run python eval_precip.py --model-dir "$NLL_DIR" --device "$DEVICE"
# NB: `[[ test ]] && cmd` would abort the whole script under `set -e` whenever the
# test is false, which is precisely the SKIP_FT=1 case. Use if-blocks.
if [[ -n "$FT_DIR" ]]; then
  run python eval_precip.py --model-dir "$FT_DIR" --device "$DEVICE"
fi

if [[ -n "$EVAL_YEAR" ]]; then
  echo "=== Phase 3b: eval_precip (holdout year ${EVAL_YEAR}) ==="
  echo "NOTE: only valid once tp-${EVAL_YEAR}.nc has been re-downloaded and gated."
  run python eval_precip.py --model-dir "$NLL_DIR" --eval-year "$EVAL_YEAR" --device "$DEVICE"
  if [[ -n "$FT_DIR" ]]; then
    run python eval_precip.py --model-dir "$FT_DIR" --eval-year "$EVAL_YEAR" --device "$DEVICE"
  fi
fi
fi   # end SKIP_EVAL

# --- Phase 4: feature importance (PFI + LIME) --------------------------------
# Cheap here: PFI cost scales with input channels, and this model has 7 against
# 95-155 for the atmospheric precip runs -- ~36 permutation evaluations rather
# than ~480, so minutes rather than the ~3-4 h those took. SHAP is deliberately
# excluded (~30 h, cluster job -- see scripts/fi_run_shap.sh).
# FI_YEAR defaults to 2022, matching the other precip FI studies, and 2022 is
# fully covered by the tp series (only 2023 is short a day).
if [[ "$RUN_FI" == "1" ]]; then
  echo "=== Phase 4: feature importance (PFI + LIME, year ${FI_YEAR}) ==="
  run env MODEL_DIR="$NLL_DIR" YEAR="$FI_YEAR" DEVICE="$DEVICE" \
      bash scripts/fi_run_local.sh
  if [[ -n "$FT_DIR" ]]; then
    run env MODEL_DIR="$FT_DIR" YEAR="$FI_YEAR" DEVICE="$DEVICE" \
        bash scripts/fi_run_local.sh
  fi
else
  echo "=== Phase 4 skipped (RUN_FI=0) ==="
fi

echo "Done. Compare:"
echo "  NLL base:   ${NLL_DIR}/eval_precip_metrics.json"
if [[ -n "$FT_DIR" ]]; then
  echo "  fine-tuned: ${FT_DIR}/eval_precip_metrics.json"
fi
echo "  atmospheric counterparts: ${MODELS_SUBDIR}/clean_solo_precip__*/precip"
