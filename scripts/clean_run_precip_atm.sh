#!/usr/bin/env bash
#
# CLEAN solo run — atmospheric precipitation (Bernoulli-Gamma), NO surface anchors.
#   ERA5 pressure-level z/t/q on the native coarse grid, flat encoder.
#   Flags: --variables precip --use-atmospheric --atmos-native-grid
#          (Bernoulli-Gamma is the per-variable default; not passed explicitly so
#           the trial slug stays clean.)
#
# Trains on 2020-2023 into CLEAN_trained_models/, then runs eval_precip.py physical
# metrics (wet-day freq/R01, SDII, R10, P98, MAE/bias), and commits the model +
# metrics on its own branch.
#
# NOTE: PFI/LIME feature importance is intentionally NOT run here. feature_importance.py
# supports only the atmospheric *tmax* model or a *joint* two-stage tmax+precip model —
# not a standalone/solo precip model. eval_precip.py is the precip analysis analog.
# (To get precip feature importance you'd need a joint model, i.e. not a "solo" run.)
#
# Run this on ONE cluster. Usage:
#   bash scripts/clean_run_precip_atm.sh
#   DRY_RUN=1   bash scripts/clean_run_precip_atm.sh   # print commands only
#   SKIP_EVAL=1 bash scripts/clean_run_precip_atm.sh   # train + commit, no eval
#   NO_GIT=1    bash scripts/clean_run_precip_atm.sh   # no branch/commit/push

RUN_BRANCH="clean-run/precip-atm-baseline"
TRIAL_PREFIX="clean_solo_precip"
VARIABLE="precip"
ANALYSIS_LABEL="eval_precip"
TRAIN_FLAGS=(--use-atmospheric --atmos-native-grid)

source "$(dirname "${BASH_SOURCE[0]}")/_clean_run_common.sh"

# Precip analysis = eval_precip.py physical metrics -> <model_dir>/eval_precip_metrics.json
# (committed with the trial dir). Overrides the default tmax PFI/LIME analysis.
do_analysis() {
  if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
    echo "SKIP_EVAL=1 — skipping eval_precip"
    return 0
  fi
  run python eval_precip.py --model-dir "$MODEL_DIR" --device "$DEVICE"
}

clean_run_main
