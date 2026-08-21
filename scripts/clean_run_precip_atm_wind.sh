#!/usr/bin/env bash
#
# CLEAN solo run — atmospheric precipitation (Bernoulli-Gamma, NLL) WITH WIND.
#   ERA5 pressure-level z/t/q PLUS u/v wind components on the native coarse grid,
#   flat encoder. The precip analog of clean_run_5_atm_wind.sh (which does the same
#   for tmax), and the NLL counterpart to clean_run_precip_atm_crps.sh.
#
#   Flags: --use-atmospheric --atmos-native-grid --atmos-variables z t q u v
#     * Bernoulli-Gamma is the per-variable default; NO --distribution flag is
#       passed, so the optimizer minimizes the Bernoulli-Gamma **NLL** (obj =
#       -gamma_ll). This is what distinguishes it from clean_run_precip_atm_crps.sh
#       (--distribution bernoulli_gamma_crps).
#     * --atmos-variables z t q u v adds the u/v wind channels on top of the z/t/q
#       baseline; the slug gains an 'av-ztquv' marker so it never collides with the
#       z/t/q-only precip baseline (clean_run_precip_atm.sh).
#
# REQUIRES the u/v wind data on disk / mount first (both are present under
#   ../datasets-chr/ERA5_PressureLevels/{u,v}); if absent, fetch with:
#     python datasets/download_era5.py --years 2020-2023 --fields none --pl-fields u,v
#
# Trains on 2020-2023 into CLEAN_trained_models/, then runs eval_precip.py physical
# metrics (wet-day freq/R01, SDII, R10, P98, MAE/bias), and commits the model +
# metrics on its own branch.
#
# NOTE: like the other solo precip runs, PFI/LIME feature importance is NOT run
# (feature_importance.py supports only the atmospheric *tmax* model or a *joint*
# tmax+precip model, not a standalone precip model). eval_precip.py is the analog.
#
# Fork this run branch from the atm-wind lineage so the --atmos-variables flag is
# present (it does not exist on the default clean-solo base):
#   BASE_BRANCH=clean-run/atm-wind bash scripts/clean_run_precip_atm_wind.sh
#
# Usage:
#   BASE_BRANCH=clean-run/atm-wind            bash scripts/clean_run_precip_atm_wind.sh
#   DRY_RUN=1   BASE_BRANCH=clean-run/atm-wind bash scripts/clean_run_precip_atm_wind.sh  # print only
#   SKIP_EVAL=1 BASE_BRANCH=clean-run/atm-wind bash scripts/clean_run_precip_atm_wind.sh  # train + commit, no eval
#   NO_GIT=1                                   bash scripts/clean_run_precip_atm_wind.sh  # no branch/commit/push

RUN_BRANCH="clean-run/precip-atm-wind"
TRIAL_PREFIX="clean_solo_precip_wind"
VARIABLE="precip"
ANALYSIS_LABEL="eval_precip"
TRAIN_FLAGS=(--use-atmospheric --atmos-native-grid --atmos-variables z t q u v)

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
