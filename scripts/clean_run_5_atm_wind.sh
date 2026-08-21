#!/usr/bin/env bash
#
# CLEAN solo run #5 — atmospheric tmax WITH WIND ("clean solo wind run").
#   ERA5 pressure-level z/t/q PLUS u/v wind components on the native coarse grid,
#   flat encoder.
#   Flags: --use-atmospheric --atmos-native-grid --atmos-variables z t q u v
#
# Same setup as run #1 (clean_run_1_atm_baseline.sh) but adds the u/v wind
# channels, so its feature importance directly shows how much wind contributes on
# top of the z/t/q baseline. Trains on 2020-2023 into CLEAN_trained_models/, then
# runs PFI + LIME feature importance (full res), and commits on its own branch.
#
# REQUIRES the wind data on disk first:
#   python datasets/download_era5.py --years 2020-2023 --fields none --pl-fields u,v
#
# Run this on ONE cluster. Usage:
#   bash scripts/clean_run_5_atm_wind.sh
#   DRY_RUN=1 bash scripts/clean_run_5_atm_wind.sh   # print commands only
#   SKIP_FI=1 bash scripts/clean_run_5_atm_wind.sh   # train + commit, no FI
#   NO_GIT=1  bash scripts/clean_run_5_atm_wind.sh   # no branch/commit/push

RUN_BRANCH="clean-run/atm-wind"
TRIAL_PREFIX="clean_solo_wind"
TRAIN_FLAGS=(--use-atmospheric --atmos-native-grid --atmos-variables z t q u v)

source "$(dirname "${BASH_SOURCE[0]}")/_clean_run_common.sh"
clean_run_main
