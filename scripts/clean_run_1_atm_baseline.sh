#!/usr/bin/env bash
#
# CLEAN solo run #1 — atmospheric baseline tmax ("clean solo run").
#   ERA5 pressure-level z/t/q on the native coarse grid, flat encoder.
#   Flags: --use-atmospheric --atmos-native-grid
#
# Trains on 2020-2023 into CLEAN_trained_models/, then runs PFI + LIME feature
# importance (full res), and commits the model + FI outputs on its own branch.
#
# Run this on ONE cluster. Usage:
#   bash scripts/clean_run_1_atm_baseline.sh
#   DRY_RUN=1 bash scripts/clean_run_1_atm_baseline.sh   # print commands only
#   SKIP_FI=1 bash scripts/clean_run_1_atm_baseline.sh   # train + commit, no FI
#   NO_GIT=1  bash scripts/clean_run_1_atm_baseline.sh   # no branch/commit/push

RUN_BRANCH="clean-run/atm-baseline"
TRIAL_PREFIX="clean_solo"
TRAIN_FLAGS=(--use-atmospheric --atmos-native-grid)

source "$(dirname "${BASH_SOURCE[0]}")/_clean_run_common.sh"
clean_run_main
