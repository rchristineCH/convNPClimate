#!/usr/bin/env bash
#
# CLEAN solo run #2 — atmospheric tmax WITH surface anchors.
#   Pressure-level z/t/q + ERA5 single-level surface anchor channels.
#   Flags: --use-atmospheric --atmos-native-grid --use-sfc-atmos
#
# Trains on 2020-2023 into CLEAN_trained_models/, then runs PFI + LIME feature
# importance (full res), and commits the model + FI outputs on its own branch.
#
# Run this on ONE cluster. Usage:
#   bash scripts/clean_run_2_atm_sfc_anchors.sh
#   DRY_RUN=1 bash scripts/clean_run_2_atm_sfc_anchors.sh   # print commands only
#   SKIP_FI=1 bash scripts/clean_run_2_atm_sfc_anchors.sh   # train + commit, no FI

RUN_BRANCH="clean-run/atm-sfc-anchors"
TRIAL_PREFIX="sfc_anchors"
TRAIN_FLAGS=(--use-atmospheric --atmos-native-grid --use-sfc-atmos)

source "$(dirname "${BASH_SOURCE[0]}")/_clean_run_common.sh"
clean_run_main
