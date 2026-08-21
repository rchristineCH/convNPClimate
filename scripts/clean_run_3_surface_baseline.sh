#!/usr/bin/env bash
#
# CLEAN solo run #3 — surface-only tmax (no atmospheric), WITH geopotential.
#   ERA5-Land surface field + lat/lon/seasonal + geopotential elevation channel.
#   Flags: --use-surface     (trial name: "baseline")
#
# Trains on 2020-2023 into CLEAN_trained_models/, then runs PFI + LIME feature
# importance (full res), and commits the model + FI outputs on its own branch.
#
# Run this on ONE cluster. Usage:
#   bash scripts/clean_run_3_surface_baseline.sh
#   DRY_RUN=1 bash scripts/clean_run_3_surface_baseline.sh   # print commands only
#   SKIP_FI=1 bash scripts/clean_run_3_surface_baseline.sh   # train + commit, no FI

RUN_BRANCH="clean-run/surface-baseline"
TRIAL_PREFIX="baseline"
TRAIN_FLAGS=(--use-surface)

source "$(dirname "${BASH_SOURCE[0]}")/_clean_run_common.sh"
clean_run_main
