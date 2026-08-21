#!/usr/bin/env bash
#
# CLEAN solo run #4 — surface-only tmax (no atmospheric), WITHOUT geopotential.
#   Same as #3 but drops the geopotential-derived elevation INPUT channel from
#   the encoder context. Geopotential is still loaded so the elevation-bias MLP
#   (hi-res DEM + TPI + elev_diff) keeps working — only the input channel is
#   ablated.
#   Flags: --use-surface --no-geopotential   (trial name: "baseline_no_geo")
#
# Trains on 2020-2023 into CLEAN_trained_models/, then runs PFI + LIME feature
# importance (full res), and commits the model + FI outputs on its own branch.
#
# Run this on ONE cluster. Usage:
#   bash scripts/clean_run_4_surface_no_geo.sh
#   DRY_RUN=1 bash scripts/clean_run_4_surface_no_geo.sh   # print commands only
#   SKIP_FI=1 bash scripts/clean_run_4_surface_no_geo.sh   # train + commit, no FI

RUN_BRANCH="clean-run/surface-no-geo"
TRIAL_PREFIX="baseline_no_geo"
TRAIN_FLAGS=(--use-surface --no-geopotential)

source "$(dirname "${BASH_SOURCE[0]}")/_clean_run_common.sh"
clean_run_main
