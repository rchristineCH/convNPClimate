#!/usr/bin/env bash
# Data download for the long-scale training run (train 2000-2023, holdout 2024,
# second unseen test year 2025).
#
# What is already on disk vs. what this fetches:
#   MeteoSwiss targets            1961-2025  -> nothing to do
#   ERA5-Land daily (0.1 deg)     1960-2024  -> fetch 2025 only
#   ERA5 pressure levels z/t/q/u/v 2020-2024 -> fetch 2000-2019 and 2025
#   ERA5 surface anchors t2m/tp   2020-2024  -> fetch 2000-2019 and 2025
#
# Grid/levels/hours are identical to the existing files (0.25 deg Swiss box,
# levels 1000/925/850/700/500/300 hPa, hours 00/06/12/15/18 UTC), so new years
# concatenate cleanly with the 2020-2024 data.
#
# ~735 CDS requests total (~35/year), roughly 30 MB per year of pressure-level
# data. Network/queue-bound: expect several hours up to a day depending on the
# CDS queue. Resume-safe -- already-downloaded files are skipped, so just rerun
# after any interruption.
#
# Usage:  nohup bash scripts/download_longrun_data.sh > download_longrun.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

# Drop u,v here if the long run will not include the wind experiment.
PL_FIELDS="z,t,q,u,v"

# Backward extension of the training period (2020-2024 already on disk).
conda run --no-capture-output -n convnpclimate python datasets/download_era5.py \
  --years 2000-2019 --fields none --pl-fields "$PL_FIELDS" --sfc-fields all

# 2025 as a second fully unseen test year: pressure levels + surface anchors
# plus the daily ERA5-Land fields (surface-mode input and the skill baseline).
conda run --no-capture-output -n convnpclimate python datasets/download_era5.py \
  --years 2025 --fields all --pl-fields "$PL_FIELDS" --sfc-fields all

echo "[all done] long-run data download complete"
