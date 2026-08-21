#!/usr/bin/env bash
# Backfill of the missing ERA5 pressure-level years: 2004-2014 (gap between the
# 2000-2003 and 2015-2024 blocks already on disk) plus 2025, the second unseen
# test year.
#
# 12 years x 5 fields x 5 synoptic hours = 300 CDS requests. Each request queues
# for 8-25 min, so running them serially would take days; the years are split
# round-robin over 4 concurrent workers instead. Different workers only ever
# touch different years, so their output files never collide.
#
# Grid/levels/hours match the existing files (0.25 deg Swiss box, levels
# 1000/925/850/700/500/300 hPa, hours 00/06/12/15/18 UTC) so the new years
# concatenate cleanly with 2000-2003 and 2015-2024.
#
# Resume-safe: already-downloaded files are skipped, so just rerun after any
# interruption.
#
# Usage:  bash scripts/download_pl_backfill.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PL_FIELDS="z,t,q,u,v"

# Round-robin so every worker gets a similar share of the queue wait.
WORKER_YEARS=(
  "2004,2008,2012"
  "2005,2009,2013"
  "2006,2010,2014"
  "2007,2011,2025"
)

for i in "${!WORKER_YEARS[@]}"; do
  years="${WORKER_YEARS[$i]}"
  log="download_pl_backfill_w$((i + 1)).log"
  echo "[launch] worker $((i + 1)): years $years -> $log"
  nohup conda run --no-capture-output -n convnpclimate \
    python datasets/download_era5.py \
      --years "$years" \
      --fields none \
      --sfc-fields none \
      --pl-fields "$PL_FIELDS" \
      --skip-geopotential \
    > "$log" 2>&1 &
  disown
done

echo "[launched] 4 workers; follow with: tail -f download_pl_backfill_w*.log"
