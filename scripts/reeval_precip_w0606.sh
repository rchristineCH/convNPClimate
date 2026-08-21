#!/usr/bin/env bash
#
# Re-evaluate every CLEAN precip run against the corrected, RhiresD-aligned tp
# reference. Five runs x two regimes (cv_holdout + holdout_year_2024).
#
# WHY
#   1. datasets/ERA5_Land/precipitation/tp-2024.nc is ~1.67x inflated: it was
#      built from hourly data with a max-over-day rule, which returns
#      max(total(D-1), partial(D)). It is the reference behind every published
#      2024 skill_mae, so all four are overstated.
#   2. The committed CV metrics used THREE different baselines -- a 5-file glob,
#      a 4-file glob, and (for ft-crps) tp-2020.nc alone, i.e. ~365 of 1461 days.
#      eval_precip drops uncovered days silently, so the CV skill column was not
#      a valid ranking.
#   3. ERA5-Land days span 00-00 UTC; RhiresD is documented as 06 UTC of day D to
#      06 UTC of D+1. The 00-00 reference is handicapped by that offset, which
#      FLATTERS skill = 1 - MAE_model/MAE_baseline.
#
# So every run here is scored against <TP_DIR>/w0606/ (06-06 UTC), built from
# hourly ERA5-Land by scripts/build_tp_from_hourly.py. Expect skill to FALL. That
# is the point: the lower number is the defensible one.
#
# MODEL INPUTS ARE NOT CHANGED, with one exception. Only the sfctp run reads tp
# as an input channel (USE_SURFACE_PRECIP), and for 2024 the file on disk is the
# defective one -- feeding it would be scoring the model on an input unlike
# anything it saw in training. That single case reads the rebuilt next00 field
# via --precip-glob, which overrides ERA5_PRECIP_GLOB for inference only and
# leaves params.json untouched. next00 is bit-identical to the legacy files on
# 2020-2023 (verified: maxdiff 0.00000), so this changes 2024 and nothing else.
#
# sfctp CV deliberately keeps its legacy inputs: next00 additionally CONTAINS
# 2023-12-31, which the legacy set lacks, so overriding there would shift the CV
# span 1460 -> 1461 days and break comparability with that run's own history.
#
# Resume-safe: a run whose output JSON already exists is skipped. Delete the file
# to force a re-run.
#
# Usage:  bash scripts/reeval_precip_w0606.sh
#         DRY_RUN=1 bash scripts/reeval_precip_w0606.sh      # print, do not execute
#
# Afterwards, regenerate the figures (not done here -- they are cheap and read
# the JSONs this produces):
#         python compare_precip_models.py
#         python scripts/regen_precip_figures.py
#         python scripts/collect_result_figures.py --out <doc>/LATEX_REPORT/images/results \
#                --index RESULT_FIGURES_INDEX.md
set -euo pipefail
cd "$(dirname "$0")/.."

# The rebuilt tp fields live under datasets/ with every other dataset, not at the
# repo root: that is where the cloud sync puts them (and where the hourly `raw/`
# monthlies they are built from live), so one location stays authoritative whether
# the files arrive by sync or by rebuild. Verified byte-identical to the previous
# repo-root copies before the move.
TP_DIR="${TP_DIR:-datasets/ERA5_Land/tp_hourly}"
BASELINE_GLOB="${TP_DIR}/w0606/tp-*.nc"     # 06-06 UTC, RhiresD-aligned
SFCTP_INPUT_GLOB="${TP_DIR}/tp-*.nc"        # 00-00 UTC, continuous with training
EVAL_YEAR=2024
LOG_DIR="reeval_logs"
DRY_RUN="${DRY_RUN:-0}"
VENV="${VENV:-/home/renku/work/.venv}"

# Interpreter indirection: this box has no conda, it runs the repo venv, while the
# machine the script was written on has a `convnpclimate` env. Resolve once here
# instead of hardcoding either, so the same script runs in both places.
if command -v conda > /dev/null 2>&1 && conda env list 2>/dev/null | grep -qE '^convnpclimate\s'; then
  PYRUN=(conda run --no-capture-output -n convnpclimate python)
else
  if ! python -c "import torch" 2>/dev/null; then
    if [[ -z "${VIRTUAL_ENV:-}" && -f "${VENV}/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "${VENV}/bin/activate"
    fi
  fi
  if ! python -c "import torch" 2>/dev/null; then
    echo "FATAL: no convnpclimate conda env and no importable torch; activate ${VENV}." >&2
    exit 1
  fi
  PYRUN=(python)
fi
echo "interpreter: ${PYRUN[*]}  ($(command -v python))"

MODELS=(
  "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8"
  "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8__ft-crps"
  "clean_solo_precip_crps__precip_bgcrps_atm_natg_flat_y2020-2023_e30f5_b8"
  "clean_solo_precip_wind__precip_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8"
  "clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8"
)
SFCTP="clean_solo_precip_sfctp__precip_sfc_tp_flat_y2020-2023_e30f5_b8"

# ---------------------------------------------------------------------------
# Preflight: refuse to score against a reference that is missing or malformed.
# A silently absent baseline yields available=false and an all-NaN skill rather
# than an error, which is exactly the failure this whole exercise is fixing.
# ---------------------------------------------------------------------------
shopt -s nullglob
w0606_files=( "${TP_DIR}"/w0606/tp-*.nc )
next00_files=( "${TP_DIR}"/tp-*.nc )
shopt -u nullglob

if [ "${#w0606_files[@]}" -lt 5 ]; then
  echo "FATAL: expected 5 w0606 baseline files (2020-2024) in ${TP_DIR}/w0606, found ${#w0606_files[@]}." >&2
  echo "       Run: python scripts/build_tp_from_hourly.py --years 2020-2024 --aggregate --rule w0606" >&2
  exit 1
fi
if [ "${#next00_files[@]}" -lt 5 ]; then
  echo "FATAL: expected 5 next00 input files (2020-2024) in ${TP_DIR}, found ${#next00_files[@]}." >&2
  echo "       Run: python scripts/build_tp_from_hourly.py --years 2020-2024 --aggregate --rule next00" >&2
  exit 1
fi

TP_DIR="$TP_DIR" "${PYRUN[@]}" - <<'PY' || exit 1
import glob, os, sys
import xarray as xr
tp_dir = os.environ["TP_DIR"]
bad = []
for rule, pat in (("w0606", f"{tp_dir}/w0606/tp-*.nc"), ("next00", f"{tp_dir}/tp-*.nc")):
    for f in sorted(glob.glob(pat)):
        d = xr.open_dataset(f, engine="h5netcdf")
        n, la, lo = d.sizes["time"], d.sizes["latitude"], d.sizes["longitude"]
        mm = float(d.tp.mean()) * 1000
        year = int(f.split("tp-")[-1][:4])
        exp = 366 if year % 4 == 0 else 365
        if (la, lo) != (29, 61) or n != exp or not (3.0 <= mm <= 5.0):
            bad.append(f"  {rule}/{f}: n={n}/{exp} grid={la}x{lo} mean={mm:.3f}")
        d.close()
if bad:
    print("FATAL: tp files failed the shape/coverage/magnitude gate:", file=sys.stderr)
    print("\n".join(bad), file=sys.stderr)
    sys.exit(1)
print("preflight OK: all tp files 29x61, full calendar years, mean in 3.0-5.0 mm/day")
PY

mkdir -p "$LOG_DIR"

run_eval () {
  local model="$1" regime="$2"; shift 2
  local dir="CLEAN_trained_models/${model}/precip"
  local out extra=()

  if [ "$regime" = "cv" ]; then
    out="${dir}/eval_precip_metrics.json"
  else
    out="${dir}/eval_precip_metrics_${EVAL_YEAR}.json"
    extra+=(--eval-year "$EVAL_YEAR")
  fi

  # Only the sfctp run's 2024 leg overrides the tp INPUT channel.
  if [ "$model" = "$SFCTP" ] && [ "$regime" != "cv" ]; then
    extra+=(--precip-glob "$SFCTP_INPUT_GLOB" --refresh-cache)
  fi

  if [ -f "$out" ]; then
    echo "[skip] $out already exists"
    return 0
  fi

  local log="${LOG_DIR}/${model}__${regime}.log"
  echo "[run ] ${model} ${regime} -> ${out}"
  if [ "$DRY_RUN" = "1" ]; then
    echo "       python eval_precip.py --model-dir $dir --baseline-glob '$BASELINE_GLOB' ${extra[*]}"
    return 0
  fi
  "${PYRUN[@]}" eval_precip.py \
    --model-dir "$dir" \
    --baseline-glob "$BASELINE_GLOB" \
    "${extra[@]}" \
    > "$log" 2>&1
  echo "[done] ${model} ${regime}  (log: $log)"
}

failed=()
for m in "${MODELS[@]}"; do
  for r in cv "$EVAL_YEAR"; do
    if ! run_eval "$m" "$r"; then
      echo "[FAIL] $m $r -- see ${LOG_DIR}/${m}__${r}.log" >&2
      failed+=("$m/$r")
    fi
  done
done

echo
if [ "${#failed[@]}" -gt 0 ]; then
  echo "FAILED (${#failed[@]}):" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "[all done] 10 evaluations complete against the w0606 baseline."
echo "Now verify every CV metrics file reports the SAME baseline.source:"
echo "  grep -h '\"source\"' CLEAN_trained_models/*/precip/eval_precip_metrics.json"
