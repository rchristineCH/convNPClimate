#!/usr/bin/env bash
#
# Feature importance — LOCAL full run: PFI then LIME, at full resolution.
#
# Runs the two cheaper methods on the local GPU (single A10 here). At full
# defaults this is all 365 days × all 5 folds, PFI with 5 permutation repeats,
# LIME with 4 representative days × 1000 masks. Estimated wall time on one A10:
# PFI ~8 h, LIME ~9 min.
#
# SHAP is intentionally NOT run here — its 2000-coalition KernelSHAP is ~30 h
# and is meant for the cluster (see scripts/fi_run_shap.sh). Once all three
# methods' CSVs are co-located, scripts/fi_evaluate.sh combines them.
#
# Outputs (into <MODEL_DIR>/feature_importance/): pfi.csv, lime.csv + their PNGs,
# baseline.json, and a method-local summary.md. Logs: fi_pfi.log, fi_lime.log.
# On success it commits the results and pushes (current branch; skipped with
# NO_GIT=1; refuses to commit on master — branch first).
#
# Usage:
#   bash scripts/fi_run_local.sh                 # full run (PFI then LIME) + commit/push
#   DRY_RUN=1 bash scripts/fi_run_local.sh       # print the commands only
#   NO_GIT=1 bash scripts/fi_run_local.sh        # run but do not commit/push
#   YEAR=2021 bash scripts/fi_run_local.sh       # different evaluation year (standalone tmax)
#   DAY_STRIDE=4 bash scripts/fi_run_local.sh     # every 4th day (~365 days for 2020-2023)
#   DEVICE=cpu WORKERS=11 bash scripts/fi_run_local.sh   # CPU fork-pool instead of GPU
#   MODEL_DIR=trained_models/joint_my__..._y2020-2023.../joint_two_stage bash scripts/fi_run_local.sh

source "$(dirname "${BASH_SOURCE[0]}")/_fi_common.sh"

DEVICE="${DEVICE:-cuda}"
if [[ "$DEVICE" == "cpu" ]]; then
  WORKERS="${WORKERS:-11}"; DAY_BATCH="${DAY_BATCH:-1}"; export CUDA_VISIBLE_DEVICES=""
else
  WORKERS="${WORKERS:-1}"; DAY_BATCH="${DAY_BATCH:-32}"   # GPU day-batching (mem-safe on A10-8Q)
fi

fi_preflight
echo "device: $DEVICE | workers: $WORKERS | day-batch: $DAY_BATCH"

# LIME runs only if PFI succeeds (set -e / sequential). --year is ignored for a joint model.
common=(--model-dir "$MODEL_DIR" --year "$YEAR" --day-stride "$DAY_STRIDE"
        --device "$DEVICE" --workers "$WORKERS" --day-batch "$DAY_BATCH")
fi_run fi_pfi  python feature_importance.py --method pfi  "${common[@]}"
fi_run fi_lime python feature_importance.py --method lime "${common[@]}"

if [[ "$FI_IS_JOINT" == "1" ]]; then
  fi_commit_push "feature-importance: joint PFI + LIME run (tmax+precip, stride ${DAY_STRIDE})"
else
  fi_commit_push "feature-importance: full PFI + LIME run (${YEAR})"
fi

echo
echo "PFI + LIME done. Next: run SHAP on the cluster (scripts/fi_run_shap.sh),"
echo "then combine all three with scripts/fi_evaluate.sh."
