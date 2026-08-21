#!/usr/bin/env bash
#
# Shared setup for the feature-importance shell scripts
# (fi_run_local.sh / fi_run_shap.sh / fi_evaluate.sh).
#
# Sourced — not run directly. Provides:
#   - repo-root cwd
#   - MODEL_DIR / YEAR / DRY_RUN env overrides (with defaults)
#   - FI_DIR (the <model_dir>/feature_importance output dir)
#   - fi_preflight()  : sanity-check model dir, checkpoints, dataset stage, GPU
#   - fi_run "<cmd…>" : echo + (unless DRY_RUN=1) execute a command, teeing to a log
#
# Env overrides (all optional):
#   MODEL_DIR   model dir with params.json / manifest.json / model_fold_*
#               (default: trained_models/tmax_atm_natg_flat_y2023_e30f5_b8/tmax)
#   YEAR        evaluation year (default: 2022)
#   DRY_RUN=1   print commands instead of running them
#   NO_GIT=1    skip the auto commit + push of results

set -euo pipefail

# Always operate from the repo root (scripts/ lives one level down).
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${MODEL_DIR:-trained_models/tmax_atm_natg_flat_y2023_e30f5_b8/tmax}"
YEAR="${YEAR:-2022}"
# DAY_STRIDE: use every Nth day (1 = all days). For the multi-year joint 2020-2023
# model, stride 4 gives ~365 evenly-spread days — the y2023 study's fidelity/runtime.
DAY_STRIDE="${DAY_STRIDE:-1}"
DRY_RUN="${DRY_RUN:-0}"
NO_GIT="${NO_GIT:-0}"
FI_DIR="${FI_DIR:-${MODEL_DIR}/feature_importance}"

# A joint two-stage model (joint_meta.json present) is scored on both heads and
# ignores YEAR (uses its own DATA_YEAR span); standalone tmax uses YEAR.
FI_IS_JOINT=0
[[ -f "${MODEL_DIR}/joint_meta.json" ]] && FI_IS_JOINT=1

# Preflight: fail fast on the things that silently waste an 8–30 h run.
fi_preflight() {
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: MODEL_DIR not found: $MODEL_DIR" >&2
    exit 1
  fi
  if ! compgen -G "${MODEL_DIR}/model_fold_*" > /dev/null; then
    echo "ERROR: no model_fold_* checkpoints in $MODEL_DIR" >&2
    exit 1
  fi
  if [[ ! -e datasets ]]; then
    echo "ERROR: ./datasets is not staged — feature_importance.py needs the ERA5/MeteoSwiss data." >&2
    echo "       Stage it (e.g. via cloud_sync) before running." >&2
    exit 1
  fi
  mkdir -p "$FI_DIR"
  echo "model dir : $MODEL_DIR"
  echo "FI out dir: $FI_DIR"
  if [[ "$FI_IS_JOINT" == "1" ]]; then
    echo "model type: JOINT two-stage (tmax + precip heads; YEAR ignored)"
  else
    # Standalone runs cover tmax (Gaussian) and precip (Bernoulli-Gamma); report which.
    fi_var=$(sed -n 's/.*"VARIABLE"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
             "${MODEL_DIR}/params.json" | head -1)
    echo "model type: standalone ${fi_var:-unknown}  |  year: $YEAR"
  fi
  echo "day-stride: $DAY_STRIDE"
  if command -v nvidia-smi > /dev/null 2>&1; then
    echo "GPU       :"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader 2>/dev/null | sed 's/^/            /' || true
  else
    echo "GPU       : nvidia-smi not found (will fall back to whatever select_device() picks)"
  fi
}

# fi_run <log-basename> <cmd...> : log the command, then run it (tee to FI_DIR/<base>.log).
# Honors DRY_RUN=1 (print only). The *.log files are gitignored.
fi_run() {
  local base="$1"; shift
  local log="${FI_DIR}/${base}.log"
  echo "+ $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (DRY_RUN=1 — not executed)"
    return 0
  fi
  # shellcheck disable=SC2068
  "$@" 2>&1 | tee "$log"
}

# fi_commit_push "<commit subject>" : stage the FI output dir, commit and push.
# Honors DRY_RUN=1 (print only) and NO_GIT=1 (skip entirely). Refuses to commit on
# master/main (per the repo working agreement: branch first). Logs are gitignored,
# so only the result artifacts (CSVs, PNGs, summary.md, REPORT.md, baseline.json)
# get committed. No-ops cleanly when there is nothing to commit.
fi_commit_push() {
  local subject="$1"
  if [[ "$NO_GIT" == "1" ]]; then
    echo "NO_GIT=1 — skipping commit + push of $FI_DIR"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ git add $FI_DIR && git commit -m \"$subject\" && git push   (DRY_RUN=1 — skipped)"
    return 0
  fi
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "$branch" == "master" || "$branch" == "main" ]]; then
    echo "ERROR: on '$branch' — branch first before auto-committing results (working agreement)." >&2
    echo "       Create/switch to a working branch and re-run, or set NO_GIT=1 to skip git." >&2
    exit 1
  fi
  git add "$FI_DIR"
  if git diff --cached --quiet -- "$FI_DIR"; then
    echo "nothing to commit in $FI_DIR"
    return 0
  fi
  git commit -m "$subject"
  git push
  echo "committed + pushed: $subject"
}
