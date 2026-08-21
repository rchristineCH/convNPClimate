#!/usr/bin/env bash
#
# Shared driver for the "CLEAN solo" tmax cluster runs (clean_run_1..4).
#
# Sourced — not run directly. Each per-config script sets a few variables and
# then calls clean_run_main, which:
#   1. ensures gpu_keepalive.py is present (fetched from the feature-importance
#      branch if missing) and starts it in the BACKGROUND for the whole run so
#      the Renku session is not culled for inactivity (killed on exit);
#   2. creates / switches to a per-run git branch (branch-first: never commits
#      on master/main);
#   3. trains one tmax model on 2020-2023 into CLEAN_trained_models/;
#   4. runs PFI + LIME feature importance at full resolution (all days, all
#      folds) by reusing scripts/fi_run_local.sh;
#   5. commits the trained model + FI outputs and pushes the run branch.
#
# The sourcing script MUST set (before calling clean_run_main):
#   RUN_BRANCH     git branch name for this run
#   TRIAL_PREFIX   --trial-name prefix (also used to locate the output dir)
#   TRAIN_FLAGS    bash array of config-specific train.py flags
#
# Optional env overrides (all have sensible defaults):
#   BASE_BRANCH=...      branch the run branch is forked from
#                        (default CLEAN_SOLO_RUNS_2020-2023; prefers origin/<it>)
#   RESET_RUN_BRANCH=1   if RUN_BRANCH already exists, re-create it on top of
#                        BASE_BRANCH (discards commits unique to RUN_BRANCH)
#   YEAR=2022            FI evaluation year (standalone tmax)
#   DEVICE=cuda          torch device for train + FI
#   DRY_RUN=1            print every command instead of executing it
#   NO_GIT=1             run but skip the branch/commit/push
#   SKIP_FI=1            train + commit only, no PFI/LIME
#   NO_KEEPALIVE=1       do not start gpu_keepalive.py
#   KEEPALIVE_BUSY=2     seconds of GPU work per keepalive cycle
#   KEEPALIVE_IDLE=20    seconds idle per cycle (default ~9% duty: light touch
#                        that fills training's idle gaps without stealing much
#                        throughput)

set -euo pipefail

# Always operate from the repo root (scripts/ lives one level down).
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODELS_SUBDIR="CLEAN_trained_models"
# Project virtualenv (has torch + the convCNP package). Auto-activated by
# ensure_python_env if the current interpreter can't import torch.
VENV="${VENV:-/home/renku/work/.venv}"
VARIABLE="${VARIABLE:-tmax}"                 # tmax (default) or precip
ANALYSIS_LABEL="${ANALYSIS_LABEL:-PFI/LIME}" # shown in banner + commit message
YEAR="${YEAR:-2022}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"
NO_GIT="${NO_GIT:-0}"
SKIP_FI="${SKIP_FI:-0}"
NO_KEEPALIVE="${NO_KEEPALIVE:-0}"
# Branch the per-run branch is cut from. Every clean run must start from the
# current state of the clean-solo-run branch so it picks up shared code changes
# (e.g. the per-epoch CRPS diagnostic); otherwise a run branched from an older
# point silently trains with stale code.
BASE_BRANCH="${BASE_BRANCH:-CLEAN_SOLO_RUNS_2020-2023}"
# Branch that owns gpu_keepalive.py (the "feature dependence" branch).
FEATURE_BRANCH="${FEATURE_BRANCH:-DEV_CHR_feature_importance}"
KEEPALIVE_PID=""
MODEL_DIR=""
TRIAL_DIR=""

# run <cmd...> : echo the command, then execute it unless DRY_RUN=1.
run() {
  echo "+ $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (DRY_RUN=1 — not executed)"
    return 0
  fi
  "$@"
}

# Refuse to operate on master/main (repo working agreement: branch first).
_assert_not_default_branch() {
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "$branch" == "master" || "$branch" == "main" ]]; then
    echo "ERROR: on '$branch' — this script must run on a working branch." >&2
    echo "       It creates '$RUN_BRANCH' for you; nothing should commit to $branch." >&2
    exit 1
  fi
}

# Fail fast if the Python env can't import torch. Tries to activate the project
# venv first ($VENV). Runs BEFORE branch/keepalive so a bad env wastes nothing.
ensure_python_env() {
  if python -c "import torch" 2>/dev/null; then
    echo "python env OK: $(command -v python) ($(python -c 'import torch; print("torch", torch.__version__)'))"
    return 0
  fi
  if [[ -z "${VIRTUAL_ENV:-}" && -f "${VENV}/bin/activate" ]]; then
    echo "torch not importable — activating venv ${VENV}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
  fi
  if python -c "import torch" 2>/dev/null; then
    echo "python env OK: $(command -v python) ($(python -c 'import torch; print("torch", torch.__version__)'))"
    return 0
  fi
  echo "ERROR: PyTorch is not available in this Python environment." >&2
  echo "       active python: $(command -v python)" >&2
  echo "       Activate the project venv and retry, e.g.:" >&2
  echo "         source ${VENV}/bin/activate" >&2
  echo "       If the venv is missing, create it once:" >&2
  echo "         pip install -r requirements.txt && pip install -e convCNP/" >&2
  [[ "$DRY_RUN" == "1" ]] && { echo "       (DRY_RUN=1 — continuing anyway)"; return 0; }
  exit 1
}

ensure_keepalive() {
  if [[ -f gpu_keepalive.py ]]; then
    return 0
  fi
  echo "gpu_keepalive.py missing — fetching from origin/${FEATURE_BRANCH}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (DRY_RUN=1 — not fetched)"
    return 0
  fi
  git fetch origin "$FEATURE_BRANCH"
  git show "origin/${FEATURE_BRANCH}:gpu_keepalive.py" > gpu_keepalive.py
}

start_keepalive() {
  if [[ "$NO_KEEPALIVE" == "1" ]]; then
    echo "NO_KEEPALIVE=1 — not starting keepalive"
    return 0
  fi
  if ! command -v nvidia-smi > /dev/null 2>&1; then
    echo "no GPU (nvidia-smi absent) — not starting keepalive"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ nohup python gpu_keepalive.py --busy ${KEEPALIVE_BUSY:-2} --idle ${KEEPALIVE_IDLE:-20} &  (DRY_RUN — skipped)"
    return 0
  fi
  nohup python gpu_keepalive.py --busy "${KEEPALIVE_BUSY:-2}" --idle "${KEEPALIVE_IDLE:-20}" \
        > gpu_keepalive.log 2>&1 &
  KEEPALIVE_PID=$!
  echo "started gpu_keepalive.py (pid ${KEEPALIVE_PID}) -> gpu_keepalive.log"
}

stop_keepalive() {
  if [[ -n "$KEEPALIVE_PID" ]]; then
    kill "$KEEPALIVE_PID" 2>/dev/null && echo "stopped keepalive (pid ${KEEPALIVE_PID})" || true
  fi
  # belt-and-braces in case it was re-parented or the pid was lost
  pkill -f gpu_keepalive.py 2>/dev/null || true
}

make_branch() {
  if [[ "$NO_GIT" == "1" ]]; then
    echo "NO_GIT=1 — staying on current branch"
    return 0
  fi

  # Resolve the base to the freshest available ref: prefer origin/$BASE_BRANCH so
  # a run cluster with only a stale local copy still starts from the latest
  # pushed state. Falls back to the local branch if there is no remote-tracking
  # ref (e.g. offline).
  git fetch origin "$BASE_BRANCH" 2>/dev/null || true
  local base_ref="$BASE_BRANCH"
  if git rev-parse --verify "origin/${BASE_BRANCH}" > /dev/null 2>&1; then
    base_ref="origin/${BASE_BRANCH}"
  elif ! git rev-parse --verify "$BASE_BRANCH" > /dev/null 2>&1; then
    echo "ERROR: base branch '${BASE_BRANCH}' not found locally or on origin." >&2
    echo "       Set BASE_BRANCH=<branch> to the clean-solo-run branch to fork from." >&2
    exit 1
  fi

  if git rev-parse --verify "$RUN_BRANCH" > /dev/null 2>&1; then
    # The run branch already exists. Guard against the stale-base bug: if it does
    # not already contain every commit from the base, it was cut from an older
    # point and would train with out-of-date code. Refuse rather than silently
    # continue; RESET_RUN_BRANCH=1 re-creates it on top of the base.
    if [[ "${RESET_RUN_BRANCH:-0}" == "1" ]]; then
      echo "branch ${RUN_BRANCH} exists — RESET_RUN_BRANCH=1, re-forking from ${base_ref}"
      run git switch -C "$RUN_BRANCH" "$base_ref"
    elif git merge-base --is-ancestor "$base_ref" "$RUN_BRANCH"; then
      echo "branch ${RUN_BRANCH} exists and already contains ${base_ref} — switching to it"
      run git switch "$RUN_BRANCH"
    else
      echo "ERROR: branch '${RUN_BRANCH}' exists but is missing commits from '${base_ref}'." >&2
      echo "       It was cut from an older base and would train with stale code." >&2
      echo "       Re-run with RESET_RUN_BRANCH=1 to re-create it on top of ${base_ref}" >&2
      echo "       (discards any commits unique to '${RUN_BRANCH}'), or rebase it manually." >&2
      exit 1
    fi
  else
    echo "creating ${RUN_BRANCH} from ${base_ref}"
    run git switch -c "$RUN_BRANCH" "$base_ref"
  fi
}

do_train() {
  # Optional remote-dataset override. Point REMOTE_DATASET_DIR at the local
  # datasets/ dir (or any readable source) when the default network mount
  # (../datasets-chr) is unavailable; train.py's cloud sync then no-ops or
  # syncs from there instead of hard-failing on a stale mount.
  local remote_args=()
  if [[ -n "${REMOTE_DATASET_DIR:-}" ]]; then
    remote_args=(--remote-dataset-dir "$REMOTE_DATASET_DIR")
  fi
  run python train.py --variables "$VARIABLE" "${TRAIN_FLAGS[@]}" \
      --data-year-start 2020 --data-year-end 2023 \
      --models-subdir "$MODELS_SUBDIR" --trial-name "$TRIAL_PREFIX" \
      "${remote_args[@]}" \
      --device "$DEVICE"
}

# The trial slug is auto-generated by train.py; locate it by our unique prefix.
resolve_model_dir() {
  local matches=()
  if compgen -G "${MODELS_SUBDIR}/${TRIAL_PREFIX}__*/${VARIABLE}" > /dev/null 2>&1; then
    mapfile -t matches < <(ls -d "${MODELS_SUBDIR}/${TRIAL_PREFIX}__"*/"${VARIABLE}")
  fi
  if [[ ${#matches[@]} -eq 0 ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      MODEL_DIR="${MODELS_SUBDIR}/${TRIAL_PREFIX}__<auto-slug>/${VARIABLE}"
      TRIAL_DIR="${MODELS_SUBDIR}/${TRIAL_PREFIX}__<auto-slug>"
      echo "model dir (DRY_RUN placeholder): $MODEL_DIR"
      return 0
    fi
    echo "ERROR: no trained model dir found for '${MODELS_SUBDIR}/${TRIAL_PREFIX}__*/${VARIABLE}'." >&2
    exit 1
  fi
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "ERROR: expected exactly one model dir for '${TRIAL_PREFIX}__*', found ${#matches[@]}:" >&2
    printf '   %s\n' "${matches[@]}" >&2
    exit 1
  fi
  MODEL_DIR="${matches[0]}"
  TRIAL_DIR="$(dirname "$MODEL_DIR")"
  echo "model dir: $MODEL_DIR"
}

# Reuse the canonical PFI -> LIME runner at full resolution (all days, all
# folds). NO_GIT=1 there so we make a single combined commit below.
do_feature_importance() {
  if [[ "$SKIP_FI" == "1" ]]; then
    echo "SKIP_FI=1 — skipping PFI + LIME"
    return 0
  fi
  run env NO_GIT=1 MODEL_DIR="$MODEL_DIR" YEAR="$YEAR" DEVICE="$DEVICE" DRY_RUN="$DRY_RUN" \
      bash scripts/fi_run_local.sh
}

# Default post-training analysis (tmax => PFI + LIME). Config scripts may REDEFINE
# do_analysis after sourcing this file (e.g. a precip run calls eval_precip.py).
do_analysis() {
  do_feature_importance
}

commit_push() {
  if [[ "$NO_GIT" == "1" ]]; then
    echo "NO_GIT=1 — skipping commit + push"
    return 0
  fi
  _assert_not_default_branch
  run git add "$TRIAL_DIR"
  if [[ "$DRY_RUN" != "1" ]] && git diff --cached --quiet; then
    echo "nothing to commit in $TRIAL_DIR"
    return 0
  fi
  run git commit -m "CLEAN solo ${VARIABLE}: ${TRIAL_PREFIX} — train 2020-2023 + ${ANALYSIS_LABEL}"
  run git push -u origin "$RUN_BRANCH"
  echo "committed + pushed ${TRIAL_DIR} on ${RUN_BRANCH}"
}

clean_run_main() {
  : "${RUN_BRANCH:?set RUN_BRANCH before calling clean_run_main}"
  : "${TRIAL_PREFIX:?set TRIAL_PREFIX before calling clean_run_main}"
  if [[ -z "${TRAIN_FLAGS+x}" ]]; then
    echo "ERROR: TRAIN_FLAGS array must be set before clean_run_main" >&2
    exit 1
  fi
  echo "=========================================================="
  echo " CLEAN solo run: ${TRIAL_PREFIX}  (${VARIABLE})"
  echo " branch        : ${RUN_BRANCH}"
  echo " train flags   : ${TRAIN_FLAGS[*]}"
  echo " data years    : 2020-2023   models -> ${MODELS_SUBDIR}/"
  echo " analysis      : ${ANALYSIS_LABEL}"
  echo " device        : ${DEVICE}"
  echo "=========================================================="
  ensure_python_env
  trap stop_keepalive EXIT
  ensure_keepalive
  make_branch
  start_keepalive
  do_train
  resolve_model_dir
  do_analysis
  commit_push
  echo "=== done: ${TRIAL_PREFIX} ==="
}
