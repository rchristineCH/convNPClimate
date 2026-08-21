#!/usr/bin/env bash
#
# Full evaluation of the five CLEAN solo tmax models (trained 2020-2023), under the
# same two regimes the precip pipeline uses.
#
# For each model, for each regime in {cv, 2024}:
#   1. evaluate.py       -> eval_cv/ or eval_2024/ report, figures (+ .fig.pkl)
#                           and the cached prediction bundle in pred_cache/.
#                           THIS is the only GPU inference over the grid.
#   2. error_analysis.py -> error_analysis_<regime>/ (per-grid-point error
#                           structure; served entirely from step 1's bundle,
#                           so it costs no GPU)
#   3. station_analysis.py -> station_analysis_<regime>/ (29 NBCN stations;
#                           its own small bundle, since station coordinates are
#                           not on the MeteoSwiss lattice; shares one NBCN
#                           download across models)
#   4. commit + push that model's outputs, so a crash mid-matrix never loses
#      finished work.
#
# The two regimes:
#   cv    2020-2023 cross-validation holdout — each fold predicts ONLY the block
#         it was held out from, so every day is scored by a model that never saw
#         it. (The old --years 2020-2024 flow ensembled all folds over the
#         training span, which was an in-sample score labelled "unseen".)
#   2024  a genuinely unseen year — all folds, ensembled, training-frozen norm.
#
# Run it detached — this is 2 regimes x 5 models of fold inference:
#   nohup bash scripts/eval_all_clean_tmax.sh > eval.log 2>&1 &
#
# Env overrides:
#   REGIMES="cv 2024"     regimes to run (default both)
#   MODELS="a b"          evaluate only these trial dir names
#   DEVICE=cuda           torch device
#   DAY_BATCH=1           days per forward pass (larger = faster on a big GPU)
#   DRY_RUN=1             print commands instead of running them
#   NO_GIT=1              skip the commit/push after each model
#   SKIP_STATIONS=1       skip station_analysis.py (it needs network access)
#   MIN_FREE_GB=6         abort before a model if less disk than this is free
#   NO_KEEPALIVE=1        do not start gpu_keepalive.py

# Reuse run/ensure_python_env/keepalive helpers (definitions only; sourcing it
# does not start a training run). It also sets -euo pipefail and cds to the root.
source "$(dirname "${BASH_SOURCE[0]}")/_clean_run_common.sh"

REGIMES="${REGIMES:-cv 2024}"
DAY_BATCH="${DAY_BATCH:-1}"
SKIP_STATIONS="${SKIP_STATIONS:-0}"
# One model's two bundles are ~1.5 GB, so stop early rather than dying half-way
# through a write.
MIN_FREE_GB="${MIN_FREE_GB:-6}"
SHARED_DIR="CLEAN_eval_shared"   # NBCN station download only (model-independent)

# The five tmax runs collected onto this branch (see the temp-eval commit).
DEFAULT_MODELS=(
  clean_solo__tmax_atm_natg_flat_y2020-2023_e30f5_b8
  sfc_anchors__tmax_atm_natg_sfcanc_flat_y2020-2023_e30f5_b8
  baseline__tmax_sfc_flat_y2020-2023_e30f5_b8
  baseline_no_geo__tmax_sfc_flat_y2020-2023_e30f5_b8_nogeo
  clean_solo_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e30f5_b8
)
if [[ -n "${MODELS:-}" ]]; then
  read -r -a TRIALS <<< "$MODELS"
else
  TRIALS=("${DEFAULT_MODELS[@]}")
fi

# The prediction bundles are ~1.1 GB (CV) + ~0.4 GB (2024) per model, so stop
# early and loudly rather than dying half-way through with a partial npz.
check_disk() {
  local free_gb
  free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  echo "disk free: ${free_gb} GB"
  if (( free_gb < MIN_FREE_GB )); then
    echo "ERROR: only ${free_gb} GB free (< ${MIN_FREE_GB} GB) — stopping before the next model." >&2
    echo "       Free space, or re-run the rest with --no-cache (slower re-analysis)." >&2
    exit 1
  fi
}

eval_one_model() {
  local trial="$1"
  local trial_dir="${MODELS_SUBDIR}/${trial}"
  local model_dir="${trial_dir}/tmax"

  if [[ ! -f "${model_dir}/params.json" ]]; then
    echo "ERROR: ${model_dir}/params.json not found — is the model collected on this branch?" >&2
    exit 1
  fi

  echo
  echo "==================================================================="
  echo "=== ${trial}"
  echo "=== regimes ${REGIMES} | $(date '+%Y-%m-%d %H:%M:%S')"
  echo "==================================================================="
  check_disk

  for regime in $REGIMES; do
    local -a Y=()
    [[ "$regime" == "cv" ]] || Y=(--eval-year "$regime")

    echo
    echo "--- ${trial} | regime ${regime} ---"

    # evaluate.py is the foundation (its bundle feeds error_analysis), so a
    # failure there is fatal. The two analyses are additive: losing one — e.g.
    # station_analysis when the MeteoSwiss download is unreachable — must not
    # cost the rest of a multi-hour matrix.
    run python evaluate.py --model-dir "$model_dir" "${Y[@]}" \
        --device "$DEVICE" --day-batch "$DAY_BATCH"
    run python error_analysis.py --model-dir "$model_dir" "${Y[@]}" \
        --device "$DEVICE" --out "${model_dir}/error_analysis_${regime}" \
        || echo "WARNING: error_analysis.py failed for ${trial}/${regime} — continuing" >&2
    if [[ "$SKIP_STATIONS" == "1" ]]; then
      echo "SKIP_STATIONS=1 — skipping station_analysis.py"
    else
      run python station_analysis.py --model-dir "$model_dir" "${Y[@]}" \
          --device "$DEVICE" --out "${model_dir}/station_analysis_${regime}" \
          --cache-dir "${SHARED_DIR}/nbcn" --day-batch "$DAY_BATCH" \
          || echo "WARNING: station_analysis.py failed for ${trial}/${regime} — continuing" >&2
    fi
  done

  if [[ "$NO_GIT" == "1" ]]; then
    echo "NO_GIT=1 — not committing ${trial}"
    return 0
  fi
  run git add "$trial_dir"
  if [[ "$DRY_RUN" != "1" ]] && git diff --cached --quiet; then
    echo "nothing new to commit for ${trial}"
    return 0
  fi
  run git commit -m "eval ${trial}: CV + 2024 holdout reports + error/station analysis

Both regimes: the 2020-2023 cross-validation holdout (each fold scores only the
block it was held out from) and the genuine 2024 holdout year (all folds,
moment-matched ensemble, training-frozen normalization). Plus the per-grid-point
error structure and the 29 NBCN station check for each. Figures carry a .fig.pkl
next to each PNG; the prediction bundles stay local (gitignored)."
  run git push -u origin HEAD
}

main() {
  ensure_python_env
  ensure_keepalive
  start_keepalive
  trap stop_keepalive EXIT

  echo "evaluating ${#TRIALS[@]} model(s) over regimes: ${REGIMES}"
  for trial in "${TRIALS[@]}"; do
    eval_one_model "$trial"
  done

  echo
  echo "=== all models done: $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "next: python compare_evaluations.py"
}

main "$@"
