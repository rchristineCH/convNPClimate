#!/usr/bin/env bash
#
# Paired gradient-clipped retrain + FULL evaluation of the two atmospheric tmax
# configurations (z,t,q vs z,t,q,u,v).
#
# WHY: across the five CLEAN tmax runs, 8 of 25 folds peaked before epoch 15 of 30,
# and the frequency tracks input dimensionality (6 ch -> 1 fold; 155 ch -> 3 folds).
# clean_solo_wind fold 3 peaked at epoch 3 and ended WORSE than it started; clean_solo
# fold 2 ended at train NLL +1.161 after peaking at -0.328. That is divergence, not
# early stopping — and train-vs-test gaps (wind fold 2: train -0.495 / test -0.173)
# say the models already overfit, so the fix is regularisation, NOT more capacity.
#
# The optimiser is bare: torch.optim.Adam(model.parameters(), lr=p.LR) at train.py:542
# — no weight decay, no LR schedule, no dropout, and GRAD_CLIP_NORM=None on every tmax
# run, while precip sets 1.0 precisely because its Gamma NLL needed stabilising.
#
# WHAT CHANGES vs the CLEAN runs: budget 30->60 epochs, patience 10->20, and
# --grad-clip 1.0. Nothing else. Both models get the identical change and run
# CONCURRENTLY on the one GPU, so the pair stays internally comparable and a wind win
# is attributable.
#
# Trial names are lbclip_* because _config_slug (train.py:203) encodes epochs/folds/
# batch size but NOT grad_clip — two runs differing only in clipping would otherwise
# collide in the same output directory.
#
# STAGES (each commits on success, so a crash never loses finished work):
#   1. train      both models, concurrent                      ~2 h
#   2. evaluate   CV holdout + 2024 holdout, per model         ~4 min per model-regime
#   3. analyses   error_analysis + station_analysis, both      cache-served, no GPU
#   4. feature importance  PFI + LIME at full resolution       ~8-10 h (the long pole)
#
# Usage:
#   nohup bash scripts/run_paired_clipped.sh > paired_clipped.log 2>&1 &
# Env:
#   NO_GIT=1        do not commit between stages
#   SKIP_FI=1       stop after stage 3 (skip the ~8 h feature-importance stage)
#   SKIP_TRAIN=1    reuse existing lbclip_* checkpoints, go straight to evaluation
#   DEVICE=cuda     torch device
set -uo pipefail
cd /home/renku/work/convNPClimate

DEVICE="${DEVICE:-cuda}"
NO_GIT="${NO_GIT:-0}"
SKIP_FI="${SKIP_FI:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SUBDIR=CLEAN_trained_models

ATM_DIR="$SUBDIR/lbclip_atm__tmax_atm_natg_flat_y2020-2023_e60f5_b8/tmax"
WIND_DIR="$SUBDIR/lbclip_wind__tmax_atm_natg_av-ztquv_flat_y2020-2023_e60f5_b8/tmax"

say() { echo; echo "=== $* | $(date '+%F %T') ==="; }

commit() {  # commit <paths...> with the message on stdin
  [[ "$NO_GIT" == "1" ]] && { echo "NO_GIT=1 — not committing"; return 0; }
  git add -A "$@" || return 0
  git diff --cached --quiet && { echo "nothing new to commit"; return 0; }
  git commit -qF - && git push -q origin HEAD && echo "committed + pushed"
}

# ---------------------------------------------------------------- 1. train
if [[ "$SKIP_TRAIN" != "1" ]]; then
  say "stage 1/4: paired clipped training (60 epochs, patience 20, grad-clip 1.0)"
  COMMON=(--variables tmax --use-atmospheric --atmos-native-grid
          --data-year-start 2020 --data-year-end 2023
          --n-epochs 60 --patience 20 --grad-clip 1.0
          --models-subdir "$SUBDIR" --device "$DEVICE")

  python train.py "${COMMON[@]}" --trial-name lbclip_atm  > lbclip_atm.log  2>&1 &
  P1=$!
  python train.py "${COMMON[@]}" --atmos-variables z t q u v --trial-name lbclip_wind > lbclip_wind.log 2>&1 &
  P2=$!
  echo "training: atm=$P1 wind=$P2"
  wait $P1; R1=$?
  wait $P2; R2=$?
  echo "training exits: atm=$R1 wind=$R2"
  if (( R1 != 0 || R2 != 0 )); then
    echo "ERROR: a training run failed — stopping before evaluation." >&2
    exit 1
  fi
  commit "$SUBDIR" <<EOF
lbclip: paired gradient-clipped retrain of the two atmospheric tmax models

60 epochs / patience 20 / --grad-clip 1.0, both models trained concurrently on one
GPU so conditions are identical. Only the budget and clipping change vs the CLEAN
runs; every other hyperparameter is untouched, so lbclip_wind vs lbclip_atm is a
clean test of whether the u/v wind channels help once training is stable.

Motivated by 8 of 25 CLEAN folds peaking before epoch 15, one of which diverged
outright (wind fold 3: best epoch 3, final NLL worse than its start).
EOF
fi

for d in "$ATM_DIR" "$WIND_DIR"; do
  [[ -f "$d/params.json" ]] || { echo "ERROR: $d not found — training did not produce it." >&2; exit 1; }
done

# ------------------------------------------------------------- 2. evaluate
say "stage 2/4: evaluation (CV holdout + 2024 holdout)"
for d in "$ATM_DIR" "$WIND_DIR"; do
  for regime in cv 2024; do
    Y=(); [[ "$regime" == "cv" ]] || Y=(--eval-year "$regime")
    python evaluate.py --model-dir "$d" "${Y[@]}" --device "$DEVICE" --day-batch 8 \
      || { echo "ERROR: evaluate.py failed for $d/$regime" >&2; exit 1; }
  done
done

# ------------------------------------------------------------- 3. analyses
say "stage 3/4: error + station analysis (cache-served from stage 2)"
for d in "$ATM_DIR" "$WIND_DIR"; do
  for regime in cv 2024; do
    Y=(); [[ "$regime" == "cv" ]] || Y=(--eval-year "$regime")
    python error_analysis.py --model-dir "$d" "${Y[@]}" --device "$DEVICE" \
        --out "$d/error_analysis_${regime}" \
      || echo "WARNING: error_analysis failed for $d/$regime — continuing" >&2
    python station_analysis.py --model-dir "$d" "${Y[@]}" --device "$DEVICE" \
        --out "$d/station_analysis_${regime}" --cache-dir CLEAN_eval_shared/nbcn --day-batch 8 \
      || echo "WARNING: station_analysis failed for $d/$regime — continuing" >&2
  done
done
python compare_evaluations.py || echo "WARNING: compare_evaluations failed — continuing" >&2

commit "$SUBDIR" <<EOF
lbclip: full evaluation under both regimes + error/station analysis

CV holdout 2020-2023 (each fold scores only its own held-out block) and the
genuine 2024 holdout, plus per-grid-point error structure and the 29 NBCN station
check for each. All analyses are served from the stage-2 prediction bundles.
EOF

# ----------------------------------------------------- 4. feature importance
if [[ "$SKIP_FI" == "1" ]]; then
  say "SKIP_FI=1 — stopping before feature importance"
  echo "PAIRED CLIPPED PIPELINE DONE (no FI)"
  exit 0
fi

say "stage 4/4: feature importance (PFI + LIME, full resolution) — the long pole"
# Matches what the CLEAN models carry: year 2022, all 365 days, all 5 folds, PFI+LIME.
# SHAP is deliberately excluded — its KernelSHAP pass is ~30 h and no CLEAN tmax model
# has it either, so including it would make these two incomparable with the rest.
python feature_importance.py --method all --model-dir "$ATM_DIR" --device "$DEVICE" \
  > fi_lbclip_atm.log 2>&1 &
F1=$!
python feature_importance.py --method all --model-dir "$WIND_DIR" --device "$DEVICE" \
  > fi_lbclip_wind.log 2>&1 &
F2=$!
echo "feature importance: atm=$F1 wind=$F2"
wait $F1; S1=$?
wait $F2; S2=$?
echo "feature-importance exits: atm=$S1 wind=$S2"

commit "$SUBDIR" <<EOF
lbclip: feature importance (PFI + LIME) for both clipped models

Full resolution (year 2022, 365 days, all 5 folds), matching what the CLEAN tmax
models carry so the two are directly comparable. SHAP excluded for the same reason
— no CLEAN tmax model has it.
EOF

say "PAIRED CLIPPED PIPELINE DONE atm_fi=$S1 wind_fi=$S2"
