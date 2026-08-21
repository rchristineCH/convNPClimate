#!/usr/bin/env bash
#
# CLEAN solo run — atmospheric precipitation trained on a CRPS loss.
#   Identical to clean_run_precip_atm.sh (atmospheric native grid, no surface
#   anchors, same Bernoulli-Gamma output head) EXCEPT the training loss is the
#   sample-based Bernoulli-Gamma CRPS instead of the NLL:
#     --distribution bernoulli_gamma_crps
#
#   The output distribution/parameterization is unchanged (3 params
#   [rho, alpha, beta]); only what the optimizer minimizes differs (mean CRPS in
#   mm vs. negative log-likelihood). MC sample count = params.CRPS_N_SAMPLES (25).
#
# Trains on 2020-2023 into CLEAN_trained_models/, then runs eval_precip.py physical
# metrics, and commits the model + metrics on its own branch. The slug carries
# 'bgcrps' so it never collides with the NLL run.
#
# Run this on ONE cluster. Usage:
#   bash scripts/clean_run_precip_atm_crps.sh
#   DRY_RUN=1   bash scripts/clean_run_precip_atm_crps.sh   # print commands only
#   SKIP_EVAL=1 bash scripts/clean_run_precip_atm_crps.sh   # train + commit, no eval

RUN_BRANCH="clean-run/precip-atm-crps"
TRIAL_PREFIX="clean_solo_precip_crps"
VARIABLE="precip"
ANALYSIS_LABEL="eval_precip"
TRAIN_FLAGS=(--use-atmospheric --atmos-native-grid --distribution bernoulli_gamma_crps)

source "$(dirname "${BASH_SOURCE[0]}")/_clean_run_common.sh"

# Precip analysis = eval_precip.py physical metrics -> <model_dir>/eval_precip_metrics.json
do_analysis() {
  if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
    echo "SKIP_EVAL=1 — skipping eval_precip"
    return 0
  fi
  run python eval_precip.py --model-dir "$MODEL_DIR" --device "$DEVICE"
}

clean_run_main
