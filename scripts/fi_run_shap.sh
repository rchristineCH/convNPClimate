#!/usr/bin/env bash
#
# Feature importance — SHAP full run (HEAVY; cluster-intended).
#
# Channel-coalition Shapley values: exact for the small var/level/hour groupings,
# sampled KernelSHAP for the 95 individual channels. At full defaults this uses
# --shap-nsamples 2000 coalitions over all 365 days × all 5 folds, ~30 h on a
# single A10 — which is why it is split out from the local PFI+LIME run and meant
# to run on the cluster.
#
# Runtime/precision lever: --shap-nsamples. Fewer coalitions = faster but noisier
# Shapley estimates for the per-channel ranking (the grouped exact values are
# unaffected). Override via SHAP_NSAMPLES below.
#
# Parallelism lever: WORKERS (CPU worker processes; default 11 ≈ cores-1). The
# per-coalition forward is compute-bound and runs single-threaded per worker
# (torch.set_num_threads(1)), so throughput scales ~linearly with workers as long
# as you do NOT oversubscribe the cores — keep WORKERS ≤ (cores-1) and make sure
# nothing else heavy (e.g. gpu_keepalive.py) is running, or loadavg exceeds the
# core count and everything thrashes. DEVICE defaults to cpu (the only mode that
# parallelises; forking after CUDA init is unsafe). LOG_EVERY sets the progress
# cadence (first 10 evals, then every N) written to the fi_shap.log.
#
# Outputs (into <MODEL_DIR>/feature_importance/): shap.csv + PNGs, baseline.json,
# a method-local summary.md. Log: fi_shap.log. This writes a fresh shap.csv,
# superseding any smoke-test SHAP outputs.
#
# On success it commits the fresh shap.csv + PNGs and pushes (current branch;
# skipped with NO_GIT=1; refuses to commit on master — branch first).
#
# Usage:
#   bash scripts/fi_run_shap.sh                       # full run (2000 coalitions, 11 workers, CPU)
#   DRY_RUN=1 bash scripts/fi_run_shap.sh             # print the command only
#   NO_GIT=1 bash scripts/fi_run_shap.sh              # run but do not commit/push
#   SHAP_NSAMPLES=600 bash scripts/fi_run_shap.sh     # faster, noisier per-channel
#   WORKERS=8 bash scripts/fi_run_shap.sh             # fewer CPU workers
#   DEVICE=cuda bash scripts/fi_run_shap.sh           # run on GPU (day-batched, mem-safe batch 32)
#   DEVICE=cuda DAY_BATCH=48 bash scripts/fi_run_shap.sh   # larger GPU day-batch (watch memory)
#   LOG_EVERY=25 bash scripts/fi_run_shap.sh          # quieter progress log
#   YEAR=2021 MODEL_DIR=path/to/tmax bash scripts/fi_run_shap.sh

source "$(dirname "${BASH_SOURCE[0]}")/_fi_common.sh"

SHAP_NSAMPLES="${SHAP_NSAMPLES:-2000}"
DEVICE="${DEVICE:-cpu}"
LOG_EVERY="${LOG_EVERY:-10}"

# On GPU, scoring runs serially (fork-after-CUDA is unsafe) and the speedup comes
# from DAY_BATCH (days per forward); on CPU, from WORKERS (fork pool). Pick device-
# appropriate defaults so the common invocations "just work".
if [[ "$DEVICE" == "cpu" ]]; then
  WORKERS="${WORKERS:-11}"
  DAY_BATCH="${DAY_BATCH:-1}"
  export CUDA_VISIBLE_DEVICES=""   # take the verified fork-based CPU path
else
  WORKERS="${WORKERS:-1}"          # parallel workers are ignored on GPU anyway
  DAY_BATCH="${DAY_BATCH:-32}"     # days/forward. Memory-safe on the 8 GB A10-8Q vGPU: the
                                   # elev-MLP allocs ~(B x 88800 x hidden), so batch=64 peaks
                                   # ~7.2 GiB (92%, OOM-risky for a long run); 32 peaks ~4-5 GiB.
fi

fi_preflight
echo "SHAP coalitions: $SHAP_NSAMPLES | device: $DEVICE | workers: $WORKERS | day-batch: $DAY_BATCH | day-stride: $DAY_STRIDE | log-every: $LOG_EVERY"

# --year is ignored for a joint model (both heads scored over its DATA_YEAR span).
fi_run fi_shap python feature_importance.py --method shap \
  --model-dir "$MODEL_DIR" --year "$YEAR" --day-stride "$DAY_STRIDE" --shap-nsamples "$SHAP_NSAMPLES" \
  --device "$DEVICE" --workers "$WORKERS" --day-batch "$DAY_BATCH" --log-every "$LOG_EVERY"

if [[ "$FI_IS_JOINT" == "1" ]]; then
  fi_commit_push "feature-importance: joint SHAP run (tmax+precip, stride ${DAY_STRIDE}, ${SHAP_NSAMPLES} coalitions)"
else
  fi_commit_push "feature-importance: full SHAP run (${YEAR}, ${SHAP_NSAMPLES} coalitions)"
fi

echo
echo "SHAP done. Once pfi.csv / lime.csv / shap.csv are all in $FI_DIR,"
echo "combine them with scripts/fi_evaluate.sh."
