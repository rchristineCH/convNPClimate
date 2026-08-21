#!/usr/bin/env bash
#
# Feature importance — combined cross-method EVALUATION.
#
# Combines whichever of pfi.csv / shap.csv / lime.csv are present in
# <MODEL_DIR>/feature_importance/ (two or more required). Run it after the local
# PFI+LIME run (scripts/fi_run_local.sh); if the ~30 h cluster SHAP run
# (scripts/fi_run_shap.sh) has not happened yet, the report is generated as
# 2-method and says so explicitly rather than refusing.
#
# It re-scores nothing expensive: it rebuilds the lightweight backbone once
# (loads the 5-fold ensemble + context, one baseline pass ~1–2 min) purely to
# get the grouping metadata + baseline, then reuses feature_importance.write_summary
# to regenerate the unified cross-method summary.md and the level×hour heatmaps,
# and finally assembles a full-run REPORT.md (config + cross-method tables +
# figure gallery + interpretation), following the SMOKE_REPORT.md structure.
#
# Outputs (into <MODEL_DIR>/feature_importance/): summary.md (cross-method),
# *_heatmap_*.png, baseline.json (refreshed), REPORT.md. Log: fi_evaluate.log.
#
# On success it commits summary.md + REPORT.md (+ heatmaps, baseline.json) and
# pushes (current branch; skipped with NO_GIT=1; refuses to commit on master).
#
# Usage:
#   bash scripts/fi_evaluate.sh                   # combine + write REPORT.md + commit/push
#   DRY_RUN=1 bash scripts/fi_evaluate.sh         # print what it would do
#   NO_GIT=1 bash scripts/fi_evaluate.sh          # combine but do not commit/push
#   FI_DIR=path/to/feature_importance bash scripts/fi_evaluate.sh   # e.g. a smoke dir

source "$(dirname "${BASH_SOURCE[0]}")/_fi_common.sh"

fi_preflight

# At least TWO method CSVs must be present per head — a "cross-method" report needs
# something to cross. A missing method is reported and the report is generated from
# the rest (SHAP is the ~30 h cluster method, so PFI+LIME is the common case and must
# not be blocked on it). Joint models write CSVs per head subdir (tmax/ + precip/);
# the standalone model writes them directly in FI_DIR.
if [[ "$FI_IS_JOINT" == "1" ]]; then
  csv_dirs=("${FI_DIR}/tmax" "${FI_DIR}/precip")
else
  csv_dirs=("${FI_DIR}")
fi
missing=()
for d in "${csv_dirs[@]}"; do
  present=0
  for m in pfi shap lime; do
    if [[ -f "${d}/${m}.csv" ]]; then present=$((present + 1))
    else missing+=("${d#"${FI_DIR}"/}/${m}.csv"); fi
  done
  if (( present < 2 )); then
    echo "ERROR: ${d} has only ${present} method CSV(s); a cross-method report needs 2+." >&2
    echo "       Run scripts/fi_run_local.sh (PFI+LIME) first." >&2
    exit 1
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "NOTE: missing method CSV(s) under ${FI_DIR}: ${missing[*]}"
  echo "      Proceeding without them; the report states which methods it covers."
fi

# Rebuild the report's baseline at the SAME day-stride the method CSVs were computed
# with, so baseline.json matches the importances (unless the caller overrode it).
export FI_EVAL_STRIDE="${FI_EVAL_STRIDE:-$DAY_STRIDE}"
fi_run fi_evaluate python scripts/fi_combine.py "$MODEL_DIR" "$YEAR" "$FI_DIR"

if [[ "$FI_IS_JOINT" == "1" ]]; then
  fi_commit_push "feature-importance: combined evaluation + REPORT.md (joint tmax+precip)"
  echo
  echo "Combined evaluation written (per head):"
  echo "  ${FI_DIR}/{tmax,precip}/summary.md   (cross-method tables + heatmaps)"
  echo "  ${FI_DIR}/{tmax,precip}/REPORT.md     (full-run analysis report)"
else
  fi_commit_push "feature-importance: combined evaluation + REPORT.md (${YEAR})"
  echo
  echo "Combined evaluation written:"
  echo "  ${FI_DIR}/summary.md   (cross-method tables + heatmaps)"
  echo "  ${FI_DIR}/REPORT.md     (full-run analysis report)"
fi
