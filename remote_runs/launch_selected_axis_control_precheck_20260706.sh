#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${ROOT:-$BASE/OpenTAD_SparseHeadClean_20260702}"
LOG_DIR="$ROOT/logs/selected_axis_control_precheck"
RUN_TAG="${RUN_TAG:-selected_axis_control_precheck_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="$LOG_DIR/${RUN_TAG}.log"

mkdir -p "$LOG_DIR"

echo "RUN_TAG=$RUN_TAG"
echo "LOG_FILE=$LOG_FILE"
echo "PRECHECK_ONLY=1"

bash "$ROOT/remote_runs/precheck_selected_axis_control_matrix_20260706.sh" 2>&1 | tee "$LOG_FILE"
