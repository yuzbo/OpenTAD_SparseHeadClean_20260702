#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${ROOT:-$BASE/OpenTAD_SparseHeadClean_20260702}"
cd "$ROOT"

submit_one() {
  local label="$1"
  local cfg="$2"
  local exp_id="${3:-0}"
  echo "[protocol-bridge-submit] label=$label cfg=$cfg exp_id=$exp_id"
  sbatch \
    --job-name="$label" \
    --export=ALL,CFG="$cfg",EXP_LABEL="$label",EXP_ID="$exp_id" \
    remote_runs/sbatch_protocol_bridge_train_20260708.sh
}

submit_one \
  "bridge_exact0410_random_irregular_densepass" \
  "configs/adatad/thumos/input_random_fixed_50pct_0410_exact_irregular_densepass_n16r4.py" \
  "${EXACT_IRREGULAR_EXP_ID:-0}"

submit_one \
  "bridge_current_random_actionformer_bs8" \
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_actionformer_selected_axis_bs8_n16r4.py" \
  "${CURRENT_RANDOM_ACTIONFORMER_EXP_ID:-0}"

submit_one \
  "bridge_current_uniform_actionformer_bs8" \
  "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_actionformer_selected_axis_bs8_n16r4.py" \
  "${CURRENT_UNIFORM_ACTIONFORMER_EXP_ID:-0}"
