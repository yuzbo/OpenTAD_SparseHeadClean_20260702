#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702}"
LOG_DIR="$ROOT/logs/slurm_sparse_diag_20260707"
SBATCH_BODY="$ROOT/remote_runs/sbatch_sparse_diag_train_20260707.sh"
EXCLUDE_NODE="${EXCLUDE_NODE:-g0030}"
PARTITION="${SLURM_PARTITION:-gpu}"
GPUS="${SLURM_GPUS:-1}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-sparse_diag_$(date +%Y%m%d_%H%M%S)}"
EXP_ID="${EXP_ID:-0}"

SELECTED_AXIS_CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py"
UNIFORM_NATIVE_CFG="configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py"
SHORTGATE_CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py"

mkdir -p "$LOG_DIR"
cd "$ROOT"

submit_one() {
  local label="$1"
  local cfg="$2"
  local exp_id="$3"
  local run_tag="${RUN_TAG_PREFIX}_${label}"
  echo "SUBMIT label=$label cfg=$cfg exp_id=$exp_id exclude=$EXCLUDE_NODE"
  sbatch \
    --partition="$PARTITION" \
    --gpus="$GPUS" \
    --exclude="${EXCLUDE_NODE:-g0030}" \
    --job-name="$label" \
    --output="$LOG_DIR/%x-%j.out" \
    --export=ALL,CFG="$cfg",EXP_LABEL="$label",EXP_ID="$exp_id",RUN_TAG="$run_tag" \
    "$SBATCH_BODY"
}

submit_one "sparse_selected_axis_bridge_absx" "$SELECTED_AXIS_CFG" "${SELECTED_AXIS_EXP_ID:-$EXP_ID}"
submit_one "sparse_uniform_native_bridge_absx" "$UNIFORM_NATIVE_CFG" "${UNIFORM_NATIVE_EXP_ID:-$EXP_ID}"
submit_one "sparse_shortgate_native_bridge_absx" "$SHORTGATE_CFG" "${SHORTGATE_EXP_ID:-$EXP_ID}"
