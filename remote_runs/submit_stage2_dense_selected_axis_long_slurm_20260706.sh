#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702}"
LOG_DIR="$ROOT/logs/slurm_stage2_dense_selected_axis_long"
SBATCH_BODY="$ROOT/remote_runs/sbatch_stage2_dense_selected_axis_train_20260706.sh"
EXCLUDE_NODE="${EXCLUDE_NODE:-g0030}"
PARTITION="${SLURM_PARTITION:-gpu}"
GPUS="${SLURM_GPUS:-1}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-stage2_dense_axis_$(date +%Y%m%d_%H%M%S)}"

RANDOM_FIXED_CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py"
UNIFORM_OFFICIAL_CFG="configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py"

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

submit_one "near63_random_fixed_selected_axis_dense_control" "$RANDOM_FIXED_CFG" "${RANDOM_EXP_ID:-0}"
submit_one "near65_uniform_even_spacing_official_dense" "$UNIFORM_OFFICIAL_CFG" "${UNIFORM_EXP_ID:-0}"
