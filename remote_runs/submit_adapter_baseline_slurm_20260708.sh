#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${ROOT:-$BASE/OpenTAD_SparseHeadClean_20260702}"
LOG_DIR="$ROOT/logs/slurm_adapter_baseline"
SBATCH_BODY="$ROOT/remote_runs/sbatch_adapter_baseline_train_20260708.sh"
EXCLUDE_NODE="${EXCLUDE_NODE:-g0030}"
PARTITION="${SLURM_PARTITION:-gpu}"
GPUS="${SLURM_GPUS:-1}"
EXP_ID="${EXP_ID:-0}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-adapter_baseline_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ADAPTER_RANDOM_CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_n16r4_retrain_20260604.py"

mkdir -p "$LOG_DIR"
if [[ -f "$BASE/conda_envs/opentad/bin/activate" ]]; then
  source "$BASE/conda_envs/opentad/bin/activate"
fi
cd "$ROOT"

run_dir_for() {
  local cfg="$1"
  local exp_id="$2"
  "$PYTHON_BIN" - "$cfg" "$exp_id" <<'PY'
import os
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
print(os.path.join(cfg.work_dir, "gpu1_id%d" % int(sys.argv[2])))
PY
}

active_job_named() {
  local label="$1"
  [[ -n "$(squeue -h -u "${USER:-sczc063}" -n "$label" 2>/dev/null || true)" ]]
}

submit_one() {
  local label="$1"
  local cfg="$2"
  local exp_id="${3:-$EXP_ID}"
  local run_tag="${RUN_TAG_PREFIX}_${label}"
  local run_dir

  run_dir="$(run_dir_for "$cfg" "$exp_id")"
  if active_job_named "$label"; then
    echo "SKIP active label=$label cfg=$cfg"
    return 0
  fi
  if [[ -e "$ROOT/$run_dir/log.json" || -e "$ROOT/$run_dir/result_detection.json" ]]; then
    echo "SKIP has_output label=$label run_dir=$run_dir"
    return 0
  fi

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

echo "Submitting trainable adapter random-fixed baseline from commit marker: $(cat "$ROOT/.codex_synced_commit" 2>/dev/null || echo unknown)"
echo "Reference run: 2026-06-04 random_fixed_50pct adapter baseline, final Average-mAP=63.77."
echo "This is pure ActionFormer + VisionTransformerAdapter, not frozen 0410 exact dense and not IrregularActionFormer."

submit_one "adapter_random_fixed_baseline63_repro" "$ADAPTER_RANDOM_CFG" "${ADAPTER_RANDOM_EXP_ID:-$EXP_ID}"
