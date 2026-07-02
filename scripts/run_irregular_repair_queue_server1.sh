#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/root/autodl-tmp/OpenTAD_Back_check}"
LOG_DIR="$ROOT_DIR/logs"
QUEUE_LOG="$LOG_DIR/irregular_repair_queue_server1.log"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"
MASTER_PORT="${MASTER_PORT:-29601}"

mkdir -p "$LOG_DIR"

CONFIGS=(
  "configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step0_dense_head_baseline.py"
  "configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step0b_dense_points_soft_sym_repaired.py"
  "configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check.py"
)

cfg_abspath() {
  local config_path="$1"
  if [[ "$config_path" = /* ]]; then
    printf '%s\n' "$config_path"
  else
    printf '%s\n' "$ROOT_DIR/$config_path"
  fi
}

wait_for_existing_jobs() {
  while pgrep -f "tools/train.py" >/dev/null; do
    echo "$(date '+%F %T') waiting for existing training jobs to finish..." | tee -a "$QUEUE_LOG"
    sleep 60
  done
}

check_cfg() {
  local config_path="$1"
  local resolved_path
  resolved_path="$(cfg_abspath "$config_path")"
  echo "$(date '+%F %T') checking ${resolved_path}" | tee -a "$QUEUE_LOG"
  cd "$ROOT_DIR"
  "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$ROOT_DIR'); from mmengine.config import Config; cfg = Config.fromfile('$resolved_path'); print(cfg.work_dir)"
}

extract_workdir() {
  local config_path="$1"
  local resolved_path
  resolved_path="$(cfg_abspath "$config_path")"
  "$PYTHON_BIN" - <<PY
import sys
sys.path.insert(0, "$ROOT_DIR")
from mmengine.config import Config
cfg = Config.fromfile("$resolved_path")
print(cfg.work_dir)
PY
}

latest_ckpt() {
  local work_dir="$1"
  local ckpt_dir="$ROOT_DIR/$work_dir/gpu1_id0/checkpoint"
  if [ -d "$ckpt_dir" ]; then
    ls -1 "$ckpt_dir"/epoch_*.pth 2>/dev/null | sort -V | tail -1 || true
  fi
}

run_one() {
  local config_path="$1"
  local name
  name="$(basename "$config_path" .py)"
  local work_dir
  work_dir="$(extract_workdir "$config_path")"
  local ckpt
  ckpt="$(latest_ckpt "$work_dir")"
  local log_file="$LOG_DIR/${name}_queue_$(date '+%Y%m%d_%H%M%S').log"

  echo "$(date '+%F %T') starting $name" | tee -a "$QUEUE_LOG"
  cd "$ROOT_DIR"
  if [ -n "${ckpt:-}" ] && [ -f "$ckpt" ]; then
    echo "$(date '+%F %T') resume from $ckpt" | tee -a "$QUEUE_LOG"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$TORCHRUN" --nproc_per_node=1 --master_port="$MASTER_PORT" tools/train.py "$config_path" --id 0 --resume "${ckpt#$ROOT_DIR/}" 2>&1 | tee "$log_file"
  else
    echo "$(date '+%F %T') fresh start" | tee -a "$QUEUE_LOG"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$TORCHRUN" --nproc_per_node=1 --master_port="$MASTER_PORT" tools/train.py "$config_path" --id 0 2>&1 | tee "$log_file"
  fi
  echo "$(date '+%F %T') finished $name" | tee -a "$QUEUE_LOG"
}

echo "$(date '+%F %T') irregular repair queue server1 start" | tee -a "$QUEUE_LOG"
wait_for_existing_jobs
for config in "${CONFIGS[@]}"; do
  check_cfg "$config"
done
for config in "${CONFIGS[@]}"; do
  run_one "$config"
done
echo "$(date '+%F %T') irregular repair queue server1 completed" | tee -a "$QUEUE_LOG"
