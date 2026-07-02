#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/root/autodl-tmp/OpenTAD_Back_check}"
LOG_DIR="$ROOT_DIR/logs"
QUEUE_LOG="$LOG_DIR/temporalgridfix_diag_server3.log"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"

mkdir -p "$LOG_DIR"

wait_for_gpu() {
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID + 1))p")
    if [[ -n "${used}" && "${used}" -lt 500 ]]; then
      break
    fi
    echo "$(date '+%F %T') waiting for GPU ${GPU_ID}, memory.used=${used:-unknown} MiB" | tee -a "$QUEUE_LOG"
    sleep 60
  done
}

check_cfg() {
  local config_path="$1"
  echo "$(date '+%F %T') checking ${config_path}" | tee -a "$QUEUE_LOG"
  cd "$ROOT_DIR"
  "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$ROOT_DIR'); from mmengine.config import Config; cfg = Config.fromfile('$config_path'); print(cfg.work_dir)"
}

run_exp() {
  local config_path="$1"
  local name="$2"
  local master_port="$3"
  local log_file="$LOG_DIR/${name}_$(date '+%Y%m%d_%H%M%S').log"

  echo "$(date '+%F %T') starting ${name}" | tee -a "$QUEUE_LOG"
  cd "$ROOT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$TORCHRUN" --nproc_per_node=1 --master_port="$master_port" tools/train.py "$config_path" --id 0 2>&1 | tee "$log_file"
  echo "$(date '+%F %T') finished ${name}" | tee -a "$QUEUE_LOG"
}

CFG_PATH="configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_headv3_x_temporalgridfix_diag_epoch1.py"
EXP_NAME="input_random_fixed_50pct_irregular_actionformer_headv3_x_temporalgridfix_diag_epoch1"

echo "$(date '+%F %T') temporalgridfix diag server3 start" | tee -a "$QUEUE_LOG"
check_cfg "$CFG_PATH"
wait_for_gpu
run_exp "$CFG_PATH" "$EXP_NAME" 29833
echo "$(date '+%F %T') temporalgridfix diag server3 done" | tee -a "$QUEUE_LOG"
