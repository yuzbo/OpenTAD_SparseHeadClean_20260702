#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/root/autodl-tmp/OpenTAD_Back_check
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

QUEUE_LOG="$LOG_DIR/headv3_x_queue_server1.log"
GPU_ID=0
TORCHRUN=/root/miniconda3/bin/torchrun

wait_for_train_idle() {
  echo "$(date '+%F %T') waiting for existing training jobs to finish..." | tee -a "$QUEUE_LOG"
  while pgrep -f "tools/train.py configs/adatad/thumos/" >/dev/null || screen -ls | grep -q "headv2_x_k3_wait"; do
    echo "$(date '+%F %T') still waiting..." | tee -a "$QUEUE_LOG"
    sleep 60
  done
}

run_exp() {
  local config_path="$1"
  local name="$2"
  local log_file="$LOG_DIR/${name}_$(date '+%Y%m%d_%H%M%S').log"

  echo "$(date '+%F %T') starting $name" | tee -a "$QUEUE_LOG"
  cd "$ROOT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$TORCHRUN" --nproc_per_node=1 tools/train.py "$config_path" --id 0 2>&1 | tee "$log_file"
  echo "$(date '+%F %T') finished $name" | tee -a "$QUEUE_LOG"
}

wait_for_train_idle
run_exp "configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_headv3_x.py" "input_random_fixed_50pct_irregular_actionformer_headv3_x"
run_exp "configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_headv3_timeembed_x.py" "input_random_fixed_50pct_irregular_actionformer_headv3_timeembed_x"
echo "$(date '+%F %T') headv3 x queue completed" | tee -a "$QUEUE_LOG"
