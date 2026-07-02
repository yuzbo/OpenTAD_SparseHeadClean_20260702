#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/root/autodl-tmp/OpenTAD_Back_check}"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

QUEUE_LOG="$LOG_DIR/adapter_irregular_driver.log"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"
BASE_PORT="${BASE_PORT:-29820}"
PARALLEL_SMOKE="${PARALLEL_SMOKE:-0}"
PARALLEL_SMOKE_SCREEN="${PARALLEL_SMOKE_SCREEN:-adapter_irregular_parallel_smoke_35407}"
PARALLEL_SMOKE_PORT_OFFSET="${PARALLEL_SMOKE_PORT_OFFSET:-100}"

CONFIGS=(
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_contract_smoke.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_dense_control.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_y.py"
)

NAMES=(
  "input_random_fixed_50pct_adapter_irregular_contract_smoke"
  "input_random_fixed_50pct_adapter_irregular_dense_control"
  "input_random_fixed_50pct_adapter_irregular_headv3_x"
  "input_random_fixed_50pct_adapter_irregular_headv3_y"
)

SMOKE_CONFIG="configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_contract_smoke.py"
SMOKE_NAME="input_random_fixed_50pct_adapter_irregular_contract_smoke"

log_msg() {
  echo "$(date '+%F %T') $*" | tee -a "$QUEUE_LOG"
}

launch_parallel_smoke() {
  local port="$((BASE_PORT + PARALLEL_SMOKE_PORT_OFFSET))"
  local log_file="$LOG_DIR/${SMOKE_NAME}_parallel_$(date '+%Y%m%d_%H%M%S').log"
  log_msg "launching parallel smoke ${SMOKE_NAME} in screen ${PARALLEL_SMOKE_SCREEN} with log ${log_file}"
  screen -dmS "$PARALLEL_SMOKE_SCREEN" bash -lc \
    "cd '$ROOT_DIR' && CUDA_VISIBLE_DEVICES='$GPU_ID' '$TORCHRUN' --master_port='$port' --nproc_per_node=1 tools/train.py '$SMOKE_CONFIG' --id 0 2>&1 | tee '$log_file'"
}

wait_for_adapter_newideas_queue() {
  while true; do
    if ! screen -ls 2>/dev/null | grep -q '\.adapter_newideas_p0'; then
      break
    fi
    log_msg "waiting for adapter_newideas_p0 to finish before starting IrregularActionFormer queue"
    sleep 300
  done
}

wait_for_gpu() {
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID + 1))p")
    if [[ -n "${used}" && "${used}" -lt 800 ]]; then
      break
    fi
    log_msg "waiting for GPU ${GPU_ID}, memory.used=${used:-unknown} MiB"
    sleep 60
  done
}

check_cfg() {
  local config_path="$1"
  log_msg "checking ${config_path}"
  cd "$ROOT_DIR"
  "$PYTHON_BIN" - <<PY
import sys
sys.path.insert(0, "$ROOT_DIR")
from mmengine.config import Config

cfg = Config.fromfile("$config_path")
workflow = cfg.workflow
assert cfg.model.type == "IrregularActionFormer", cfg.model.type
assert cfg.model.backbone.backbone.type == "VisionTransformerAdapter", cfg.model.backbone.backbone.type
assert int(workflow.checkpoint_interval) == 10, workflow
assert not bool(workflow.get("disable_checkpoint", False)), workflow
print("work_dir=", cfg.work_dir)
print("detector=", cfg.model.type)
print("backbone=", cfg.model.backbone.backbone.type)
print("projection=", cfg.model.projection.type)
print("neck=", cfg.model.neck.type)
print("head=", cfg.model.rpn_head.type)
print("checkpoint_interval=", workflow.checkpoint_interval)
print("disable_checkpoint=", workflow.get("disable_checkpoint", False))
print("end_epoch=", workflow.end_epoch)
PY
}

run_exp() {
  local config_path="$1"
  local name="$2"
  local port="$3"
  local log_file="$LOG_DIR/${name}_$(date '+%Y%m%d_%H%M%S').log"

  log_msg "starting ${name}"
  cd "$ROOT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$TORCHRUN" --master_port="$port" --nproc_per_node=1 \
    tools/train.py "$config_path" --id 0 2>&1 | tee "$log_file"
  log_msg "finished ${name}"
}

log_msg "adapter IrregularActionFormer serial queue start"
df -h /root/autodl-tmp | tee -a "$QUEUE_LOG"

for cfg in "${CONFIGS[@]}"; do
  check_cfg "$cfg"
done

if [[ "$PARALLEL_SMOKE" == "1" ]]; then
  launch_parallel_smoke
  CONFIGS=( "${CONFIGS[@]:1}" )
  NAMES=( "${NAMES[@]:1}" )
fi

wait_for_adapter_newideas_queue

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  name="${NAMES[$i]}"
  port="$((BASE_PORT + i))"
  wait_for_gpu
  run_exp "$cfg" "$name" "$port"
done

log_msg "adapter IrregularActionFormer serial queue complete"
