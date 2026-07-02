#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/root/autodl-tmp/OpenTAD_Back_check}"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

QUEUE_LOG="$LOG_DIR/adapter_irregular_fix_controls_driver.log"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"
BASE_PORT="${BASE_PORT:-30020}"
GPU_FREE_MIB="${GPU_FREE_MIB:-900}"
WAIT_SCREENS="${WAIT_SCREENS:-}"
CHECK_ONLY="${CHECK_ONLY:-0}"

CONFIGS=(
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_pdrop02_control.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_dense_control_pdrop0.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_minimal_container_control.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_y_pdrop0.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_noboundary.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_nogeometry.py"
)

NAMES=(
  "input_random_fixed_50pct_adapter_pdrop02_control"
  "input_random_fixed_50pct_adapter_irregular_dense_control_pdrop0"
  "input_random_fixed_50pct_adapter_irregular_minimal_container_control"
  "input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0"
  "input_random_fixed_50pct_adapter_irregular_headv3_y_pdrop0"
  "input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_noboundary"
  "input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_nogeometry"
)

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-$((${#CONFIGS[@]} - 1))}"

log_msg() {
  echo "$(date '+%F %T') $*" | tee -a "$QUEUE_LOG"
}

screen_is_active() {
  local screen_name="$1"
  screen -ls 2>/dev/null | grep -q "\\.${screen_name}[[:space:]]"
}

wait_for_screens() {
  if [[ -z "$WAIT_SCREENS" ]]; then
    return
  fi
  while true; do
    local active=""
    for screen_name in $WAIT_SCREENS; do
      if screen_is_active "$screen_name"; then
        active="${active} ${screen_name}"
      fi
    done
    if [[ -z "$active" ]]; then
      break
    fi
    log_msg "waiting for screens to finish:${active}"
    sleep 300
  done
}

wait_for_gpu() {
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID + 1))p")
    if [[ -n "${used}" && "${used}" -lt "$GPU_FREE_MIB" ]]; then
      break
    fi
    log_msg "waiting for GPU ${GPU_ID}, memory.used=${used:-unknown} MiB, threshold=${GPU_FREE_MIB} MiB"
    sleep 60
  done
}

check_cfg() {
  local config_path="$1"
  local config_name="${config_path##*/}"
  log_msg "checking ${config_path}"
  cd "$ROOT_DIR"
  "$PYTHON_BIN" - <<PY
import sys
sys.path.insert(0, "$ROOT_DIR")
from mmengine.config import Config

config_path = "$config_path"
config_name = "$config_name"
cfg = Config.fromfile(config_path)
workflow = cfg.workflow
assert int(workflow.checkpoint_interval) == 10, workflow
assert not bool(workflow.get("disable_checkpoint", False)), workflow

proj = cfg.model.projection
model_type = cfg.model.type

if config_name == "input_random_fixed_50pct_adapter_pdrop02_control.py":
    assert model_type == "ActionFormer", model_type
    assert abs(float(proj.input_pdrop) - 0.2) < 1e-9, proj
else:
    assert model_type == "IrregularActionFormer", model_type
    assert proj.type in {
        "DensePassthroughConv1DTransformerProj",
        "GridAwareConv1DTransformerProj",
        "IrregularConvTransformerProj",
    }, proj.type
    if "pdrop0" in config_name or "minimal_container" in config_name:
        assert abs(float(proj.input_pdrop) - 0.0) < 1e-9, proj

if "minimal_container" in config_name:
    bb = cfg.model.backbone.backbone
    assert not bool(bb.get("add_irregular_time_embed", False)), bb
    assert not bool(bb.get("use_irregular_time_embed", False)), bb
    assert cfg.model.neck.type == "DensePassthroughFPNIdentity", cfg.model.neck

if "noboundary" in config_name:
    assert abs(float(cfg.model.rpn_head.boundary_loss_weight) - 0.0) < 1e-9, cfg.model.rpn_head

if "nogeometry" in config_name:
    assert abs(float(cfg.model.rpn_head.geometry_scale) - 0.0) < 1e-9, cfg.model.rpn_head

print("work_dir=", cfg.work_dir)
print("model=", model_type)
print("projection=", proj.type)
print("input_pdrop=", proj.get("input_pdrop", None))
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

log_msg "adapter IrregularActionFormer fix-controls queue start"
log_msg "run slice START_INDEX=${START_INDEX} END_INDEX=${END_INDEX}"
df -h /root/autodl-tmp | tee -a "$QUEUE_LOG"

for cfg in "${CONFIGS[@]}"; do
  check_cfg "$cfg"
done

if [[ "$CHECK_ONLY" == "1" ]]; then
  log_msg "adapter IrregularActionFormer fix-controls queue check-only complete"
  exit 0
fi

wait_for_screens

for i in "${!CONFIGS[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi
  cfg="${CONFIGS[$i]}"
  name="${NAMES[$i]}"
  port="$((BASE_PORT + i))"
  wait_for_gpu
  run_exp "$cfg" "$name" "$port"
done

log_msg "adapter IrregularActionFormer fix-controls queue complete"
