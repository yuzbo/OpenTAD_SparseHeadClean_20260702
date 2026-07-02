#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/root/autodl-tmp/OpenTAD_Back_check}"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

QUEUE_LOG="$LOG_DIR/adapter_native_dense_headv2_safe_driver.log"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"
BASE_PORT="${BASE_PORT:-30654}"
GPU_FREE_MIB="${GPU_FREE_MIB:-900}"
CHECK_ONLY="${CHECK_ONLY:-0}"
EXPECT_BATCH_SIZE="${EXPECT_BATCH_SIZE:-2}"
APPROVAL_FILE="${APPROVAL_FILE:-$ROOT_DIR/gate_approvals/adapter_native_dense_headv2_after_review.ok}"

CONFIG="configs/adatad/thumos/input_random_fixed_50pct_adapter_native_dense_headv2_safe.py"
NAME="input_random_fixed_50pct_adapter_native_dense_headv2_safe"

log_msg() {
  echo "$(date '+%F %T') $*" | tee -a "$QUEUE_LOG"
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
  log_msg "checking ${CONFIG}"
  cd "$ROOT_DIR"
  "$PYTHON_BIN" - <<PY
import sys
sys.path.insert(0, "$ROOT_DIR")
from mmengine.config import Config

cfg = Config.fromfile("$CONFIG")
assert cfg.model.type == "IrregularActionFormer", cfg.model.type
assert cfg.model.backbone.backbone.type == "VisionTransformerAdapter", cfg.model.backbone.backbone
assert not bool(cfg.model.backbone.backbone.use_irregular_time_embed), cfg.model.backbone.backbone
assert not bool(cfg.model.backbone.backbone.add_irregular_time_embed), cfg.model.backbone.backbone
assert cfg.model.projection.type == "DensePassthroughConv1DTransformerProj", cfg.model.projection
assert cfg.model.neck.type == "DensePassthroughFPNIdentity", cfg.model.neck
assert cfg.model.rpn_head.type == "IrregularActionFormerHeadV2", cfg.model.rpn_head
assert cfg.model.rpn_head.prior_generator.type == "IrregularPointGeneratorV2", cfg.model.rpn_head.prior_generator
assert abs(float(cfg.model.projection.get("input_pdrop", 0.0))) < 1e-9, cfg.model.projection
assert int(cfg.model.backbone.backbone.total_frames) == 384, cfg.model.backbone.backbone
assert int(cfg.model.projection.max_seq_len) == 384, cfg.model.projection

train = cfg.dataset.train
val = cfg.dataset.val
test = cfg.dataset.test
assert int(train.sample_stride) == 1, train
assert int(val.sample_stride) == 1, val
assert int(test.sample_stride) == 1, test
assert int(val.window_size) == 768, val
assert int(test.window_size) == 768, test
train_load = next(x for x in train.pipeline if x.type == "LoadFrames")
val_load = next(x for x in val.pipeline if x.type == "LoadFrames")
test_load = next(x for x in test.pipeline if x.type == "LoadFrames")
assert train_load.method == "random_fixed_subsample", train_load
assert abs(float(train_load.keep_ratio) - 0.5) < 1e-9, train_load
assert train_load.method_base == "random_trunc", train_load
assert not bool(train_load.remap_gt_to_selected_axis), train_load
assert val_load.method == "random_fixed_subsample", val_load
assert abs(float(val_load.keep_ratio) - 0.5) < 1e-9, val_load
assert val_load.method_base == "sliding_window", val_load
assert not bool(val_load.remap_gt_to_selected_axis), val_load
assert test_load.method == "random_fixed_subsample", test_load
assert abs(float(test_load.keep_ratio) - 0.5) < 1e-9, test_load
assert test_load.method_base == "sliding_window", test_load
assert not bool(test_load.remap_gt_to_selected_axis), test_load

workflow = cfg.workflow
solver = cfg.solver
assert int(workflow.checkpoint_interval) == 10, workflow
assert not bool(workflow.get("disable_checkpoint", False)), workflow
assert int(workflow.val_start_epoch) == 40, workflow
assert int(workflow.val_eval_interval) == 2, workflow
assert int(workflow.end_epoch) == 60, workflow
expected_batch_size = int("$EXPECT_BATCH_SIZE")
assert int(solver.train.batch_size) == expected_batch_size, solver
assert int(solver.val.batch_size) == expected_batch_size, solver
assert int(solver.test.batch_size) == expected_batch_size, solver

print("work_dir=", cfg.work_dir)
print("model=", cfg.model.type)
print("backbone=", cfg.model.backbone.backbone.type)
print("projection=", cfg.model.projection.type)
print("neck=", cfg.model.neck.type)
print("head=", cfg.model.rpn_head.type)
print("prior=", cfg.model.rpn_head.prior_generator.type)
print("train_load=", train_load.method, train_load.keep_ratio, train_load.method_base, train_load.remap_gt_to_selected_axis)
print("val_load=", val_load.method, val_load.keep_ratio, val_load.method_base, val_load.remap_gt_to_selected_axis)
print("checkpoint_interval=", workflow.checkpoint_interval)
print("disable_checkpoint=", workflow.get("disable_checkpoint", False))
print("batch_size=", solver.train.batch_size, solver.val.batch_size, solver.test.batch_size)
PY
}

run_exp() {
  local log_file="$LOG_DIR/${NAME}_$(date '+%Y%m%d_%H%M%S').log"
  log_msg "starting ${NAME}"
  cd "$ROOT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$TORCHRUN" --master_port="$BASE_PORT" --nproc_per_node=1 \
    tools/train.py "$CONFIG" --id 0 2>&1 | tee "$log_file"
  log_msg "finished ${NAME}"
}

require_approval() {
  if [[ -f "$APPROVAL_FILE" ]]; then
    log_msg "approval file found: ${APPROVAL_FILE}"
    return
  fi
  log_msg "missing approval file: ${APPROVAL_FILE}"
  log_msg "native dense headv2 is gated behind code review and a written launch decision; run CHECK_ONLY=1 for validation only"
  exit 3
}

log_msg "adapter native dense headv2 safe queue start"
df -h /root/autodl-tmp | tee -a "$QUEUE_LOG"
check_cfg

if [[ "$CHECK_ONLY" == "1" ]]; then
  log_msg "adapter native dense headv2 safe check-only complete"
  exit 0
fi

require_approval
wait_for_gpu
run_exp
log_msg "adapter native dense headv2 safe queue complete"
