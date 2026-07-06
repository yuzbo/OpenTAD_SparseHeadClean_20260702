#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
ROOT="$BASE/OpenTAD_SparseHeadClean_20260702"
LOG_DIR="$ROOT/logs/gpu1_uniform_fixed_dense_control_long"
RUN_TAG="${RUN_TAG:-gpu1_uniform_fixed_dense_control_long_$(date +%Y%m%d_%H%M%S)}"
CHAIN_LOG="$LOG_DIR/${RUN_TAG}.log"
TRAIN_LOG="$LOG_DIR/${RUN_TAG}_train.log"
CFG="configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py"
EXP_ID="${EXP_ID:-1}"
PORT="${PORT:-32327}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$CHAIN_LOG") 2>&1

log_msg() { echo "$(date '+%F %T') $*"; }

log_msg "uniform_fixed dense-control long-train start on $(hostname) cfg=$CFG exp_id=$EXP_ID"
log_msg "root=$ROOT"
log_msg "chain_log=$CHAIN_LOG"
log_msg "train_log=$TRAIN_LOG"

source "$BASE/conda_envs/opentad/bin/activate"
export HOME="$BASE/tmp/home"
export XDG_CACHE_HOME="$BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$BASE/tmp/xdg_config"
export HF_HOME="$BASE/hf_cache"
export THUMOS_ROOT="$BASE/thumos14"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

cd "$ROOT"

log_msg "SLURM_JOB_ID=${SLURM_JOB_ID:-unset} SLURM_STEP_ID=${SLURM_STEP_ID:-unset} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || nvidia-smi || true

log_msg "config load preflight"
python - "$CFG" "$EXP_ID" <<'PY'
import os
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
exp_id = int(sys.argv[2])
head = cfg.model.rpn_head
work_dir = os.path.join(cfg.work_dir, f"gpu1_id{exp_id}")
steps = [
    next(step for step in getattr(cfg.dataset, split).pipeline if step.get("type") == "LoadFrames")
    for split in ("train", "val", "test")
]
print("config_ok", sys.argv[1])
print("head", head.type)
print("methods", [step.method for step in steps])
print("remap_gt_to_selected_axis", [bool(step.remap_gt_to_selected_axis) for step in steps])
print("compare_to", "equal_interval_reference_approx_65", "random_fixed_dense_control_old_51.59")
print("work_dir", work_dir)
PY

log_msg "py_compile preflight"
python -m py_compile tools/train.py opentad/datasets/transforms/end_to_end.py

log_msg "START train port=$PORT exp_id=$EXP_ID"
torchrun --master_port="$PORT" --nproc_per_node=1 tools/train.py "$CFG" --id "$EXP_ID" 2>&1 | tee "$TRAIN_LOG"
status=${PIPESTATUS[0]}
log_msg "END train status=$status"
exit "$status"
