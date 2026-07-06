#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
ROOT="$BASE/OpenTAD_SparseHeadClean_20260702"
LOG_DIR="$ROOT/logs/gpu1_bridge_absrange_long"
RUN_TAG="${RUN_TAG:-gpu1_bridge_absrange_long_$(date +%Y%m%d_%H%M%S)}"
CHAIN_LOG="$LOG_DIR/${RUN_TAG}.log"
TRAIN_LOG="$LOG_DIR/${RUN_TAG}_train.log"
CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py"
EXP_ID="${EXP_ID:-1}"
PORT="${PORT:-32325}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$CHAIN_LOG") 2>&1

log_msg() { echo "$(date '+%F %T') $*"; }

log_msg "absrange long-train start on $(hostname) cfg=$CFG exp_id=$EXP_ID"
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
prior = head.prior_generator
work_dir = os.path.join(cfg.work_dir, f"gpu1_id{exp_id}")
print("config_ok", sys.argv[1])
print("head", head.type, "assignment", head.assignment_mode, "regression", head.regression_mode)
print("range_mode", prior.range_mode, "regression_range", prior.regression_range)
print("compare_to", "openrange_final_42.44", "HeadV3_fixed_40.20")
print("work_dir", work_dir)
PY

log_msg "py_compile preflight"
python -m py_compile tools/train.py opentad/models/dense_heads/irregular_actionformer_bridge_head.py

log_msg "START train port=$PORT exp_id=$EXP_ID"
torchrun --master_port="$PORT" --nproc_per_node=1 tools/train.py "$CFG" --id "$EXP_ID" 2>&1 | tee "$TRAIN_LOG"
status=${PIPESTATUS[0]}
log_msg "END train status=$status"
exit "$status"
