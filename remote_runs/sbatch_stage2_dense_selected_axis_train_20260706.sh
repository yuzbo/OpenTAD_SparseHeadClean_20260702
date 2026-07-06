#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --exclude=g0030
#SBATCH -J stage2_dense_axis
#SBATCH -o logs/slurm_stage2_dense_selected_axis_long/%x-%j.out
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${ROOT:-$BASE/OpenTAD_SparseHeadClean_20260702}"
CFG="${CFG:?Set CFG to a Stage-2 dense selected-axis config}"
EXP_LABEL="${EXP_LABEL:-$(basename "$CFG" .py)}"
EXP_ID="${EXP_ID:-0}"
PORT="${PORT:-$((32000 + (${SLURM_JOB_ID:-0} % 10000)))}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/slurm_stage2_dense_selected_axis_long}"
RUN_TAG="${RUN_TAG:-stage2_dense_axis_${EXP_LABEL}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
CHAIN_LOG="$LOG_DIR/${RUN_TAG}.log"
TRAIN_LOG="$LOG_DIR/${RUN_TAG}_train.log"
FAIL_CLOSED_JSON="$LOG_DIR/${RUN_TAG}_fail_closed_config.json"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$CHAIN_LOG") 2>&1

log_msg() { echo "$(date '+%F %T') [slurm-stage2-dense] $*"; }

if [[ "${SLURM_JOB_ID:-}" == "1118197" ]]; then
  echo "Refusing long training inside allocation 1118197; submit a separate Slurm job." >&2
  exit 65
fi
if [[ "$(hostname -s)" == "g0030" && "${ALLOW_G0030_LONG:-0}" != "1" ]]; then
  echo "Refusing long training on g0030; this node hosts the current two-card long allocation." >&2
  exit 65
fi

source "$BASE/conda_envs/opentad/bin/activate"
export HOME="${HOME:-$BASE/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$BASE/tmp/xdg_config}"
export HF_HOME="${HF_HOME:-$BASE/hf_cache}"
export THUMOS_ROOT="${THUMOS_ROOT:-$BASE/thumos14}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

cd "$ROOT"

log_msg "start host=$(hostname) slurm_job=${SLURM_JOB_ID:-unset} cfg=$CFG label=$EXP_LABEL exp_id=$EXP_ID"
log_msg "chain_log=$CHAIN_LOG"
log_msg "train_log=$TRAIN_LOG"
log_msg "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || nvidia-smi || true

log_msg "fail-closed config scan"
python tools/check_fail_closed_config.py "$CFG" --json-out "$FAIL_CLOSED_JSON"
log_msg "fail_closed_config_json=$FAIL_CLOSED_JSON"

log_msg "config load and selected-axis dense contract preflight"
python - "$CFG" "$EXP_ID" <<'PY'
import os
import sys
from pathlib import Path

from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
exp_id = int(sys.argv[2])
name = Path(sys.argv[1]).name
expected_methods = {
    "input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py": "random_fixed_subsample",
    "input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py": "uniform_fixed_subsample",
}
assert name in expected_methods, name
assert cfg.model.rpn_head.type == "ActionFormerHead", cfg.model.rpn_head.type
assert cfg.model.projection.type == "DensePassthroughConv1DTransformerProj", cfg.model.projection.type
assert cfg.model.neck.type == "DensePassthroughFPNIdentity", cfg.model.neck.type
assert bool(cfg.post_processing.save_dict), name
for split in ("train", "val", "test"):
    step = next(item for item in getattr(cfg.dataset, split).pipeline if item.get("type") == "LoadFrames")
    assert step.method == expected_methods[name], (split, step.method)
    assert bool(step.remap_gt_to_selected_axis), (split, step.remap_gt_to_selected_axis)
    assert int(step.target_len) == 384, (split, step.target_len)
work_dir = os.path.join(cfg.work_dir, f"gpu1_id{exp_id}")
print("config_ok", name)
print("work_dir", work_dir)
print("compare_to", "near63_random_fixed_selected_axis_dense_control", "near65_uniform_even_spacing_official_dense")
PY

log_msg "py_compile preflight"
python -m py_compile "$CFG" tools/train.py tools/check_fail_closed_config.py

log_msg "START train port=$PORT exp_id=$EXP_ID"
torchrun --master_port="$PORT" --nproc_per_node=1 tools/train.py "$CFG" --id "$EXP_ID" 2>&1 | tee "$TRAIN_LOG"
status=${PIPESTATUS[0]}
log_msg "END train status=$status"
exit "$status"
