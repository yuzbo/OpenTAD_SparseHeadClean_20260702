#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --exclude=g0030
#SBATCH -J protocol_bridge
#SBATCH -o logs/slurm_protocol_bridge_20260708/%x-%j.out
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${ROOT:-$BASE/OpenTAD_SparseHeadClean_20260702}"
CFG="${CFG:?Set CFG to a protocol bridge config}"
EXP_LABEL="${EXP_LABEL:-$(basename "$CFG" .py)}"
EXP_ID="${EXP_ID:-0}"
PORT="${PORT:-$((34000 + (${SLURM_JOB_ID:-0} % 10000)))}"
RUN_ANALYSIS_AFTER="${RUN_ANALYSIS_AFTER:-1}"
ALLOW_OVERWRITE_PROTOCOL_BRIDGE_OUTPUT="${ALLOW_OVERWRITE_PROTOCOL_BRIDGE_OUTPUT:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/slurm_protocol_bridge_20260708}"
RUN_TAG="${RUN_TAG:-protocol_bridge_${EXP_LABEL}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
CHAIN_LOG="$LOG_DIR/${RUN_TAG}.log"
TRAIN_LOG="$LOG_DIR/${RUN_TAG}_train.log"
FAIL_CLOSED_JSON="$LOG_DIR/${RUN_TAG}_fail_closed_config.json"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$CHAIN_LOG") 2>&1

log_msg() { echo "$(date '+%F %T') [protocol-bridge] $*"; }

prepare_output_dir() {
  local run_dir="$1"
  if [[ -e "$run_dir" ]]; then
    if [[ "$ALLOW_OVERWRITE_PROTOCOL_BRIDGE_OUTPUT" != "1" ]]; then
      echo "Refusing protocol bridge training: output directory already exists: $run_dir" >&2
      echo "Set ALLOW_OVERWRITE_PROTOCOL_BRIDGE_OUTPUT=1 only for an intentional rerun." >&2
      exit 66
    fi
    case "$run_dir" in
      "$ROOT"/exps/thumos/adatad/*/gpu1_id*) ;;
      *)
        echo "Refusing to remove unsafe protocol bridge output path: $run_dir" >&2
        exit 66
        ;;
    esac
    log_msg "removing existing protocol bridge output directory run_dir=$run_dir"
    rm -rf "$run_dir"
  fi
  mkdir -p "$(dirname "$run_dir")"
}

if [[ "${SLURM_JOB_ID:-}" == "1118197" ]]; then
  echo "Refusing training inside allocation 1118197; submit a separate Slurm job." >&2
  exit 65
fi
if [[ "$(hostname -s)" == "g0030" && "${ALLOW_G0030_LONG:-0}" != "1" ]]; then
  echo "Refusing protocol bridge training on g0030; this node hosts the current two-card allocation." >&2
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

log_msg "config load and protocol bridge contract preflight"
python - "$CFG" "$EXP_ID" <<'PY'
import os
import sys
from pathlib import Path

from mmengine.config import Config

cfg_path = sys.argv[1]
exp_id = int(sys.argv[2])
cfg = Config.fromfile(cfg_path)
name = Path(cfg_path).name
expected = {
    "input_random_fixed_50pct_0410_exact_irregular_densepass_n16r4.py": {
        "model": "IrregularActionFormer",
        "projection": "DensePassthroughConv1DTransformerProj",
        "neck": "DensePassthroughFPNIdentity",
        "method": "random_fixed_subsample",
        "batch_size": 2,
        "sigma": 0.5,
        "legacy_drop": True,
    },
    "input_random_fixed_50pct_adapter_actionformer_selected_axis_bs8_n16r4.py": {
        "model": "ActionFormer",
        "projection": "Conv1DTransformerProj",
        "neck": "FPNIdentity",
        "method": "random_fixed_subsample",
        "batch_size": 8,
        "sigma": 0.7,
        "legacy_drop": False,
    },
    "input_uniform_fixed_50pct_adapter_actionformer_selected_axis_bs8_n16r4.py": {
        "model": "ActionFormer",
        "projection": "Conv1DTransformerProj",
        "neck": "FPNIdentity",
        "method": "uniform_fixed_subsample",
        "batch_size": 8,
        "sigma": 0.7,
        "legacy_drop": False,
    },
}
assert name in expected, name
spec = expected[name]
assert cfg.model.type == spec["model"], cfg.model.type
assert cfg.model.projection.type == spec["projection"], cfg.model.projection.type
assert cfg.model.neck.type == spec["neck"], cfg.model.neck.type
assert cfg.model.rpn_head.type == "ActionFormerHead", cfg.model.rpn_head.type
assert int(cfg.solver.train.batch_size) == spec["batch_size"], cfg.solver.train.batch_size
assert float(cfg.post_processing.nms.sigma) == spec["sigma"], cfg.post_processing.nms
assert bool(cfg.post_processing.save_dict), name
for split in ("train", "val", "test"):
    step = next(item for item in getattr(cfg.dataset, split).pipeline if item.get("type") == "LoadFrames")
    assert step.method == spec["method"], (split, step.method)
    assert bool(step.remap_gt_to_selected_axis), (split, step.remap_gt_to_selected_axis)
    assert int(step.target_len) == 384, (split, step.target_len)
    if spec["legacy_drop"]:
        assert bool(step.allow_drop_selected_axis_gt), (split, step)
        assert bool(step.legacy_selected_axis_gt_drop_diagnostic), (split, step)
work_dir = os.path.join(cfg.work_dir, f"gpu1_id{exp_id}")
print("config_ok", name)
print("work_dir", work_dir)
print("bridge", spec)
PY

log_msg "py_compile preflight"
python -m py_compile "$CFG" tools/train.py tools/check_fail_closed_config.py tools/analyze_detection_quality.py

RUN_DIR="$(python - "$CFG" "$EXP_ID" <<'PY'
import os
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
exp_id = int(sys.argv[2])
print(os.path.join(cfg.work_dir, f"gpu1_id{exp_id}"))
PY
)"
prepare_output_dir "$ROOT/$RUN_DIR"

log_msg "START train port=$PORT exp_id=$EXP_ID"
torchrun --master_port="$PORT" --nproc_per_node=1 tools/train.py "$CFG" --id "$EXP_ID" 2>&1 | tee "$TRAIN_LOG"
status=${PIPESTATUS[0]}
log_msg "END train status=$status"
if [[ "$status" != "0" ]]; then
  exit "$status"
fi

if [[ "$RUN_ANALYSIS_AFTER" == "1" ]]; then
  log_msg "START post-training detection quality"
  QUALITY_JSON="$ROOT/$RUN_DIR/detection_quality_summary_${RUN_TAG}.json"
  QUALITY_CSV="$ROOT/$RUN_DIR/detection_quality_rows_${RUN_TAG}.csv"
  python tools/analyze_detection_quality.py \
    --config "$CFG" \
    --dataset-split val \
    --experiment-dir "$ROOT/$RUN_DIR" \
    --ground-truth-fallback "$THUMOS_ROOT/annotations/thumos_14_anno.json" \
    --subset validation \
    --output-json "$QUALITY_JSON" \
    --output-csv "$QUALITY_CSV"
  log_msg "quality_json=$QUALITY_JSON"
  log_msg "quality_csv=$QUALITY_CSV"
  log_msg "END post-training detection quality"
fi
