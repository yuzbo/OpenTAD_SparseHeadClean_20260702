#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --exclude=g0030
#SBATCH -J exact0410_dense
#SBATCH -o logs/slurm_0410_exact_dense/%x-%j.out
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${ROOT:-$BASE/OpenTAD_SparseHeadClean_20260702}"
CFG="${CFG:?Set CFG to a 0410 exact dense config}"
EXP_LABEL="${EXP_LABEL:-$(basename "$CFG" .py)}"
EXP_ID="${EXP_ID:-0}"
PORT="${PORT:-$((34000 + (${SLURM_JOB_ID:-0} % 10000)))}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/slurm_0410_exact_dense}"
RUN_TAG="${RUN_TAG:-exact0410_dense_${EXP_LABEL}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
ALLOW_OVERWRITE_EXACT0410_OUTPUT="${ALLOW_OVERWRITE_EXACT0410_OUTPUT:-0}"
RUN_STAGE4_AFTER="${RUN_STAGE4_AFTER:-0}"
CHAIN_LOG="$LOG_DIR/${RUN_TAG}.log"
TRAIN_LOG="$LOG_DIR/${RUN_TAG}_train.log"
FAIL_CLOSED_JSON="$LOG_DIR/${RUN_TAG}_fail_closed_config.json"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$CHAIN_LOG") 2>&1

log_msg() { echo "$(date '+%F %T') [exact0410-dense] $*"; }

prepare_output_dir() {
  local run_dir="$1"
  if [[ -e "$run_dir" ]]; then
    if [[ "$ALLOW_OVERWRITE_EXACT0410_OUTPUT" != "1" ]]; then
      echo "Refusing 0410 exact dense training: output directory already exists: $run_dir" >&2
      echo "Set ALLOW_OVERWRITE_EXACT0410_OUTPUT=1 only for an intentional rerun." >&2
      exit 66
    fi
    case "$run_dir" in
      "$ROOT"/exps/thumos/adatad/*/gpu1_id*) ;;
      *)
        echo "Refusing to remove unsafe 0410 exact output path: $run_dir" >&2
        exit 66
        ;;
    esac
    log_msg "removing existing 0410 exact output directory run_dir=$run_dir"
    rm -rf "$run_dir"
  fi
  mkdir -p "$(dirname "$run_dir")"
}

if [[ "${SLURM_JOB_ID:-}" == "1118197" ]]; then
  echo "Refusing 0410 exact dense training inside allocation 1118197; submit a separate Slurm job." >&2
  exit 65
fi
if [[ "$(hostname -s)" == "g0030" && "${ALLOW_G0030_LONG:-0}" != "1" ]]; then
  echo "Refusing 0410 exact dense training on g0030; this node hosts the current two-card allocation." >&2
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

log_msg "config load and 0410 exact dense contract preflight"
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
    "input_random_fixed_50pct_0410_exact_n16r4.py": {
        "train_method": "random_fixed_subsample",
        "eval_method": "random_fixed_subsample",
        "train_stride": 1,
        "eval_stride": 1,
        "train_window": None,
        "eval_window": 768,
        "target_len": 384,
        "source_len": 768,
        "remap": True,
        "historical_avg_map": 63.12,
    },
    "input_stride2_uniform_0410_exact_n16r4.py": {
        "train_method": "random_trunc",
        "eval_method": "sliding_window",
        "train_stride": 2,
        "eval_stride": 2,
        "train_window": None,
        "eval_window": 384,
        "target_len": None,
        "source_len": None,
        "remap": None,
        "historical_avg_map": 65.09,
    },
}
assert name in expected, name
spec = expected[name]
assert cfg.model.type == "ActionFormer", cfg.model.type
assert cfg.model.projection.type == "Conv1DTransformerProj", cfg.model.projection.type
assert cfg.model.neck.type == "FPNIdentity", cfg.model.neck.type
assert cfg.model.rpn_head.type == "ActionFormerHead", cfg.model.rpn_head.type
assert cfg.model.rpn_head.prior_generator.type == "PointGenerator", cfg.model.rpn_head.prior_generator.type
assert int(cfg.solver.train.batch_size) == 2, cfg.solver.train
assert int(cfg.solver.val.batch_size) == 2, cfg.solver.val
assert int(cfg.workflow.val_start_epoch) == 39, cfg.workflow
assert int(cfg.workflow.val_eval_interval) == 5, cfg.workflow
assert int(cfg.workflow.end_epoch) == 60, cfg.workflow
assert float(cfg.post_processing.nms.sigma) == 0.5, cfg.post_processing.nms
assert float(cfg.post_processing.nms.min_score) == 0.001, cfg.post_processing.nms
assert bool(cfg.post_processing.save_dict), cfg.post_processing
for split in ("train", "val", "test"):
    ds = getattr(cfg.dataset, split)
    step = next(item for item in ds.pipeline if item.get("type") == "LoadFrames")
    is_train = split == "train"
    assert int(ds.sample_stride) == (spec["train_stride"] if is_train else spec["eval_stride"]), (split, ds.sample_stride)
    expected_window = spec["train_window"] if is_train else spec["eval_window"]
    assert ds.get("window_size", None) == expected_window, (split, ds.get("window_size", None), expected_window)
    assert step.method == (spec["train_method"] if is_train else spec["eval_method"]), (split, step.method)
    if spec["target_len"] is not None:
        assert int(step.target_len) == spec["target_len"], (split, step.target_len)
    if is_train and spec["source_len"] is not None:
        assert int(step.source_len) == spec["source_len"], step
    if spec["remap"] is True:
        assert bool(step.remap_gt_to_selected_axis), (split, step.remap_gt_to_selected_axis)
    elif "remap_gt_to_selected_axis" in step:
        raise AssertionError((split, "unexpected remap_gt_to_selected_axis", step.remap_gt_to_selected_axis))
work_dir = os.path.join(cfg.work_dir, f"gpu1_id{exp_id}")
print("config_ok", name)
print("work_dir", work_dir)
print("historical_avg_map_reference", spec["historical_avg_map"])
print("contract", cfg.model.type, cfg.model.projection.type, cfg.model.neck.type, cfg.model.rpn_head.type)
PY

log_msg "py_compile preflight"
python -m py_compile "$CFG" tools/train.py tools/check_fail_closed_config.py

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

if [[ "$RUN_STAGE4_AFTER" == "1" ]]; then
  log_msg "START optional post-training detection quality"
  (
    unset CUDA_VISIBLE_DEVICES
    RUN_TAG="${RUN_TAG}_stage4_quality" \
    REQUIRE_RESULTS=1 \
    PYTHON_BIN=python \
    bash "$ROOT/remote_runs/run_stage4_detection_quality_stage2_dense_20260706.sh"
  )
  log_msg "END optional post-training detection quality"
fi
