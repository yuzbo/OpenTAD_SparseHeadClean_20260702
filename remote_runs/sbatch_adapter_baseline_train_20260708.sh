#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --exclude=g0030
#SBATCH -J adapter_baseline
#SBATCH -o logs/slurm_adapter_baseline/%x-%j.out
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${ROOT:-$BASE/OpenTAD_SparseHeadClean_20260702}"
CFG="${CFG:?Set CFG to the adapter baseline reproduction config}"
EXP_LABEL="${EXP_LABEL:-$(basename "$CFG" .py)}"
EXP_ID="${EXP_ID:-0}"
PORT="${PORT:-$((35000 + (${SLURM_JOB_ID:-0} % 10000)))}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/slurm_adapter_baseline}"
RUN_TAG="${RUN_TAG:-adapter_baseline_${EXP_LABEL}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
ALLOW_OVERWRITE_ADAPTER_BASELINE_OUTPUT="${ALLOW_OVERWRITE_ADAPTER_BASELINE_OUTPUT:-0}"
CHAIN_LOG="$LOG_DIR/${RUN_TAG}.log"
TRAIN_LOG="$LOG_DIR/${RUN_TAG}_train.log"
FAIL_CLOSED_JSON="$LOG_DIR/${RUN_TAG}_fail_closed_config.json"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$CHAIN_LOG") 2>&1

log_msg() { echo "$(date '+%F %T') [adapter-baseline] $*"; }

prepare_output_dir() {
  local run_dir="$1"
  if [[ -e "$run_dir" ]]; then
    if [[ "$ALLOW_OVERWRITE_ADAPTER_BASELINE_OUTPUT" != "1" ]]; then
      echo "Refusing adapter baseline training: output directory already exists: $run_dir" >&2
      echo "Set ALLOW_OVERWRITE_ADAPTER_BASELINE_OUTPUT=1 only for an intentional rerun." >&2
      exit 66
    fi
    case "$run_dir" in
      "$ROOT"/exps/thumos/adatad/*/gpu1_id*) ;;
      *)
        echo "Refusing to remove unsafe adapter baseline output path: $run_dir" >&2
        exit 66
        ;;
    esac
    log_msg "removing existing adapter baseline output directory run_dir=$run_dir"
    rm -rf "$run_dir"
  fi
  mkdir -p "$(dirname "$run_dir")"
}

if [[ "${SLURM_JOB_ID:-}" == "1118197" ]]; then
  echo "Refusing adapter baseline training inside allocation 1118197; submit a separate Slurm job." >&2
  exit 65
fi
if [[ "$(hostname -s)" == "g0030" && "${ALLOW_G0030_LONG:-0}" != "1" ]]; then
  echo "Refusing adapter baseline training on g0030; this node hosts the current two-card allocation." >&2
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

log_msg "config load and adapter-baseline contract preflight"
python - "$CFG" "$EXP_ID" <<'PY'
import os
import sys
from pathlib import Path

from mmengine.config import Config

cfg_path = sys.argv[1]
exp_id = int(sys.argv[2])
cfg = Config.fromfile(cfg_path)
name = Path(cfg_path).name
assert name == "input_random_fixed_50pct_adapter_n16r4_retrain_20260604.py", name

assert cfg.model.type == "ActionFormer", cfg.model.type
assert cfg.model.backbone.type == "mmaction.Recognizer3D", cfg.model.backbone.type
assert cfg.model.backbone.backbone.type == "VisionTransformerAdapter", cfg.model.backbone.backbone.type
assert bool(cfg.model.backbone.custom.freeze_backbone) is False, cfg.model.backbone.custom
assert bool(cfg.model.backbone.custom.norm_eval) is False, cfg.model.backbone.custom
assert cfg.model.projection.type == "Conv1DTransformerProj", cfg.model.projection.type
assert cfg.model.neck.type == "FPNIdentity", cfg.model.neck.type
assert cfg.model.rpn_head.type == "ActionFormerHead", cfg.model.rpn_head.type
assert cfg.model.rpn_head.prior_generator.type == "PointGenerator", cfg.model.rpn_head.prior_generator.type
assert int(cfg.solver.train.batch_size) == 2, cfg.solver.train
assert int(cfg.solver.val.batch_size) == 2, cfg.solver.val
assert int(cfg.solver.test.batch_size) == 2, cfg.solver.test
assert float(cfg.post_processing.nms.sigma) == 0.7, cfg.post_processing.nms
assert bool(cfg.workflow.disable_checkpoint) is False, cfg.workflow
assert int(cfg.workflow.end_epoch) == 60, cfg.workflow
assert int(cfg.workflow.val_start_epoch) == 40, cfg.workflow
assert int(cfg.workflow.val_eval_interval) == 2, cfg.workflow

for split in ("train", "val", "test"):
    ds = getattr(cfg.dataset, split)
    step = next(item for item in ds.pipeline if item.get("type") == "LoadFrames")
    assert "/data/run01/sczc063/yuzibo/thumos14" in ds.ann_file, (split, ds.ann_file)
    assert "/root/autodl-tmp" not in ds.ann_file, (split, ds.ann_file)
    assert int(ds.sample_stride) == 1, (split, ds.sample_stride)
    assert step.method == "random_fixed_subsample", (split, step.method)
    assert float(step.keep_ratio) == 0.5, (split, step)
    assert int(step.target_len) == 384, (split, step)
    if split == "train":
        assert step.method_base == "random_trunc", step
        assert int(step.source_len) == 768, step
    else:
        assert step.method_base == "sliding_window", step
    assert "remap_gt_to_selected_axis" not in step, (split, step)
    assert "allow_drop_selected_axis_gt" not in step, (split, step)

work_dir = os.path.join(cfg.work_dir, f"gpu1_id{exp_id}")
print("config_ok", name)
print("work_dir", work_dir)
print("historical_avg_map_reference", 63.77)
print("contract", cfg.model.type, cfg.model.backbone.backbone.type, cfg.model.rpn_head.type)
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
exit "$status"
