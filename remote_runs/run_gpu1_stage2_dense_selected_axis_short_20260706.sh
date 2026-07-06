#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/gpu1_stage2_dense_selected_axis_short}"
RUN_TAG="${RUN_TAG:-gpu1_stage2_dense_selected_axis_short_$(date +%Y%m%d_%H%M%S)}"
FAIL_CLOSED_JSON="$LOG_DIR/${RUN_TAG}_fail_closed_config.json"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_STAGE4_AFTER="${RUN_STAGE4_AFTER:-1}"
EXP_ID="${EXP_ID:-1}"
PORT_BASE="${PORT_BASE:-32340}"

RANDOM_FIXED_CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py"
UNIFORM_OFFICIAL_CFG="configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py"
CONFIGS=("$RANDOM_FIXED_CFG" "$UNIFORM_OFFICIAL_CFG")
LABELS=("near63_random_fixed_selected_axis_dense_control" "near65_uniform_even_spacing_official_dense")
SHORT_WORK_DIRS=(
  "exps/thumos/adatad/input_random_fixed_50pct_adapter_densehead_selected_axis_control_shortgate_n16r4"
  "exps/thumos/adatad/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_shortgate_n16r4"
)

log_msg() { echo "$(date '+%F %T') [gpu1-stage2-dense-short] $*"; }
python_works() { "$1" -c "import sys" >/dev/null 2>&1; }

select_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_works "$PYTHON_BIN" || { echo "Configured PYTHON_BIN is not executable: $PYTHON_BIN" >&2; exit 127; }
    return 0
  fi
  for candidate in "$BASE/conda_envs/opentad/bin/python" python python3 python.exe; do
    if command -v "$candidate" >/dev/null 2>&1 && python_works "$candidate"; then
      PYTHON_BIN="$candidate"
      return 0
    fi
  done
  echo "No Python interpreter found. Set PYTHON_BIN=/path/to/python." >&2
  exit 127
}

mkdir -p "$LOG_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HOME="${HOME:-$BASE/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$BASE/tmp/xdg_config}"
export HF_HOME="${HF_HOME:-$BASE/hf_cache}"
export THUMOS_ROOT="${THUMOS_ROOT:-$BASE/thumos14}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
select_python

log_msg "root=$ROOT"
log_msg "python_bin=$PYTHON_BIN"
log_msg "RUN_TRAIN=$RUN_TRAIN PRECHECK_ONLY=$PRECHECK_ONLY RUN_STAGE4_AFTER=$RUN_STAGE4_AFTER EXP_ID=$EXP_ID PORT_BASE=$PORT_BASE"
log_msg "fail-closed config scan"
"$PYTHON_BIN" tools/check_fail_closed_config.py "${CONFIGS[@]}" --json-out "$FAIL_CLOSED_JSON"
log_msg "fail_closed_config_json=$FAIL_CLOSED_JSON"

log_msg "config load and selected-axis dense contract preflight"
"$PYTHON_BIN" - "${CONFIGS[@]}" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config

expected = {
    "input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py": {
        "label": "near63_random_fixed_selected_axis_dense_control",
        "method": "random_fixed_subsample",
    },
    "input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py": {
        "label": "near65_uniform_even_spacing_official_dense",
        "method": "uniform_fixed_subsample",
    },
}

for path in sys.argv[1:]:
    cfg = Config.fromfile(path)
    name = Path(path).name
    rule = expected[name]
    head = cfg.model.rpn_head
    assert head.type == "ActionFormerHead", (name, head.type)
    assert cfg.model.projection.type == "DensePassthroughConv1DTransformerProj", cfg.model.projection.type
    assert cfg.model.neck.type == "DensePassthroughFPNIdentity", cfg.model.neck.type
    assert bool(cfg.post_processing.save_dict), name
    for split in ("train", "val", "test"):
        step = next(item for item in getattr(cfg.dataset, split).pipeline if item.get("type") == "LoadFrames")
        assert step.method == rule["method"], (name, split, step.method)
        assert abs(float(step.keep_ratio) - 0.5) < 1e-12, (name, split, step.keep_ratio)
        assert bool(step.remap_gt_to_selected_axis), (name, split, step.remap_gt_to_selected_axis)
        assert int(step.target_len) == 384, (name, split, step.target_len)
    print("config_ok", rule["label"], name, "work_dir", cfg.work_dir)
PY

log_msg "py_compile preflight"
"$PYTHON_BIN" -m py_compile "${CONFIGS[@]}" tools/train.py tools/check_fail_closed_config.py

if [[ "${RUN_PYTEST:-1}" == "1" ]]; then
  log_msg "pytest static preflight"
  "$PYTHON_BIN" -m pytest tests/test_fail_closed_static_gates.py -q -k "stage2_resource_boundary or remote_run_execution"
fi

if [[ "$RUN_TRAIN" != "1" ]]; then
  log_msg "preflight complete; set RUN_TRAIN=1 PRECHECK_ONLY=0 through the GPU1 launcher for a two-epoch smoke"
  exit 0
fi

if [[ "${ALLOW_GPU1_SHORT_VALIDATION:-0}" != "1" ]]; then
  echo "Refusing to train: short GPU1 validation requires ALLOW_GPU1_SHORT_VALIDATION=1." >&2
  exit 64
fi
if [[ "${SLURM_JOB_ID:-}" != "1118197" && "${ALLOW_NON_1118197_SHORT_VALIDATION:-0}" != "1" ]]; then
  echo "Refusing to train: GPU1 short validation must run inside Slurm allocation 1118197 unless explicitly overridden." >&2
  exit 64
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
log_msg "short validation device boundary SLURM_JOB_ID=${SLURM_JOB_ID:-unset} SLURM_STEP_ID=${SLURM_STEP_ID:-unset} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || nvidia-smi || true

for idx in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$idx]}"
  label="${LABELS[$idx]}"
  short_work_dir="${SHORT_WORK_DIRS[$idx]}"
  port=$((PORT_BASE + idx))
  train_log="$LOG_DIR/${RUN_TAG}_${label}_train.log"
  train_cmd=(
    torchrun
    --master_port="$port"
    --nproc_per_node=1
    tools/train.py
    "$cfg"
    --id "$EXP_ID"
    --cfg-options
    workflow.end_epoch=2
    workflow.val_start_epoch=1
    workflow.val_eval_interval=1
    workflow.val_loss_interval=-1
    workflow.checkpoint_interval=1
    workflow.logging_interval=10
    workflow.disable_checkpoint=True
    "work_dir=$short_work_dir"
  )

  log_msg "DRY_RUN_TRAIN_CMD label=$label ${train_cmd[*]}"
  if [[ "$PRECHECK_ONLY" == "1" ]]; then
    continue
  fi

  log_msg "START short validation label=$label cfg=$cfg work_dir=$short_work_dir port=$port exp_id=$EXP_ID"
  "${train_cmd[@]}" 2>&1 | tee "$train_log"
  status=${PIPESTATUS[0]}
  log_msg "END short validation label=$label status=$status train_log=$train_log"
  if [[ "$status" != "0" ]]; then
    exit "$status"
  fi
done

if [[ "$PRECHECK_ONLY" == "1" ]]; then
  log_msg "precheck_only: not executing training commands; set PRECHECK_ONLY=0 with RUN_TRAIN=1 to execute"
elif [[ "$RUN_STAGE4_AFTER" == "1" ]]; then
  log_msg "START post-short Stage-4 detection quality"
  (
    unset CUDA_VISIBLE_DEVICES
    RUN_TAG="${RUN_TAG}_stage4_quality" \
    REQUIRE_RESULTS=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    bash "$ROOT/remote_runs/run_stage4_detection_quality_stage2_dense_20260706.sh"
  )
  log_msg "END post-short Stage-4 detection quality"
fi
