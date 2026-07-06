#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py"
EXP_ID="${EXP_ID:-1}"
PORT="${PORT:-32336}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_PYTEST="${RUN_PYTEST:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/bridge_absrange_expanded_shortgate_failclosed}"
RUN_TAG="${RUN_TAG:-shortgate_failclosed_$(date +%Y%m%d_%H%M%S)}"
FAIL_CLOSED_JSON="$LOG_DIR/${RUN_TAG}_fail_closed_config.json"

log_msg() { echo "$(date '+%F %T') $*"; }
_python_works() { "$1" -c "import sys" >/dev/null 2>&1; }

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$LOG_DIR"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1 && _python_works python; then
    PYTHON_BIN=python
  elif command -v python.exe >/dev/null 2>&1 && _python_works python.exe; then
    PYTHON_BIN=python.exe
  elif command -v python3 >/dev/null 2>&1 && _python_works python3; then
    PYTHON_BIN=python3
  else
    echo "No python interpreter found. Set PYTHON_BIN=/path/to/python." >&2
    exit 127
  fi
elif ! _python_works "$PYTHON_BIN"; then
  echo "Configured PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 127
fi

log_msg "shortgate fail-closed helper root=$ROOT cfg=$CFG exp_id=$EXP_ID"
log_msg "PRECHECK_ONLY=$PRECHECK_ONLY RUN_TRAIN=$RUN_TRAIN RUN_PYTEST=$RUN_PYTEST PYTHON_BIN=$PYTHON_BIN"

log_msg "fail-closed config scan"
"$PYTHON_BIN" tools/check_fail_closed_config.py "$CFG" --json-out "$FAIL_CLOSED_JSON"
log_msg "fail_closed_config_json=$FAIL_CLOSED_JSON"

log_msg "config load preflight"
"$PYTHON_BIN" - "$CFG" <<'PY'
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
head = cfg.model.rpn_head
prior = head.prior_generator
steps = [
    next(step for step in getattr(cfg.dataset, split).pipeline if step.get("type") == "LoadFrames")
    for split in ("train", "val", "test")
]

assert cfg.model.projection.type == "GridAwareConv1DTransformerProj"
assert cfg.model.neck.type == "GridAwareFPNIdentity"
assert head.type == "IrregularActionFormerBridgeHead"
assert head.assignment_mode == "hard"
assert head.regression_mode == "symmetric_linear"
assert head.center_radius_scale == "point_radius"
assert head.reg_denom_mode == "left_right_mean"
assert prior.range_mode == "absolute"
assert prior.decode_scale_mode == "level_stride"
assert prior.radius_scale_mode == "level_stride"
assert [tuple(item) for item in prior.regression_range] == [
    (0, 8),
    (2, 16),
    (4, 32),
    (8, 64),
    (16, 128),
    (32, 10000),
]
assert cfg.scheduler.max_epoch == 2
assert cfg.workflow.end_epoch == 2
assert cfg.workflow.val_start_epoch == 1
assert cfg.workflow.val_eval_interval == 1
assert cfg.workflow.disable_checkpoint
assert cfg.post_processing.save_dict
assert "shortgate" in cfg.work_dir
assert all(step.method == "random_fixed_subsample" for step in steps)
assert not any(bool(step.remap_gt_to_selected_axis) for step in steps)

print("config_ok", sys.argv[1])
print("work_dir", cfg.work_dir)
print("head", head.type, head.assignment_mode, head.regression_mode)
print("range_mode", prior.range_mode, "regression_range", prior.regression_range)
print("workflow", dict(cfg.workflow))
PY

log_msg "py_compile preflight"
"$PYTHON_BIN" -m py_compile "$CFG" tools/train.py

if [[ "$RUN_PYTEST" == "1" ]]; then
  log_msg "pytest preflight"
  "$PYTHON_BIN" -m pytest tests/test_adapter_native_dense_headv2_contracts.py -q -k "shortgate"
fi

if [[ "$RUN_TRAIN" != "1" ]]; then
  log_msg "fail_closed: set RUN_TRAIN=1 to print a training command; set PRECHECK_ONLY=0 as well to execute it"
  exit 0
fi

TRAIN_CMD=(
  torchrun
  --master_port="$PORT"
  --nproc_per_node=1
  tools/train.py
  "$CFG"
  --id
  "$EXP_ID"
)

log_msg "DRY_RUN_TRAIN_CMD ${TRAIN_CMD[*]}"

if [[ "$PRECHECK_ONLY" == "1" ]]; then
  log_msg "precheck_only: not executing training command; set PRECHECK_ONLY=0 with RUN_TRAIN=1 to execute"
  exit 0
fi

log_msg "execute_train_cmd port=$PORT exp_id=$EXP_ID"
exec "${TRAIN_CMD[@]}"
