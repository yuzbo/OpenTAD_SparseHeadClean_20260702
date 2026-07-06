#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/stage3_bridge_selected_native_short_audit}"
RUN_TAG="${RUN_TAG:-stage3_bridge_selected_native_short_audit_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$LOG_DIR/$RUN_TAG}"
FAIL_CLOSED_JSON="$OUT_DIR/fail_closed_config.json"
SELECTED_AUDIT_OUT_DIR="$OUT_DIR/selected_axis"
NATIVE_AUDIT_OUT_DIR="$OUT_DIR/native_axis"
RUN_ASSIGNMENT_AUDIT="${RUN_ASSIGNMENT_AUDIT:-0}"
RUN_PYTEST="${RUN_PYTEST:-1}"

SELECTED_BRIDGE_CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py"
NATIVE_BRIDGE_CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py"
CONFIGS=("$SELECTED_BRIDGE_CFG" "$NATIVE_BRIDGE_CFG")

log_msg() { echo "$(date '+%F %T') [stage3-bridge-short-audit] $*"; }
_python_works() { "$1" -c "import sys" >/dev/null 2>&1; }

select_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    if _python_works "$PYTHON_BIN"; then
      return 0
    fi
    echo "Configured PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 127
  fi

  for candidate in python python.exe python3; do
    if command -v "$candidate" >/dev/null 2>&1 && _python_works "$candidate"; then
      PYTHON_BIN="$candidate"
      return 0
    fi
  done

  echo "No Python interpreter found. Set PYTHON_BIN=/path/to/python." >&2
  exit 127
}

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR"
select_python

log_msg "root=$ROOT"
log_msg "python_bin=$PYTHON_BIN"
log_msg "selected_bridge_cfg=$SELECTED_BRIDGE_CFG"
log_msg "native_bridge_cfg=$NATIVE_BRIDGE_CFG"
log_msg "RUN_ASSIGNMENT_AUDIT=$RUN_ASSIGNMENT_AUDIT RUN_PYTEST=$RUN_PYTEST OUT_DIR=$OUT_DIR"
log_msg "fail-closed config scan"
"$PYTHON_BIN" tools/check_fail_closed_config.py "${CONFIGS[@]}" --json-out "$FAIL_CLOSED_JSON"
log_msg "fail_closed_config_json=$FAIL_CLOSED_JSON"

log_msg "config load and selected/native bridge contract preflight"
"$PYTHON_BIN" - "${CONFIGS[@]}" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config

expected = {
    "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py": {
        "label": "selected_axis_bridge_control",
        "remap_gt_to_selected_axis": True,
        "work_dir_contains": "selected_axis_control",
        "postprocess_axis": "native",
    },
    "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py": {
        "label": "native_axis_bridge_shortgate_control",
        "remap_gt_to_selected_axis": False,
        "work_dir_contains": "shortgate",
        "postprocess_axis": "native",
    },
}

for path in sys.argv[1:]:
    cfg = Config.fromfile(path)
    name = Path(path).name
    rule = expected[name]
    head = cfg.model.rpn_head
    prior = head.prior_generator

    assert cfg.model.projection.type == "GridAwareConv1DTransformerProj", (name, cfg.model.projection.type)
    assert cfg.model.neck.type == "GridAwareFPNIdentity", (name, cfg.model.neck.type)
    assert head.type == "IrregularActionFormerBridgeHead", (name, head.type)
    assert head.assignment_mode == "hard", (name, head.assignment_mode)
    assert head.regression_mode == "symmetric_linear", (name, head.regression_mode)
    assert head.center_radius_scale == "point_radius", (name, head.center_radius_scale)
    assert head.reg_denom_mode == "left_right_mean", (name, head.reg_denom_mode)
    assert prior.range_mode == "absolute", (name, prior.range_mode)
    assert prior.decode_scale_mode == "level_stride", (name, prior.decode_scale_mode)
    assert prior.radius_scale_mode == "level_stride", (name, prior.radius_scale_mode)
    assert rule["work_dir_contains"] in cfg.work_dir, (name, cfg.work_dir)
    assert bool(cfg.post_processing.save_dict), name

    route_contract = getattr(head, "route_contract", {})
    expected_axis = route_contract.get("expected_axis_contract", {})
    assert expected_axis.get("postprocess_axis", "native") == rule["postprocess_axis"], (name, expected_axis)

    remap_values = []
    for split in ("train", "val", "test"):
        step = next(item for item in getattr(cfg.dataset, split).pipeline if item.get("type") == "LoadFrames")
        remap_values.append(bool(step.remap_gt_to_selected_axis))
        assert step.method == "random_fixed_subsample", (name, split, step.method)
        assert abs(float(step.keep_ratio) - 0.5) < 1e-12, (name, split, step.keep_ratio)
        assert int(step.target_len) == 384, (name, split, step.target_len)
    assert remap_values == [rule["remap_gt_to_selected_axis"]] * 3, (name, remap_values)

    print("config_ok", rule["label"], name, "work_dir", cfg.work_dir)
PY

log_msg "py_compile preflight"
"$PYTHON_BIN" -m py_compile \
  "${CONFIGS[@]}" \
  tools/audit_sparse_head_assignment.py \
  tools/check_fail_closed_config.py \
  tests/test_fail_closed_static_gates.py

if [[ "$RUN_PYTEST" == "1" ]]; then
  log_msg "pytest short contract preflight"
  "$PYTHON_BIN" -m pytest tests/test_adapter_native_dense_headv2_contracts.py tests/test_fail_closed_static_gates.py -q \
    -k "selected_axis_random_uniform_control_matrix_contracts or shortgate or stage23 or remote_run_execution"
else
  log_msg "pytest skipped; set RUN_PYTEST=1 to enable"
fi

if [[ "$RUN_ASSIGNMENT_AUDIT" != "1" ]]; then
  log_msg "assignment audit skipped; set RUN_ASSIGNMENT_AUDIT=1 for separated one-batch selected/native audits"
  exit 0
fi

mkdir -p "$SELECTED_AUDIT_OUT_DIR" "$NATIVE_AUDIT_OUT_DIR"

log_msg "running separated one-batch selected-axis assignment audit"
"$PYTHON_BIN" tools/audit_sparse_head_assignment.py \
  --configs "$SELECTED_BRIDGE_CFG" \
  --split "${AUDIT_SPLIT:-train}" \
  --num-batches "${NUM_BATCHES:-1}" \
  --seed "${AUDIT_SEED:-20260706}" \
  --device "${AUDIT_DEVICE:-cuda}" \
  --out "$SELECTED_AUDIT_OUT_DIR"

log_msg "running separated one-batch native-axis assignment audit"
"$PYTHON_BIN" tools/audit_sparse_head_assignment.py \
  --configs "$NATIVE_BRIDGE_CFG" \
  --split "${AUDIT_SPLIT:-train}" \
  --num-batches "${NUM_BATCHES:-1}" \
  --seed "${AUDIT_SEED:-20260706}" \
  --device "${AUDIT_DEVICE:-cuda}" \
  --out "$NATIVE_AUDIT_OUT_DIR"

log_msg "selected assignment audit json=$SELECTED_AUDIT_OUT_DIR/assignment_audit.json"
log_msg "selected assignment audit csv=$SELECTED_AUDIT_OUT_DIR/assignment_audit.csv"
log_msg "native assignment audit json=$NATIVE_AUDIT_OUT_DIR/assignment_audit.json"
log_msg "native assignment audit csv=$NATIVE_AUDIT_OUT_DIR/assignment_audit.csv"
