#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${ROOT:-$BASE/OpenTAD_SparseHeadClean_20260702}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/precheck_selected_axis_control_matrix}"
RUN_TAG="${RUN_TAG:-precheck_selected_axis_control_matrix_$(date +%Y%m%d_%H%M%S)}"
FAIL_CLOSED_JSON="$LOG_DIR/${RUN_TAG}_fail_closed_config.json"

CONFIGS=(
  "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py"
)

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$LOG_DIR"

echo "[precheck] fail-closed config scan"
python tools/check_fail_closed_config.py "${CONFIGS[@]}" --json-out "$FAIL_CLOSED_JSON"
echo "[precheck] fail_closed_config_json=$FAIL_CLOSED_JSON"

echo "[precheck] selected-axis control config load"
python - "${CONFIGS[@]}" <<'PY'
import sys
from mmengine.config import Config

seen_work_dirs = set()
expected = {
    "input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py": (
        "uniform_fixed_subsample",
        "ActionFormerHead",
        "DensePassthroughConv1DTransformerProj",
        "DensePassthroughFPNIdentity",
    ),
    "input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py": (
        "random_fixed_subsample",
        "ActionFormerHead",
        "DensePassthroughConv1DTransformerProj",
        "DensePassthroughFPNIdentity",
    ),
    "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py": (
        "random_fixed_subsample",
        "IrregularActionFormerBridgeHead",
        "GridAwareConv1DTransformerProj",
        "GridAwareFPNIdentity",
    ),
}

for path in sys.argv[1:]:
    cfg = Config.fromfile(path)
    name = path.rsplit("/", 1)[-1]
    method, head_type, projection_type, neck_type = expected[name]
    assert cfg.model.rpn_head.type == head_type, (name, cfg.model.rpn_head.type)
    assert cfg.model.projection.type == projection_type, (name, cfg.model.projection.type)
    assert cfg.model.neck.type == neck_type, (name, cfg.model.neck.type)
    assert bool(cfg.post_processing.save_dict), name
    assert cfg.work_dir not in seen_work_dirs, cfg.work_dir
    seen_work_dirs.add(cfg.work_dir)

    for split in ("train", "val", "test"):
        step = next(item for item in getattr(cfg.dataset, split).pipeline if item.get("type") == "LoadFrames")
        assert step.method == method, (name, split, step.method)
        assert abs(float(step.keep_ratio) - 0.5) < 1e-12, (name, split, step.keep_ratio)
        assert bool(step.remap_gt_to_selected_axis), (name, split, step.remap_gt_to_selected_axis)

    print("config_ok", name, "head", head_type, "method", method, "work_dir", cfg.work_dir)
PY

echo "[precheck] py_compile"
python -m py_compile "${CONFIGS[@]}" tests/test_adapter_native_dense_headv2_contracts.py

echo "[precheck] pytest"
python -m pytest tests/test_adapter_native_dense_headv2_contracts.py -q

echo "[precheck] selected-axis control matrix passed"
