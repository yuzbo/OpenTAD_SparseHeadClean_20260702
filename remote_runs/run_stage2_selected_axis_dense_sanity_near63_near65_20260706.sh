#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/stage2_selected_axis_dense_sanity_near63_near65}"
RUN_TAG="${RUN_TAG:-stage2_selected_axis_dense_sanity_$(date +%Y%m%d_%H%M%S)}"
FAIL_CLOSED_JSON="$LOG_DIR/${RUN_TAG}_fail_closed_config.json"

RANDOM_FIXED_CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py"
UNIFORM_OFFICIAL_CFG="configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py"
CONFIGS=("$RANDOM_FIXED_CFG" "$UNIFORM_OFFICIAL_CFG")

log_msg() { echo "$(date '+%F %T') [stage2-dense-sanity] $*"; }
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
mkdir -p "$LOG_DIR"
select_python

log_msg "root=$ROOT"
log_msg "python_bin=$PYTHON_BIN"
log_msg "random_fixed_reference=near63_random_fixed_selected_axis_dense_control cfg=$RANDOM_FIXED_CFG"
log_msg "uniform_reference=near65_uniform_even_spacing_official_dense cfg=$UNIFORM_OFFICIAL_CFG"
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
        "head": "ActionFormerHead",
        "projection": "DensePassthroughConv1DTransformerProj",
        "neck": "DensePassthroughFPNIdentity",
        "work_dir_contains": "input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4",
    },
    "input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py": {
        "label": "near65_uniform_even_spacing_official_dense",
        "method": "uniform_fixed_subsample",
        "head": "ActionFormerHead",
        "projection": "DensePassthroughConv1DTransformerProj",
        "neck": "DensePassthroughFPNIdentity",
        "work_dir_contains": "official_dense_selected_axis_sanity",
    },
}

seen_work_dirs = set()
for path in sys.argv[1:]:
    cfg = Config.fromfile(path)
    name = Path(path).name
    rule = expected[name]
    head = cfg.model.rpn_head

    assert head.type == rule["head"], (name, head.type)
    assert cfg.model.projection.type == rule["projection"], (name, cfg.model.projection.type)
    assert cfg.model.neck.type == rule["neck"], (name, cfg.model.neck.type)
    assert bool(cfg.post_processing.save_dict), name
    assert rule["work_dir_contains"] in cfg.work_dir, (name, cfg.work_dir)
    assert cfg.work_dir not in seen_work_dirs, cfg.work_dir
    seen_work_dirs.add(cfg.work_dir)

    for split in ("train", "val", "test"):
        step = next(item for item in getattr(cfg.dataset, split).pipeline if item.get("type") == "LoadFrames")
        assert step.method == rule["method"], (name, split, step.method)
        assert abs(float(step.keep_ratio) - 0.5) < 1e-12, (name, split, step.keep_ratio)
        assert bool(step.remap_gt_to_selected_axis), (name, split, step.remap_gt_to_selected_axis)
        assert int(step.target_len) == 384, (name, split, step.target_len)

    print("config_ok", rule["label"], name, "work_dir", cfg.work_dir)
PY

log_msg "py_compile preflight"
"$PYTHON_BIN" -m py_compile "${CONFIGS[@]}" tools/check_fail_closed_config.py tests/test_fail_closed_static_gates.py

if [[ "${RUN_PYTEST:-1}" == "1" ]]; then
  log_msg "pytest static runner-gate preflight"
  "$PYTHON_BIN" -m pytest tests/test_fail_closed_static_gates.py -q -k "stage23 or remote_run_execution"
else
  log_msg "pytest skipped; set RUN_PYTEST=1 to enable"
fi

log_msg "preflight complete; this script intentionally does not launch long training"
