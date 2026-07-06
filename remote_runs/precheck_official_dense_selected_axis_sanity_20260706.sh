#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CFG="configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py"

cd "$ROOT"

python_has_mmengine() {
  "$1" - <<'PY' >/dev/null 2>&1
import mmengine.config
PY
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! python_has_mmengine "$PYTHON_BIN"; then
    echo "PYTHON_BIN cannot import mmengine.config: $PYTHON_BIN" >&2
    exit 127
  fi
else
  for candidate in python python.exe python3; do
    if command -v "$candidate" >/dev/null 2>&1 && python_has_mmengine "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    echo "No Python interpreter with mmengine.config found; set PYTHON_BIN=/path/to/python" >&2
    exit 127
  fi
fi

echo "config load preflight: $CFG"
"$PYTHON_BIN" scripts/verify_official_dense_reference.py \
  --local-root "$ROOT" \
  --config "$CFG" \
  --skip-reference-files

echo "py_compile preflight"
"$PYTHON_BIN" -m py_compile \
  scripts/verify_official_dense_reference.py \
  opentad/datasets/transforms/end_to_end.py \
  opentad/models/detectors/irregular_actionformer.py

if [[ "${RUN_PYTEST:-0}" == "1" ]]; then
  echo "pytest preflight"
  "$PYTHON_BIN" -m pytest tests/test_adapter_native_dense_headv2_contracts.py -q
else
  echo "pytest preflight skipped; set RUN_PYTEST=1 to enable"
fi

echo "official dense selected-axis sanity precheck ok"
