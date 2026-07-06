#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/stage4_detection_quality_stage2_dense}"
RUN_TAG="${RUN_TAG:-stage4_detection_quality_stage2_dense_$(date +%Y%m%d_%H%M%S)}"
REQUIRE_RESULTS="${REQUIRE_RESULTS:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"
FAIL_CLOSED_JSON="$LOG_DIR/${RUN_TAG}_fail_closed_config.json"

RANDOM_CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py"
UNIFORM_CFG="configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py"

TARGETS=(
  "near63_random_short|$RANDOM_CFG|exps/thumos/adatad/input_random_fixed_50pct_adapter_densehead_selected_axis_control_shortgate_n16r4/gpu1_id1"
  "near65_uniform_short|$UNIFORM_CFG|exps/thumos/adatad/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_shortgate_n16r4/gpu1_id1"
  "near63_random_long|$RANDOM_CFG|exps/thumos/adatad/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4/gpu1_id0"
  "near65_uniform_long|$UNIFORM_CFG|exps/thumos/adatad/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4/gpu1_id0"
)

log_msg() { echo "$(date '+%F %T') [stage4-quality] $*"; }
python_works() { "$1" -c "import sys" >/dev/null 2>&1; }

select_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
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

cd "$ROOT"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export THUMOS_ROOT="${THUMOS_ROOT:-$BASE/thumos14}"
select_python

log_msg "root=$ROOT run_tag=$RUN_TAG require_results=$REQUIRE_RESULTS"
log_msg "fail-closed config scan"
"$PYTHON_BIN" tools/check_fail_closed_config.py "$RANDOM_CFG" "$UNIFORM_CFG" --json-out "$FAIL_CLOSED_JSON"
log_msg "fail_closed_config_json=$FAIL_CLOSED_JSON"

log_msg "py_compile preflight"
"$PYTHON_BIN" -m py_compile tools/analyze_detection_quality.py tools/check_fail_closed_config.py

missing=0
analyzed=0
for item in "${TARGETS[@]}"; do
  IFS="|" read -r label cfg exp_dir <<<"$item"
  if [[ ! -d "$exp_dir" ]] || ! find "$exp_dir" -name result_detection.json -type f -print -quit | grep -q .; then
    log_msg "skip_missing_result label=$label exp_dir=$exp_dir"
    missing=$((missing + 1))
    continue
  fi

  output_json="$exp_dir/detection_quality_summary_${RUN_TAG}.json"
  output_csv="$exp_dir/detection_quality_rows_${RUN_TAG}.csv"
  log_msg "START quality label=$label cfg=$cfg exp_dir=$exp_dir"
  "$PYTHON_BIN" tools/analyze_detection_quality.py \
    --config "$cfg" \
    --dataset-split val \
    --experiment-dir "$exp_dir" \
    --ground-truth-fallback "$THUMOS_ROOT/annotations/thumos_14_anno.json" \
    --subset validation \
    --output-json "$output_json" \
    --output-csv "$output_csv"
  log_msg "END quality label=$label output_json=$output_json output_csv=$output_csv"
  analyzed=$((analyzed + 1))
done

log_msg "summary analyzed=$analyzed missing=$missing"
if [[ "$REQUIRE_RESULTS" == "1" && "$analyzed" -eq 0 ]]; then
  echo "No Stage-2 dense result_detection.json files found." >&2
  exit 2
fi
