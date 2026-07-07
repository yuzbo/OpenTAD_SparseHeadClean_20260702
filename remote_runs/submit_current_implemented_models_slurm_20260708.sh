#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${ROOT:-$BASE/OpenTAD_SparseHeadClean_20260702}"
EXCLUDE_NODE="${EXCLUDE_NODE:-g0030}"
PARTITION="${SLURM_PARTITION:-gpu}"
GPUS="${SLURM_GPUS:-1}"
EXP_ID="${EXP_ID:-0}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-current_models_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DENSE_LOG_DIR="$ROOT/logs/slurm_stage2_dense_selected_axis_long"
EXACT0410_LOG_DIR="$ROOT/logs/slurm_0410_exact_dense"
SPARSE_LOG_DIR="$ROOT/logs/slurm_sparse_diag_20260707"
DENSE_BODY="$ROOT/remote_runs/sbatch_stage2_dense_selected_axis_train_20260706.sh"
EXACT0410_BODY="$ROOT/remote_runs/sbatch_0410_exact_dense_train_20260708.sh"
SPARSE_BODY="$ROOT/remote_runs/sbatch_sparse_diag_train_20260707.sh"

mkdir -p "$DENSE_LOG_DIR" "$EXACT0410_LOG_DIR" "$SPARSE_LOG_DIR"
if [[ -f "$BASE/conda_envs/opentad/bin/activate" ]]; then
  # Config.fromfile requires the OpenTAD Python environment even before sbatch.
  source "$BASE/conda_envs/opentad/bin/activate"
fi
cd "$ROOT"

run_dir_for() {
  local cfg="$1"
  local exp_id="$2"
  "$PYTHON_BIN" - "$cfg" "$exp_id" <<'PY'
import os
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
print(os.path.join(cfg.work_dir, "gpu1_id%d" % int(sys.argv[2])))
PY
}

active_job_named() {
  local label="$1"
  [[ -n "$(squeue -h -u "${USER:-sczc063}" -n "$label" 2>/dev/null || true)" ]]
}

submit_one() {
  local family="$1"
  local label="$2"
  local cfg="$3"
  local exp_id="${4:-$EXP_ID}"
  local body log_dir run_tag run_dir

  case "$family" in
    exact0410)
      body="$EXACT0410_BODY"
      log_dir="$EXACT0410_LOG_DIR"
      ;;
    dense)
      body="$DENSE_BODY"
      log_dir="$DENSE_LOG_DIR"
      ;;
    sparse)
      body="$SPARSE_BODY"
      log_dir="$SPARSE_LOG_DIR"
      ;;
    *)
      echo "ERROR unknown family=$family for label=$label" >&2
      exit 2
      ;;
  esac

  run_dir="$(run_dir_for "$cfg" "$exp_id")"
  if active_job_named "$label"; then
    echo "SKIP active label=$label cfg=$cfg"
    return 0
  fi
  if [[ -e "$ROOT/$run_dir/log.json" || -e "$ROOT/$run_dir/result_detection.json" ]]; then
    echo "SKIP has_output label=$label run_dir=$run_dir"
    return 0
  fi

  run_tag="${RUN_TAG_PREFIX}_${label}"
  echo "SUBMIT family=$family label=$label cfg=$cfg exp_id=$exp_id exclude=$EXCLUDE_NODE"
  sbatch \
    --partition="$PARTITION" \
    --gpus="$GPUS" \
    --exclude="${EXCLUDE_NODE:-g0030}" \
    --job-name="$label" \
    --output="$log_dir/%x-%j.out" \
    --export=ALL,CFG="$cfg",EXP_LABEL="$label",EXP_ID="$exp_id",RUN_TAG="$run_tag" \
    "$body"
}

echo "Current safe submitter uses commit marker: $(cat "$ROOT/.codex_synced_commit" 2>/dev/null || echo unknown)"
echo "Policy skips:"
echo "  - sparse_selected_axis_bridge_absx: selected-axis GT remap fail-closed already triggered; do not resubmit unchanged."
echo "  - levelstride/radiuslevel: same-batch audit says positives/coverage are too weak for long training."
echo "  - legacy bridge hard/openrange/log/soft/densepass: compatibility=legacy_ablation_only."
echo "Dense baseline policy:"
echo "  - exact0410_* are the only current 0410 random/uniform dense-baseline reproduction jobs."
echo "  - near63/near65 selected-axis dense-control jobs are current-route sanity checks, not exact 0410 baselines."

submit_one exact0410 "exact0410_random_fixed_dense63" \
  "configs/adatad/thumos/input_random_fixed_50pct_0410_exact_n16r4.py" "$EXP_ID"
submit_one exact0410 "exact0410_stride2_uniform_dense65" \
  "configs/adatad/thumos/input_stride2_uniform_0410_exact_n16r4.py" "$EXP_ID"

submit_one dense "near63_random_fixed_selected_axis_dense_control" \
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py" "$EXP_ID"
submit_one dense "near65_uniform_even_spacing_official_dense" \
  "configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py" "$EXP_ID"
submit_one dense "near65_uniform_even_spacing_adapter_dense" \
  "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py" "$EXP_ID"

submit_one sparse "sparse_random_native_bridge_absx" \
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py" "$EXP_ID"
submit_one sparse "sparse_uniform_native_bridge_absx" \
  "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py" "$EXP_ID"
submit_one sparse "sparse_shortgate_native_bridge_absx" \
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py" "$EXP_ID"
