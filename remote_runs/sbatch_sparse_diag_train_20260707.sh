#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --exclude=g0030
#SBATCH -J sparse_diag
#SBATCH -o logs/slurm_sparse_diag_20260707/%x-%j.out
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${ROOT:-$BASE/OpenTAD_SparseHeadClean_20260702}"
CFG="${CFG:?Set CFG to a supported sparse diagnostic config}"
EXP_LABEL="${EXP_LABEL:-$(basename "$CFG" .py)}"
EXP_ID="${EXP_ID:-0}"
PORT="${PORT:-$((33000 + (${SLURM_JOB_ID:-0} % 10000)))}"
RUN_ANALYSIS_AFTER="${RUN_ANALYSIS_AFTER:-1}"
ALLOW_OVERWRITE_SPARSE_DIAG_OUTPUT="${ALLOW_OVERWRITE_SPARSE_DIAG_OUTPUT:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/slurm_sparse_diag_20260707}"
RUN_TAG="${RUN_TAG:-sparse_diag_${EXP_LABEL}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
CHAIN_LOG="$LOG_DIR/${RUN_TAG}.log"
TRAIN_LOG="$LOG_DIR/${RUN_TAG}_train.log"
FAIL_CLOSED_JSON="$LOG_DIR/${RUN_TAG}_fail_closed_config.json"
ANALYSIS_DIR="$ROOT/analysis/sparse_diag_20260707"

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
exec > >(tee -a "$CHAIN_LOG") 2>&1

log_msg() { echo "$(date '+%F %T') [sparse-diagnostic-slurm] $*"; }

prepare_output_dir() {
  local run_dir="$1"
  if [[ -e "$run_dir" ]]; then
    if [[ "$ALLOW_OVERWRITE_SPARSE_DIAG_OUTPUT" != "1" ]]; then
      echo "Refusing sparse diagnostic training: output directory already exists: $run_dir" >&2
      echo "Set ALLOW_OVERWRITE_SPARSE_DIAG_OUTPUT=1 only for an intentional rerun." >&2
      exit 66
    fi
    case "$run_dir" in
      "$ROOT"/exps/thumos/adatad/*/gpu1_id*) ;;
      *)
        echo "Refusing to remove unsafe sparse diagnostic output path: $run_dir" >&2
        exit 66
        ;;
    esac
    log_msg "removing existing sparse diagnostic output directory run_dir=$run_dir"
    rm -rf "$run_dir"
  fi
  mkdir -p "$(dirname "$run_dir")"
}

if [[ "${SLURM_JOB_ID:-}" == "1118197" ]]; then
  echo "Refusing training inside allocation 1118197; submit a separate Slurm job." >&2
  exit 65
fi
if [[ "$(hostname -s)" == "g0030" && "${ALLOW_G0030_LONG:-0}" != "1" ]]; then
  echo "Refusing sparse diagnostic training on g0030; this node hosts the current two-card allocation." >&2
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

log_msg "config load and sparse diagnostic contract preflight"
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
    "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py": {
        "method": "random_fixed_subsample",
        "remap": False,
        "gt_axis": "native",
        "proposal_axis": "native",
        "nms_axis": "native",
        "compatibility": "irregular_geometry_diagnostic_candidate",
        "shortgate": False,
    },
    "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py": {
        "method": "random_fixed_subsample",
        "remap": True,
        "gt_axis": "selected",
        "proposal_axis": "selected",
        "nms_axis": "native",
        "compatibility": "dense_compatible_diagnostic_candidate",
        "shortgate": False,
    },
    "input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py": {
        "method": "uniform_fixed_subsample",
        "remap": False,
        "gt_axis": "native",
        "proposal_axis": "native",
        "nms_axis": "native",
        "compatibility": "irregular_geometry_diagnostic_candidate",
        "shortgate": False,
    },
    "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py": {
        "method": "random_fixed_subsample",
        "remap": False,
        "gt_axis": "native",
        "proposal_axis": "native",
        "nms_axis": "native",
        "compatibility": "irregular_geometry_diagnostic_candidate",
        "shortgate": True,
    },
}
assert name in expected, name
spec = expected[name]
head = cfg.model.rpn_head
prior = head.prior_generator
contract = head.route_contract
assert cfg.model.type == "IrregularActionFormer", cfg.model.type
assert head.type == "IrregularActionFormerBridgeHead", head.type
assert head.assignment_mode == "hard", head.assignment_mode
assert head.regression_mode == "symmetric_linear", head.regression_mode
assert head.center_radius_scale == "point_radius", head.center_radius_scale
assert head.reg_denom_mode == "left_right_mean", head.reg_denom_mode
assert not bool(head.allow_legacy_full_cell_span), head.allow_legacy_full_cell_span
assert not bool(head.allow_center_fallback_inside_gt), head.allow_center_fallback_inside_gt
assert prior.range_mode == "absolute", prior.range_mode
assert prior.decode_scale_mode == "level_stride", prior.decode_scale_mode
assert prior.radius_scale_mode == "level_stride", prior.radius_scale_mode
assert [tuple(item) for item in prior.regression_range] == [
    (0, 8),
    (2, 16),
    (4, 32),
    (8, 64),
    (16, 128),
    (32, 10000),
]
assert contract.gt_axis == spec["gt_axis"], contract
assert contract.proposal_axis == spec["proposal_axis"], contract
assert contract.nms_axis == spec["nms_axis"], contract
assert contract.postprocess_axis == "native", contract
assert contract.eval_axis == "seconds", contract
assert contract.compatibility == spec["compatibility"], contract
assert bool(contract.diagnostic_only), contract
assert not bool(contract.primary_result_allowed), contract
assert not bool(contract.dense_equivalent_claim_allowed), contract
assert bool(cfg.post_processing.save_dict), name
for split in ("train", "val", "test"):
    step = next(item for item in getattr(cfg.dataset, split).pipeline if item.get("type") == "LoadFrames")
    assert step.method == spec["method"], (split, step.method)
    assert bool(step.remap_gt_to_selected_axis) == spec["remap"], (split, step.remap_gt_to_selected_axis)
    assert int(step.target_len) == 384, (split, step.target_len)
if spec["shortgate"]:
    assert int(cfg.workflow.end_epoch) == 2, cfg.workflow
    assert bool(cfg.workflow.disable_checkpoint), cfg.workflow
work_dir = os.path.join(cfg.work_dir, f"gpu1_id{exp_id}")
print("config_ok", name)
print("work_dir", work_dir)
print("head", head.type, head.assignment_mode, head.regression_mode)
print("axis", contract.gt_axis, contract.proposal_axis, contract.nms_axis, contract.postprocess_axis)
print("sampling", spec["method"], "remap", spec["remap"], "shortgate", spec["shortgate"])
PY

log_msg "py_compile preflight"
python -m py_compile "$CFG" tools/train.py tools/check_fail_closed_config.py tools/analyze_detection_quality.py tools/collect_experiment_results.py tools/plot_detection_diagnostics.py

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

PRE_FILTER_JSONL="$ROOT/$RUN_DIR/proposals_pre_filter_${RUN_TAG}.jsonl"
PRE_NMS_JSONL="$ROOT/$RUN_DIR/proposals_pre_nms_${RUN_TAG}.jsonl"
POST_NMS_JSONL="$ROOT/$RUN_DIR/proposals_post_nms_${RUN_TAG}.jsonl"
FINAL_JSONL="$ROOT/$RUN_DIR/proposals_final_${RUN_TAG}.jsonl"

log_msg "START train port=$PORT exp_id=$EXP_ID"
torchrun --master_port="$PORT" --nproc_per_node=1 tools/train.py "$CFG" --id "$EXP_ID" \
  --cfg-options \
  post_processing.debug_dump_pre_filter_path="$PRE_FILTER_JSONL" \
  post_processing.debug_dump_pre_nms_path="$PRE_NMS_JSONL" \
  post_processing.debug_dump_post_nms_path="$POST_NMS_JSONL" \
  post_processing.debug_dump_final_path="$FINAL_JSONL" \
  post_processing.debug_dump_topk=200 \
  2>&1 | tee "$TRAIN_LOG"
status=${PIPESTATUS[0]}
log_msg "END train status=$status"
if [[ "$status" != "0" ]]; then
  exit "$status"
fi
log_msg "proposal_dump_pre_filter=$PRE_FILTER_JSONL"
log_msg "proposal_dump_pre_nms=$PRE_NMS_JSONL"
log_msg "proposal_dump_post_nms=$POST_NMS_JSONL"
log_msg "proposal_dump_final=$FINAL_JSONL"

if [[ "$RUN_ANALYSIS_AFTER" == "1" ]]; then
  log_msg "START post-training detection quality"
  QUALITY_JSON="$ROOT/$RUN_DIR/detection_quality_summary_${RUN_TAG}.json"
  QUALITY_CSV="$ROOT/$RUN_DIR/detection_quality_rows_${RUN_TAG}.csv"
  python tools/analyze_detection_quality.py \
    --config "$CFG" \
    --dataset-split val \
    --experiment-dir "$ROOT/$RUN_DIR" \
    --ground-truth-fallback "$THUMOS_ROOT/annotations/thumos_14_anno.json" \
    --subset validation \
    --output-json "$QUALITY_JSON" \
    --output-csv "$QUALITY_CSV"
  log_msg "quality_json=$QUALITY_JSON"
  log_msg "quality_csv=$QUALITY_CSV"

  PROPOSAL_PREFIX="$ROOT/$RUN_DIR/proposal_lifecycle_${RUN_TAG}"
  if [[ -s "$PRE_NMS_JSONL" && -s "$POST_NMS_JSONL" ]]; then
    python tools/analyze_detection_quality.py \
      --config "$CFG" \
      --dataset-split val \
      --ground-truth-fallback "$THUMOS_ROOT/annotations/thumos_14_anno.json" \
      --subset validation \
      --pre-nms-jsonl "$PRE_NMS_JSONL" \
      --post-nms-jsonl "$POST_NMS_JSONL" \
      --proposal-output-prefix "$PROPOSAL_PREFIX" \
      --nms-iou-threshold 0.6
    log_msg "proposal_lifecycle_summary=${PROPOSAL_PREFIX}_summary.json"
    log_msg "proposal_lifecycle_pre_nms=${PROPOSAL_PREFIX}_pre_nms.csv"
    log_msg "proposal_lifecycle_post_nms=${PROPOSAL_PREFIX}_post_nms.csv"
    log_msg "proposal_lifecycle_nms=${PROPOSAL_PREFIX}_nms.csv"
  else
    log_msg "WARN proposal lifecycle skipped; missing non-empty pre/post-NMS dumps"
  fi

  SUMMARY_PREFIX="$ANALYSIS_DIR/${RUN_TAG}"
  python tools/collect_experiment_results.py \
    "sparse_${EXP_LABEL}=$ROOT/$RUN_DIR" \
    "headv3_fixed=$ROOT/exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4/gpu1_id1" \
    "bridge_openrange=$ROOT/exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4/gpu1_id2" \
    "bridge_absrange=$ROOT/exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4/gpu1_id1" \
    "dense_selected_random=$ROOT/exps/thumos/adatad/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4/gpu1_id0" \
    "official_dense_uniform=$ROOT/exps/thumos/adatad/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4/gpu1_id0" \
    --json-out "${SUMMARY_PREFIX}_results.json" \
    --csv-out "${SUMMARY_PREFIX}_results.csv" \
    --md-out "${SUMMARY_PREFIX}_results.md"
  python tools/plot_detection_diagnostics.py \
    --input "${SUMMARY_PREFIX}_results.csv" \
    --spec-json "${SUMMARY_PREFIX}_figure_specs.json" \
    --plot-dir "${SUMMARY_PREFIX}_figures"
  log_msg "summary_json=${SUMMARY_PREFIX}_results.json"
  log_msg "summary_csv=${SUMMARY_PREFIX}_results.csv"
  log_msg "summary_md=${SUMMARY_PREFIX}_results.md"
  log_msg "figure_specs=${SUMMARY_PREFIX}_figure_specs.json"
  log_msg "figure_dir=${SUMMARY_PREFIX}_figures"
  log_msg "END post-training detection quality"
fi
