#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
ROOT="$BASE/OpenTAD_SparseHeadClean_20260702"
LOG_DIR="$ROOT/logs/gpu1_same_batch_audit"
RUN_TAG="${RUN_TAG:-same_batch_audit_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/logs/sparse_head_assignment_audit_20260705/$RUN_TAG}"
CHAIN_LOG="$LOG_DIR/${RUN_TAG}.log"
FAIL_CLOSED_JSON="$OUT_DIR/fail_closed_config.json"

mkdir -p "$LOG_DIR" "$OUT_DIR"
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "[audit] started_at=$(date -Is)"
echo "[audit] host=$(hostname)"
echo "[audit] root=$ROOT"
echo "[audit] out_dir=$OUT_DIR"
echo "[audit] chain_log=$CHAIN_LOG"

source "$BASE/conda_envs/opentad/bin/activate"
export HOME="$BASE/tmp/home"
export XDG_CACHE_HOME="$BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$BASE/tmp/xdg_config"
export HF_HOME="$BASE/hf_cache"
export THUMOS_ROOT="$BASE/thumos14"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

cd "$ROOT"

CONFIGS=(
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4.py"
)

echo "[audit] fail-closed config scan"
python tools/check_fail_closed_config.py "${CONFIGS[@]}" --json-out "$FAIL_CLOSED_JSON"
echo "[audit] fail_closed_config_json=$FAIL_CLOSED_JSON"

echo "[audit] config load preflight"
python - "${CONFIGS[@]}" <<'PY'
import sys
from mmengine.config import Config

for path in sys.argv[1:]:
    cfg = Config.fromfile(path)
    head = cfg.model["rpn_head"]
    prior = head.get("prior_generator", {})
    print(
        "[config]",
        path,
        "work_dir=", cfg.get("work_dir"),
        "assignment=", head.get("assignment_mode"),
        "regression=", head.get("regression_mode"),
        "range_mode=", prior.get("range_mode"),
        "center_radius_scale=", head.get("center_radius_scale", "default"),
        "reg_denom_mode=", head.get("reg_denom_mode", "default"),
    )
PY

echo "[audit] py_compile preflight"
python -m py_compile \
  tools/audit_sparse_head_assignment.py \
  opentad/models/dense_heads/irregular_actionformer_bridge_head.py \
  tests/test_adapter_native_dense_headv2_contracts.py

echo "[audit] pytest preflight"
python -m pytest tests/test_adapter_native_dense_headv2_contracts.py -q

echo "[audit] gpu snapshot"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || nvidia-smi || true

echo "[audit] running same-batch assignment audit"
python tools/audit_sparse_head_assignment.py \
  --configs "${CONFIGS[@]}" \
  --split train \
  --num-batches "${NUM_BATCHES:-1}" \
  --seed "${AUDIT_SEED:-20260705}" \
  --device cuda \
  --out "$OUT_DIR"

echo "[audit] summary"
python - "$OUT_DIR/assignment_audit.json" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
by_cfg = defaultdict(list)
for row in rows:
    by_cfg[row["config_name"]].append(row)

for cfg_name, cfg_rows in by_cfg.items():
    n = len(cfg_rows)
    level_count = cfg_rows[0]["level_count"]
    pos = [sum(r["per_level_pos_count"][i] for r in cfg_rows) for i in range(level_count)]
    before = [sum(r["per_level_candidate_count_before_range"][i] for r in cfg_rows) for i in range(level_count)]
    after = [sum(r["per_level_candidate_count_after_range"][i] for r in cfg_rows) for i in range(level_count)]
    range_fail = [sum(r["range_fail_count_by_level"][i] for r in cfg_rows) for i in range(level_count)]
    center_fail = [sum(r["center_fail_count_by_level"][i] for r in cfg_rows) for i in range(level_count)]
    gt_total = 0
    gt_covered = 0
    bucket_total = defaultdict(int)
    bucket_covered = defaultdict(int)
    for row in cfg_rows:
        for gt in row["gt_coverage"]:
            gt_total += 1
            bucket = gt["gt_length_bucket"]
            bucket_total[bucket] += 1
            if gt["gt_covered_any"]:
                gt_covered += 1
                bucket_covered[bucket] += 1
    coverage = (gt_covered / gt_total) if gt_total else 0.0
    print(f"[summary] {cfg_name}")
    print(f"  rows={n} gt_coverage={gt_covered}/{gt_total} ({coverage:.3f})")
    print(f"  pos_by_level={pos}")
    print(f"  candidates_before_range={before}")
    print(f"  candidates_after_range={after}")
    print(f"  range_fail_by_level={range_fail}")
    print(f"  center_fail_by_level={center_fail}")
    for bucket in sorted(bucket_total):
        print(f"  bucket {bucket}: {bucket_covered[bucket]}/{bucket_total[bucket]}")

print(f"[audit] json={path}")
print(f"[audit] csv={path.with_suffix('.csv')}")
PY

echo "[audit] finished_at=$(date -Is)"
