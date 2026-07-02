#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/root/autodl-tmp/OpenTAD_Back_check}"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

CONFIG="configs/adatad/thumos/input_random_fixed_50pct_native_physical_point_single_video_overfit.py"
NAME="input_random_fixed_50pct_native_physical_point_single_video_overfit"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"
MASTER_PORT="${MASTER_PORT:-29842}"
LOG_FILE="$LOG_DIR/${NAME}_$(date '+%Y%m%d_%H%M%S').log"

cd "$ROOT_DIR"
"$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$ROOT_DIR'); from mmengine.config import Config; cfg = Config.fromfile('$CONFIG'); print(cfg.work_dir)"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$TORCHRUN" --master_port="$MASTER_PORT" --nproc_per_node=1 \
  tools/train.py "$CONFIG" --id 0 2>&1 | tee "$LOG_FILE"
