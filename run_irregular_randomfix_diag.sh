#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/OpenTAD_Back_check
mkdir -p /root/autodl-tmp/OpenTAD_Back_check/logs
exec > /root/autodl-tmp/OpenTAD_Back_check/logs/input_random_fixed_50pct_irregular_actionformer_diag.log 2>&1
export CUDA_VISIBLE_DEVICES=0
/root/miniconda3/bin/torchrun --nproc_per_node=1 /root/autodl-tmp/OpenTAD_Back_check/tools/train.py /root/autodl-tmp/OpenTAD_Back_check/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_diag.py --id 0
