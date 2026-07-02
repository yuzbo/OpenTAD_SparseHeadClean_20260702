#!/bin/bash
# R05E: Full THUMOS14 pilot with NativePhysicalPointHead
# Server: server1:24013
# Expected: ~60 epochs, val every 5 epochs from epoch 30
# Target: beat nearest 54.03, ideally approach 59.22 oracle ceiling

cd /root/autodl-tmp/OpenTAD_Back_check

export MASTER_PORT=29844

/root/miniconda3/bin/torchrun --nproc_per_node=1 \
    tools/train.py \
    configs/adatad/thumos/input_random_fixed_50pct_native_physical_point_full_pilot.py \
    2>&1 | tee logs/r05e_full_pilot_live.log
