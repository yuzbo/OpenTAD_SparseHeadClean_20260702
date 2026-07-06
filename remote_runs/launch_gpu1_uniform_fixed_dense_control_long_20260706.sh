#!/usr/bin/env bash
set -euo pipefail

echo "DEPRECATED_GPU1_LONG_TRAINING_DISABLED: $0"
echo "Long training must be submitted as a separate sbatch job and must not reuse allocation 1118197/g0030/GPU1."
echo "Use remote_runs/submit_stage2_dense_selected_axis_long_slurm_20260706.sh for Stage-2 dense controls."
exit 64
