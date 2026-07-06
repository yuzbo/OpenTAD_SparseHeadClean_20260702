#!/usr/bin/env bash
set -euo pipefail

echo "DEPRECATED_GPU1_LONG_TRAINING_DISABLED: $0"
echo "Long training must be submitted as a separate sbatch job and must not reuse allocation 1118197/g0030/GPU1."
echo "Use a dedicated Slurm submitter; GPU1 is reserved only for short validation and audits."
exit 64
