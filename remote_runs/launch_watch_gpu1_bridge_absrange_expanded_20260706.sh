#!/usr/bin/env bash
set -euo pipefail

echo "DEPRECATED_GPU1_LONG_TRAINING_DISABLED: $0"
echo "The old GPU1 waiter launcher is disabled because it would launch long training inside allocation 1118197/g0030/GPU1."
echo "Submit long training with sbatch; GPU1 is reserved only for short validation and audits."
exit 64
