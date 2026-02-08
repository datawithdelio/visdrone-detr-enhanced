#!/bin/bash
# Wrapper for reproducible ablation runs defined in configs/ablations/*.yaml.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running DroneDETR++ ablations..."
# Pass through any CLI filters, e.g. --only baseline all_combined
python3 "${BASE_DIR}/tools/run_ablations.py" "$@"
