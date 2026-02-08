#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running DroneDETR++ ablations..."
python3 "${BASE_DIR}/tools/run_ablations.py" "$@"
