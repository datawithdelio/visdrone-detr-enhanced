#!/bin/bash
# Quick baseline training launcher.
#
# Usage:
#   ./run.sh [extra train.py args]
# Example:
#   ./run.sh --batch_size 4 --device cuda
set -euo pipefail

echo "🚀 Starting VisDrone DETR Training..."

# Activate local virtual environment before running Python.
source ~/kumar/bin/activate

# Run baseline training defaults; user flags are appended via "$@".
python train.py \
  --dataset_file coco \
  --coco_path data/processed \
  --num_classes 11 \
  --output_dir outputs/checkpoints \
  --epochs 50 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --batch_size 2 \
  --num_workers 2 \
  "$@"
