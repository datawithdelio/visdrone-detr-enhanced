#!/bin/bash
# Quick training script

echo "🚀 Starting VisDrone DETR Training..."

# Activate virtual environment
source ~/kumar/bin/activate

# Run training with default parameters
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
