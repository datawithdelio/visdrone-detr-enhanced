#!/bin/bash
# Build smaller train/val subsets for fast local experiments.
#
# This script calls tools/create_subset.py twice:
# 1) train subset from data/processed/train2017
# 2) val subset from data/processed/val2017
set -euo pipefail

echo "🚀 Creating VisDrone Dataset Subset"
echo "===================================="
echo ""

# Configuration
TRAIN_RATIO=0.3  # Use 30% of training data
VAL_RATIO=0.5    # Use 50% of validation data
MIN_OBJECTS=3    # Minimum objects per image
IMAGE_TRANSFER=copy  # copy = portable/professional demos, symlink = space-saving local dev

# Paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${BASE_DIR}/data"

echo "📁 Base directory: ${BASE_DIR}"
echo ""

# Create training subset
echo "📊 Creating training subset (${TRAIN_RATIO}% of data)..."
python3 "${BASE_DIR}/tools/create_subset.py" \
  --input-annotations "${DATA_DIR}/processed/annotations/instances_train2017.json" \
  --output-annotations "${DATA_DIR}/subset/annotations/instances_train2017.json" \
  --input-images "${DATA_DIR}/processed/train2017" \
  --output-images "${DATA_DIR}/subset/train2017" \
  --ratio ${TRAIN_RATIO} \
  --min-objects ${MIN_OBJECTS} \
  --image-transfer ${IMAGE_TRANSFER}

if [ $? -ne 0 ]; then
    echo "❌ Failed to create training subset"
    exit 1
fi

echo ""
echo "================================================"
echo ""

# Create validation subset
echo "📊 Creating validation subset (${VAL_RATIO}% of data)..."
python3 "${BASE_DIR}/tools/create_subset.py" \
  --input-annotations "${DATA_DIR}/processed/annotations/instances_val2017.json" \
  --output-annotations "${DATA_DIR}/subset/annotations/instances_val2017.json" \
  --input-images "${DATA_DIR}/processed/val2017" \
  --output-images "${DATA_DIR}/subset/val2017" \
  --ratio ${VAL_RATIO} \
  --min-objects ${MIN_OBJECTS} \
  --image-transfer ${IMAGE_TRANSFER}

if [ $? -ne 0 ]; then
    echo "❌ Failed to create validation subset"
    exit 1
fi

echo ""
echo "✅ Dataset subset created successfully!"
echo ""
echo "📂 Subset location: ${DATA_DIR}/subset/"
echo ""
echo "🚀 Next steps:"
echo "   1. Review the subset statistics above"
echo "   2. Run training with: python train_fast.py --coco_path data/subset"
echo "   3. Monitor GPU memory and adjust batch_size if needed"
echo ""
