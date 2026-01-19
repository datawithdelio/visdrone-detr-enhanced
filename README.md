# VisDrone DETR Object Detection

DETR (Detection Transformer) for drone object detection using VisDrone dataset.

## 🎯 Goal

Detect 10 object classes in aerial drone imagery:
- pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

## ✨ Enhanced Features

✅ **100 object queries** (up from 10) - detects more objects per image
✅ **Cosine learning rate** - better convergence  
✅ **Optimized hyperparameters** - tuned for drone imagery
✅ **Best model auto-save** - saves model with highest validation mAP

## 🚀 Quick Start

### Setup
```bash
# Activate environment
source ~/kumar/bin/activate

# Verify setup
python train_optimized.py --help
```

### Training

**Enhanced version (Recommended):**
```bash
python train_optimized.py \
  --dataset_file coco \
  --coco_path data/processed \
  --num_classes 11 \
  --epochs 50 \
  --device cuda
```

**Original version:**
```bash
python train.py \
  --dataset_file coco \
  --coco_path data/processed \
  --num_classes 11 \
  --epochs 50 \
  --device cuda
```

## 📁 Structure
```
drone-visdrone-detr/
├── train.py                # Original
├── train_optimized.py      # Enhanced ⭐
├── run_optimized.sh        # Helper script
├── src/                    # Source code
├── tools/                  # Preprocessing
├── data/                   # Dataset (symlinked)
└── outputs/                # Training outputs
```

## 📊 Expected Results

- Training time: ~2-3 hours (GPU) / ~15-20 hours (CPU)
- Final mAP: 35-45% (enhanced) vs 30-40% (baseline)

## 🔧 Hardware Options

- **GPU (11GB+)**: `--batch_size 4 --device cuda`
- **GPU (8GB)**: `--batch_size 2 --device cuda`  
- **CPU**: `--batch_size 2 --device cpu` (slow)

## 📚 References

- [DETR Paper](https://arxiv.org/abs/2005.12872)
- [VisDrone Dataset](http://aiskyeye.com/)
