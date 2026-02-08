# VisDrone DETR Object Detection (DroneDETR++)

DETR (Detection Transformer) for drone object detection using VisDrone dataset.

## 🎯 Goal

Detect 10 object classes in aerial drone imagery:
- pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

## ✨ Enhanced Features

✅ **100 object queries** (up from 10) - detects more objects per image
✅ **Cosine learning rate** - better convergence  
✅ **Optimized hyperparameters** - tuned for drone imagery
✅ **Best model auto-save** - saves model with highest validation mAP
✅ **Small-object weighted losses** - improves learning on tiny aerial targets
✅ **Focal classification loss** - addresses class imbalance
✅ **Size-aware evaluation dashboard** - exports AP_tiny/AP_small/per-class metrics
✅ **Ablation runner** - reproducible baseline vs feature-variant comparisons

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

**DroneDETR++ (small-object + focal):**
```bash
python train.py \
  --dataset_file coco \
  --coco_path data/subset \
  --num_classes 11 \
  --epochs 25 \
  --device mps \
  --use_focal_loss \
  --class_weight_strategy effective_num_samples \
  --small_object_weighted_loss \
  --small_obj_weight_factor 3.5 \
  --tensorboard \
  --enable_eval_dashboard \
  --output_dir outputs/dronedetrpp
```

### Metrics Tracking
TensorBoard logging (loss/AP curves):
```bash
tensorboard --logdir outputs/dronedetrpp/tensorboard
```

Static plots from training logs:
```bash
python tools/plot_training_metrics.py \
  --log-file outputs/dronedetrpp/log.txt
```

### Ablations
```bash
python tools/run_ablations.py
```
This runs the experiment YAMLs in `configs/ablations/` and writes summary files to:
- `outputs/ablations/summary.json`
- `outputs/ablations/summary.csv`

### Hyperparameter Optimization (Optuna)
```bash
python tools/optuna_tune.py \
  --study-name dronedetr_smallobj \
  --coco-path data/subset \
  --device mps \
  --n-trials 20 \
  --epochs 12 \
  --metric bbox[3]
```
Key outputs:
- `outputs/optuna/<study_name>/best_params.json`
- `outputs/optuna/<study_name>/trials.csv`

### Visual Comparison
```bash
python tools/visualize_comparison.py \
  --baseline-checkpoint outputs/ablations/baseline/checkpoint.pth \
  --improved-checkpoint outputs/ablations/all_combined/checkpoint.pth \
  --coco-path data/subset \
  --num-classes 11 \
  --device mps \
  --output-dir outputs/comparison
```

### Tiled Inference (High-Resolution)
```bash
python tools/tiled_inference.py \
  --checkpoint outputs/ablations/all_combined/checkpoint.pth \
  --image-dir data/subset/val2017 \
  --device mps \
  --tile-size 1024 \
  --overlap 0.25 \
  --scales 1.0,1.5 \
  --output-json outputs/tiled_inference/predictions.json
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
