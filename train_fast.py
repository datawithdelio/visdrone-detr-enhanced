#!/usr/bin/env python3
"""
Fast training script for VisDrone DETR
Optimized for quick iteration and experimentation
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

import utils.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    """Build CLI options tuned for fast iteration experiments."""
    parser = argparse.ArgumentParser('Fast DETR Training', add_help=False)
    
    # Learning rate
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='Learning rate (higher for faster convergence)')
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='Backbone learning rate')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size (increase if you have GPU memory)')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=25, type=int,
                        help='Number of epochs (fewer for fast training)')
    parser.add_argument('--lr_drop', default=20, type=int,
                        help='Drop LR at this epoch')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='Gradient clipping max norm')
    
    # Model Architecture - OPTIMIZED FOR SPEED
    parser.add_argument('--backbone', default='resnet34', type=str,
                        choices=['resnet34', 'resnet50'],
                        help='Backbone (resnet34 is faster)')
    parser.add_argument('--num_queries', default=50, type=int,
                        help='Number of object queries (fewer = faster)')
    parser.add_argument('--hidden_dim', default=128, type=int,
                        help='Transformer hidden dim (smaller = faster)')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--enc_layers', default=3, type=int,
                        help='Encoder layers (fewer = faster)')
    parser.add_argument('--dec_layers', default=3, type=int,
                        help='Decoder layers (fewer = faster)')
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help='FFN dimension (smaller = faster)')
    
    # Dataset
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, required=True,
                        help='Path to dataset (use data/subset for fast training)')
    parser.add_argument('--num_classes', type=int, default=11,
                        help='Number of classes (10 + background)')
    
    # Training options
    parser.add_argument('--output_dir', default='outputs/fast',
                        help='Path to save outputs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int,
                        help='DataLoader workers (increase for CPU bottleneck)')
    parser.add_argument('--resume', default='', type=str,
                        help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    
    # Mixed precision training (faster on modern GPUs)
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision training')
    
    # Auxiliary decoding loss
    parser.add_argument('--aux_loss', default=True, type=bool)
    
    # Loss coefficients
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help='Loss weight for no-object class')
    
    # Other
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str,
                        choices=['sine', 'learned'])
    parser.add_argument('--eval', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--frozen_weights', type=str, default=None)
    
    return parser


def main(args):
    """Run shortened, speed-oriented training with periodic validation."""
    print("=" * 60)
    print("🚀 Fast DETR Training for VisDrone")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   - Model: {args.backbone}")
    print(f"   - Queries: {args.num_queries}")
    print(f"   - Enc/Dec Layers: {args.enc_layers}/{args.dec_layers}")
    print(f"   - Hidden Dim: {args.hidden_dim}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Device: {args.device}")
    print(f"   - Mixed Precision: {args.amp}")
    print(f"   - Dataset: {args.coco_path}")
    print("=" * 60)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device(args.device)
    
    # Build model
    print("\n🏗️  Building model...")
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model loaded: {n_parameters:,} trainable parameters")
    
    # Load datasets
    print(f"\n📚 Loading datasets from {args.coco_path}...")
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    base_ds = get_coco_api_from_dataset(dataset_val)
    
    print(f"   - Train: {len(dataset_train)} images")
    print(f"   - Val: {len(dataset_val)} images")
    
    # Create data loaders
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Build optimizer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                      if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Training loop
    print(f"\n🎯 Starting training for {args.epochs} epochs...")
    print("=" * 60)
    
    best_map = 0.0
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        
        print(f"\n📊 Epoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm
        )
        
        lr_scheduler.step()
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print("\n🔍 Running validation...")
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val,
                base_ds, device, args.output_dir, args=args, epoch=epoch
            )
            
            if 'bbox' in coco_evaluator.coco_eval:
                current_map = coco_evaluator.coco_eval['bbox'].stats[0]
                print(f"\n📈 Validation Results:")
                print(f"   - mAP@0.5:0.95: {current_map:.4f}")
                print(f"   - mAP@0.5: {coco_evaluator.coco_eval['bbox'].stats[1]:.4f}")
                print(f"   - mAP@0.75: {coco_evaluator.coco_eval['bbox'].stats[2]:.4f}")
                
                # Save best model
                if current_map > best_map:
                    best_map = current_map
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'mAP': best_map,
                    }
                    torch.save(checkpoint, output_dir / 'best_model.pth')
                    print(f"   ✅ New best mAP! Saved to {output_dir / 'best_model.pth'}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        epoch_time = time.time() - epoch_start
        print(f"\n⏱️  Epoch time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print(f"   - Total time: {total_time_str}")
    print(f"   - Best mAP: {best_map:.4f}")
    print(f"   - Model saved to: {output_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
