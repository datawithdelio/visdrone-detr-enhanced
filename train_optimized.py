#!/usr/bin/env python3
"""Enhanced DETR training with improvements"""
import argparse
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
import utils.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Enhanced DETR', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_dir', default='outputs/optimized')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--frozen_weights', type=str, default=None)
    return parser

def main(args):
    print(f"🚀 Enhanced Training: {args.backbone}, queries={args.num_queries}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=utils.collate_fn, drop_last=True)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=utils.collate_fn)
    param_dicts = [{"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]}, {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone}]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_map = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if (epoch + 1) % 5 == 0:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, device)
            if 'bbox' in coco_evaluator.coco_eval:
                current_map = coco_evaluator.coco_eval['bbox'].stats[0]
                if current_map > best_map:
                    best_map = current_map
                    torch.save({'model': model.state_dict(), 'mAP': best_map}, Path(args.output_dir) / 'best_model.pth')
                    print(f"✅ Best mAP: {best_map:.4f}")
    print(f"🎉 Done! Best mAP: {best_map:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
