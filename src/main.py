# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Main training/evaluation entrypoint used by ``train.py``.

This file extends the original DETR CLI with DroneDETR++ options for
small-object weighting, focal loss, and dashboard exports.
"""

import argparse
import datetime
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import datasets
import utils.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def _load_coco_json(coco_path, split):
    ann_path = Path(coco_path) / "annotations" / f"instances_{split}2017.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {ann_path}")
    with ann_path.open("r") as f:
        data = json.load(f)
    return data, ann_path


def _validate_coco_split_structure(coco_data, ann_path, expected_num_classes):
    categories = coco_data.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in {ann_path}")

    cat_ids = sorted(int(c["id"]) for c in categories if "id" in c)
    expected_ids = list(range(1, len(categories) + 1))
    if cat_ids != expected_ids:
        raise ValueError(
            f"Non-contiguous category ids in {ann_path}. "
            f"Expected {expected_ids[:5]}... got {cat_ids[:5]}..."
        )

    if len(categories) != expected_num_classes:
        raise ValueError(
            f"num_classes={expected_num_classes} does not match categories={len(categories)} in {ann_path}"
        )

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    if not images:
        raise ValueError(f"No images found in {ann_path}")
    if not annotations:
        raise ValueError(f"No annotations found in {ann_path}")

    valid_cat_ids = set(cat_ids)
    image_ids = {int(i["id"]) for i in images if "id" in i}
    anns_per_img = defaultdict(int)
    bad_bbox = 0

    for ann in annotations:
        cat_id = int(ann.get("category_id", -1))
        if cat_id not in valid_cat_ids:
            raise ValueError(f"Invalid category_id={cat_id} in {ann_path}")

        img_id = int(ann.get("image_id", -1))
        if img_id not in image_ids:
            raise ValueError(f"Annotation references missing image_id={img_id} in {ann_path}")

        bbox = ann.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox format in {ann_path}: {bbox}")

        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            bad_bbox += 1
        anns_per_img[img_id] += 1

    if bad_bbox > 0:
        raise ValueError(f"Found {bad_bbox} non-positive width/height boxes in {ann_path}")

    labeled_images = sum(1 for img_id in image_ids if anns_per_img.get(img_id, 0) > 0)
    if labeled_images == 0:
        raise ValueError(f"No labeled images found in {ann_path}")

    print(
        f"✅ {ann_path.name}: images={len(images)}, anns={len(annotations)}, "
        f"categories={len(categories)}, labeled_images={labeled_images}"
    )


def _compute_class_weights_from_coco(args):
    """Compute per-class weights from COCO annotations for imbalance handling."""
    if args.dataset_file != "coco":
        return None
    if args.class_weight_strategy == "none":
        return None

    ann_path = Path(args.coco_path) / "annotations" / "instances_train2017.json"
    if not ann_path.exists():
        print(f"⚠️  class weight computation skipped (missing {ann_path})")
        return None

    with ann_path.open("r") as f:
        coco = json.load(f)
    counts = [0 for _ in range(args.num_classes)]
    for ann in coco.get("annotations", []):
        cat_id = int(ann["category_id"])
        if 1 <= cat_id <= args.num_classes:
            counts[cat_id - 1] += 1

    total = sum(counts)
    if total == 0:
        print("⚠️  class weight computation skipped (no annotations)")
        return None

    eps = 1e-6
    if args.class_weight_strategy == "frequency":
        weights = [total / (args.num_classes * max(c, 1)) for c in counts]
    elif args.class_weight_strategy == "effective_num_samples":
        beta = args.effective_num_beta
        weights = [(1 - beta) / (1 - math.pow(beta, max(c, 1)) + eps) for c in counts]
    else:
        raise ValueError(f"Unsupported class_weight_strategy: {args.class_weight_strategy}")

    mean_w = sum(weights) / len(weights)
    weights = [w / (mean_w + eps) for w in weights]
    return weights


def _validate_coco_configuration(args):
    """Fail fast on common COCO/VisDrone config mistakes that lead to zero AP."""
    if args.dataset_file != "coco" or not args.coco_path:
        return

    ann_path = Path(args.coco_path) / "annotations" / "instances_train2017.json"
    if not ann_path.exists():
        print(f"⚠️  COCO config check skipped (missing {ann_path})")
        return

    with ann_path.open("r") as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in {ann_path}")

    dataset_num_classes = len(categories)
    if args.num_classes != dataset_num_classes:
        raise ValueError(
            f"num_classes={args.num_classes} does not match dataset categories={dataset_num_classes}. "
            f"Use --num_classes {dataset_num_classes}."
        )

    cat_ids = sorted(int(c["id"]) for c in categories if "id" in c)
    expected_ids = list(range(1, dataset_num_classes + 1))
    if cat_ids != expected_ids:
        print(
            "⚠️  Non-contiguous category ids detected in annotations. "
            "Ensure label mapping is consistent with training/evaluation."
        )

    if args.num_queries < 50:
        print(
            f"⚠️  num_queries={args.num_queries} is very low for dense drone scenes; "
            "this can severely cap recall."
        )


def _run_strict_data_checks(args):
    """Validate train/val COCO JSON structure and bbox/category consistency."""
    if args.dataset_file != "coco":
        return
    train_data, train_ann = _load_coco_json(args.coco_path, "train")
    val_data, val_ann = _load_coco_json(args.coco_path, "val")
    _validate_coco_split_structure(train_data, train_ann, args.num_classes)
    _validate_coco_split_structure(val_data, val_ann, args.num_classes)


def get_args_parser():
    """Build argument parser shared across training utilities."""
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use focal loss for classification instead of cross-entropy')
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help='Focal loss focusing parameter')
    parser.add_argument('--class_weight_strategy', default='none', type=str,
                        choices=['none', 'frequency', 'effective_num_samples'],
                        help='How to compute class weights used in focal alpha')
    parser.add_argument('--effective_num_beta', default=0.999, type=float,
                        help='Beta for effective number of samples weighting')
    parser.add_argument('--small_object_weighted_loss', action='store_true',
                        help='Upweight box losses for small objects')
    parser.add_argument('--small_obj_area_thresh', default=0.001, type=float,
                        help='Normalized area threshold where small-object upweighting starts')
    parser.add_argument('--small_obj_weight_factor', default=3.0, type=float,
                        help='Scaling factor for small-object loss weighting')
    parser.add_argument('--small_obj_gamma', default=1.5, type=float,
                        help='Exponent for size-aware weighting curve')
    parser.add_argument('--small_obj_max_weight', default=10.0, type=float,
                        help='Maximum per-box upweight for small objects')
    parser.add_argument('--eval_score_thresh', default=0.3, type=float,
                        help='Score threshold for error analysis outputs')
    parser.add_argument('--enable_eval_dashboard', action='store_true',
                        help='Export size-aware metrics and per-class analysis JSON/CSV')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard scalar logging')
    parser.add_argument('--tensorboard_dir', default='',
                        help='Directory for TensorBoard logs (defaults to <output_dir>/tensorboard)')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', type=str)

    # ✅ FIX: some parts of this repo expect args.coco_path
    parser.add_argument('--coco_path', type=str, default=None, help='Path to COCO dataset')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of object classes')

    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--strict_data_checks', action='store_true',
                        help='Run strict COCO JSON checks for category ids and bbox validity before training')
    parser.add_argument('--overfit_subset_size', default=0, type=int,
                        help='If > 0, train/eval on only the first N images for pipeline debugging')
    parser.add_argument('--eval_train_split', action='store_true',
                        help='Also evaluate on train split each epoch to diagnose val-label issues')
    parser.add_argument('--debug_disable_train_augment', action='store_true',
                        help='Use val transforms for train split (helps 1-5 image overfit debugging)')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    """Run training or evaluation based on parsed CLI arguments."""
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    _validate_coco_configuration(args)
    if args.strict_data_checks:
        _run_strict_data_checks(args)

    # Precompute class weights once and pass them through args.
    args.class_weights = _compute_class_weights_from_coco(args)
    if args.class_weights is not None:
        print("✅ Computed class weights for imbalance handling")

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    if args.overfit_subset_size > 0:
        overfit_n = min(args.overfit_subset_size, len(dataset_train), len(dataset_val))
        subset_idx = list(range(overfit_n))
        dataset_train = torch.utils.data.Subset(dataset_train, subset_idx)
        dataset_val = torch.utils.data.Subset(dataset_val, subset_idx)
        print(f"🧪 Overfit debug mode enabled: using first {overfit_n} images for train+val")

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
    train_ds_for_eval = get_coco_api_from_dataset(dataset_train) if args.eval_train_split else None

    data_loader_train_eval = None
    if args.eval_train_split:
        data_loader_train_eval = DataLoader(
            dataset_train,
            args.batch_size,
            sampler=torch.utils.data.SequentialSampler(dataset_train),
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    writer = None
    if args.tensorboard and args.output_dir:
        tb_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else (output_dir / "tensorboard")
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

    if args.resume:
        # Resume supports both local checkpoint paths and URL checkpoints.
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        # Drop heads that depend on class count / query count when resuming.
        del checkpoint["model"]["class_embed.weight"]
        del checkpoint["model"]["class_embed.bias"]
        del checkpoint["model"]["query_embed.weight"]

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, args=args, epoch=args.start_epoch)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args=args, epoch=epoch
        )
        train_eval_stats = {}
        if args.eval_train_split:
            train_eval_stats, _ = evaluate(
                model, criterion, postprocessors, data_loader_train_eval, train_ds_for_eval,
                device, args.output_dir, args=args, epoch=epoch
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     **{f'train_eval_{k}': v for k, v in train_eval_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if writer is not None and utils.is_main_process():
            # Keep TensorBoard output scalar-only to avoid serialization issues.
            for k, v in log_stats.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(k, v, epoch)
            bbox = log_stats.get("test_coco_eval_bbox", None)
            if isinstance(bbox, list) and len(bbox) >= 6:
                writer.add_scalar("metrics/AP", float(bbox[0]), epoch)
                writer.add_scalar("metrics/AP50", float(bbox[1]), epoch)
                writer.add_scalar("metrics/AP75", float(bbox[2]), epoch)
                writer.add_scalar("metrics/AP_small", float(bbox[3]), epoch)
                writer.add_scalar("metrics/AP_medium", float(bbox[4]), epoch)
                writer.add_scalar("metrics/AP_large", float(bbox[5]), epoch)
            writer.flush()

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # ✅ FIX: default coco_path to data_path so datasets/coco.py can find it
    if getattr(args, "coco_path", None) is None:
        args.coco_path = args.data_path

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
