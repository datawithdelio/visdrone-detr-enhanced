# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw

import utils.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from utils.drone_analysis import get_category_names, compute_error_report, export_dashboard


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    """Train DETR for a single epoch and return averaged metric scalars."""
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args=None, epoch=0):
    """Run validation, COCO evaluation, and optional dashboard/report exports."""
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    # -------------------------------
    # Save ONE validation batch visuals
    # -------------------------------
    first_samples = None
    first_targets = None
    first_results = None
    analysis_records = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # ✅ Save the FIRST val batch (GT vs Pred) ONLY ONCE
        if utils.is_main_process() and first_samples is None:
            # keep CPU copies
            first_samples = samples.detach().cpu()

            # Only keep needed keys; assumes DETR-style targets with normalized cxcywh boxes
            first_targets = []
            for t in targets:
                first_targets.append({
                    "orig_size": t["orig_size"].detach().cpu(),
                    "boxes": t["boxes"].detach().cpu(),
                })

            # postprocessor returns absolute xyxy boxes + scores + labels
            first_results = []
            for r in results:
                first_results.append({
                    "boxes": r["boxes"].detach().cpu(),
                    "scores": r["scores"].detach().cpu(),
                    "labels": r["labels"].detach().cpu(),
                })

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if args is not None and getattr(args, "enable_eval_dashboard", False):
            for t, r in zip(targets, results):
                H, W = [int(x) for x in t["orig_size"].tolist()]
                gt_xyxy = _cxcywh_to_xyxy_abs(t["boxes"], (H, W)).detach().cpu()
                analysis_records.append({
                    "gt_boxes": gt_xyxy,
                    "gt_labels": t["labels"].detach().cpu(),
                    "pred_boxes": r["boxes"].detach().cpu(),
                    "pred_labels": r["labels"].detach().cpu(),
                    "pred_scores": r["scores"].detach().cpu(),
                })

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # ✅ Write the report images at the end of eval
    if utils.is_main_process() and first_samples is not None:
        _save_val_batch_images(first_samples, first_targets, first_results, out_dir="outputs/report", max_images=8)

    if (
        utils.is_main_process()
        and args is not None
        and getattr(args, "enable_eval_dashboard", False)
        and coco_evaluator is not None
        and "bbox" in coco_evaluator.coco_eval
    ):
        category_names = get_category_names(base_ds)
        error_report = compute_error_report(
            analysis_records,
            num_classes=getattr(args, "num_classes", 91),
            score_thresh=getattr(args, "eval_score_thresh", 0.3),
            iou_thresh=0.5,
        )
        dashboard_summary = export_dashboard(
            output_dir=output_dir,
            epoch=epoch,
            coco_eval_bbox=coco_evaluator.coco_eval["bbox"],
            category_names=category_names,
            error_report=error_report,
        )
        stats["AP_tiny"] = dashboard_summary["AP_tiny"]

    return stats, coco_evaluator


# ======================================================
# VISUAL REPORT HELPERS (GT vs Pred)
# ======================================================

def _cxcywh_to_xyxy_abs(boxes, orig_size_hw):
    """Convert normalized cxcywh boxes to absolute xyxy format."""
    # boxes: Tensor [N,4] in normalized cx,cy,w,h
    H, W = orig_size_hw
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - bw / 2.0) * W
    y1 = (cy - bh / 2.0) * H
    x2 = (cx + bw / 2.0) * W
    y2 = (cy + bh / 2.0) * H
    return torch.stack([x1, y1, x2, y2], dim=-1)

def _unnormalize_img(img_chw):
    """Undo ImageNet normalization for PIL/JPEG export."""
    # DETR uses ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = img_chw.cpu() * std + mean
    x = (x.clamp(0, 1) * 255).byte()
    return x.permute(1, 2, 0).numpy()  # HWC uint8

def _draw_xyxy(draw, box, color=(255, 0, 0), w=3):
    """Draw one xyxy bounding box on a PIL ImageDraw canvas."""
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=w)

def _save_val_batch_images(first_samples, first_targets, first_results, out_dir, max_images=8):
    """Save grid images comparing validation GT and predictions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = first_samples.tensors  # [B,3,H,W]
    B = min(imgs.shape[0], max_images)

    gt_tiles = []
    pr_tiles = []

    for i in range(B):
        img_np = _unnormalize_img(imgs[i])
        img_pil = Image.fromarray(img_np)

        H, W = [int(x) for x in first_targets[i]["orig_size"].tolist()]

        # ---- GT ---- (green)
        gt_img = img_pil.copy()
        gt_draw = ImageDraw.Draw(gt_img)
        gt_boxes = first_targets[i]["boxes"]
        gt_xyxy = _cxcywh_to_xyxy_abs(gt_boxes, (H, W))
        for b in gt_xyxy:
            _draw_xyxy(gt_draw, b, color=(0, 255, 0), w=3)
        gt_tiles.append(gt_img)

        # ---- PRED ---- (red)
        pr_img = img_pil.copy()
        pr_draw = ImageDraw.Draw(pr_img)

        boxes = first_results[i]["boxes"]
        scores = first_results[i]["scores"]

        # Keep only confident boxes
        keep = scores >= 0.3
        boxes = boxes[keep]

        for b in boxes:
            _draw_xyxy(pr_draw, b, color=(255, 0, 0), w=3)
        pr_tiles.append(pr_img)

    def save_grid(images, path, cols=4):
        if len(images) == 0:
            return
        w, h = images[0].size
        rows = int(np.ceil(len(images) / cols))
        canvas = Image.new("RGB", (cols * w, rows * h), (30, 30, 30))
        for k, im in enumerate(images):
            r, c = divmod(k, cols)
            canvas.paste(im, (c * w, r * h))
        canvas.save(str(path))

    save_grid(gt_tiles, out_dir / "val_batch0_labels.jpg")
    save_grid(pr_tiles, out_dir / "val_batch0_pred.jpg")
