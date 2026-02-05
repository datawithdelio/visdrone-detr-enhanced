# src/utils/report.py
import os
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# -----------------------------
# [DRAW HELPERS]
# -----------------------------
def _xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    return [x, y, x + w, y + h]


def _draw_boxes_pil(img_pil, boxes_xyxy, labels=None, scores=None, color=(255, 0, 0), width=3):
    draw = ImageDraw.Draw(img_pil)
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        txt = ""
        if labels is not None:
            txt += str(labels[i])
        if scores is not None:
            txt += f" {scores[i]:.2f}"
        if txt.strip():
            draw.text((x1, max(0, y1 - 12)), txt, fill=color)
    return img_pil


# -----------------------------
# [SAVE VAL BATCH VISUALS]
# -----------------------------
@torch.no_grad()
def save_val_batch_visuals(model, val_loader, device, out_dir, idx_to_name=None,
                           score_thresh=0.3, max_images=8):
    """
    Saves two images:
      - val_batch0_labels.jpg : GT boxes
      - val_batch0_pred.jpg   : predicted boxes

    Works with DETR-style outputs:
      outputs = model(samples)
      outputs['pred_logits'], outputs['pred_boxes'] (normalized cxcywh in [0,1])
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    batch = next(iter(val_loader))
    samples, targets = batch

    samples = samples.to(device)
    outputs = model(samples)

    # DETR output format assumptions
    pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes+1]
    pred_boxes = outputs["pred_boxes"]    # [B, num_queries, 4] in cxcywh normalized

    probs = pred_logits.softmax(-1)
    scores, labels = probs[..., :-1].max(-1)  # ignore "no-object" last class

    # Unnormalize boxes to image pixels
    # targets usually contain "orig_size" or "size". We'll use orig_size if present.
    gt_imgs = []
    pred_imgs = []

    # Try to get PIL images from samples.tensors
    # If your dataset already returns PIL in targets, you can adjust.
    images = samples.tensors  # [B, 3, H, W]
    B = images.shape[0]

    for i in range(min(B, max_images)):
        img_t = images[i].cpu()
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # sizes
        if "orig_size" in targets[i]:
            h, w = targets[i]["orig_size"].tolist()
        elif "size" in targets[i]:
            h, w = targets[i]["size"].tolist()
        else:
            h, w = img_pil.size[1], img_pil.size[0]

        # --- GT ---
        gt = img_pil.copy()
        if "boxes" in targets[i]:
            # COCO usually stores GT as xyxy in pixels in many DETR repos;
            # if yours is normalized cxcywh, you must adapt here.
            gt_boxes = targets[i]["boxes"].cpu().numpy()
            # Assume GT boxes are xyxy pixel already:
            gt_boxes_xyxy = gt_boxes.tolist()
            gt_labels = targets[i].get("labels", None)
            if gt_labels is not None:
                gt_labels = gt_labels.cpu().tolist()
                if idx_to_name:
                    gt_labels = [idx_to_name.get(int(x), str(int(x))) for x in gt_labels]
            gt = _draw_boxes_pil(gt, gt_boxes_xyxy, labels=gt_labels, color=(0, 255, 0), width=3)
        gt_imgs.append(gt)

        # --- PRED ---
        pr = img_pil.copy()

        sc = scores[i].cpu().numpy()
        lb = labels[i].cpu().numpy()
        bx = pred_boxes[i].cpu().numpy()  # cxcywh normalized

        keep = sc >= score_thresh
        sc = sc[keep]
        lb = lb[keep]
        bx = bx[keep]

        # cxcywh normalized -> xyxy pixels
        boxes_xyxy = []
        label_names = []
        for (cx, cy, bw, bh), s, l in zip(bx, sc, lb):
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            boxes_xyxy.append([float(x1), float(y1), float(x2), float(y2)])

            name = int(l)
            if idx_to_name:
                name = idx_to_name.get(int(l), str(int(l)))
            label_names.append(str(name))

        pr = _draw_boxes_pil(pr, boxes_xyxy, labels=label_names, scores=sc.tolist(),
                             color=(255, 0, 0), width=3)
        pred_imgs.append(pr)

    # make simple grid
    def _save_grid(imgs, path, cols=4):
        rows = int(np.ceil(len(imgs) / cols))
        w, h = imgs[0].size
        canvas = Image.new("RGB", (cols * w, rows * h), (30, 30, 30))
        for k, im in enumerate(imgs):
            r, c = divmod(k, cols)
            canvas.paste(im, (c * w, r * h))
        canvas.save(path)

    _save_grid(gt_imgs, out_dir / "val_batch0_labels.jpg")
    _save_grid(pred_imgs, out_dir / "val_batch0_pred.jpg")


# -----------------------------
# [COCO METRICS + PR CURVES]
# -----------------------------
def save_coco_metrics_and_pr(coco_gt_json, coco_pred_json, out_dir,
                             class_names=None, iou_thr=0.50):
    """
    Given:
      - ground truth COCO json
      - prediction COCO json (list of dicts with image_id, category_id, bbox [x,y,w,h], score)
    Saves:
      - metrics.json
      - pr_curve_iou50.png (mean PR over all classes)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cocoGt = COCO(coco_gt_json)
    cocoDt = cocoGt.loadRes(coco_pred_json)

    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # Save numeric metrics
    metrics = {
        "AP@[.50:.95]": float(cocoEval.stats[0]),
        "AP@0.50": float(cocoEval.stats[1]),
        "AP@0.75": float(cocoEval.stats[2]),
        "AR@1": float(cocoEval.stats[6]),
        "AR@10": float(cocoEval.stats[7]),
        "AR@100": float(cocoEval.stats[8]),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # PR curve from COCOeval precision tensor:
    # precision: [T, R, K, A, M]
    precision = cocoEval.eval["precision"]
    ious = cocoEval.params.iouThrs
    # find iou index closest to iou_thr
    t = int(np.argmin(np.abs(ious - iou_thr)))

    # Average across classes K, area A=all(0), maxDets M=last index (usually 100)
    pr = precision[t, :, :, 0, -1]  # [R, K]
    pr_mean = np.nanmean(pr, axis=1)  # [R]
    recall = cocoEval.params.recThrs

    plt.figure(figsize=(7, 5))
    plt.plot(recall, pr_mean)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall (IoU={iou_thr})")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "pr_curve_iou50.png", dpi=200, bbox_inches="tight")
    plt.close()
