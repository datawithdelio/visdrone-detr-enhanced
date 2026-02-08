import contextlib
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

from utils import box_ops


def get_category_names(coco_gt):
    if coco_gt is None:
        return {}
    names = {}
    for cat_id, cat in coco_gt.cats.items():
        names[int(cat_id)] = cat.get("name", str(cat_id))
    return names


def per_class_ap(coco_eval_bbox):
    precision = coco_eval_bbox.eval["precision"]  # [T, R, K, A, M]
    cat_ids = list(coco_eval_bbox.params.catIds)
    rows = []
    for k, cat_id in enumerate(cat_ids):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        ap = float(np.mean(p)) if p.size else float("nan")

        p_small = precision[:, :, k, 1, -1]
        p_small = p_small[p_small > -1]
        ap_small = float(np.mean(p_small)) if p_small.size else float("nan")

        p_medium = precision[:, :, k, 2, -1]
        p_medium = p_medium[p_medium > -1]
        ap_medium = float(np.mean(p_medium)) if p_medium.size else float("nan")

        p_large = precision[:, :, k, 3, -1]
        p_large = p_large[p_large > -1]
        ap_large = float(np.mean(p_large)) if p_large.size else float("nan")
        rows.append({
            "category_id": int(cat_id),
            "AP": ap,
            "AP_small": ap_small,
            "AP_medium": ap_medium,
            "AP_large": ap_large,
        })
    return rows


def tiny_ap(coco_eval_bbox, tiny_max_area=32 ** 2):
    coco_gt = coco_eval_bbox.cocoGt
    coco_dt = coco_eval_bbox.cocoDt
    if coco_dt is None:
        return float("nan")
    tiny_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    tiny_eval.params.imgIds = list(coco_eval_bbox.params.imgIds)
    tiny_eval.params.catIds = list(coco_eval_bbox.params.catIds)
    tiny_eval.params.maxDets = list(coco_eval_bbox.params.maxDets)
    tiny_eval.params.areaRng = [[0 ** 2, tiny_max_area]]
    tiny_eval.params.areaRngLbl = ["tiny"]
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            tiny_eval.evaluate()
            tiny_eval.accumulate()
    return float(tiny_eval.stats[0])


def export_dashboard(output_dir, epoch, coco_eval_bbox, category_names, error_report):
    out_dir = Path(output_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = coco_eval_bbox.stats
    summary = {
        "epoch": int(epoch),
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "AP_tiny": tiny_ap(coco_eval_bbox),
    }

    rows = per_class_ap(coco_eval_bbox)
    for row in rows:
        row["category_name"] = category_names.get(row["category_id"], str(row["category_id"]))

    json_path = out_dir / f"metrics_epoch_{epoch:03d}.json"
    csv_path = out_dir / f"per_class_epoch_{epoch:03d}.csv"
    error_path = out_dir / f"errors_epoch_{epoch:03d}.json"

    with json_path.open("w") as f:
        json.dump({"summary": summary, "per_class": rows}, f, indent=2)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category_id", "category_name", "AP", "AP_small", "AP_medium", "AP_large"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with error_path.open("w") as f:
        json.dump(error_report, f, indent=2)

    latest = out_dir / "latest_metrics.json"
    with latest.open("w") as f:
        json.dump({"summary": summary, "per_class": rows, "error_report": error_report}, f, indent=2)

    return summary


def _safe_iou_matrix(gt_xyxy, pred_xyxy):
    if gt_xyxy.numel() == 0 or pred_xyxy.numel() == 0:
        return torch.zeros((gt_xyxy.shape[0], pred_xyxy.shape[0]), dtype=torch.float32)
    iou, _ = box_ops.box_iou(gt_xyxy, pred_xyxy)
    return iou


def compute_error_report(records, num_classes, score_thresh=0.3, iou_thresh=0.5):
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    tp = np.zeros((num_classes,), dtype=np.int64)
    fp = np.zeros((num_classes,), dtype=np.int64)
    fn = np.zeros((num_classes,), dtype=np.int64)
    localization_errors = np.zeros((num_classes,), dtype=np.int64)

    for rec in records:
        gt_boxes = rec["gt_boxes"]
        gt_labels = rec["gt_labels"]
        pred_boxes = rec["pred_boxes"]
        pred_labels = rec["pred_labels"]
        pred_scores = rec["pred_scores"]

        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        ious = _safe_iou_matrix(gt_boxes, pred_boxes)
        matched_gt = set()
        matched_pred = set()

        if ious.numel() > 0:
            while True:
                iou_flat = ious.clone()
                if matched_gt:
                    iou_flat[list(matched_gt), :] = -1
                if matched_pred:
                    iou_flat[:, list(matched_pred)] = -1
                max_iou, flat_idx = torch.max(iou_flat.view(-1), dim=0)
                if max_iou.item() < iou_thresh:
                    break
                g = int(flat_idx.item() // iou_flat.shape[1])
                p = int(flat_idx.item() % iou_flat.shape[1])
                matched_gt.add(g)
                matched_pred.add(p)

                gt_cls = int(gt_labels[g].item()) - 1
                pred_cls = int(pred_labels[p].item()) - 1
                if 0 <= gt_cls < num_classes and 0 <= pred_cls < num_classes:
                    conf[gt_cls, pred_cls] += 1
                    if gt_cls == pred_cls:
                        tp[gt_cls] += 1
                    else:
                        fp[pred_cls] += 1
                        fn[gt_cls] += 1

        for g in range(gt_boxes.shape[0]):
            gt_cls = int(gt_labels[g].item()) - 1
            if g in matched_gt:
                continue
            if not (0 <= gt_cls < num_classes):
                continue
            fn[gt_cls] += 1
            if pred_boxes.shape[0] > 0:
                same_cls = (pred_labels == gt_labels[g]).nonzero(as_tuple=False).flatten()
                if same_cls.numel() > 0:
                    best = torch.max(ious[g, same_cls]).item()
                    if 0.1 <= best < iou_thresh:
                        localization_errors[gt_cls] += 1

        for p in range(pred_boxes.shape[0]):
            if p in matched_pred:
                continue
            pred_cls = int(pred_labels[p].item()) - 1
            if 0 <= pred_cls < num_classes:
                fp[pred_cls] += 1

    per_class = []
    for c in range(num_classes):
        prec = float(tp[c] / max(tp[c] + fp[c], 1))
        rec = float(tp[c] / max(tp[c] + fn[c], 1))
        per_class.append({
            "class_id": c + 1,
            "tp": int(tp[c]),
            "fp": int(fp[c]),
            "fn": int(fn[c]),
            "precision": prec,
            "recall": rec,
            "localization_errors": int(localization_errors[c]),
        })

    top_confusions = []
    for gt in range(num_classes):
        for pred in range(num_classes):
            if gt == pred:
                continue
            cnt = int(conf[gt, pred])
            if cnt > 0:
                top_confusions.append({"gt_class_id": gt + 1, "pred_class_id": pred + 1, "count": cnt})
    top_confusions.sort(key=lambda x: x["count"], reverse=True)

    return {
        "score_thresh": float(score_thresh),
        "iou_thresh": float(iou_thresh),
        "confusion_matrix": conf.tolist(),
        "per_class": per_class,
        "top_confusions": top_confusions[:20],
    }
