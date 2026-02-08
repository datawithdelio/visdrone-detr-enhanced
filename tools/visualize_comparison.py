#!/usr/bin/env python3
"""Create side-by-side validation visualizations for baseline vs improved models."""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from datasets import build_dataset, get_coco_api_from_dataset  # noqa: E402
from models import build_model  # noqa: E402
import utils.misc as utils  # noqa: E402


def _unnormalize_img(img_chw):
    """Undo ImageNet normalization for visualization output."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_chw.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_chw.device).view(3, 1, 1)
    x = img_chw * std + mean
    x = (x.clamp(0, 1) * 255).byte().cpu()
    return x.permute(1, 2, 0).numpy()


def _cxcywh_to_xyxy_abs(boxes, orig_size_hw):
    """Convert normalized DETR boxes to absolute xyxy coordinates."""
    H, W = orig_size_hw
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - bw / 2.0) * W
    y1 = (cy - bh / 2.0) * H
    x2 = (cx + bw / 2.0) * W
    y2 = (cy + bh / 2.0) * H
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _draw_boxes(img, boxes, color, width=3):
    """Draw a list of xyxy boxes on a PIL image."""
    draw = ImageDraw.Draw(img)
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img


def _load_model(ckpt_path, args):
    """Load checkpoint weights into a DETR model instance."""
    model, _, postprocessors = build_model(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(args.device)
    model.eval()
    return model, postprocessors


def _build_args(parser, cli_args):
    """Merge CLI overrides into default DETR parser args."""
    args = parser.parse_args([])
    for k, v in vars(cli_args).items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)
    return args


def main():
    parser = argparse.ArgumentParser(description="Visual comparison: baseline vs DroneDETR++")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--improved-checkpoint", required=True)
    parser.add_argument("--coco-path", default="data/subset")
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--dataset-file", default="coco")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/comparison")
    parser.add_argument("--num-images", type=int, default=24)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    args = parser.parse_args()

    from main import get_args_parser  # noqa: E402
    detr_parser = get_args_parser()
    common = _build_args(detr_parser, args)
    common.coco_path = args.coco_path
    common.num_classes = args.num_classes
    common.dataset_file = args.dataset_file
    common.output_dir = args.output_dir
    common.device = args.device

    model_a, post_a = _load_model(args.baseline_checkpoint, common)
    model_b, post_b = _load_model(args.improved_checkpoint, common)

    ds_val = build_dataset("val", common)
    _ = get_coco_api_from_dataset(ds_val)
    loader = DataLoader(ds_val, batch_size=1, sampler=torch.utils.data.SequentialSampler(ds_val),
                        collate_fn=utils.collate_fn, num_workers=0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(common.device)
            t = targets[0]
            out_a = model_a(samples)
            out_b = model_b(samples)

            orig_target_sizes = torch.stack([tt["orig_size"] for tt in targets], dim=0).to(common.device)
            res_a = post_a["bbox"](out_a, orig_target_sizes)[0]
            res_b = post_b["bbox"](out_b, orig_target_sizes)[0]

            img_np = _unnormalize_img(samples.tensors[0])
            base = Image.fromarray(img_np)
            H, W = [int(x) for x in t["orig_size"].tolist()]

            gt_img = base.copy()
            gt_boxes = _cxcywh_to_xyxy_abs(t["boxes"], (H, W))
            _draw_boxes(gt_img, gt_boxes, color=(0, 255, 0), width=3)

            a_img = base.copy()
            keep_a = res_a["scores"] >= args.score_thresh
            _draw_boxes(a_img, res_a["boxes"][keep_a].cpu(), color=(255, 0, 0), width=3)

            b_img = base.copy()
            keep_b = res_b["scores"] >= args.score_thresh
            _draw_boxes(b_img, res_b["boxes"][keep_b].cpu(), color=(0, 128, 255), width=3)

            # Panel layout: GT | baseline | improved.
            panel = Image.new("RGB", (base.width * 3, base.height), (25, 25, 25))
            panel.paste(gt_img, (0, 0))
            panel.paste(a_img, (base.width, 0))
            panel.paste(b_img, (base.width * 2, 0))
            panel.save(out_dir / f"compare_{count:03d}.jpg")

            count += 1
            if count >= args.num_images:
                break

    meta = {
        "baseline_checkpoint": args.baseline_checkpoint,
        "improved_checkpoint": args.improved_checkpoint,
        "num_images": count,
        "score_thresh": args.score_thresh,
        "layout": "left=ground_truth, middle=baseline, right=improved",
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {count} comparison images to {out_dir}")


if __name__ == "__main__":
    main()
