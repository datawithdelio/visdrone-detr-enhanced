#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.ops import nms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models import build_model  # noqa: E402
from datasets.transforms import Normalize  # noqa: E402
import utils.misc as utils  # noqa: E402


def _get_model_args(cli_args):
    from main import get_args_parser  # noqa: E402
    parser = get_args_parser()
    args = parser.parse_args([])
    args.coco_path = cli_args.coco_path
    args.num_classes = cli_args.num_classes
    args.dataset_file = "coco"
    args.device = cli_args.device
    args.masks = False
    return args


def _load_model(ckpt_path, args):
    model, _, post = build_model(args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(args.device)
    model.eval()
    return model, post


def _iter_tiles(width, height, tile_size, overlap):
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = list(range(0, max(width - tile_size + 1, 1), stride))
    ys = list(range(0, max(height - tile_size + 1, 1), stride))
    if not xs or xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))
    if not ys or ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))
    for y in ys:
        for x in xs:
            yield x, y, min(tile_size, width - x), min(tile_size, height - y)


def _prepare_tile(img):
    t = TF.to_tensor(img)
    t = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t, None)[0]
    return t


@torch.no_grad()
def _predict_image(model, post, image_pil, device, tile_size, overlap, score_thresh, scales, nms_thr):
    all_boxes = []
    all_scores = []
    all_labels = []

    for scale in scales:
        scaled_w = max(1, int(round(image_pil.width * scale)))
        scaled_h = max(1, int(round(image_pil.height * scale)))
        scaled_img = image_pil.resize((scaled_w, scaled_h), Image.BILINEAR)

        for x, y, w, h in _iter_tiles(scaled_w, scaled_h, tile_size, overlap):
            tile = scaled_img.crop((x, y, x + w, y + h))
            tile_tensor = _prepare_tile(tile).to(device)
            samples = utils.nested_tensor_from_tensor_list([tile_tensor])
            outputs = model(samples)
            target_sizes = torch.tensor([[h, w]], device=device)
            pred = post["bbox"](outputs, target_sizes)[0]

            keep = pred["scores"] >= score_thresh
            boxes = pred["boxes"][keep]
            scores = pred["scores"][keep]
            labels = pred["labels"][keep]
            if boxes.numel() == 0:
                continue

            boxes[:, [0, 2]] += x
            boxes[:, [1, 3]] += y
            boxes /= scale
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=image_pil.width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=image_pil.height)

            all_boxes.append(boxes.cpu())
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

    if not all_boxes:
        return {"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": torch.empty((0,), dtype=torch.long)}

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    kept = []
    for cls in labels.unique():
        idx = torch.where(labels == cls)[0]
        cls_keep = nms(boxes[idx], scores[idx], nms_thr)
        kept.append(idx[cls_keep])
    keep = torch.cat(kept, dim=0) if kept else torch.empty((0,), dtype=torch.long)

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
    }


def main():
    parser = argparse.ArgumentParser(description="Tiled multi-scale inference for DroneDETR++")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True, help="Directory with images")
    parser.add_argument("--output-json", default="outputs/tiled_inference/predictions.json")
    parser.add_argument("--coco-path", default="data/subset")
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--scales", default="1.0", help="Comma-separated scales, e.g. 0.5,1.0,1.5")
    args = parser.parse_args()

    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]
    model_args = _get_model_args(args)
    model, post = _load_model(args.checkpoint, model_args)

    image_dir = Path(args.image_dir)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    out = []
    for p in images:
        image = Image.open(p).convert("RGB")
        pred = _predict_image(
            model=model,
            post=post,
            image_pil=image,
            device=model_args.device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            score_thresh=args.score_thresh,
            scales=scales,
            nms_thr=args.nms_thresh,
        )
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            x1, y1, x2, y2 = b
            out.append({
                "image_file": p.name,
                "category_id": int(l),
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(s),
            })

    with out_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(out)} detections to {out_json}")


if __name__ == "__main__":
    main()
