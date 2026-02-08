#!/usr/bin/env python3
"""Plot loss and AP curves from line-delimited JSON training logs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_log(log_path):
    """Load all JSON records from ``log.txt``."""
    rows = []
    with log_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _series(rows, key):
    """Extract epoch-aligned x/y lists for one scalar metric key."""
    xs, ys = [], []
    for r in rows:
        if key in r and isinstance(r[key], (int, float)):
            xs.append(int(r["epoch"]))
            ys.append(float(r[key]))
    return xs, ys


def _plot_multi(out_path, title, data, ylabel):
    """Render one chart with multiple named series."""
    plt.figure(figsize=(9, 5))
    for label, (xs, ys) in data.items():
        if xs and ys:
            plt.plot(xs, ys, label=label, linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    if any(len(v[0]) > 0 for v in data.values()):
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from log.txt")
    parser.add_argument("--log-file", required=True, help="Path to output_dir/log.txt")
    parser.add_argument("--output-dir", default="", help="Output dir for plots (default: <run>/plots)")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    rows = _load_log(log_path)
    if not rows:
        raise RuntimeError(f"No rows found in {log_path}")

    out_dir = Path(args.output_dir) if args.output_dir else (log_path.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss breakdown helps distinguish optimization issues from regression in AP.
    losses = {
        "train_loss": _series(rows, "train_loss"),
        "test_loss": _series(rows, "test_loss"),
        "train_loss_ce": _series(rows, "train_loss_ce"),
        "train_loss_bbox": _series(rows, "train_loss_bbox"),
        "train_loss_giou": _series(rows, "train_loss_giou"),
    }
    _plot_multi(out_dir / "loss_curves.png", "Loss Curves", losses, "Loss")

    # COCO evaluator stores standard bbox metrics in fixed index order.
    ap, ap50, ap75, aps, apm, apl, epochs = [], [], [], [], [], [], []
    for r in rows:
        bbox = r.get("test_coco_eval_bbox", None)
        if isinstance(bbox, list) and len(bbox) >= 6:
            epochs.append(int(r["epoch"]))
            ap.append(float(bbox[0]))
            ap50.append(float(bbox[1]))
            ap75.append(float(bbox[2]))
            aps.append(float(bbox[3]))
            apm.append(float(bbox[4]))
            apl.append(float(bbox[5]))

    metrics = {
        "AP": (epochs, ap),
        "AP50": (epochs, ap50),
        "AP75": (epochs, ap75),
        "AP_small": (epochs, aps),
        "AP_medium": (epochs, apm),
        "AP_large": (epochs, apl),
    }
    _plot_multi(out_dir / "ap_curves.png", "Detection AP Curves", metrics, "AP")

    if any("test_AP_tiny" in r for r in rows):
        tiny = _series(rows, "test_AP_tiny")
        _plot_multi(out_dir / "ap_tiny_curve.png", "Tiny Object AP", {"AP_tiny": tiny}, "AP_tiny")

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
