#!/usr/bin/env python3
"""Run Optuna trials for DroneDETR++ and export study summaries."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import optuna


def _read_best_metric(log_path, metric):
    """Return the best metric value found in a training log file."""
    if not log_path.exists():
        return None
    best = None
    with log_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if metric == "AP_tiny":
                val = row.get("test_AP_tiny", None)
            elif metric.startswith("bbox["):
                idx = int(metric[5:-1])
                bbox = row.get("test_coco_eval_bbox", None)
                val = float(bbox[idx]) if isinstance(bbox, list) and len(bbox) > idx else None
            else:
                val = row.get(metric, None)
            if val is None:
                continue
            val = float(val)
            if best is None or val > best:
                best = val
    return best


def _run_trial(train_script, root, base_args, trial_args, output_dir):
    """Launch one training run with merged base + trial arguments."""
    cmd = [sys.executable, str(train_script)]
    merged = {**base_args, **trial_args, "output_dir": str(output_dir)}
    for k, v in merged.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    subprocess.run(cmd, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for DroneDETR++")
    parser.add_argument("--train-script", default="train.py")
    parser.add_argument("--coco-path", default="data/subset")
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--study-name", default="dronedetr_optuna")
    parser.add_argument("--storage", default="", help="Optuna storage URL (optional, e.g. sqlite:///optuna.db)")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--metric", default="bbox[3]",
                        help="Target metric: bbox[0]=AP, bbox[1]=AP50, bbox[3]=AP_small, AP_tiny, etc.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="outputs/optuna")
    parser.add_argument("--batch-size-choices", default="2,4")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_script = root / args.train_script
    out_root = root / args.output_root / args.study_name
    out_root.mkdir(parents=True, exist_ok=True)
    batch_choices = [int(x.strip()) for x in args.batch_size_choices.split(",") if x.strip()]

    base_args = {
        "dataset_file": "coco",
        "coco_path": args.coco_path,
        "num_classes": args.num_classes,
        "device": args.device,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "enable_eval_dashboard": True,
        "tensorboard": True,
        "use_focal_loss": True,
        "class_weight_strategy": "effective_num_samples",
        "small_object_weighted_loss": True,
    }

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=(args.storage if args.storage else None),
        load_if_exists=True,
    )

    def objective(trial):
        # Sample optimization hyperparameters for this training run.
        trial_args = {
            "lr": trial.suggest_float("lr", 5e-5, 4e-4, log=True),
            "lr_backbone": trial.suggest_float("lr_backbone", 5e-6, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-5, 5e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", batch_choices),
            "num_queries": trial.suggest_categorical("num_queries", [50, 100, 150]),
            "small_obj_weight_factor": trial.suggest_float("small_obj_weight_factor", 2.0, 6.0),
            "small_obj_gamma": trial.suggest_float("small_obj_gamma", 1.0, 2.5),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.5, 3.0),
        }
        output_dir = out_root / f"trial_{trial.number:04d}"
        _run_trial(train_script, root, base_args, trial_args, output_dir)
        # Read the best value across epochs, not just the final epoch.
        metric_value = _read_best_metric(output_dir / "log.txt", args.metric)
        if metric_value is None:
            raise RuntimeError(f"Trial {trial.number} did not produce metric {args.metric}")
        trial.set_user_attr("output_dir", str(output_dir))
        return metric_value

    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "study_name": args.study_name,
        "metric": args.metric,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_trial": int(study.best_trial.number),
    }
    with (out_root / "best_params.json").open("w") as f:
        json.dump(best, f, indent=2)

    rows = []
    for t in study.trials:
        rows.append({
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "params": json.dumps(t.params),
            "output_dir": t.user_attrs.get("output_dir", ""),
        })
    with (out_root / "trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["number", "value", "state", "params", "output_dir"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value ({args.metric}): {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved: {out_root / 'best_params.json'}")


if __name__ == "__main__":
    main()
