#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _to_cli_flag(key):
    return f"--{key.replace('_', '-')}"


def _build_command(train_script, args_dict):
    cmd = [sys.executable, str(train_script)]
    for key, value in args_dict.items():
        flag = _to_cli_flag(key)
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])
    return cmd


def _read_best_metrics(output_dir):
    log_path = output_dir / "log.txt"
    if not log_path.exists():
        return {"best_AP": None, "best_AP50": None, "best_AP_small": None, "best_epoch": None}

    best = {"best_AP": -1.0, "best_AP50": None, "best_AP_small": None, "best_epoch": None}
    with log_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            bbox = row.get("test_coco_eval_bbox")
            if not bbox:
                continue
            ap = float(bbox[0])
            if ap > best["best_AP"]:
                best["best_AP"] = ap
                best["best_AP50"] = float(bbox[1])
                best["best_AP_small"] = float(bbox[3])
                best["best_epoch"] = int(row.get("epoch", -1))
    if best["best_AP"] < 0:
        return {"best_AP": None, "best_AP50": None, "best_AP_small": None, "best_epoch": None}
    return best


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments sequentially.")
    parser.add_argument("--config-dir", default="configs/ablations", help="Directory with ablation YAML files")
    parser.add_argument("--train-script", default="train.py", help="Training script path")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these experiment names")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_dir = root / args.config_dir
    train_script = root / args.train_script
    summary_dir = root / "outputs" / "ablations"
    summary_dir.mkdir(parents=True, exist_ok=True)

    configs = sorted(config_dir.glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No YAML files found in {config_dir}")

    results = []
    for cfg_path in configs:
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f)
        name = cfg["name"]
        if args.only and name not in args.only:
            continue
        exp_args = cfg["args"].copy()
        output_dir = root / cfg["output_dir"]
        exp_args["output_dir"] = str(output_dir)

        cmd = _build_command(train_script, exp_args)
        print(f"\n=== {name} ===")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=root, check=True)

        metrics = _read_best_metrics(output_dir)
        results.append({
            "name": name,
            "config": str(cfg_path.relative_to(root)),
            **metrics,
        })

    if not results:
        print("No experiments selected.")
        return

    baseline = next((r for r in results if r["name"] == "baseline"), None)
    baseline_ap = baseline["best_AP"] if baseline else None
    for r in results:
        if baseline_ap is not None and r["best_AP"] is not None:
            r["delta_AP_vs_baseline"] = r["best_AP"] - baseline_ap
        else:
            r["delta_AP_vs_baseline"] = None

    out_json = summary_dir / "summary.json"
    out_csv = summary_dir / "summary.csv"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "config",
                "best_epoch",
                "best_AP",
                "best_AP50",
                "best_AP_small",
                "delta_AP_vs_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved ablation summary to {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
