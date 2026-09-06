"""Print variable-shape run completion and per-shape task accuracy (no model needed)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TASKS = ("normalized_point_value", "raw_point_value_with_stats", "point_compare",
         "region_mean_compare", "extreme_quadrant")


def summarize(run_dir: Path, split: str = "val", csv_path: Path | None = None):
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"status={summary['status']} updates={summary['global_step']}/{summary['planned_updates']} "
              f"best_step={summary.get('best_step', 'unavailable')}")
    path = run_dir / f"final_{split}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Final {split} metrics are not available yet: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))["modes"]["correct"]
    for name, row in metrics["by_shape_partition"].items():
        print(f"{name}: {100 * row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    for name, row in metrics.get("by_source_task", {}).items():
        print(f"source_task={name}: {100 * row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    rows = []
    print("shape,n,accuracy," + ",".join(TASKS))
    for shape in sorted(metrics["by_shape"], key=lambda value: tuple(int(x) for x in value.split("x"))):
        item = metrics["by_shape"][shape]
        row = {"shape": shape, "n": item["total"], "accuracy": item["accuracy"]}
        for task in TASKS:
            row[task] = metrics["by_shape_task"][f"{shape}|{task}"]["accuracy"]
        rows.append(row)
        print(f"{shape},{row['n']}," + ",".join(f"{100 * row[key]:.2f}" for key in ("accuracy", *TASKS)))
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["shape", "n", "accuracy", *TASKS])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV (accuracy in [0,1]): {csv_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    summarize(args.run_dir, args.split, args.csv)


if __name__ == "__main__":
    main()
