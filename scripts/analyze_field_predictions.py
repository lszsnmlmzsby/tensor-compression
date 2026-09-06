"""Verify rank-sharded predictions and report task/source/geometry/numeric errors."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def analyze(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids, pairs, counts = set(), {}, defaultdict(lambda: [0, 0, 0.0, 0])
    for shard in manifest["shards"]:
        path = manifest_path.parent / shard["file"]
        if path.parent != manifest_path.parent or path.name != shard["file"]:
            raise ValueError("Prediction shard must be a local filename.")
        digest = hashlib.sha256()
        shard_count = 0
        with path.open("rb") as reader:
            for line in reader:
                digest.update(line)
                row = json.loads(line)
                if row["qa_id"] in ids:
                    raise ValueError(f"Duplicate prediction: {row['qa_id']}.")
                ids.add(row["qa_id"])
                shard_count += 1
                keys = ["all", f"task:{row['task']}", f"source:{row['source']}",
                        f"source_task:{row['source']}|{row['task']}",
                        f"shape_task:{row['shape']}|{row['task']}",
                        f"source_shape_task:{row['source']}|{row['shape']}|{row['task']}",
                        f"field_task:{row['field']}|{row['task']}",
                        f"coordinate32_task:{row['coordinate_32']}|{row['task']}",
                        f"coordinate96_task:{row['coordinate_96']}|{row['task']}",
                        f"partition_task:{row.get('extrapolation_axis') or row['shape_partition']}|{row['task']}"]
                gap = row["diagnostic"].get("value_gap")
                if gap is not None:
                    band = "lt_0.25" if gap < 0.25 else "0.25_to_0.5" if gap < 0.5 else "0.5_to_1" if gap < 1 else "ge_1"
                    keys.append(f"gap_task:{band}|{row['task']}")
                for key in keys:
                    values = counts[key]
                    values[0] += int(row["correct"])
                    values[1] += 1
                    if "absolute_z_error" in row:
                        values[2] += row["absolute_z_error"]
                        values[3] += 1
                pair = row["diagnostic"].get("pair_id")
                if pair:
                    members = pairs.setdefault(pair, {})
                    if row["task"] in members:
                        raise ValueError(f"Repeated numeric pair member: {pair}.")
                    members[row["task"]] = row
        if digest.hexdigest() != shard["sha256"] or shard_count != shard["records"]:
            raise ValueError(f"Prediction shard is incomplete or modified: {path}.")
    if len(ids) != manifest["records"]:
        raise ValueError("Prediction count differs from the completed evaluation manifest.")
    paired = defaultdict(int)
    for members in pairs.values():
        norm, raw = members.get("normalized_point_value"), members.get("raw_point_value_with_stats")
        if norm is None or raw is None:
            paired["incomplete"] += 1
            continue
        paired["total"] += 1
        paired["same_value_rank"] += int(norm["chosen_value_rank"] == raw["chosen_value_rank"])
        paired["both_correct"] += int(norm["correct"] and raw["correct"])
        paired["norm_correct_raw_wrong"] += int(norm["correct"] and not raw["correct"])
        paired["norm_wrong_raw_correct"] += int(not norm["correct"] and raw["correct"])
        paired["both_wrong"] += int(not norm["correct"] and not raw["correct"])
    return {"records": len(ids), "numeric_pairs": dict(paired), "breakdowns": {
        key: {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1],
              "mean_absolute_z_error": value[2] / value[3] if value[3] else None}
        for key, value in sorted(counts.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = analyze(args.manifest)
    output = Path(args.output) if args.output else Path(args.manifest).with_suffix(".diagnostics.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key, counts in result["breakdowns"].items():
        if key == "all" or key.startswith(("source_task:", "coordinate32_task:", "partition_task:")):
            print(f"{key} accuracy={counts['accuracy']:.4f} n={counts['total']} z_mae={counts['mean_absolute_z_error']}")
    print("numeric_pairs=" + json.dumps(result["numeric_pairs"]))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
