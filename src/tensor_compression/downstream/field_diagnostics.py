"""Offline prediction records and coordinate/error breakdowns; never model inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .variable_shape import record_shape, shape_name


def coordinate_bucket(target, boundary):
    if target.get("type") == "point":
        max_coordinate = max(target["row"] + 1, target["col"] + 1)
    elif target.get("type") in {"point_pair", "region_pair"}:
        points = [target["a"], target["b"]]
        # Include the full queried region, not just its top-left anchor.
        max_coordinate = max(max(p[0] + (p[2] if len(p) == 4 else 1),
                                 p[1] + (p[3] if len(p) == 4 else 1)) for p in points)
    else:
        return "global"
    return f"le_{boundary}" if max_coordinate <= boundary else f"gt_{boundary}"


def prediction_record(record, prediction, logits=None):
    meta = record["metadata"]
    diagnostic = meta.get("diagnostic", {})
    result = {"qa_id": record["qa_id"], "state_ref": record["state_ref"],
              "task": record["task_type"], "field": record["field"],
              "source": meta.get("source_kind", "real"), "shape": shape_name(record_shape(record)),
              "shape_partition": meta["shape_partition"], "extrapolation_axis": meta.get("extrapolation_axis"),
              "answer": record["answer"], "prediction": prediction, "correct": prediction == record["answer"],
              "question": record["query"], "choice_logits": logits,
              "diagnostic": diagnostic,
              "coordinate_32": coordinate_bucket(diagnostic.get("target", {}), 32),
              "coordinate_96": coordinate_bucket(diagnostic.get("target", {}), 96)}
    if "option_z" in diagnostic:
        chosen = diagnostic["option_z"][prediction]
        result.update(chosen_z=chosen, absolute_z_error=abs(chosen - diagnostic["target_z"]),
                      absolute_displayed_value_error=abs(diagnostic["option_values"][prediction] - diagnostic["target_displayed_value"]))
        # Ordering is invariant to positive affine raw-value conversion and label permutation.
        ordered = sorted(diagnostic["option_z"], key=diagnostic["option_z"].__getitem__)
        result["chosen_value_rank"] = ordered.index(prediction)
    return result


def write_prediction_shard(prefix, mode, rows, rank):
    path = Path(f"{prefix}.{mode}.rank{rank:05d}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as writer:
        for row in rows:
            line = (json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
            digest.update(line)
            writer.write(line)
    temporary.replace(path)
    return {"file": path.name, "records": len(rows), "sha256": digest.hexdigest()}
