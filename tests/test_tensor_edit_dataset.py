from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.downstream.tensor_edit_dataset import TensorEditJsonlDataset


class TestTensorEditJsonlDataset(unittest.TestCase):
    def test_optionally_repairs_prompt_mojibake(self) -> None:
        clean_prompt = "\u52a0 0.2"
        mojibake_prompt = clean_prompt.encode("utf-8").decode("latin1")
        record = {
            "prompt": mojibake_prompt,
            "tensor": [[0.0, 1.0], [2.0, 3.0]],
            "label": [[0.0, 1.0], [2.0, 3.0]],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "samples.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            dataset = TensorEditJsonlDataset(
                jsonl_path=jsonl_path,
                input_size=(2, 2),
                channels=1,
                fix_prompt_mojibake=True,
            )
            sample = dataset[0]

        self.assertEqual(sample["raw_prompt"], mojibake_prompt)
        self.assertEqual(sample["prompt"], clean_prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
