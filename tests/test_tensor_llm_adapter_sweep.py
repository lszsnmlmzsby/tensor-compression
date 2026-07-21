from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_tensor_llm_adapter_sweep import (  # noqa: E402
    SweepRun,
    SweepSettings,
    build_training_command,
    find_run_dir,
    override_to_args,
)


class TestStage2SweepScheduler(unittest.TestCase):
    def settings(self, root: Path, gpus_per_run: int = 1) -> SweepSettings:
        return SweepSettings(
            training_script=root / "train.py",
            training_config=root / "pipeline.yaml",
            alignment_checkpoint=root / "alignment_best.pt",
            output_root=root / "runs",
            status_root=root / "sweeps",
            run_name_prefix="stage2_sweep",
            min_free_memory_mib=79000,
            poll_seconds=15.0,
            non_tty_status_seconds=300.0,
            gpus_per_run=gpus_per_run,
            max_concurrent_runs=8,
            excluded_gpus=frozenset(),
            common_overrides={"gradient_accumulation_steps": 5, "lr": 1.0e-4},
            parameter_names=("lr",),
            runs=(SweepRun("S001", {}),),
        )

    def test_override_conversion_handles_scalars_and_boolean_flags(self) -> None:
        self.assertEqual(override_to_args("lr", 5.0e-5), ["--lr", "5e-05"])
        self.assertEqual(override_to_args("append_eos", True), ["--append-eos"])
        self.assertEqual(override_to_args("append_eos", False), ["--no-append-eos"])

    def test_single_gpu_command_uses_python_and_numbered_run_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = self.settings(root)
            command = build_training_command(
                settings,
                SweepRun("S002", {"lr": 5.0e-5}),
                "stage2_sweep_S002",
            )

        self.assertEqual(command[0], sys.executable)
        self.assertNotIn("torch.distributed.run", command)
        self.assertEqual(command[command.index("--run-name") + 1], "stage2_sweep_S002")
        self.assertEqual(command[command.index("--gradient-accumulation-steps") + 1], "5")
        self.assertEqual(command[command.index("--lr") + 1], "5e-05")

    def test_multi_gpu_command_uses_torch_distributed_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self.settings(Path(temporary), gpus_per_run=4)
            command = build_training_command(settings, SweepRun("S001", {}), "stage2_sweep_S001")

        self.assertIn("torch.distributed.run", command)
        self.assertIn("--nproc_per_node=4", command)

    def test_run_directory_lookup_records_numbered_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "20260721_220000_stage2_sweep_S001"
            expected.mkdir()

            observed = find_run_dir(root, "stage2_sweep_S001", started_epoch=0.0)

        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
