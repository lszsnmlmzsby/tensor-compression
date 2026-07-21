from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
SUPPORTED_OVERRIDES = {
    "append_eos",
    "batch_size",
    "ce_loss_weight",
    "choice_ce_loss_weight",
    "diagnostics_enabled",
    "dropout",
    "epochs",
    "eval_batch_size",
    "eval_choice_batch_size",
    "global_prompt_dropout",
    "grad_clip_norm",
    "gradient_accumulation_steps",
    "initial_eval_records",
    "local_gate_init",
    "local_text_gate_init",
    "lr",
    "max_test_records",
    "max_train_records",
    "max_val_records",
    "ranking_loss_margin",
    "ranking_loss_weight",
    "seed",
    "shuffle_seed",
    "swapped_question_loss_margin",
    "swapped_question_loss_weight",
    "warmup_ratio",
    "weight_decay",
}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def compact_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def override_to_args(name: str, value: Any) -> list[str]:
    if name not in SUPPORTED_OVERRIDES:
        raise ValueError(
            f"Unsupported sweep override {name!r}. Supported values: {sorted(SUPPORTED_OVERRIDES)}"
        )
    flag = "--" + name.replace("_", "-")
    if isinstance(value, bool):
        return [flag if value else "--no-" + name.replace("_", "-")]
    if value is None:
        return []
    return [flag, str(value)]


@dataclass(frozen=True)
class SweepRun:
    sweep_id: str
    overrides: dict[str, Any]


@dataclass
class RunningJob:
    spec: SweepRun
    run_name: str
    gpu_ids: tuple[int, ...]
    process: subprocess.Popen[Any]
    log_handle: TextIO
    log_path: Path
    started_at: str
    started_epoch: float
    run_dir: Path | None = None


@dataclass(frozen=True)
class SweepSettings:
    training_script: Path
    training_config: Path
    alignment_checkpoint: Path
    output_root: Path
    status_root: Path
    run_name_prefix: str
    min_free_memory_mib: int
    poll_seconds: float
    non_tty_status_seconds: float
    gpus_per_run: int
    max_concurrent_runs: int
    excluded_gpus: frozenset[int]
    common_overrides: dict[str, Any]
    parameter_names: tuple[str, ...]
    runs: tuple[SweepRun, ...]


class TerminalStatus:
    def __init__(self, non_tty_interval: float) -> None:
        self.dynamic = bool(sys.stdout.isatty())
        self.non_tty_interval = max(1.0, float(non_tty_interval))
        self.last_non_tty_status = 0.0
        self.last_width = 0

    def status(self, message: str) -> None:
        if self.dynamic:
            padded = message.ljust(self.last_width)
            self.last_width = max(self.last_width, len(message))
            sys.stdout.write("\r" + padded)
            sys.stdout.flush()
            return
        now = time.monotonic()
        if now - self.last_non_tty_status >= self.non_tty_interval:
            print(message, flush=True)
            self.last_non_tty_status = now

    def event(self, message: str) -> None:
        if self.dynamic and self.last_width:
            sys.stdout.write("\r" + (" " * self.last_width) + "\r")
            sys.stdout.flush()
        print(message, flush=True)
        self.last_width = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for free GPUs and schedule numbered tensor-LLM Stage-2 sweep runs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tensor_llm_adapter_stage2_sweep.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-free-memory-mib", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=None)
    parser.add_argument("--gpus-per-run", type=int, default=None)
    parser.add_argument("--max-concurrent-runs", type=int, default=None)
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> tuple[SweepSettings, dict[str, Any]]:
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    if not isinstance(raw_config, dict) or not isinstance(raw_config.get("sweep"), dict):
        raise ValueError(f"Sweep config must contain a mapping named 'sweep': {config_path}")
    raw = raw_config["sweep"]
    common_overrides = dict(raw.get("common_overrides") or {})
    unknown_common = sorted(set(common_overrides) - SUPPORTED_OVERRIDES)
    if unknown_common:
        raise ValueError(f"Unsupported common_overrides: {unknown_common}")
    parameter_names = tuple(str(name) for name in raw.get("parameters") or ())
    if not parameter_names:
        raise ValueError("sweep.parameters must explicitly list the parameters being swept.")
    unknown_parameters = sorted(set(parameter_names) - SUPPORTED_OVERRIDES)
    if unknown_parameters:
        raise ValueError(f"Unsupported sweep.parameters: {unknown_parameters}")

    runs: list[SweepRun] = []
    seen_ids: set[str] = set()
    for item in raw.get("runs") or ():
        if not isinstance(item, dict):
            raise TypeError("Each sweep.runs entry must be a mapping.")
        sweep_id = str(item.get("id", "")).strip()
        if not SWEEP_ID_PATTERN.fullmatch(sweep_id):
            raise ValueError(f"Invalid sweep id: {sweep_id!r}")
        if sweep_id in seen_ids:
            raise ValueError(f"Duplicate sweep id: {sweep_id}")
        seen_ids.add(sweep_id)
        overrides = dict(item.get("overrides") or {})
        unknown = sorted(set(overrides) - SUPPORTED_OVERRIDES)
        if unknown:
            raise ValueError(f"{sweep_id} has unsupported overrides: {unknown}")
        merged = {**common_overrides, **overrides}
        missing = [name for name in parameter_names if name not in merged]
        if missing:
            raise ValueError(f"{sweep_id} does not define listed sweep parameters: {missing}")
        runs.append(SweepRun(sweep_id=sweep_id, overrides=overrides))
    if not runs:
        raise ValueError("sweep.runs must contain at least one run.")

    gpus_per_run = int(args.gpus_per_run or raw.get("gpus_per_run", 1))
    max_concurrent = int(args.max_concurrent_runs or raw.get("max_concurrent_runs", 8))
    settings = SweepSettings(
        training_script=resolve_project_path(raw["training_script"]),
        training_config=resolve_project_path(raw["training_config"]),
        alignment_checkpoint=Path(raw["alignment_checkpoint"]).expanduser(),
        output_root=Path(raw["output_root"]).expanduser(),
        status_root=Path(raw["status_root"]).expanduser(),
        run_name_prefix=str(raw.get("run_name_prefix", "tensor_patch_qa_sweep")),
        min_free_memory_mib=int(
            args.min_free_memory_mib
            if args.min_free_memory_mib is not None
            else raw.get("min_free_memory_mib", 79000)
        ),
        poll_seconds=max(
            1.0,
            float(args.poll_seconds if args.poll_seconds is not None else raw.get("poll_seconds", 15)),
        ),
        non_tty_status_seconds=max(1.0, float(raw.get("non_tty_status_seconds", 300))),
        gpus_per_run=max(1, gpus_per_run),
        max_concurrent_runs=max(1, max_concurrent),
        excluded_gpus=frozenset(int(value) for value in raw.get("excluded_gpus") or ()),
        common_overrides=common_overrides,
        parameter_names=parameter_names,
        runs=tuple(runs),
    )
    if not settings.training_script.exists():
        raise FileNotFoundError(f"Training script not found: {settings.training_script}")
    if not settings.training_config.exists():
        raise FileNotFoundError(f"Training config not found: {settings.training_config}")
    if not settings.alignment_checkpoint.exists():
        raise FileNotFoundError(
            f"Stage-1 alignment checkpoint not found: {settings.alignment_checkpoint}"
        )
    return settings, raw_config


def query_gpu_free_memory() -> dict[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Unexpected nvidia-smi output: {line!r}")
        result[int(parts[0])] = int(parts[1])
    if not result:
        raise RuntimeError("nvidia-smi did not report any GPUs.")
    return result


def build_training_command(
    settings: SweepSettings,
    spec: SweepRun,
    run_name: str,
) -> list[str]:
    if settings.gpus_per_run == 1:
        command = [sys.executable, str(settings.training_script)]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={settings.gpus_per_run}",
            str(settings.training_script),
        ]
    command.extend(
        [
            "--config",
            str(settings.training_config),
            "--qa-alignment-checkpoint",
            str(settings.alignment_checkpoint),
            "--adapter-init-checkpoint",
            str(settings.alignment_checkpoint),
            "--output-root",
            str(settings.output_root),
            "--run-name",
            run_name,
        ]
    )
    merged_overrides = {**settings.common_overrides, **spec.overrides}
    for name in sorted(merged_overrides):
        command.extend(override_to_args(name, merged_overrides[name]))
    return command


def find_run_dir(output_root: Path, run_name: str, started_epoch: float) -> Path | None:
    if not output_root.exists():
        return None
    candidates = [
        path
        for path in output_root.glob(f"*_{run_name}")
        if path.is_dir() and path.stat().st_mtime >= started_epoch - 5.0
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def serialize_records(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in records.values():
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "updated_at": timestamp(),
        "counts": counts,
        "runs": [records[key] for key in sorted(records)],
    }


def terminate_jobs(jobs: dict[str, RunningJob], timeout: float = 15.0) -> None:
    for job in jobs.values():
        if job.process.poll() is None:
            try:
                os.killpg(job.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(job.process.poll() is None for job in jobs.values()):
        time.sleep(0.25)
    for job in jobs.values():
        if job.process.poll() is None:
            try:
                os.killpg(job.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main() -> int:
    args = parse_args()
    settings, raw_config = load_settings(args)
    session_dir = settings.status_root / f"{compact_timestamp()}_stage2_sweep"
    log_dir = session_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)
    state_path = session_dir / "sweep_state.json"
    event_path = session_dir / "events.jsonl"
    atomic_json_dump(session_dir / "config_snapshot.json", raw_config)

    records: dict[str, dict[str, Any]] = {
        spec.sweep_id: {
            "id": spec.sweep_id,
            "status": "pending",
            "gpu_ids": [],
            "pid": None,
            "started_at": None,
            "ended_at": None,
            "duration_seconds": None,
            "return_code": None,
            "run_dir": None,
            "log_path": str(log_dir / f"{spec.sweep_id}.log"),
            "overrides": {**settings.common_overrides, **spec.overrides},
        }
        for spec in settings.runs
    }
    atomic_json_dump(state_path, serialize_records(records))
    append_jsonl(event_path, {"event": "session_start", "time": timestamp()})
    terminal = TerminalStatus(settings.non_tty_status_seconds)
    terminal.event(f"SWEEP START time={timestamp()} session={session_dir}")

    def interrupt_handler(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_handler)

    if args.dry_run:
        for spec in settings.runs:
            terminal.event(f"{spec.sweep_id} QUEUED")
        terminal.event(f"SWEEP DRY_RUN_END time={timestamp()}")
        return 0

    pending = list(settings.runs)
    running: dict[str, RunningJob] = {}
    scheduler_started = time.monotonic()
    failures = 0
    try:
        while pending or running:
            for sweep_id, job in list(running.items()):
                if job.run_dir is None:
                    job.run_dir = find_run_dir(
                        settings.output_root,
                        job.run_name,
                        job.started_epoch,
                    )
                    if job.run_dir is not None:
                        records[sweep_id]["run_dir"] = str(job.run_dir)
                        append_jsonl(
                            event_path,
                            {
                                "event": "run_dir_discovered",
                                "id": sweep_id,
                                "time": timestamp(),
                                "run_dir": str(job.run_dir),
                            },
                        )
                        atomic_json_dump(state_path, serialize_records(records))
                return_code = job.process.poll()
                if return_code is None:
                    continue
                ended_at = timestamp()
                duration = time.time() - job.started_epoch
                job.log_handle.close()
                status = "completed" if return_code == 0 else "failed"
                failures += int(return_code != 0)
                records[sweep_id].update(
                    {
                        "status": status,
                        "ended_at": ended_at,
                        "duration_seconds": round(duration, 3),
                        "return_code": int(return_code),
                        "run_dir": str(job.run_dir) if job.run_dir is not None else None,
                    }
                )
                append_jsonl(
                    event_path,
                    {
                        "event": "run_end",
                        "id": sweep_id,
                        "time": ended_at,
                        "duration_seconds": round(duration, 3),
                        "return_code": int(return_code),
                        "run_dir": records[sweep_id]["run_dir"],
                    },
                )
                atomic_json_dump(state_path, serialize_records(records))
                terminal.event(
                    f"{sweep_id} END time={ended_at} duration={format_duration(duration)} "
                    f"exit={return_code} run_dir={records[sweep_id]['run_dir']}"
                )
                del running[sweep_id]

            assigned = {gpu for job in running.values() for gpu in job.gpu_ids}
            try:
                free_memory = query_gpu_free_memory()
            except (OSError, subprocess.SubprocessError, ValueError, RuntimeError):
                elapsed = format_duration(time.monotonic() - scheduler_started)
                terminal.status(
                    f"waiting elapsed={elapsed} pending={len(pending)} "
                    f"running={len(running)} gpu_scan_error=1"
                )
                time.sleep(settings.poll_seconds)
                continue
            available = [
                gpu
                for gpu, free_mib in sorted(free_memory.items(), key=lambda item: (-item[1], item[0]))
                if gpu not in assigned
                and gpu not in settings.excluded_gpus
                and free_mib >= settings.min_free_memory_mib
            ]
            while (
                pending
                and len(running) < settings.max_concurrent_runs
                and len(available) >= settings.gpus_per_run
            ):
                spec = pending.pop(0)
                gpu_ids = tuple(available[: settings.gpus_per_run])
                del available[: settings.gpus_per_run]
                run_name = f"{settings.run_name_prefix}_{spec.sweep_id}"
                command = build_training_command(settings, spec, run_name)
                log_path = log_dir / f"{spec.sweep_id}.log"
                log_handle = log_path.open("w", encoding="utf-8", buffering=1)
                log_handle.write("command=" + json.dumps(command) + "\n")
                child_env = dict(os.environ)
                child_env.update(
                    {
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpu_ids),
                        "PYTHONUNBUFFERED": "1",
                    }
                )
                started_epoch = time.time()
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=str(PROJECT_ROOT),
                        env=child_env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log_handle.close()
                    pending.insert(0, spec)
                    raise
                job = RunningJob(
                    spec=spec,
                    run_name=run_name,
                    gpu_ids=gpu_ids,
                    process=process,
                    log_handle=log_handle,
                    log_path=log_path,
                    started_at=timestamp(),
                    started_epoch=started_epoch,
                )
                running[spec.sweep_id] = job
                records[spec.sweep_id].update(
                    {
                        "status": "running",
                        "gpu_ids": list(gpu_ids),
                        "pid": int(process.pid),
                        "started_at": job.started_at,
                    }
                )
                append_jsonl(
                    event_path,
                    {
                        "event": "run_start",
                        "id": spec.sweep_id,
                        "time": job.started_at,
                        "gpu_ids": list(gpu_ids),
                        "pid": int(process.pid),
                    },
                )
                atomic_json_dump(state_path, serialize_records(records))
                terminal.event(
                    f"{spec.sweep_id} START time={job.started_at} "
                    f"gpu={','.join(str(gpu) for gpu in gpu_ids)} pid={process.pid}"
                )

            elapsed = format_duration(time.monotonic() - scheduler_started)
            if pending:
                terminal.status(
                    f"waiting elapsed={elapsed} pending={len(pending)} running={len(running)}"
                )
            elif running:
                terminal.status(f"running elapsed={elapsed} active={len(running)}")
            if pending or running:
                time.sleep(settings.poll_seconds)
    except KeyboardInterrupt:
        terminal.event(f"SWEEP INTERRUPT time={timestamp()} active={len(running)}")
        terminate_jobs(running)
        for sweep_id, job in running.items():
            job.log_handle.close()
            ended_at = timestamp()
            records[sweep_id].update(
                {
                    "status": "interrupted",
                    "ended_at": ended_at,
                    "return_code": job.process.poll(),
                    "run_dir": str(job.run_dir) if job.run_dir is not None else None,
                }
            )
            terminal.event(f"{sweep_id} END time={ended_at} status=interrupted")
        atomic_json_dump(state_path, serialize_records(records))
        append_jsonl(event_path, {"event": "session_interrupt", "time": timestamp()})
        return 130
    except BaseException as exc:
        terminal.event(
            f"SWEEP ERROR time={timestamp()} type={type(exc).__name__} active={len(running)}"
        )
        terminate_jobs(running)
        for sweep_id, job in running.items():
            job.log_handle.close()
            ended_at = timestamp()
            records[sweep_id].update(
                {
                    "status": "scheduler_error",
                    "ended_at": ended_at,
                    "return_code": job.process.poll(),
                    "run_dir": str(job.run_dir) if job.run_dir is not None else None,
                }
            )
            terminal.event(f"{sweep_id} END time={ended_at} status=scheduler_error")
        atomic_json_dump(state_path, serialize_records(records))
        append_jsonl(
            event_path,
            {
                "event": "session_error",
                "time": timestamp(),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2000],
            },
        )
        raise

    ended_at = timestamp()
    append_jsonl(
        event_path,
        {"event": "session_end", "time": ended_at, "failures": failures},
    )
    terminal.event(f"SWEEP END time={ended_at} failures={failures} state={state_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
