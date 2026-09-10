"""Internal helpers shared by the per-method Qwen3 trigger scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QWEN3_ROOT = Path(__file__).resolve().parent
REPO_ROOT = QWEN3_ROOT
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_runs" / "qwen3"

DATASETS = {
    "science": "qwen3_science",
    "medicine": "qwen3_medicine",
    "pathology": "qwen3_pathology",
    "finqa": "qwen3_finqa",
}

DATASET_PATHS = {
    "qwen3_science": QWEN3_ROOT / "data/qwen3_science/train_data",
    "qwen3_medicine": QWEN3_ROOT / "data/qwen3_medicine/train_data",
    "qwen3_pathology": QWEN3_ROOT / "data/qwen3_pathology/train_data",
    "qwen3_finqa": QWEN3_ROOT / "data/qwen3_finqa/train_data",
}

# Deliberately code-level constants rather than another user-facing config.
# These values must not vary between methods in the main paper comparison.
SHARED_HPARAMS: dict[str, Any] = {
    "model_name": "Qwen/Qwen3-8B",
    "learning_rate": 5e-5,
    "num_train_epochs": 2,
    "global_batch_size": 16,
    "per_device_train_batch_size": 1,
    "warmup_steps": 10,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "precision": "bf16",
    "max_length": 4096,
    "seed": 42,
}


@dataclass
class Trigger:
    method: str
    dataset_id: str
    output_dir: Path
    command: list[str]
    grad_accum_steps: int
    config_path: Path
    dry_run: bool


def common_parser(description: str, config_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "science, medicine, pathology, finqa, or a raw dataset identifier "
            "paired with --data-path"
        ),
    )
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--deepspeed-bin", default="deepspeed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact DeepSpeed command without starting training.",
    )
    return parser


def load_method_config(path: Path, expected_keys: set[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"method config must be a JSON object: {resolved}")
    if set(config) != expected_keys:
        raise ValueError(
            f"{resolved} must contain exactly {sorted(expected_keys)}; "
            f"found {sorted(config)}"
        )
    return config


def _value(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _gradient_accumulation(num_gpus: int) -> int:
    if num_gpus < 1:
        raise ValueError("--num-gpus must be positive")
    denominator = num_gpus * SHARED_HPARAMS["per_device_train_batch_size"]
    global_batch = SHARED_HPARAMS["global_batch_size"]
    if global_batch % denominator:
        raise ValueError(
            f"global batch {global_batch} cannot be preserved exactly with "
            f"{num_gpus} GPUs and micro-batch "
            f"{SHARED_HPARAMS['per_device_train_batch_size']}"
        )
    return global_batch // denominator


def _dataset_label(requested: str, dataset_id: str) -> str:
    if requested in DATASETS:
        return requested
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", dataset_id).strip("_")


def build_trigger(
    args: argparse.Namespace,
    *,
    method: str,
    trainer: str,
    deepspeed_config: str,
) -> Trigger:
    dataset_id = DATASETS.get(args.dataset, args.dataset)
    dataset_label = _dataset_label(args.dataset, dataset_id)
    grad_accum = _gradient_accumulation(args.num_gpus)
    trainer_path = (REPO_ROOT / trainer).resolve()
    deepspeed_path = (REPO_ROOT / deepspeed_config).resolve()
    if not trainer_path.is_file():
        raise FileNotFoundError(f"trainer does not exist: {trainer_path}")
    if not deepspeed_path.is_file():
        raise FileNotFoundError(f"DeepSpeed config does not exist: {deepspeed_path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / dataset_label / method
    )
    run_name = args.run_name or f"main_qwen3_{dataset_label}_{method}"
    shared = SHARED_HPARAMS
    command = [
        args.deepspeed_bin,
        "--num_gpus",
        str(args.num_gpus),
        str(trainer_path),
        "--model_name",
        shared["model_name"],
        "--dataset_name",
        dataset_id,
        "--output_dir",
        str(output_dir),
        "--run_name",
        run_name,
        "--learning_rate",
        _value(shared["learning_rate"]),
        "--num_train_epochs",
        _value(shared["num_train_epochs"]),
        "--seed",
        _value(shared["seed"]),
        "--warmup_steps",
        _value(shared["warmup_steps"]),
        "--lr_scheduler_type",
        shared["lr_scheduler_type"],
        "--max_grad_norm",
        _value(shared["max_grad_norm"]),
        "--report_to",
        args.report_to,
        "--deepspeed",
        str(deepspeed_path),
        "--grad_accum_steps",
        str(grad_accum),
        "--max_length",
        str(shared["max_length"]),
    ]
    if args.data_path is not None:
        data_path = args.data_path.expanduser().resolve()
    else:
        data_path = DATASET_PATHS.get(dataset_id)
    if data_path is not None:
        if not args.dry_run and not data_path.is_dir():
            raise FileNotFoundError(
                f"dataset directory does not exist: {data_path}; "
                "run `git lfs pull` or pass --data-path"
            )
        command.extend(["--data_path", str(data_path)])
    return Trigger(
        method=method,
        dataset_id=dataset_id,
        output_dir=output_dir,
        command=command,
        grad_accum_steps=grad_accum,
        config_path=args.config.expanduser().resolve(),
        dry_run=args.dry_run,
    )


def display(trigger: Trigger) -> None:
    print(f"[{trigger.method}/{trigger.dataset_id}]")
    print(shlex.join(trigger.command))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def launch(trigger: Trigger, method_config: dict[str, Any]) -> int:
    if trigger.dry_run:
        display(trigger)
        return 0

    trigger.output_dir.mkdir(parents=True, exist_ok=True)
    config_bytes = trigger.config_path.read_bytes()
    manifest = {
        "schema_version": 1,
        "method": trigger.method,
        "dataset_id": trigger.dataset_id,
        "shared_hparams": SHARED_HPARAMS,
        "method_hparams": method_config,
        "num_gpus": int(trigger.command[2]),
        "gradient_accumulation_steps": trigger.grad_accum_steps,
        "effective_global_batch_size": SHARED_HPARAMS["global_batch_size"],
        "method_config": str(trigger.config_path),
        "method_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "git_commit": _git_commit(),
        "command": trigger.command,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = trigger.output_dir / "reproduction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    display(trigger)
    subprocess.run(trigger.command, cwd=REPO_ROOT, check=True)
    return 0
