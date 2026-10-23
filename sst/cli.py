"""Public launcher for SST and the included fine-tuning baselines."""

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
from typing import Any, Sequence

from sst.config import (
    DEFAULT_CONFIG_DIR,
    METHOD_DEEPSPEED,
    METHODS,
    ROOT,
    DatasetConfig,
    ModelConfig,
    load_dataset,
    load_method,
    load_model,
)


@dataclass(frozen=True)
class RunPlan:
    method: str
    dataset: DatasetConfig
    model: ModelConfig
    method_config: dict[str, Any]
    method_config_path: Path
    model_config_path: Path
    dataset_config_path: Path | None
    output_dir: Path
    command: list[str]
    num_gpus: int
    gradient_accumulation_steps: int
    dry_run: bool


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="sst-train",
        description="Train SST or an included baseline on one dataset.",
    )
    result.add_argument("--method", required=True, choices=METHODS)
    result.add_argument("--dataset", required=True)
    result.add_argument("--model", default="qwen3-8b")
    result.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    result.add_argument("--data-path", type=Path)
    result.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override chat-template thinking for the selected dataset.",
    )
    result.add_argument("--num-gpus", type=int, default=8)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--run-name")
    result.add_argument("--report-to", default="none")
    result.add_argument("--deepspeed-bin", default="deepspeed")
    result.add_argument(
        "--refresh-every",
        type=int,
        help="For SST, refresh after this many successful optimizer updates.",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved DeepSpeed command without starting training.",
    )
    return result


def _value(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _gradient_accumulation(model: ModelConfig, num_gpus: int) -> int:
    if num_gpus < 1:
        raise ValueError("--num-gpus must be positive")
    denominator = num_gpus * model.per_device_train_batch_size
    if model.global_batch_size % denominator:
        raise ValueError(
            f"global batch {model.global_batch_size} cannot be preserved exactly "
            f"with {num_gpus} GPUs and micro-batch "
            f"{model.per_device_train_batch_size}"
        )
    return model.global_batch_size // denominator


def _method_arguments(method: str, config: dict[str, Any]) -> list[str]:
    if method == "sst":
        result = ["--rank-percent", _value(config["rank_percent"])]
        if config["refresh_every"] is not None:
            result.extend(["--refresh-every", _value(config["refresh_every"])])
        return result
    if method == "sft":
        return []
    if method == "sdft":
        return [
            "--generation-prompts-per-device",
            _value(config["generation_prompts_per_device"]),
            "--demo-mode",
            str(config["demo_mode"]),
            "--kl-alpha",
            _value(config["kl_alpha"]),
            "--ref-model-mixup-alpha",
            _value(config["ref_model_mixup_alpha"]),
            "--max-prompt-length",
            _value(config["max_prompt_length"]),
            "--max-completion-length",
            _value(config["max_completion_length"]),
            "--vllm-mode",
            str(config["vllm_mode"]),
            "--vllm-tensor-parallel-size",
            _value(config["vllm_tensor_parallel_size"]),
            "--vllm-gpu-memory-utilization",
            _value(config["vllm_gpu_memory_utilization"]),
        ]
    if method == "oplora":
        return [
            "--lora-rank",
            _value(config["lora_rank"]),
            "--lora-alpha",
            _value(config["lora_alpha"]),
            "--lora-dropout",
            _value(config["lora_dropout"]),
            "--projection-rank",
            _value(config["projection_rank"]),
            "--target-modules",
            *config["target_modules"],
        ]
    if method == "sculpting":
        result = [
            "--minimum-retention-ratio",
            _value(config["minimum_retention_ratio"]),
            "--target-retention-ratio",
            _value(config["target_retention_ratio"]),
            "--target-modules",
            *config["target_modules"],
        ]
        if config["calibration_samples"] is not None:
            result.extend(
                ["--calibration-samples", _value(config["calibration_samples"])]
            )
        return result
    raise ValueError(f"unsupported method: {method}")


def build_plan(argv: Sequence[str] | None = None) -> RunPlan:
    args = parser().parse_args(argv)
    config_dir = args.config_dir.expanduser().resolve()
    model, model_path = load_model(config_dir, args.model)
    dataset, dataset_path = load_dataset(
        config_dir,
        args.dataset,
        data_path=args.data_path,
        thinking=args.thinking,
    )
    method_config, method_path = load_method(config_dir, args.method)
    if args.refresh_every is not None:
        if args.method != "sst":
            raise ValueError("--refresh-every is only supported with --method sst")
        if args.refresh_every < 1:
            raise ValueError("--refresh-every must be positive")
        method_config["refresh_every"] = args.refresh_every
    grad_accum = _gradient_accumulation(model, args.num_gpus)

    worker = ROOT / "sst" / "methods" / f"{args.method}.py"
    deepspeed = config_dir / "deepspeed" / METHOD_DEEPSPEED[args.method]
    if not worker.is_file():
        raise FileNotFoundError(f"method implementation does not exist: {worker}")
    if not deepspeed.is_file():
        raise FileNotFoundError(f"DeepSpeed configuration does not exist: {deepspeed}")

    label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.dataset).strip("_")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else ROOT / "runs" / label / args.method
    )
    run_name = args.run_name or f"{args.method}_{label}"
    command = [
        args.deepspeed_bin,
        "--num_gpus",
        str(args.num_gpus),
        str(worker),
        "--model-name",
        model.name,
        "--dataset-name",
        dataset.dataset_id,
        "--data-path",
        str(dataset.path),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--learning-rate",
        _value(model.learning_rate),
        "--num-train-epochs",
        _value(model.num_train_epochs),
        "--seed",
        str(model.seed),
        "--warmup-steps",
        str(model.warmup_steps),
        "--lr-scheduler-type",
        model.lr_scheduler_type,
        "--weight-decay",
        _value(model.weight_decay),
        "--max-grad-norm",
        _value(model.max_grad_norm),
        "--precision",
        model.precision,
        "--report-to",
        args.report_to,
        "--deepspeed",
        str(deepspeed),
        "--gradient-accumulation-steps",
        str(grad_accum),
        "--per-device-train-batch-size",
        str(model.per_device_train_batch_size),
        "--max-length",
        str(model.max_length),
        "--thinking" if dataset.thinking else "--no-thinking",
        *_method_arguments(args.method, method_config),
    ]
    return RunPlan(
        method=args.method,
        dataset=dataset,
        model=model,
        method_config=method_config,
        method_config_path=method_path,
        model_config_path=model_path,
        dataset_config_path=dataset_path,
        output_dir=output_dir,
        command=command,
        num_gpus=args.num_gpus,
        gradient_accumulation_steps=grad_accum,
        dry_run=args.dry_run,
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_dataset_ready(dataset: DatasetConfig) -> None:
    if not dataset.path.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset.path}")
    state_path = dataset.path / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"missing Hugging Face dataset state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for entry in state.get("_data_files", []):
        data_file = dataset.path / entry["filename"]
        if not data_file.is_file():
            raise FileNotFoundError(f"missing dataset shard: {data_file}")
        with data_file.open("rb") as handle:
            if handle.read(64).startswith(b"version https://git-lfs.github.com/"):
                raise RuntimeError(
                    f"{data_file} is a Git LFS pointer; run `git lfs pull` first"
                )


def display(plan: RunPlan) -> None:
    print(f"[{plan.method}/{plan.dataset.name}]")
    print(shlex.join(plan.command))


def launch(plan: RunPlan) -> int:
    display(plan)
    if plan.dry_run:
        return 0

    _ensure_dataset_ready(plan.dataset)
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "method": plan.method,
        "dataset": plan.dataset.to_dict(),
        "model": plan.model.to_dict(),
        "method_config": plan.method_config,
        "method_config_path": str(plan.method_config_path),
        "method_config_sha256": _sha256(plan.method_config_path),
        "model_config_path": str(plan.model_config_path),
        "model_config_sha256": _sha256(plan.model_config_path),
        "dataset_config_path": (
            str(plan.dataset_config_path) if plan.dataset_config_path else None
        ),
        "num_gpus": plan.num_gpus,
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "git_commit": _git_commit(),
        "command": plan.command,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (plan.output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(plan.command, cwd=ROOT, check=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return launch(build_plan(argv))
