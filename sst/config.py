"""Configuration loading for SST training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = ROOT / "configs"
METHODS = ("sst", "sft", "sdft", "oplora", "sculpting")

METHOD_KEYS = {
    "sst": {"rank_percent"},
    "sft": set(),
    "sdft": {
        "demo_mode",
        "kl_alpha",
        "ref_model_mixup_alpha",
        "generation_prompts_per_device",
        "max_prompt_length",
        "max_completion_length",
        "vllm_mode",
        "vllm_tensor_parallel_size",
        "vllm_gpu_memory_utilization",
    },
    "oplora": {
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "projection_rank",
        "target_modules",
    },
    "sculpting": {
        "minimum_retention_ratio",
        "target_retention_ratio",
        "calibration_samples",
        "target_modules",
    },
}

METHOD_DEEPSPEED = {
    "sst": "zero2.json",
    "sft": "zero2.json",
    "sdft": "zero3.json",
    "oplora": "zero2.json",
    "sculpting": "zero2.json",
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    learning_rate: float
    num_train_epochs: float
    global_batch_size: int
    per_device_train_batch_size: int
    warmup_steps: int
    lr_scheduler_type: str
    weight_decay: float
    max_grad_norm: float
    precision: str
    max_length: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_id: str
    path: Path
    thinking: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return loaded


def _require_keys(path: Path, config: dict[str, Any], expected: set[str]) -> None:
    if set(config) != expected:
        raise ValueError(
            f"{path} must contain exactly {sorted(expected)}; "
            f"found {sorted(config)}"
        )


def load_model(config_dir: Path, name: str) -> tuple[ModelConfig, Path]:
    path = config_dir / "model" / f"{name}.yaml"
    config = _read_yaml(path)
    _require_keys(path, config, set(ModelConfig.__dataclass_fields__))
    model = ModelConfig(**config)
    if model.precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"unsupported precision: {model.precision}")
    if model.global_batch_size < 1 or model.per_device_train_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    return model, path


def load_dataset(
    config_dir: Path,
    name: str,
    *,
    data_path: Path | None = None,
    thinking: bool | None = None,
) -> tuple[DatasetConfig, Path | None]:
    path = config_dir / "data" / f"{name}.yaml"
    if path.is_file():
        config = _read_yaml(path)
        _require_keys(path, config, {"dataset_id", "path", "thinking"})
        configured_path = Path(config["path"])
        if not configured_path.is_absolute():
            configured_path = config_dir.parent / configured_path
        dataset = DatasetConfig(
            name=name,
            dataset_id=str(config["dataset_id"]),
            path=(data_path or configured_path).expanduser().resolve(),
            thinking=bool(config["thinking"]) if thinking is None else thinking,
        )
        return dataset, path

    if data_path is None:
        raise ValueError(
            f"unknown dataset {name!r}; add configs/data/{name}.yaml "
            "or pass --data-path"
        )
    return (
        DatasetConfig(
            name=name,
            dataset_id=name,
            path=data_path.expanduser().resolve(),
            thinking=True if thinking is None else thinking,
        ),
        None,
    )


def load_method(config_dir: Path, name: str) -> tuple[dict[str, Any], Path]:
    if name not in METHOD_KEYS:
        raise ValueError(f"unknown method {name!r}; choose from {METHODS}")
    path = config_dir / "method" / f"{name}.yaml"
    config = _read_yaml(path)
    _require_keys(path, config, METHOD_KEYS[name])
    return config, path
