#!/usr/bin/env python3
"""Trigger the SDFT Qwen3 baseline on one dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


QWEN3_ROOT = Path(__file__).resolve().parents[2]
if str(QWEN3_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN3_ROOT))

from _shared import build_trigger, common_parser, launch, load_method_config  # noqa: E402


HERE = Path(__file__).resolve().parent
CONFIG_KEYS = {
    "demo_mode",
    "kl_alpha",
    "ref_model_mixup_alpha",
    "generation_prompts_per_device",
    "max_prompt_length",
    "max_completion_length",
    "vllm_mode",
    "vllm_tensor_parallel_size",
    "vllm_gpu_memory_utilization",
}


def make_trigger(argv: Sequence[str] | None = None):
    parser = common_parser("Train the SDFT baseline on one Qwen3 dataset.", HERE / "config.json")
    args = parser.parse_args(argv)
    config = load_method_config(args.config, CONFIG_KEYS)
    trigger = build_trigger(
        args,
        method="sdft",
        trainer="workers/sdft.py",
        deepspeed_config="deepspeed/zero3.json",
    )
    trigger.command.extend(
        [
            "--generation_prompts_per_device",
            str(config["generation_prompts_per_device"]),
            "--demo_mode",
            str(config["demo_mode"]),
            "--kl_alpha",
            str(config["kl_alpha"]),
            "--ref_model_mixup_alpha",
            str(config["ref_model_mixup_alpha"]),
            "--max_prompt_length",
            str(config["max_prompt_length"]),
            "--max_completion_length",
            str(config["max_completion_length"]),
            "--vllm_mode",
            str(config["vllm_mode"]),
            "--vllm_tensor_parallel_size",
            str(config["vllm_tensor_parallel_size"]),
            "--vllm_gpu_memory_utilization",
            str(config["vllm_gpu_memory_utilization"]),
        ]
    )
    return trigger, config


def main(argv: Sequence[str] | None = None) -> int:
    trigger, config = make_trigger(argv)
    return launch(trigger, config)


if __name__ == "__main__":
    raise SystemExit(main())
