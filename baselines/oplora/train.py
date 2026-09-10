#!/usr/bin/env python3
"""Trigger the OPLoRA Qwen3 baseline on one dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


QWEN3_ROOT = Path(__file__).resolve().parents[2]
if str(QWEN3_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN3_ROOT))

from _shared import (  # noqa: E402
    build_trigger,
    common_parser,
    launch,
    load_method_config,
)


HERE = Path(__file__).resolve().parent
CONFIG_KEYS = {
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "projection_rank",
    "target_modules",
}


def make_trigger(argv: Sequence[str] | None = None):
    parser = common_parser("Train the OPLoRA baseline on one Qwen3 dataset.", HERE / "config.json")
    args = parser.parse_args(argv)
    config = load_method_config(args.config, CONFIG_KEYS)
    trigger = build_trigger(
        args,
        method="oplora",
        trainer="workers/oplora.py",
        deepspeed_config="deepspeed/zero2.json",
    )
    trigger.command.extend(
        [
            "--lora_rank",
            str(config["lora_rank"]),
            "--lora_alpha",
            str(config["lora_alpha"]),
            "--lora_dropout",
            str(config["lora_dropout"]),
            "--projection_rank",
            str(config["projection_rank"]),
            "--target_modules",
            *config["target_modules"],
        ]
    )
    return trigger, config


def main(argv: Sequence[str] | None = None) -> int:
    trigger, config = make_trigger(argv)
    return launch(trigger, config)


if __name__ == "__main__":
    raise SystemExit(main())
