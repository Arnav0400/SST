#!/usr/bin/env python3
"""Trigger the main SST Qwen3 run on one dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


QWEN3_ROOT = Path(__file__).resolve().parents[1]
if str(QWEN3_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN3_ROOT))

from _shared import (  # noqa: E402
    build_trigger,
    common_parser,
    launch,
    load_method_config,
)


HERE = Path(__file__).resolve().parent

def make_trigger(argv: Sequence[str] | None = None):
    parser = common_parser("Train SST on one Qwen3 dataset.", HERE / "config.json")
    args = parser.parse_args(argv)
    config = load_method_config(args.config, {"rank_percent"})
    trigger = build_trigger(
        args,
        method="sst",
        trainer="workers/sst.py",
        deepspeed_config="deepspeed/zero2.json",
    )
    trigger.command.extend(["--rank_percent", str(config["rank_percent"])])
    return trigger, config


def main(argv: Sequence[str] | None = None) -> int:
    trigger, config = make_trigger(argv)
    return launch(trigger, config)


if __name__ == "__main__":
    raise SystemExit(main())
