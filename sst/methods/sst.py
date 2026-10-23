#!/usr/bin/env python3
"""SST training implementation."""

from __future__ import annotations

from sst.training import (
    load_model_and_tokenizer,
    rank_zero,
    train_sft_model,
    worker_parser,
)
from sst_torch import DEFAULT_TARGET_MODULES


def main() -> None:
    parser = worker_parser("Train with SST")
    parser.add_argument("--rank-percent", type=float, required=True)
    args = parser.parse_args()
    try:
        from transformers import SST
    except ImportError as error:
        raise RuntimeError(
            "The SST recipe requires the native Transformers SST optimizer wrapper. "
            "Install the optimizer-wrapper branch described in docs/huggingface-integration.md."
        ) from error
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.precision)
    if rank_zero():
        print(
            f"[sst] optimizer wrapper; requested rank_percent={args.rank_percent:g}; "
            "original checkpoint basis restored automatically when resuming",
            flush=True,
        )
    train_sft_model(
        args,
        model,
        tokenizer,
        optimizer_wrapper=(
            SST,
            {
                "rank_fraction": args.rank_percent / 100.0,
                "target_modules": DEFAULT_TARGET_MODULES,
            },
        ),
    )


if __name__ == "__main__":
    main()
