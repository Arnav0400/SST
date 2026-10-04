#!/usr/bin/env python3
"""Vanilla supervised fine-tuning."""

from __future__ import annotations

from sst.training import (
    load_model_and_tokenizer,
    train_sft_model,
    worker_parser,
)


def main() -> None:
    parser = worker_parser("Train with vanilla SFT")
    args = parser.parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.precision)
    train_sft_model(args, model, tokenizer)


if __name__ == "__main__":
    main()
