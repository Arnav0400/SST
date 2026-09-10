#!/usr/bin/env python3
"""Independent vanilla SFT worker."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.training import (  # noqa: E402
    load_model_and_tokenizer,
    train_sft_model,
    worker_parser,
)


def main() -> None:
    parser = worker_parser("Independent vanilla SFT worker")
    args = parser.parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    train_sft_model(args, model, tokenizer)


if __name__ == "__main__":
    main()
