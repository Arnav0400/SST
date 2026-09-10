#!/usr/bin/env python3
"""Independent adaptive Sculpting worker implementing arXiv:2504.07097."""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.training import (  # noqa: E402
    load_model_and_tokenizer,
    prepared_sft_dataset,
    rank_zero,
    train_sft_model,
    worker_parser,
)


def _work_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


@torch.no_grad()
def adaptive_retention_by_block(
    model,
    dataset,
    *,
    minimum_retention_ratio: float,
    target_retention_ratio: float,
    calibration_samples: int | None,
) -> list[float]:
    layers = model.model.layers
    count = len(dataset) if calibration_samples is None else min(calibration_samples, len(dataset))
    if count < 1:
        raise ValueError("Sculpting calibration requires at least one sample")
    device = _work_device()
    model.to(device)
    model.eval()
    sums = [0.0] * len(layers)
    tokens = [0] * len(layers)

    for index in range(count):
        input_ids = torch.tensor(dataset[index]["input_ids"], device=device).unsqueeze(0)
        hidden = model(input_ids, output_hidden_states=True, use_cache=False).hidden_states
        for layer in range(len(layers)):
            before = hidden[layer].reshape(-1, hidden[layer].shape[-1]).float()
            after = hidden[layer + 1].reshape(-1, hidden[layer + 1].shape[-1]).float()
            cosine = functional.cosine_similarity(before, after, dim=-1)
            sums[layer] += float(cosine.sum())
            tokens[layer] += cosine.numel()
        del hidden, input_ids

    raw = [total / max(size, 1) for total, size in zip(sums, tokens)]
    mean = sum(raw) / len(raw)
    if abs(mean) < 1e-12:
        raise RuntimeError("Sculpting layer-importance mean is numerically zero")
    normalized = [value / mean for value in raw]
    retention = [
        min(
            max(
                minimum_retention_ratio
                + importance * (target_retention_ratio - minimum_retention_ratio),
                1e-4,
            ),
            0.9999,
        )
        for importance in normalized
    ]
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if rank_zero():
        print(
            f"[sculpting] calibrated {count} examples; retention "
            f"mean={sum(retention)/len(retention):.4f} "
            f"min={min(retention):.4f} max={max(retention):.4f}",
            flush=True,
        )
    return retention


@dataclass
class GradientSubspace:
    name: str
    left: torch.Tensor
    right: torch.Tensor

    def project(self, gradient: torch.Tensor) -> torch.Tensor:
        left = self.left.to(gradient.device, torch.float32, non_blocking=True)
        right = self.right.to(gradient.device, torch.float32, non_blocking=True)
        grad = gradient.float()
        blocked = left @ ((left.T @ grad) @ right) @ right.T
        return (grad - blocked).to(gradient.dtype)


def register_sculpting_hooks(
    model,
    *,
    target_modules: list[str],
    block_retention: list[float],
) -> list[GradientSubspace]:
    device = _work_device()
    subspaces = []
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or not any(
            name.endswith(f".{target}.weight") for target in target_modules
        ):
            continue
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match is None:
            raise RuntimeError(f"cannot map Sculpting target to a transformer block: {name}")
        layer = int(match.group(1))
        weight = parameter.detach().to(device=device, dtype=torch.float32)
        minimum = min(weight.shape)
        rank = min(max(1, math.floor(minimum * block_retention[layer])), minimum - 1)
        left, _, vh = torch.linalg.svd(weight, full_matrices=False)
        subspace = GradientSubspace(
            name=name,
            left=left[:, :rank].contiguous().cpu(),
            right=vh[:rank].T.contiguous().cpu(),
        )
        parameter.register_hook(subspace.project)
        subspaces.append(subspace)
        del weight, left, vh
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if rank_zero() and len(subspaces) % 32 == 0:
            print(f"[sculpting] prepared {len(subspaces)} matrices", flush=True)
    if not subspaces:
        raise RuntimeError("Sculpting found no target matrices")
    return subspaces


def main() -> None:
    parser = worker_parser("Independent adaptive Sculpting worker")
    parser.add_argument("--minimum_retention_ratio", type=float, required=True)
    parser.add_argument("--target_retention_ratio", type=float, required=True)
    parser.add_argument("--calibration_samples", type=int, default=None)
    parser.add_argument("--target_modules", nargs="+", required=True)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    dataset = prepared_sft_dataset(args, tokenizer)
    retention = adaptive_retention_by_block(
        model,
        dataset,
        minimum_retention_ratio=args.minimum_retention_ratio,
        target_retention_ratio=args.target_retention_ratio,
        calibration_samples=args.calibration_samples,
    )
    subspaces = register_sculpting_hooks(
        model,
        target_modules=args.target_modules,
        block_retention=retention,
    )
    if rank_zero():
        print(f"[sculpting] projected matrices={len(subspaces)}", flush=True)
    train_sft_model(args, model, tokenizer)


if __name__ == "__main__":
    main()
