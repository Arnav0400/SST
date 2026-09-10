#!/usr/bin/env python3
"""Independent OPLoRA worker implementing arXiv:2510.13003."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.data import SFTDataCollator  # noqa: E402
from common.training import (  # noqa: E402
    load_model_and_tokenizer,
    prepared_sft_dataset,
    rank_zero,
    training_arguments,
    worker_parser,
)


def _svd_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


class OPLoRALinear(nn.Module):
    """Frozen linear layer plus a doubly projected LoRA update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        projection_rank: int,
        dropout: float,
    ):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        output_size, input_size = base.weight.shape
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, input_size, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(output_size, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

        maximum = min(output_size, input_size)
        k = min(max(1, projection_rank), maximum - 1)
        device = _svd_device()
        weight = base.weight.detach().to(device=device, dtype=torch.float32)
        q = min(maximum, k + 8)
        left, _, right = torch.svd_lowrank(weight, q=q, niter=2)
        self.register_buffer("left", left[:, :k].contiguous().cpu(), persistent=False)
        self.register_buffer("right", right[:, :k].contiguous().cpu(), persistent=False)
        self.projection_rank = k
        del weight, left, right
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        right = self.right.to(inputs.dtype)
        update_inputs = self.dropout(inputs)
        update_inputs = update_inputs - (update_inputs @ right) @ right.T
        update = (update_inputs @ self.lora_a.to(inputs.dtype).T) @ self.lora_b.to(
            inputs.dtype
        ).T
        left = self.left.to(inputs.dtype)
        update = update - (update @ left) @ left.T
        return output + self.scaling * update

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        update = (self.lora_b.float() @ self.lora_a.float()) * self.scaling
        left = self.left.float()
        right = self.right.float()
        update = update - left @ (left.T @ update)
        update = update - (update @ right) @ right.T
        return (self.base.weight.float() + update).to(self.base.weight.dtype)


def inject_oplora(
    model,
    *,
    target_modules: list[str],
    rank: int,
    alpha: float,
    projection_rank: int,
    dropout: float,
) -> list[str]:
    wrapped = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not any(
            name.endswith(target) for target in target_modules
        ):
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        setattr(
            parent,
            child,
            OPLoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                projection_rank=projection_rank,
                dropout=dropout,
            ),
        )
        wrapped.append(name)
        if rank_zero() and len(wrapped) % 32 == 0:
            print(f"[oplora] wrapped {len(wrapped)} matrices", flush=True)
    if not wrapped:
        raise RuntimeError("OPLoRA found no target modules")
    return wrapped


@torch.no_grad()
def merge_and_unload(model) -> int:
    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, OPLoRALinear):
            continue
        base = module.base
        base.weight.data.copy_(module.merged_weight())
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], base)
        merged += 1
    return merged


def main() -> None:
    parser = worker_parser("Independent OPLoRA worker")
    parser.add_argument("--lora_rank", type=int, required=True)
    parser.add_argument("--lora_alpha", type=float, required=True)
    parser.add_argument("--lora_dropout", type=float, required=True)
    parser.add_argument("--projection_rank", type=int, required=True)
    parser.add_argument("--target_modules", nargs="+", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model_name, freeze=True)
    wrapped = inject_oplora(
        model,
        target_modules=args.target_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        projection_rank=args.projection_rank,
        dropout=args.lora_dropout,
    )
    model.enable_input_require_grads()
    if rank_zero():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[oplora] variant=OPLoRA-{args.projection_rank}; "
            f"wrapped={len(wrapped)}; trainable={trainable:,}",
            flush=True,
        )

    trainer = Trainer(
        model=model,
        args=training_arguments(args),
        train_dataset=prepared_sft_dataset(args, tokenizer),
        data_collator=SFTDataCollator(tokenizer),
    )
    checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    trainer.train(resume_from_checkpoint=checkpoint)
    merged = merge_and_unload(model)
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    if rank_zero():
        tokenizer.save_pretrained(final_dir)
        print(f"[oplora] merged {merged} adapters; saved {final_dir}", flush=True)


if __name__ == "__main__":
    main()
