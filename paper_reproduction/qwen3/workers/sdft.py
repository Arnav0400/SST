#!/usr/bin/env python3
"""Independent SDFT worker using the vendored DistilTrainer implementation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_reproduction.qwen3.common.data import build_sdft_dataset  # noqa: E402
from paper_reproduction.qwen3.common.training import rank_zero, worker_parser  # noqa: E402
from paper_reproduction.qwen3.vendor.sdft import DistilConfig, DistilTrainer  # noqa: E402


def main() -> None:
    parser = worker_parser("Independent SDFT worker")
    parser.add_argument("--generation_prompts_per_device", type=int, required=True)
    parser.add_argument("--demo_mode", choices=["full", "no_tags", "no_think"], required=True)
    parser.add_argument("--kl_alpha", type=float, required=True)
    parser.add_argument("--ref_model_mixup_alpha", type=float, required=True)
    parser.add_argument("--max_prompt_length", type=int, required=True)
    parser.add_argument("--max_completion_length", type=int, required=True)
    parser.add_argument("--vllm_mode", choices=["colocate", "server"], required=True)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, required=True)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, required=True)
    args = parser.parse_args()

    student = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_apply = tokenizer.apply_chat_template

    def thinking_apply(*positional, **keywords):
        keywords.setdefault("enable_thinking", True)
        return original_apply(*positional, **keywords)

    tokenizer.apply_chat_template = thinking_apply
    dataset = build_sdft_dataset(
        args.dataset_name,
        args.seed,
        args.demo_mode,
        args.data_path,
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config = DistilConfig(
        seed=args.seed,
        alpha=args.kl_alpha,
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum_steps,
        generation_batch_size=args.generation_prompts_per_device * world_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        num_iterations=1,
        num_generations=1,
        save_strategy="epoch",
        save_total_limit=max(1, int(args.num_train_epochs)),
        max_grad_norm=args.max_grad_norm,
        weight_decay=0.0,
        report_to=args.report_to,
        run_name=args.run_name,
        output_dir=args.output_dir,
        log_completions=False,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=0,
        deepspeed=args.deepspeed,
    )
    trainer = DistilTrainer(
        model=student,
        ref_model=teacher,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if rank_zero():
        print(
            f"[sdft] generation_batch={args.generation_prompts_per_device * world_size}; "
            f"optimizer_grad_accum={args.grad_accum_steps}",
            flush=True,
        )
    trainer.train(resume_from_checkpoint=checkpoint)
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    if rank_zero():
        tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
