#!/usr/bin/env python3
"""SDFT training implementation using the vendored DistilTrainer."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from sst.data import build_sdft_dataset
from sst.training import rank_zero, worker_parser
from sst.vendor.sdft import DistilConfig, DistilTrainer


def main() -> None:
    parser = worker_parser("Train with SDFT")
    parser.add_argument("--generation-prompts-per-device", type=int, required=True)
    parser.add_argument(
        "--demo-mode",
        choices=["full", "no_tags", "no_think"],
        required=True,
    )
    parser.add_argument("--kl-alpha", type=float, required=True)
    parser.add_argument("--ref-model-mixup-alpha", type=float, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--max-completion-length", type=int, required=True)
    parser.add_argument("--vllm-mode", choices=["colocate", "server"], required=True)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, required=True)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, required=True)
    args = parser.parse_args()

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.precision]
    student = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
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
        args.data_path,
        args.seed,
        args.demo_mode,
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
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=args.generation_prompts_per_device * world_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        num_iterations=1,
        num_generations=1,
        save_strategy="epoch",
        save_total_limit=max(1, int(args.num_train_epochs)),
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
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
            f"optimizer_grad_accum={args.gradient_accumulation_steps}",
            flush=True,
        )
    trainer.train(resume_from_checkpoint=checkpoint)
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    if rank_zero():
        tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
