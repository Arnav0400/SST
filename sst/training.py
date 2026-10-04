"""Shared Hugging Face training utilities."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from sst.data import (
    SFTDataCollator,
    load_training_dataset,
    preprocess_sft,
)


def worker_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--num-train-epochs", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--lr-scheduler-type", required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--max-grad-norm", type=float, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), required=True)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--deepspeed", required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--per-device-train-batch-size", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=-1,
    )
    return parser


def load_model_and_tokenizer(
    model_name: str,
    precision: str,
    *,
    freeze: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, tokenizer


def training_arguments(
    args: argparse.Namespace,
    *,
    save_total_limit: int = 1,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=1,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        report_to=args.report_to,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        deepspeed=args.deepspeed,
    )


def prepared_sft_dataset(args: argparse.Namespace, tokenizer):
    raw = load_training_dataset(args.data_path, args.seed)
    return preprocess_sft(raw, tokenizer, args.thinking, args.max_length)


def train_sft_model(
    args: argparse.Namespace,
    model,
    tokenizer,
    *,
    callbacks=None,
    save_total_limit: int = 1,
) -> Trainer:
    trainer = Trainer(
        model=model,
        args=training_arguments(args, save_total_limit=save_total_limit),
        train_dataset=prepared_sft_dataset(args, tokenizer),
        data_collator=SFTDataCollator(tokenizer),
        callbacks=list(callbacks or []),
    )
    checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    trainer.train(resume_from_checkpoint=checkpoint)
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0:
        tokenizer.save_pretrained(final_dir)
    return trainer


def rank_zero() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
