"""Common Hugging Face worker setup for the independent method trainers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from common.data import (
    SFTDataCollator,
    load_training_dataset,
    preprocess_sft,
)


def worker_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_train_epochs", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmup_steps", type=int, required=True)
    parser.add_argument("--lr_scheduler_type", required=True)
    parser.add_argument("--max_grad_norm", type=float, required=True)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--deepspeed", required=True)
    parser.add_argument("--grad_accum_steps", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser


def load_model_and_tokenizer(model_name: str, *, freeze: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
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
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        report_to=args.report_to,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        weight_decay=0.0,
        max_grad_norm=args.max_grad_norm,
        deepspeed=args.deepspeed,
    )


def prepared_sft_dataset(args: argparse.Namespace, tokenizer):
    raw = load_training_dataset(args.dataset_name, args.seed, args.data_path)
    return preprocess_sft(raw, tokenizer, args.dataset_name, args.max_length)


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
