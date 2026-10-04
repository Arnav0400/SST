"""Dataset loading and chat-format preprocessing."""

from __future__ import annotations

import re
from pathlib import Path
from string import Template
from typing import Any

import torch
from datasets import Dataset, load_from_disk

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_TAGS = re.compile(r"</?think>")


def dataset_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {path}")
    return path


def load_training_dataset(
    path_value: str | Path,
    seed: int,
) -> Dataset:
    return load_from_disk(str(dataset_path(path_value))).shuffle(seed=seed)


def _reasoning_open_suffix(tokenizer) -> str:
    try:
        messages = [{"role": "user", "content": "x"}]
        generated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        plain = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        extra = generated[len(plain):] if generated.startswith(plain) else generated
        match = re.search(r"(<think>|<\|think\|>|<reasoning>)\s*$", extra)
        return match.group(1) if match else ""
    except Exception:
        return ""


def preprocess_sft(
    dataset: Dataset,
    tokenizer,
    thinking: bool,
    max_length: int,
) -> Dataset:
    open_token = _reasoning_open_suffix(tokenizer)
    close_token = {
        "<think>": "</think>",
        "<|think|>": "<|/think|>",
        "<reasoning>": "</reasoning>",
    }.get(open_token, "")

    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        messages = example["messages"]
        response = example["output_text"].strip()
        if open_token and close_token and open_token not in response and close_token not in response:
            response = f"{open_token}\n\n{close_token}\n\n{response}"
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": response}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=thinking,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        prefix = min(len(prompt_ids), len(full_ids))
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": [-100] * prefix + full_ids[prefix:],
        }

    processed = dataset.map(preprocess, remove_columns=dataset.column_names)
    return processed.filter(lambda row: any(label != -100 for label in row["labels"]))


class SFTDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        length = max(len(feature["input_ids"]) for feature in features)

        def padded(key: str, pad_value: int) -> list[list[int]]:
            return [
                feature[key] + [pad_value] * (length - len(feature[key]))
                for feature in features
            ]

        return {
            "input_ids": torch.tensor(
                padded("input_ids", self.tokenizer.pad_token_id), dtype=torch.long
            ),
            "attention_mask": torch.tensor(padded("attention_mask", 0), dtype=torch.long),
            "labels": torch.tensor(padded("labels", -100), dtype=torch.long),
        }


def build_sdft_dataset(
    path_value: str | Path,
    seed: int,
    demo_mode: str,
) -> Dataset:
    dataset = load_training_dataset(path_value, seed)
    template = Template(
        "$orig_content\n\n"
        "This is an example for a response to the question:\n"
        "$output_text\n\n"
        "Now answer with a response of your own, including the thinking process.\n"
    )

    def demo(text: str) -> str:
        if demo_mode == "full":
            return text.strip()
        if demo_mode == "no_tags":
            return _THINK_TAGS.sub("", text).strip()
        if demo_mode == "no_think":
            return _THINK_BLOCK.sub("", text).strip()
        raise ValueError(f"unsupported demo_mode={demo_mode!r}")

    def format_example(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                {
                    "role": "user",
                    "content": template.substitute(
                        orig_content=example["messages"][-1]["content"],
                        output_text=demo(example["output_text"]),
                    ),
                }
            ],
        }

    return dataset.map(format_example, remove_columns=dataset.column_names)
