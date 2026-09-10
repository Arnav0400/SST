#!/usr/bin/env python3
"""Verify the bundled Qwen3 dataset snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"
MANIFEST_PATH = DATA_ROOT / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    total_examples = 0
    total_bytes = 0

    for dataset_name, expected in manifest["datasets"].items():
        directory = DATA_ROOT / expected["directory"]
        state_path = directory / "state.json"
        info_path = directory / "dataset_info.json"
        arrow_path = directory / expected["arrow_file"]
        for required in (state_path, info_path, arrow_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"missing {required}; run `git lfs pull` from the repository root"
                )

        state = json.loads(state_path.read_text(encoding="utf-8"))
        state_files = [entry["filename"] for entry in state["_data_files"]]
        if state_files != [expected["arrow_file"]]:
            raise ValueError(
                f"{dataset_name}: state.json points to {state_files}, "
                f"expected {[expected['arrow_file']]}"
            )
        actual_bytes = arrow_path.stat().st_size
        if actual_bytes != expected["arrow_bytes"]:
            raise ValueError(
                f"{dataset_name}: byte size {actual_bytes}, "
                f"expected {expected['arrow_bytes']}"
            )
        actual_sha256 = sha256(arrow_path)
        if actual_sha256 != expected["sha256"]:
            raise ValueError(
                f"{dataset_name}: SHA-256 {actual_sha256}, "
                f"expected {expected['sha256']}"
            )

        total_examples += expected["examples"]
        total_bytes += actual_bytes
        print(
            f"ok {dataset_name}: {expected['examples']:,} examples, "
            f"{actual_bytes:,} bytes"
        )

    if total_examples != manifest["total_examples"]:
        raise ValueError(
            f"manifest example total {total_examples}, "
            f"expected {manifest['total_examples']}"
        )
    if total_bytes != manifest["total_arrow_bytes"]:
        raise ValueError(
            f"manifest byte total {total_bytes}, "
            f"expected {manifest['total_arrow_bytes']}"
        )
    print(f"ok total: {total_examples:,} examples, {total_bytes:,} Arrow bytes")


if __name__ == "__main__":
    verify()
