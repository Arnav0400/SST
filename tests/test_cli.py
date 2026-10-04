"""Tests for the public SST configuration and launcher interface."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from sst.cli import build_plan
from sst.config import METHODS, ROOT


class SSTCliTest(unittest.TestCase):
    @staticmethod
    def option(command: list[str], flag: str) -> str:
        return command[command.index(flag) + 1]

    def plan(
        self,
        method: str,
        dataset: str = "science",
        *,
        num_gpus: int = 8,
        data_path: str | None = None,
    ):
        argv = [
            "--method",
            method,
            "--dataset",
            dataset,
            "--num-gpus",
            str(num_gpus),
            "--output-dir",
            f"/tmp/sst-tests/{dataset}/{method}",
            "--dry-run",
        ]
        if data_path:
            argv.extend(["--data-path", data_path])
        return build_plan(argv)

    def test_all_methods_resolve(self) -> None:
        for method in METHODS:
            with self.subTest(method=method):
                plan = self.plan(method)
                self.assertEqual(plan.method, method)
                self.assertTrue(Path(plan.command[3]).is_file())
                self.assertTrue(
                    Path(self.option(plan.command, "--deepspeed")).is_file()
                )

    def test_all_datasets_resolve_for_every_method(self) -> None:
        expected = {
            "science": "qwen3_science",
            "medicine": "qwen3_medicine",
            "pathology": "qwen3_pathology",
            "finqa": "qwen3_finqa",
        }
        for dataset, dataset_id in expected.items():
            for method in METHODS:
                with self.subTest(dataset=dataset, method=method):
                    plan = self.plan(method, dataset)
                    self.assertEqual(plan.dataset.dataset_id, dataset_id)
                    self.assertEqual(
                        self.option(plan.command, "--dataset-name"),
                        dataset_id,
                    )

    def test_sst_exposes_only_rank_percent(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/method/sst.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config, {"rank_percent": 99})
        command = self.plan("sst").command
        self.assertEqual(self.option(command, "--rank-percent"), "99")
        for forbidden in (
            "--constraint-mode",
            "--proj-update",
            "--svd-rank-fraction",
            "--refresh-svd-every-k",
        ):
            self.assertNotIn(forbidden, command)

    def test_shared_qwen3_defaults(self) -> None:
        expected = {
            "--model-name": "Qwen/Qwen3-8B",
            "--learning-rate": "5e-05",
            "--num-train-epochs": "2",
            "--seed": "42",
            "--warmup-steps": "10",
            "--lr-scheduler-type": "cosine",
            "--weight-decay": "0",
            "--max-grad-norm": "1",
            "--precision": "bf16",
            "--gradient-accumulation-steps": "2",
            "--per-device-train-batch-size": "1",
            "--max-length": "4096",
        }
        for method in METHODS:
            with self.subTest(method=method):
                command = self.plan(method).command
                for flag, value in expected.items():
                    self.assertEqual(self.option(command, flag), value)

    def test_custom_dataset_path(self) -> None:
        plan = self.plan("sst", "custom", data_path="/tmp/custom-dataset")
        self.assertEqual(plan.dataset.dataset_id, "custom")
        self.assertEqual(
            self.option(plan.command, "--data-path"),
            "/tmp/custom-dataset",
        )

    def test_gpu_override_preserves_global_batch(self) -> None:
        plan = self.plan("sst", num_gpus=4)
        self.assertEqual(plan.gradient_accumulation_steps, 4)
        self.assertEqual(
            self.option(plan.command, "--gradient-accumulation-steps"),
            "4",
        )

    def test_invalid_gpu_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be preserved exactly"):
            self.plan("sst", num_gpus=3)

    def test_method_specific_configs(self) -> None:
        oplora = self.plan("oplora")
        self.assertEqual(oplora.method_config["projection_rank"], 128)
        self.assertEqual(
            self.option(oplora.command, "--projection-rank"),
            "128",
        )
        sculpting = self.plan("sculpting")
        self.assertEqual(
            sculpting.method_config["minimum_retention_ratio"],
            0.1,
        )
        self.assertNotIn("--calibration-samples", sculpting.command)

    def test_public_scripts_exist(self) -> None:
        for method in METHODS:
            path = ROOT / "scripts" / f"train_{method}.sh"
            with self.subTest(method=method):
                self.assertTrue(path.is_file())

    def test_official_root_layout(self) -> None:
        self.assertTrue((ROOT / "pyproject.toml").is_file())
        self.assertTrue((ROOT / "sst/cli.py").is_file())
        self.assertTrue((ROOT / "configs/model/qwen3-8b.yaml").is_file())
        self.assertFalse((ROOT / "baselines").exists())
        self.assertFalse((ROOT / "workers").exists())


if __name__ == "__main__":
    unittest.main()
