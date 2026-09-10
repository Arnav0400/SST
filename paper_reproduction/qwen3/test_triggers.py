"""Regression tests for the independent Qwen3 method triggers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from paper_reproduction.qwen3._shared import DATASETS, DATASET_PATHS, SHARED_HPARAMS
from paper_reproduction.qwen3.baselines.oplora.train import make_trigger as make_oplora
from paper_reproduction.qwen3.baselines.sculpting.train import make_trigger as make_sculpting
from paper_reproduction.qwen3.baselines.sdft.train import make_trigger as make_sdft
from paper_reproduction.qwen3.baselines.sft.train import make_trigger as make_sft
from paper_reproduction.qwen3.sst.train import make_trigger as make_sst


HERE = Path(__file__).resolve().parent
BUILDERS = {
    "sst": make_sst,
    "sft": make_sft,
    "sdft": make_sdft,
    "oplora": make_oplora,
    "sculpting": make_sculpting,
}


class PerMethodTriggerTest(unittest.TestCase):
    @staticmethod
    def option(command: list[str], flag: str) -> str:
        return command[command.index(flag) + 1]

    def trigger(
        self,
        method: str,
        dataset: str = "science",
        num_gpus: int = 8,
        data_path: str | None = None,
    ):
        argv = [
            "--dataset",
            dataset,
            "--num-gpus",
            str(num_gpus),
            "--output-dir",
            f"/tmp/qwen3-trigger-test/{dataset}/{method}",
            "--dry-run",
        ]
        if data_path is not None:
            argv.extend(["--data-path", data_path])
        return BUILDERS[method](argv)

    def test_each_method_has_its_own_trigger_config_and_worker(self) -> None:
        paths = [
            HERE / "sst/config.json",
            HERE / "sst/train.py",
            HERE / "baselines/sft/config.json",
            HERE / "baselines/sft/train.py",
            HERE / "baselines/sdft/config.json",
            HERE / "baselines/sdft/train.py",
            HERE / "baselines/oplora/config.json",
            HERE / "baselines/oplora/train.py",
            HERE / "baselines/sculpting/config.json",
            HERE / "baselines/sculpting/train.py",
            HERE / "workers/sft.py",
            HERE / "workers/sst.py",
            HERE / "workers/sdft.py",
            HERE / "workers/oplora.py",
            HERE / "workers/sculpting.py",
            HERE / "deepspeed/zero2.json",
            HERE / "deepspeed/zero3.json",
        ]
        self.assertTrue(all(path.is_file() for path in paths))

    def test_sst_has_only_rank_percent_and_no_legacy_mode_flags(self) -> None:
        config = json.loads((HERE / "sst/config.json").read_text(encoding="utf-8"))
        self.assertEqual(config, {"rank_percent": 99})
        trigger, _ = self.trigger("sst")
        self.assertEqual(self.option(trigger.command, "--rank_percent"), "99")
        for forbidden in (
            "--constraint_mode",
            "--proj_update",
            "--svd_rank_fraction",
            "--refresh_svd_every_k",
        ):
            self.assertNotIn(forbidden, trigger.command)

    def test_all_triggers_share_the_exact_qwen3_optimizer_recipe(self) -> None:
        expected = {
            "--model_name": "Qwen/Qwen3-8B",
            "--learning_rate": "5e-05",
            "--num_train_epochs": "2",
            "--seed": "42",
            "--warmup_steps": "10",
            "--lr_scheduler_type": "cosine",
            "--max_grad_norm": "1",
            "--grad_accum_steps": "2",
            "--max_length": "4096",
        }
        for method in BUILDERS:
            with self.subTest(method=method):
                trigger, _ = self.trigger(method)
                for flag, value in expected.items():
                    self.assertEqual(self.option(trigger.command, flag), value)
                self.assertEqual(SHARED_HPARAMS["global_batch_size"], 16)
                self.assertEqual(SHARED_HPARAMS["weight_decay"], 0.0)
                self.assertEqual(SHARED_HPARAMS["precision"], "bf16")

    def test_every_trigger_accepts_all_four_dataset_aliases(self) -> None:
        for alias, dataset_id in DATASETS.items():
            for method in BUILDERS:
                with self.subTest(alias=alias, method=method):
                    trigger, _ = self.trigger(method, alias)
                    self.assertEqual(trigger.dataset_id, dataset_id)
                    self.assertEqual(self.option(trigger.command, "--dataset_name"), dataset_id)

    def test_custom_data_path_is_forwarded(self) -> None:
        trigger, _ = self.trigger("sft", "custom", data_path="/tmp/custom-dataset")
        self.assertEqual(self.option(trigger.command, "--dataset_name"), "custom")
        self.assertEqual(self.option(trigger.command, "--data_path"), "/tmp/custom-dataset")

    def test_default_aliases_use_bundled_data(self) -> None:
        for alias, dataset_id in DATASETS.items():
            with self.subTest(alias=alias):
                trigger, _ = self.trigger("sft", alias)
                data_path = Path(self.option(trigger.command, "--data_path"))
                self.assertEqual(data_path, DATASET_PATHS[dataset_id])
                self.assertTrue(data_path.is_relative_to(HERE / "data"))
                self.assertTrue((data_path / "state.json").is_file())
                self.assertTrue((data_path / "dataset_info.json").is_file())
                self.assertTrue(
                    (data_path / "data-00000-of-00001.arrow").is_file()
                )

    def test_gpu_override_preserves_global_batch(self) -> None:
        for method in BUILDERS:
            with self.subTest(method=method):
                trigger, _ = self.trigger(method, num_gpus=4)
                self.assertEqual(trigger.grad_accum_steps, 4)
                self.assertEqual(self.option(trigger.command, "--grad_accum_steps"), "4")

    def test_oplora_matches_paper_method_configuration(self) -> None:
        trigger, config = self.trigger("oplora")
        self.assertEqual(config["projection_rank"], 128)
        self.assertEqual(config["lora_rank"], 32)
        self.assertEqual(config["lora_alpha"], 32)
        self.assertEqual(config["lora_dropout"], 0.05)
        self.assertEqual(
            config["target_modules"],
            ["q_proj", "v_proj", "up_proj", "down_proj", "o_proj"],
        )
        self.assertEqual(self.option(trigger.command, "--projection_rank"), "128")
        self.assertEqual(self.option(trigger.command, "--lora_dropout"), "0.05")

    def test_sculpting_uses_adaptive_paper_configuration(self) -> None:
        trigger, config = self.trigger("sculpting")
        self.assertEqual(config["minimum_retention_ratio"], 0.1)
        self.assertEqual(config["target_retention_ratio"], 0.8)
        self.assertIsNone(config["calibration_samples"])
        self.assertEqual(self.option(trigger.command, "--minimum_retention_ratio"), "0.1")
        self.assertEqual(self.option(trigger.command, "--target_retention_ratio"), "0.8")
        self.assertNotIn("--unfreeze_rank_ratio", trigger.command)
        self.assertNotIn("--post_step_param_proj", trigger.command)

    def test_commands_are_independent_of_root_production_trainers(self) -> None:
        forbidden = {"train_sft.py", "train_oplora.py", "train_osft.py", "sdft/train.py"}
        for method in BUILDERS:
            with self.subTest(method=method):
                trigger, _ = self.trigger(method)
                worker = Path(trigger.command[3])
                deepspeed = Path(self.option(trigger.command, "--deepspeed"))
                self.assertTrue(worker.is_relative_to(HERE))
                self.assertTrue(deepspeed.is_relative_to(HERE))
                self.assertTrue(worker.is_file())
                self.assertTrue(deepspeed.is_file())
                self.assertTrue(forbidden.isdisjoint(trigger.command))

    def test_invalid_gpu_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be preserved exactly"):
            self.trigger("sft", num_gpus=3)


if __name__ == "__main__":
    unittest.main()
