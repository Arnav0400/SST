"""Standalone, optimizer-independent SST wrapper tests (no Transformers required)."""

import copy
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

import sst_torch
from sst_torch import SST


class SSTOptimizerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = nn.Sequential(nn.Linear(6, 8), nn.Linear(8, 4))

    def test_public_api_exposes_the_optimizer_wrapper(self):
        self.assertEqual(sst_torch.__all__, ["SST", "DEFAULT_TARGET_MODULES"])
        self.assertIs(sst_torch.SST, SST)
        self.assertTrue(issubclass(SST, torch.optim.Optimizer))
        self.assertIn("q_proj", sst_torch.DEFAULT_TARGET_MODULES)

    def assert_protected(self, optimizer):
        parameters = dict(self.model.named_parameters())
        for item in optimizer.sst_state_dict()["protected_weights"]:
            matrix = parameters[item["name"]].detach()
            left, right = item["left"], item["right"]
            anchor = (left * item["singular_values"].unsqueeze(0)) @ right.mH
            delta = matrix - anchor
            self.assertLess((left.mH @ delta).abs().max().item(), 3e-6)
            self.assertLess((delta @ right).abs().max().item(), 3e-6)

    def test_inner_optimizers_keep_unprojected_state(self):
        names = (
            "SGD",
            "Adam",
            "AdamW",
            "RMSprop",
            "Adagrad",
            "Adadelta",
            "Adamax",
            "ASGD",
            "Rprop",
            "NAdam",
            "RAdam",
            "Adafactor",
            "Muon",
        )
        for name, refresh_every in ((name, every) for name in names for every in (None, 1)):
            if not hasattr(torch.optim, name):
                continue
            with self.subTest(optimizer=name, refresh_every=refresh_every):
                self.setUp()
                parameters = [self.model[0].weight] if name == "Muon" else self.model.parameters()
                inner = getattr(torch.optim, name)(parameters, lr=0.05)
                optimizer = SST(
                    inner,
                    model=self.model,
                    rank_fraction=0.5,
                    target_modules=["0"],
                    refresh_every=refresh_every,
                )
                raw_state = []
                inner.register_step_post_hook(lambda *_: raw_state.append(copy.deepcopy(inner.state_dict()["state"])))
                for parameter in self.model.parameters():
                    parameter.grad = torch.randn_like(parameter)
                optimizer.step()
                self.assertIs(optimizer.state, inner.state)
                self.assertIs(optimizer.param_groups, inner.param_groups)
                for key, state in raw_state[0].items():
                    for state_name, value in state.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(
                                value,
                                inner.state_dict()["state"][key][state_name],
                                rtol=0,
                                atol=0,
                            )
                        else:
                            self.assertEqual(value, inner.state_dict()["state"][key][state_name])
                self.assert_protected(optimizer)
                self.assertEqual(optimizer.num_updates, 1)
                self.assertEqual(optimizer.num_refreshes, 0 if refresh_every is None else 1)

    def test_manual_refresh_preserves_current_weights_and_inner_optimizer_state(self):
        inner = torch.optim.AdamW(self.model.parameters(), lr=0.05)
        optimizer = SST(inner, model=self.model, rank_fraction=0.5)
        self.model[0].weight.grad = torch.randn_like(self.model[0].weight)
        optimizer.step()
        with torch.no_grad():
            self.model[0].weight.add_(torch.randn_like(self.model[0].weight))
        weights = copy.deepcopy(self.model.state_dict())
        state = copy.deepcopy(inner.state_dict())
        optimizer.refresh()
        for key, value in weights.items():
            torch.testing.assert_close(value, self.model.state_dict()[key], rtol=0, atol=0)
        for key, values in state["state"].items():
            for field, value in values.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, inner.state_dict()["state"][key][field], rtol=0, atol=0)
                else:
                    self.assertEqual(value, inner.state_dict()["state"][key][field])
        self.assertEqual(
            (
                optimizer.num_updates,
                optimizer.last_refresh_step,
                optimizer.num_refreshes,
            ),
            (1, 1, 1),
        )
        self.assert_protected(optimizer)

    def test_refresh_clock_is_restored_without_an_extra_svd(self):
        optimizer = SST(
            torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9),
            model=self.model,
            rank_fraction=0.5,
            refresh_every=3,
        )
        self.model[0].weight.grad = torch.randn_like(self.model[0].weight)
        for _ in range(2):
            optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())
        with patch("torch.linalg.svd", side_effect=AssertionError("Restore must not refresh")):
            restored = SST(
                torch.optim.SGD(self.model.parameters()),
                model=self.model,
                initialize=False,
            )
            restored.load_state_dict(state)
        self.assertEqual(
            (restored.refresh_every, restored.num_updates, restored.num_refreshes),
            (3, 2, 0),
        )
        restored.step()
        self.assertEqual(
            (restored.num_updates, restored.last_refresh_step, restored.num_refreshes),
            (3, 3, 1),
        )
        self.assert_protected(restored)

    def test_refresh_failures_do_not_replace_the_active_basis(self):
        optimizer = SST(
            torch.optim.AdamW(self.model.parameters()),
            model=self.model,
            rank_fraction=0.5,
        )
        before = copy.deepcopy(optimizer.sst_state_dict())
        with patch("torch.linalg.svd", side_effect=RuntimeError("Injected failure")):
            with self.assertRaisesRegex(RuntimeError, "active basis was not replaced"):
                optimizer.refresh()
        self.assertEqual(optimizer.num_refreshes, 0)
        for first, second in zip(before["protected_weights"], optimizer.sst_state_dict()["protected_weights"]):
            for key in ("left", "right", "singular_values"):
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_checkpoint_restores_fixed_basis_without_svd(self):
        optimizer = SST(
            torch.optim.AdamW(self.model.parameters(), lr=0.03),
            model=self.model,
            rank_fraction=0.5,
        )
        for parameter in self.model.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())
        with patch(
            "torch.linalg.svd",
            side_effect=AssertionError("Never reanchor at trained weights"),
        ):
            restored = SST(
                torch.optim.AdamW(self.model.parameters()),
                model=self.model,
                initialize=False,
            )
            restored.load_state_dict(state)
        self.assertEqual(restored.rank_fraction, 0.5)
        self.assertEqual(restored.param_groups[0]["lr"], 0.03)
        for first, second in zip(
            optimizer.sst_state_dict()["protected_weights"],
            restored.sst_state_dict()["protected_weights"],
        ):
            for name in ("left", "right", "singular_values"):
                torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        restored.step()
        self.assert_protected(restored)

    def test_standalone_wrapper_does_not_require_transformers(self):
        with patch.dict(sys.modules, {"transformers": None, "accelerate": None}):
            optimizer = SST(
                torch.optim.SGD(self.model.parameters(), lr=0.1),
                model=self.model,
                rank_fraction=0.5,
            )
            self.model[0].weight.grad = torch.randn_like(self.model[0].weight)
            optimizer.step()
            self.assert_protected(optimizer)

    def test_zero_ownership_includes_empty_local_fragments(self):
        for attribute in ("bit16_groups", "fp16_groups"):
            with self.subTest(groups=attribute):
                master = nn.Parameter(torch.zeros(8))
                inner = torch.optim.SGD([master], lr=0.1)
                setattr(inner, attribute, [[self.model[0].weight]])
                self.model[0].weight._hp_mapping = None
                optimizer = SST(inner, model=self.model, initialize=False)
                self.assertEqual([name for name, _ in optimizer._parameters()], ["0.weight"])


if __name__ == "__main__":
    unittest.main()
