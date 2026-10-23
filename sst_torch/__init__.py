"""Optimizer-independent Subspace Tuning for PyTorch training loops."""

from .optimizer import SST


__version__ = "0.2.0"

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

__all__ = ["SST", "DEFAULT_TARGET_MODULES"]
