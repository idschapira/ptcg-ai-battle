"""Value head for Crustle positions: torch trainer + numpy inference.

One number, one convention: ``value(position)`` is the win probability
for the player TO MOVE, mapped to [-1, 1] by tanh. The search negates it
whenever the node it landed on is the opponent's turn.

The architecture is deliberately the value PATH of PolicyValueNet
(network.py) and nothing else — trunk 1185->512->256, head 256->64->1 —
because the leaf never needs a policy. Dropping the option tower is not
a simplification for its own sake: it removes the option encoding from
the leaf's cost, which is what makes a leaf 0.24ms instead of 0.5ms.

    ValueNet            torch, DEV ONLY (training + export)
    NumpyValueNet       pure numpy, submission runtime, BATCHED

Batching matters. A single forward is ~0.17ms and 32 of them are
~1.0ms — the matmuls are latency-bound, not throughput-bound, so a
search that evaluates its frontier in one call gets ~5x more leaves per
second than one that evaluates node by node. NumpyValueNet.value_batch
is the interface the search is expected to use.

HONESTY OF THE BASELINE. Brier against a constant 0.5 predictor is
0.25, but that is not the baseline this head has to beat: our winrate
is nowhere near 50% in most cells, so a model that learned nothing but
the BASE RATE would already score far better than 0.25. Every report
here prints the base-rate Brier alongside, and the skill score is
computed against THAT. Beating 0.25 is necessary, not sufficient.

Run from the repo root (dev):
    python -m src.rl_models.value_net train --epochs 20
    python -m src.rl_models.value_net report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np

from .encoding import ENCODING_DIM
from .normalization import MODELS_DIR

VALUE_NPZ: Final[Path] = MODELS_DIR / "value_crustle.npz"
VALUE_CKPT: Final[Path] = MODELS_DIR / "value_crustle.pt"
VALUE_METRICS: Final[Path] = MODELS_DIR / "value_crustle.metrics.json"

HIDDEN_DIM: Final[int] = 512
EMB_DIM: Final[int] = 256
VALUE_HIDDEN: Final[int] = 64

# state_dict key -> npz key. Same contract style as network_numpy._KEY_MAP.
_KEY_MAP: Final[dict[str, str]] = {
    "trunk.0.weight": "trunk_w0", "trunk.0.bias": "trunk_b0",
    "trunk.2.weight": "trunk_w1", "trunk.2.bias": "trunk_b1",
    "head.0.weight": "val_w0", "head.0.bias": "val_b0",
    "head.2.weight": "val_w1", "head.2.bias": "val_b1",
}


# --------------------------------------------------------------------------- #
# Runtime: pure numpy, batched
# --------------------------------------------------------------------------- #


class NumpyValueNet:
    """Batched numpy forward. No torch on this import path.

    float32 throughout: unlike network_numpy (which accumulates in
    float64 to hold parity with torch below 1e-5), this net's output is
    only ever COMPARED between sibling nodes. Ranking is insensitive to
    fp32 reduction noise, and float32 matmuls are ~2x faster — which at
    the leaf of a search is the whole point.
    """

    __slots__ = ("_w", "_mean", "_std")

    def __init__(self, weights: dict[str, np.ndarray],
                 state_mean: np.ndarray | None = None,
                 state_std: np.ndarray | None = None) -> None:
        missing = [k for k in _KEY_MAP.values() if k not in weights]
        if missing:
            raise KeyError(f"weights file missing arrays: {missing}")
        # store transposed so the batched path is x @ W with no copy
        self._w = {
            "w0": np.ascontiguousarray(weights["trunk_w0"].T, np.float32),
            "b0": np.ascontiguousarray(weights["trunk_b0"], np.float32),
            "w1": np.ascontiguousarray(weights["trunk_w1"].T, np.float32),
            "b1": np.ascontiguousarray(weights["trunk_b1"], np.float32),
            "w2": np.ascontiguousarray(weights["val_w0"].T, np.float32),
            "b2": np.ascontiguousarray(weights["val_b0"], np.float32),
            "w3": np.ascontiguousarray(weights["val_w1"].T, np.float32),
            "b3": np.ascontiguousarray(weights["val_b1"], np.float32),
        }
        dim = self._w["w0"].shape[0]
        self._mean = (np.zeros(dim, np.float32) if state_mean is None
                      else np.ascontiguousarray(state_mean, np.float32))
        self._std = (np.ones(dim, np.float32) if state_std is None
                     else np.ascontiguousarray(state_std, np.float32))

    @classmethod
    def load(cls, path: Path = VALUE_NPZ,
             stats_path: Path | None = None) -> "NumpyValueNet | None":
        """None-safe: any failure returns None and the caller falls back.

        The normalization stats are baked into the same file at export
        time, so a value net can never be paired with the wrong stats —
        the failure mode that CLAUDE.md calls out for the policy pair.
        """
        try:
            with np.load(path) as data:
                weights = {k: data[k] for k in data.files}
            mean = weights.pop("state_mean", None)
            std = weights.pop("state_std", None)
            if stats_path is not None:
                with np.load(stats_path) as sdata:
                    mean = sdata["state_mean"]
                    std = sdata["state_std"]
            return cls(weights, mean, std)
        except (OSError, KeyError, ValueError):
            return None

    def value_batch(self, states: np.ndarray) -> np.ndarray:
        """states [B, ENCODING_DIM] RAW (unnormalized) -> value [B] in [-1,1].

        Normalization is applied here so callers hand over encoder
        output directly and cannot forget it.
        """
        w = self._w
        x = np.asarray(states, np.float32)
        if x.ndim == 1:
            x = x[None, :]
        x = (x - self._mean) / self._std
        h = np.maximum(x @ w["w0"] + w["b0"], 0.0)
        h = np.maximum(h @ w["w1"] + w["b1"], 0.0)
        h = np.maximum(h @ w["w2"] + w["b2"], 0.0)
        return np.tanh(h @ w["w3"] + w["b3"]).reshape(-1)

    def value(self, state: np.ndarray) -> float:
        return float(self.value_batch(state)[0])


__all__ = ["NumpyValueNet", "VALUE_NPZ", "ValueNet", "export_value_net"]


# --------------------------------------------------------------------------- #
# Dev only below this line (torch)
# --------------------------------------------------------------------------- #


def _torch():
    import torch
    return torch


class _ValueNetMeta(type):
    """Builds the nn.Module lazily so importing this file never needs torch."""

    _cls = None

    def __call__(cls, *args, **kwargs):
        if _ValueNetMeta._cls is None:
            _torch()
            from torch import nn

            class _ValueNet(nn.Module):
                """Dropout is applied FUNCTIONALLY, between the Sequential
                slices, so the state_dict keys stay trunk.0/trunk.2 and
                head.0/head.2 and _KEY_MAP keeps working. Inserting
                nn.Dropout as a layer would renumber them and silently
                break the export contract."""

                def __init__(self, dropout: float = 0.0) -> None:
                    super().__init__()
                    self.trunk = nn.Sequential(
                        nn.Linear(ENCODING_DIM, HIDDEN_DIM), nn.ReLU(),
                        nn.Linear(HIDDEN_DIM, EMB_DIM), nn.ReLU(),
                    )
                    self.head = nn.Sequential(
                        nn.Linear(EMB_DIM, VALUE_HIDDEN), nn.ReLU(),
                        nn.Linear(VALUE_HIDDEN, 1), nn.Tanh(),
                    )
                    self.drop = nn.Dropout(dropout)

                def forward(self, states):
                    h = self.drop(self.trunk[:2](states))
                    h = self.drop(self.trunk[2:](h))
                    return self.head(h).squeeze(-1)

            _ValueNetMeta._cls = _ValueNet
        return _ValueNetMeta._cls(*args, **kwargs)


class ValueNet(metaclass=_ValueNetMeta):
    """torch nn.Module (dev). Instantiating imports torch lazily."""


def export_value_net(state_dict: dict, state_mean: np.ndarray,
                     state_std: np.ndarray,
                     path: Path = VALUE_NPZ) -> Path:
    """torch state_dict + normalization stats -> one self-contained npz."""
    arrays: dict[str, np.ndarray] = {}
    for torch_key, npz_key in _KEY_MAP.items():
        if torch_key not in state_dict:
            raise KeyError(f"state_dict missing {torch_key} — architecture "
                           f"changed without updating _KEY_MAP?")
        arrays[npz_key] = (state_dict[torch_key].detach().cpu().numpy()
                           .astype(np.float32))
    if arrays["trunk_w0"].shape[1] != ENCODING_DIM:
        raise ValueError(f"trunk_w0 second dim {arrays['trunk_w0'].shape[1]} "
                         f"!= {ENCODING_DIM}")
    arrays["state_mean"] = np.asarray(state_mean, np.float32)
    arrays["state_std"] = np.asarray(state_std, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path
