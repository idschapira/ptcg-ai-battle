"""Pure-numpy inference for the policy+value net (submission runtime).

Mirrors PolicyValueNet from src/rl_models/network.py layer by layer; the
two implementations are kept in lockstep by tests/test_network_parity.py.
No torch anywhere on the import path — torch is only touched inside
export_weights(), a dev-time function that converts the trained .pt
checkpoint into models/policy_value.npz.

Runtime path: NumpyPolicyValueNet.load() -> forward(state, legal_options)
-> (logits over the legal options, win-prob value in [-1, 1]). Loading is
None-safe: a missing/corrupt weights file returns None and the caller
(NetworkAgent) falls back to a legal answer.

Export (dev, repo root):
    python -m src.rl_models.network_numpy --export
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Final

import numpy as np

from .encoding import ENCODING_DIM, OPTION_DIM
from .normalization import MODELS_DIR

WEIGHTS_PATH: Final[Path] = MODELS_DIR / "policy_value.npz"

# state_dict key -> npz key (fixed contract between network.py and here)
_KEY_MAP: Final[dict[str, str]] = {
    "trunk.0.weight": "trunk_w0", "trunk.0.bias": "trunk_b0",
    "trunk.2.weight": "trunk_w1", "trunk.2.bias": "trunk_b1",
    "option_proj.0.weight": "opt_w", "option_proj.0.bias": "opt_b",
    "query.weight": "query_w", "query.bias": "query_b",
    "option_bias.weight": "obias_w", "option_bias.bias": "obias_b",
    "value_head.0.weight": "val_w0", "value_head.0.bias": "val_b0",
    "value_head.2.weight": "val_w1", "value_head.2.bias": "val_b1",
}


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class NumpyPolicyValueNet:
    """Forward pass of PolicyValueNet with numpy matmuls only."""

    __slots__ = ("_w", "_emb_scale")

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        missing = [key for key in _KEY_MAP.values() if key not in weights]
        if missing:
            raise KeyError(f"weights file missing arrays: {missing}")
        self._w = {key: np.ascontiguousarray(weights[key], dtype=np.float32)
                   for key in _KEY_MAP.values()}
        self._emb_scale = 1.0 / math.sqrt(self._w["query_w"].shape[0])

    @classmethod
    def load(cls, path: Path = WEIGHTS_PATH) -> "NumpyPolicyValueNet | None":
        """None-safe load: any failure returns None (caller falls back)."""
        try:
            with np.load(path) as data:
                return cls({key: data[key] for key in data.files})
        except (OSError, KeyError, ValueError):
            return None

    def forward(self, state: np.ndarray,
                options: np.ndarray) -> tuple[np.ndarray, float]:
        """state [ENCODING_DIM], options [K, OPTION_DIM] (legal only, K>=1)
        -> (logits [K], value). Inputs must already be normalized."""
        w = self._w
        state_emb = _relu(w["trunk_w0"] @ state + w["trunk_b0"])
        state_emb = _relu(w["trunk_w1"] @ state_emb + w["trunk_b1"])

        option_emb = _relu(options @ w["opt_w"].T + w["opt_b"])      # [K, E]
        query = w["query_w"] @ state_emb + w["query_b"]              # [E]
        logits = (option_emb @ query) * self._emb_scale
        logits = logits + option_emb @ w["obias_w"][0] + w["obias_b"][0]

        hidden = _relu(w["val_w0"] @ state_emb + w["val_b0"])
        value = math.tanh(float((w["val_w1"] @ hidden)[0] + w["val_b1"][0]))
        return logits.astype(np.float32), value


# --------------------------------------------------------------------------- #
# Dev-time export (the only place torch is touched, lazily)
# --------------------------------------------------------------------------- #


def export_state_dict(state_dict: dict, path: Path = WEIGHTS_PATH) -> Path:
    """torch state_dict -> npz with the _KEY_MAP contract."""
    arrays: dict[str, np.ndarray] = {}
    for torch_key, npz_key in _KEY_MAP.items():
        if torch_key not in state_dict:
            raise KeyError(f"state_dict missing {torch_key} — did the "
                           f"architecture in network.py change without "
                           f"updating _KEY_MAP?")
        arrays[npz_key] = state_dict[torch_key].detach().cpu().numpy().astype(np.float32)
    expected = {"trunk_w0": (None, ENCODING_DIM), "opt_w": (None, OPTION_DIM)}
    for key, (_, dim) in expected.items():
        if arrays[key].shape[1] != dim:
            raise ValueError(f"{key} second dim {arrays[key].shape[1]} != {dim}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def export_weights(checkpoint_path: Path | None = None,
                   out_path: Path = WEIGHTS_PATH) -> Path:
    import torch

    from .network import CHECKPOINT_PATH
    source = checkpoint_path if checkpoint_path is not None else CHECKPOINT_PATH
    state_dict = torch.load(source, map_location="cpu", weights_only=True)
    return export_state_dict(state_dict, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true",
                        help="convert models/policy_value.pt -> .npz")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    args = parser.parse_args()
    if not args.export:
        parser.error("nothing to do (pass --export)")
    path = export_weights(args.checkpoint, args.out)
    size_kib = path.stat().st_size / 1024
    net = NumpyPolicyValueNet.load(path)
    assert net is not None, "exported weights failed to load back"
    print(f"exported {path} ({size_kib:,.0f} KiB), reload OK")


if __name__ == "__main__":
    main()
