"""Policy+value network and behavioral-cloning trainer (PyTorch, DEV ONLY).

NEVER imported at submission runtime: the packaged agent runs the same
forward pass in pure numpy (src/rl_models/network_numpy.py) from weights
exported to models/policy_value.npz. Keep this module's architecture and
network_numpy.forward in lockstep — tests/test_network_parity.py asserts
torch and numpy outputs match.

Architecture (small on purpose: latency is the constraint, not capacity):
    trunk        state[1185] -> 512 ReLU -> 256 ReLU        (shared emb)
    option_proj  option[154] -> 256 ReLU                    (per option)
    policy       logit_i = (option_emb_i · query(state_emb)) / sqrt(256)
                           + option_bias(option_emb_i)      (pointer style)
                 illegal/padded options masked to -inf, softmax over legal
    value        state_emb -> 64 ReLU -> 1 tanh             (win prob in [-1,1])

Training: masked cross-entropy (imitate the heuristic) + MSE on z, split
train/val by game id (no leakage across the same game).

Usage (repo root):
    python -m src.rl_models.network --epochs 10
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn

from .bc_dataset import DATASET_PATH
from .encoding import ENCODING_DIM, MAX_OPTIONS, OPTION_DIM
from .normalization import MODELS_DIR, FeatureStats

CHECKPOINT_PATH: Final[Path] = MODELS_DIR / "policy_value.pt"

HIDDEN_DIM: Final[int] = 512
EMB_DIM: Final[int] = 256
VALUE_HIDDEN: Final[int] = 64
VALUE_LOSS_WEIGHT: Final[float] = 0.5
NEG_INF: Final[float] = -1e9


class PolicyValueNet(nn.Module):
    """Pointer-style policy over encoded options + scalar value head."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(ENCODING_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, EMB_DIM), nn.ReLU(),
        )
        self.option_proj = nn.Sequential(nn.Linear(OPTION_DIM, EMB_DIM), nn.ReLU())
        self.query = nn.Linear(EMB_DIM, EMB_DIM)
        self.option_bias = nn.Linear(EMB_DIM, 1)
        self.value_head = nn.Sequential(
            nn.Linear(EMB_DIM, VALUE_HIDDEN), nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 1), nn.Tanh(),
        )

    def forward(self, states: Tensor, options: Tensor,
                mask: Tensor) -> tuple[Tensor, Tensor]:
        """states [B,D], options [B,K,O], mask [B,K] bool -> (logits [B,K], value [B])."""
        state_emb = self.trunk(states)                        # [B, E]
        option_emb = self.option_proj(options)                # [B, K, E]
        query = self.query(state_emb).unsqueeze(-1)           # [B, E, 1]
        logits = (option_emb @ query).squeeze(-1) / math.sqrt(EMB_DIM)
        logits = logits + self.option_bias(option_emb).squeeze(-1)
        logits = logits.masked_fill(~mask, NEG_INF)
        value = self.value_head(state_emb).squeeze(-1)        # [B]
        return logits, value


# --------------------------------------------------------------------------- #
# Dataset plumbing (ragged npz -> padded batches, normalized once up front)
# --------------------------------------------------------------------------- #


@dataclass
class BcArrays:
    states: np.ndarray        # [N, ENCODING_DIM] normalized float32
    options_flat: np.ndarray  # [sum(counts), OPTION_DIM] normalized float32
    offsets: np.ndarray       # [N+1] int64 into options_flat
    counts: np.ndarray        # [N] int64
    labels: np.ndarray        # [N] int64
    values: np.ndarray        # [N] float32
    game_ids: np.ndarray      # [N] int64 (bc: game_ids; replays: episode_ids)

    @classmethod
    def load(cls, path: Path, stats: FeatureStats) -> "BcArrays":
        with np.load(path) as data:
            counts = data["option_counts"].astype(np.int64)
            offsets = np.zeros(len(counts) + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])
            ids_key = "game_ids" if "game_ids" in data else "episode_ids"
            return cls(
                states=stats.normalize_state(data["states"]),
                options_flat=stats.normalize_options(data["options_flat"]),
                offsets=offsets,
                counts=counts,
                labels=data["labels"].astype(np.int64),
                values=data["values"].astype(np.float32),
                game_ids=data[ids_key].astype(np.int64),
            )

    def batch(self, indices: np.ndarray) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Materialize one padded batch (only here — never the full tensor)."""
        batch_size = len(indices)
        options = np.zeros((batch_size, MAX_OPTIONS, OPTION_DIM), dtype=np.float32)
        mask = np.zeros((batch_size, MAX_OPTIONS), dtype=bool)
        for row, sample in enumerate(indices):
            count = min(int(self.counts[sample]), MAX_OPTIONS)
            start = self.offsets[sample]
            options[row, :count] = self.options_flat[start:start + count]
            mask[row, :count] = True
        return (torch.from_numpy(self.states[indices]),
                torch.from_numpy(options),
                torch.from_numpy(mask),
                torch.from_numpy(self.labels[indices]),
                torch.from_numpy(self.values[indices]))


def split_by_game(game_ids: np.ndarray,
                  val_fraction_mod: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/val split on game id (game_id % 5 == 0 -> val)."""
    is_val = (game_ids % val_fraction_mod) == 0
    indices = np.arange(len(game_ids))
    return indices[~is_val], indices[is_val]


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(net: PolicyValueNet, arrays: BcArrays, indices: np.ndarray,
             batch_size: int) -> tuple[float, float]:
    """(top-1 accuracy vs the heuristic label, MAE of the value head)."""
    net.eval()
    correct = 0
    abs_error = 0.0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        states, options, mask, labels, values = arrays.batch(chunk)
        logits, value = net(states, options, mask)
        correct += int((logits.argmax(dim=1) == labels).sum())
        abs_error += float((value - values).abs().sum())
    return correct / max(len(indices), 1), abs_error / max(len(indices), 1)


def train(dataset_path: Path = DATASET_PATH, epochs: int = 10,
          batch_size: int = 512, lr: float = 1e-3, seed: int = 0,
          checkpoint_path: Path = CHECKPOINT_PATH) -> dict[str, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    stats = FeatureStats.load()
    arrays = BcArrays.load(dataset_path, stats)
    train_idx, val_idx = split_by_game(arrays.game_ids)
    print(f"dataset: {len(arrays.labels)} samples "
          f"({len(train_idx)} train / {len(val_idx)} val, split by game)")

    net = PolicyValueNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best = {"val_top1": 0.0, "val_value_mae": float("inf"), "epoch": -1}
    for epoch in range(epochs):
        net.train()
        rng.shuffle(train_idx)
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_idx), batch_size):
            chunk = train_idx[start:start + batch_size]
            states, options, mask, labels, values = arrays.batch(chunk)
            logits, value = net(states, options, mask)
            loss = ce_loss(logits, labels) + VALUE_LOSS_WEIGHT * mse_loss(value, values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
        top1, value_mae = evaluate(net, arrays, val_idx, batch_size)
        print(f"epoch {epoch + 1:2d}/{epochs}  loss {total_loss / n_batches:.4f}  "
              f"val top-1 {top1:.4f}  val value MAE {value_mae:.4f}  "
              f"({time.perf_counter() - t0:.1f}s)")
        if top1 > best["val_top1"]:
            best = {"val_top1": top1, "val_value_mae": value_mae, "epoch": epoch}
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), checkpoint_path)

    print(f"best epoch {best['epoch'] + 1}: top-1 {best['val_top1']:.4f}, "
          f"value MAE {best['val_value_mae']:.4f} -> {checkpoint_path}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()
    best = train(args.dataset, args.epochs, args.batch_size, args.lr,
                 args.seed, args.checkpoint)
    metrics_path = args.checkpoint.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=1)


if __name__ == "__main__":
    main()
