"""Sprint 5B: fine-tune the policy by imitating Kaggle LEADER replays.

Warm-starts PolicyValueNet from the Sprint-5A clone weights
(models/policy_value_bc5a.npz — the .npz is the durable artifact; the .pt
checkpoint is gitignored) and trains ONLY the policy path (trunk,
option_proj, query, option_bias) with masked cross-entropy against the
winner's chosen action from data/processed/replays/replay_corpus.npz.

The VALUE HEAD IS FROZEN here on purpose: the corpus is parsed with
--sides winner, so every z == +1 and an optimized value head would
collapse to a constant, regressing the 5A critic. Its parameters keep the
5A weights, but because the shared trunk moves during fine-tuning the
value OUTPUT is no longer calibrated after this phase — the critic is
retrained properly in Sprint 5C (self-play RL). Nothing at runtime
consumes the value yet (NetworkAgent ranks by policy logits only).

Train/val split is by EPISODE id (episode_id % 5 == 0 -> val), so no
decision of a validation game ever leaks into training.

Usage (repo root, dev only — torch):
    python -m src.rl_models.leader_train --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn

from ..ingestion.replays_merge import CORPUS_PATH
from .network import BcArrays, PolicyValueNet, split_by_game
from .network_numpy import _KEY_MAP
from .normalization import MODELS_DIR, FeatureStats

CHECKPOINT_5B_PATH: Final[Path] = MODELS_DIR / "policy_value_5b.pt"
WARM_START_PATH: Final[Path] = MODELS_DIR / "policy_value_bc5a.npz"


def load_net_from_npz(path: Path) -> PolicyValueNet:
    """Rebuild the torch net from exported .npz weights (inverse export)."""
    with np.load(path) as data:
        state_dict = {torch_key: torch.from_numpy(np.array(data[npz_key]))
                      for torch_key, npz_key in _KEY_MAP.items()}
    net = PolicyValueNet()
    net.load_state_dict(state_dict)
    return net


@torch.no_grad()
def top1_accuracy(net: PolicyValueNet, arrays: BcArrays,
                  indices: np.ndarray, batch_size: int) -> float:
    """Fraction of decisions where argmax(policy) == the leader's action."""
    net.eval()
    correct = 0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        states, options, mask, labels, _ = arrays.batch(chunk)
        logits, _ = net(states, options, mask)
        correct += int((logits.argmax(dim=1) == labels).sum())
    return correct / max(len(indices), 1)


def train(corpus_path: Path = CORPUS_PATH, epochs: int = 20,
          batch_size: int = 512, lr: float = 3e-4, seed: int = 0,
          warm_start: Path | None = WARM_START_PATH,
          checkpoint_path: Path = CHECKPOINT_5B_PATH,
          stats_path: Path | None = None) -> dict[str, float]:
    """stats_path pins the feature_stats the corpus is normalized with
    (PAIRED with the resulting policy — promote them together or not at
    all); None keeps the production default of FeatureStats.load()."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    stats = (FeatureStats.load(stats_path) if stats_path is not None
             else FeatureStats.load())
    print(f"feature stats: {stats_path if stats_path is not None else 'default'}")
    arrays = BcArrays.load(corpus_path, stats)
    train_idx, val_idx = split_by_game(arrays.game_ids)
    n_episodes = len(np.unique(arrays.game_ids))
    print(f"corpus: {len(arrays.labels)} leader decisions from "
          f"{n_episodes} episodes ({len(train_idx)} train / "
          f"{len(val_idx)} val, split by episode)")

    if warm_start is not None and warm_start.exists():
        net = load_net_from_npz(warm_start)
        print(f"warm start: {warm_start}")
    else:
        net = PolicyValueNet()
        print("warm start unavailable — training from scratch")

    # Policy-only phase: freeze the value head (see module docstring).
    for param in net.value_head.parameters():
        param.requires_grad = False
    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    baseline = top1_accuracy(net, arrays, val_idx, batch_size)
    print(f"epoch  0/{epochs}  val top-1 {baseline:.4f}  (before training)")

    best = {"val_top1": baseline, "epoch": -1}
    for epoch in range(epochs):
        net.train()
        rng.shuffle(train_idx)
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_idx), batch_size):
            chunk = train_idx[start:start + batch_size]
            states, options, mask, labels, _ = arrays.batch(chunk)
            logits, _ = net(states, options, mask)
            loss = ce_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
        top1 = top1_accuracy(net, arrays, val_idx, batch_size)
        print(f"epoch {epoch + 1:2d}/{epochs}  loss {total_loss / n_batches:.4f}  "
              f"val top-1 {top1:.4f}  ({time.perf_counter() - t0:.1f}s)")
        if top1 > best["val_top1"]:
            best = {"val_top1": top1, "epoch": epoch}
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), checkpoint_path)

    print(f"best epoch {best['epoch'] + 1}: val top-1 {best['val_top1']:.4f} "
          f"-> {checkpoint_path}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warm-start", type=Path, default=WARM_START_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_5B_PATH)
    parser.add_argument("--stats", type=Path, default=None,
                        help="feature_stats .npz override (paired with the "
                             "trained policy; default: production stats)")
    args = parser.parse_args()
    best = train(args.corpus, args.epochs, args.batch_size, args.lr,
                 args.seed, args.warm_start, args.checkpoint, args.stats)
    metrics_path = args.checkpoint.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=1)


if __name__ == "__main__":
    main()
