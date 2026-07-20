"""Expert Iteration — Stage B (dev): value head + destilação por soft targets.

Rodada 2 do ExIt (a R1 falhou por hard-label de poucos alvos ruidosos).
Consome os 109.777 alvos coletados por expert_iter.py (400 shards em
data/processed/exit_grimmsnarl/r2/), cada um com:
    soft_targets [N, MAX_OPTIONS]  winrate por candidato AVALIADO pela
                                   busca (NaN nos não-avaliados)
    labels                         ação da busca (argmax do winrate)
    values                         z do jogo por posição (+1/-1/0)

Três comandos (CLI):

  build   Funde os shards + o corpus do Luca num único npz de treino.
          Cada posição vira uma DISTRIBUIÇÃO-alvo sobre as options legais:
          - posição da busca: softmax(winrate / TEMP) sobre os candidatos
            avaliados (0 no resto) — soft, respeita os gaps de valor e NÃO
            comita no argmax ruidoso (o defeito da R1);
          - posição do Luca: one-hot na ação do líder (âncora).
          episode_ids reofsetados por fonte (o split held-out isola jogos
          inteiros; busca e Luca em faixas disjuntas).

  value   Treina SÓ o value head (trunk da BC-Grimmsnarl congelado) nos z
          coletados e reporta calibração (Brier + Pearson) vs o value 5D
          descalibrado que a BC herdou.

  distill Destila a política: warm-start na BC-Grimmsnarl, loss = cross-
          entropy contra a DISTRIBUIÇÃO-alvo, early-stop vigiando as DUAS
          fidelidades held-out (alvos-da-busca E Luca — a R1 saiu das
          duas). Exporta numpy (par casado com feature_stats).

Uso (raiz do repo, dev — torch):
    python -m src.rl_models.exit_distill build
    python -m src.rl_models.exit_distill value
    python -m src.rl_models.exit_distill distill --epochs 12
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn

from .encoding import ENCODING_DIM, MAX_OPTIONS, OPTION_DIM
from .leader_train import load_net_from_npz
from .network import PolicyValueNet, split_by_game
from .network_numpy import _KEY_MAP
from .normalization import MODELS_DIR, FeatureStats

R2_DIR: Final[Path] = (Path(__file__).resolve().parents[2]
                       / "data" / "processed" / "exit_grimmsnarl" / "r2")
LUCA_CORPUS: Final[Path] = (Path(__file__).resolve().parents[2] / "data"
                            / "processed" / "bc_grimmsnarl" / "luca_dataset.npz")
MIXED_PATH: Final[Path] = R2_DIR / "distill_dataset.npz"
BC_WEIGHTS: Final[Path] = MODELS_DIR / "bc_grimmsnarl.npz"
STATS_PATH: Final[Path] = MODELS_DIR / "feature_stats.npz"
DISTILL_CKPT: Final[Path] = MODELS_DIR / "bc_grimmsnarl_exit2.pt"
DISTILL_NPZ: Final[Path] = MODELS_DIR / "bc_grimmsnarl_exit2.npz"
VALUE_CKPT: Final[Path] = MODELS_DIR / "value_grimmsnarl_exit2.pt"

SOFT_TEMP: Final[float] = 0.25   # temperatura do softmax dos winrates
LUCA_BASE: Final[int] = 10_000_000
SEARCH_BASE: Final[int] = 900_000_000
OPP_BLOCK: Final[int] = 2_000_000  # separa jogos de oponentes diferentes


# --------------------------------------------------------------------------- #
# build: shards + Luca -> distribuições-alvo num npz
# --------------------------------------------------------------------------- #


def _soft_row(winrates: np.ndarray, temp: float) -> np.ndarray:
    """Winrates por candidato avaliado (NaN nos não) -> distribuição."""
    target = np.zeros(MAX_OPTIONS, dtype=np.float32)
    evaluated = np.where(~np.isnan(winrates))[0]
    if len(evaluated) == 0:
        return target
    w = winrates[evaluated] / temp
    w -= w.max()
    exp = np.exp(w)
    target[evaluated] = (exp / exp.sum()).astype(np.float32)
    return target


def build(out_path: Path = MIXED_PATH, temp: float = SOFT_TEMP) -> None:
    states: list[np.ndarray] = []
    options: list[np.ndarray] = []
    counts: list[int] = []
    targets: list[np.ndarray] = []
    labels: list[int] = []       # ação alvo (busca ou Luca) p/ top-1
    values: list[int] = []
    is_search: list[int] = []
    episode_ids: list[int] = []

    shard_paths = sorted(R2_DIR.glob("shard_*.npz"))
    opp_index: dict[str, int] = {}
    for path in shard_paths:
        opp = path.name.split("shard_")[1].split("_s0")[0]
        block = opp_index.setdefault(opp, len(opp_index))
        with np.load(path) as data:
            soft = data["soft_targets"]
            n = len(data["labels"])
            states.append(data["states"])
            options.append(data["options_flat"])
            counts.extend(int(c) for c in data["option_counts"])
            for i in range(n):
                targets.append(_soft_row(soft[i], temp))
            labels.extend(int(x) for x in data["labels"])
            values.extend(int(x) for x in data["values"])
            is_search.extend([1] * n)
            episode_ids.extend(SEARCH_BASE + block * OPP_BLOCK
                               + int(e % OPP_BLOCK) for e in data["episode_ids"])

    n_search = len(labels)
    with np.load(LUCA_CORPUS) as data:
        n = len(data["labels"])
        states.append(data["states"])
        options.append(data["options_flat"])
        counts.extend(int(c) for c in data["option_counts"])
        lab = data["labels"].astype(int)
        for i in range(n):
            onehot = np.zeros(MAX_OPTIONS, dtype=np.float32)
            if 0 <= lab[i] < MAX_OPTIONS:
                onehot[lab[i]] = 1.0
            targets.append(onehot)
        labels.extend(int(x) for x in lab)
        values.extend(int(x) for x in data["values"])
        is_search.extend([0] * n)
        episode_ids.extend(LUCA_BASE + int(e) for e in data["episode_ids"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=np.concatenate(states, axis=0).astype(np.float32),
        options_flat=np.concatenate(options, axis=0).astype(np.float32),
        option_counts=np.asarray(counts, np.uint16),
        target_probs=np.stack(targets).astype(np.float32),
        labels=np.asarray(labels, np.uint16),
        values=np.asarray(values, np.int8),
        is_search=np.asarray(is_search, np.int8),
        episode_ids=np.asarray(episode_ids, np.int64),
    )
    print(f"build: {n_search} alvos-busca + {len(labels) - n_search} Luca "
          f"= {len(labels)} posições (temp {temp}, "
          f"proporção busca:Luca = {n_search / max(len(labels) - n_search, 1):.1f}:1)")
    print(f"-> {out_path} ({out_path.stat().st_size:,} bytes)")


# --------------------------------------------------------------------------- #
# dataset em memória (padded batches, normalizado uma vez)
# --------------------------------------------------------------------------- #


class DistillArrays:
    """Como BcArrays, mas com target_probs [N,64] e flags de fonte."""

    def __init__(self, path: Path, stats: FeatureStats) -> None:
        with np.load(path) as data:
            counts = data["option_counts"].astype(np.int64)
            offsets = np.zeros(len(counts) + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])
            self.states = stats.normalize_state(data["states"])
            self.options_flat = stats.normalize_options(data["options_flat"])
            self.offsets = offsets
            self.counts = counts
            self.target_probs = data["target_probs"].astype(np.float32)
            self.labels = data["labels"].astype(np.int64)
            self.values = data["values"].astype(np.float32)
            self.is_search = data["is_search"].astype(bool)
            self.game_ids = data["episode_ids"].astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, idx: np.ndarray) -> tuple[Tensor, Tensor, Tensor, Tensor,
                                              Tensor, Tensor]:
        b = len(idx)
        options = np.zeros((b, MAX_OPTIONS, OPTION_DIM), dtype=np.float32)
        mask = np.zeros((b, MAX_OPTIONS), dtype=bool)
        for row, s in enumerate(idx):
            c = min(int(self.counts[s]), MAX_OPTIONS)
            start = self.offsets[s]
            options[row, :c] = self.options_flat[start:start + c]
            mask[row, :c] = True
        return (torch.from_numpy(self.states[idx]),
                torch.from_numpy(options),
                torch.from_numpy(mask),
                torch.from_numpy(self.target_probs[idx]),
                torch.from_numpy(self.labels[idx]),
                torch.from_numpy(self.values[idx]))


@torch.no_grad()
def _top1(net: PolicyValueNet, arr: DistillArrays, idx: np.ndarray,
          bs: int = 1024) -> float:
    net.eval()
    correct = 0
    for start in range(0, len(idx), bs):
        chunk = idx[start:start + bs]
        states, options, mask, _, labels, _ = arr.batch(chunk)
        logits, _ = net(states, options, mask)
        correct += int((logits.argmax(dim=1) == labels).sum())
    return correct / max(len(idx), 1)


# --------------------------------------------------------------------------- #
# value: calibra SÓ o value head nos z
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _value_metrics(net: PolicyValueNet, arr: DistillArrays,
                   idx: np.ndarray, bs: int = 1024) -> dict[str, float]:
    """Brier (win prob vs desfecho) + Pearson (valor vs z) no held-out."""
    net.eval()
    preds: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    for start in range(0, len(idx), bs):
        chunk = idx[start:start + bs]
        states, options, mask, _, _, values = arr.batch(chunk)
        _, value = net(states, options, mask)
        preds.append(value.numpy())
        zs.append(values.numpy())
    v = np.concatenate(preds)
    z = np.concatenate(zs)
    p = (v + 1.0) / 2.0                 # win prob em [0,1]
    outcome = (z + 1.0) / 2.0           # {0,1} (0 é draw->0.5)
    outcome = np.where(z == 0, 0.5, outcome)
    brier = float(np.mean((p - outcome) ** 2))
    pearson = float(np.corrcoef(v, z)[0, 1]) if v.std() > 1e-9 else 0.0
    return {"brier": brier, "pearson": pearson, "mean_pred": float(v.mean()),
            "n": len(idx)}


def train_value(epochs: int = 15, lr: float = 1e-3, seed: int = 0) -> None:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    stats = FeatureStats.load(STATS_PATH)
    arr = DistillArrays(MIXED_PATH, stats)
    # só posições da busca têm z de jogo real (Luca também tem z do líder,
    # mas o objetivo aqui é calibrar nos rollouts da busca coletados)
    search_idx = np.where(arr.is_search)[0]
    train_idx, val_idx = split_by_game(arr.game_ids[search_idx])
    train_idx, val_idx = search_idx[train_idx], search_idx[val_idx]
    print(f"value: {len(search_idx)} posições-busca "
          f"({len(train_idx)} train / {len(val_idx)} val)")

    net = load_net_from_npz(BC_WEIGHTS)
    base = _value_metrics(net, arr, val_idx)
    print(f"baseline (value 5D herdado): Brier {base['brier']:.4f}  "
          f"Pearson {base['pearson']:+.4f}  pred médio {base['mean_pred']:+.3f}")

    # treina SÓ o value head; trunk/policy congelados (calibração limpa)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.value_head.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(net.value_head.parameters(), lr=lr)
    mse = nn.MSELoss()

    best = {"brier": base["brier"], "pearson": base["pearson"], "epoch": -1}
    for epoch in range(epochs):
        net.train()
        rng.shuffle(train_idx)
        total = 0.0
        nb = 0
        for start in range(0, len(train_idx), 1024):
            chunk = train_idx[start:start + 1024]
            states, options, mask, _, _, values = arr.batch(chunk)
            _, value = net(states, options, mask)
            loss = mse(value, values)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        m = _value_metrics(net, arr, val_idx)
        print(f"epoch {epoch + 1:2d}/{epochs}  mse {total / nb:.4f}  "
              f"Brier {m['brier']:.4f}  Pearson {m['pearson']:+.4f}")
        if m["brier"] < best["brier"]:
            best = {**m, "epoch": epoch}
            VALUE_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), VALUE_CKPT)

    print(f"\nvalue head calibrado: Brier {base['brier']:.4f} -> "
          f"{best['brier']:.4f}  |  Pearson {base['pearson']:+.4f} -> "
          f"{best['pearson']:+.4f}  (melhor época {best['epoch'] + 1})")


# --------------------------------------------------------------------------- #
# distill: política por soft targets, early-stop nas 2 fidelidades
# --------------------------------------------------------------------------- #


def _soft_ce(logits: Tensor, target: Tensor, mask: Tensor,
             weight: Tensor | None = None) -> Tensor:
    """Cross-entropy contra a distribuição-alvo (masked), opcionalmente
    ponderada por posição (weight [B])."""
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=1)
    per_row = -(target * logp).sum(dim=1)
    if weight is not None:
        return (per_row * weight).sum() / weight.sum().clamp_min(1e-6)
    return per_row.mean()


def _margin_weight(target: Tensor) -> Tensor:
    """Confiança da busca por posição = gap entre 1º e 2º alvos (0..1).

    Posições onde a busca claramente preferiu um candidato pesam mais;
    empates (todos ~iguais) tendem a 0 — filtra o ruído de D=8 sem
    achatar (o alvo continua a distribuição, só o PESO muda)."""
    top2 = torch.topk(target, k=min(2, target.shape[1]), dim=1).values
    gap = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
    return gap


def distill(epochs: int = 12, lr: float = 2e-4, seed: int = 0,
            hard: bool = False, margin: bool = False) -> None:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    stats = FeatureStats.load(STATS_PATH)
    arr = DistillArrays(MIXED_PATH, stats)
    train_idx, val_idx = split_by_game(arr.game_ids)
    search_val = val_idx[arr.is_search[val_idx]]
    luca_val = val_idx[~arr.is_search[val_idx]]
    print(f"distill: {len(arr)} posições "
          f"({len(train_idx)} train / {len(val_idx)} val)")
    print(f"held-out: {len(search_val)} alvos-busca + {len(luca_val)} Luca")

    mode = "hard" if hard else ("soft+margin" if margin else "soft")
    print(f"modo de destilação: {mode}")
    bc = load_net_from_npz(BC_WEIGHTS)
    bc_search = _top1(bc, arr, search_val)
    bc_luca = _top1(bc, arr, luca_val)
    print(f"BC pura   top-1: busca {bc_search:.4f}  luca {bc_luca:.4f}")

    ce_hard = nn.CrossEntropyLoss()
    net = load_net_from_npz(BC_WEIGHTS)
    for p in net.value_head.parameters():   # política só (como o 5B)
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=lr)

    # early-stop: soma das duas fidelidades (não deixa nenhuma desabar)
    best = {"sum": bc_search + bc_luca, "search": bc_search,
            "luca": bc_luca, "epoch": -1}
    for epoch in range(epochs):
        net.train()
        rng.shuffle(train_idx)
        t0 = time.perf_counter()
        total = 0.0
        nb = 0
        for start in range(0, len(train_idx), 512):
            chunk = train_idx[start:start + 512]
            states, options, mask, target, labels, _ = arr.batch(chunk)
            logits, _ = net(states, options, mask)
            if hard:
                loss = ce_hard(logits.masked_fill(~mask, -1e9), labels)
            elif margin:
                loss = _soft_ce(logits, target, mask, _margin_weight(target))
            else:
                loss = _soft_ce(logits, target, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        s = _top1(net, arr, search_val)
        l = _top1(net, arr, luca_val)
        flag = ""
        if s + l > best["sum"]:
            best = {"sum": s + l, "search": s, "luca": l, "epoch": epoch}
            DISTILL_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), DISTILL_CKPT)
            flag = " *"
        print(f"epoch {epoch + 1:2d}/{epochs}  loss {total / nb:.4f}  "
              f"busca {s:.4f}  luca {l:.4f}  ({time.perf_counter() - t0:.0f}s){flag}")

    print(f"\nmelhor época {best['epoch'] + 1}: busca {best['search']:.4f} "
          f"(BC {bc_search:.4f})  luca {best['luca']:.4f} (BC {bc_luca:.4f})")
    # a força (gauntlet) NÃO é o top-1 — a própria calibração mostrou a
    # busca jogando bem acima da BC com top-1 baixo. Salva a ÚLTIMA época
    # para medir força mesmo quando a fidelidade não supera a BC.
    torch.save(net.state_dict(), DISTILL_CKPT)
    print(f"última época salva em {DISTILL_CKPT} (medir força no gauntlet)")


def export() -> None:
    from .network_numpy import export_state_dict
    sd = torch.load(DISTILL_CKPT, map_location="cpu", weights_only=True)
    export_state_dict(sd, DISTILL_NPZ)
    print(f"exportado {DISTILL_NPZ}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=("build", "value", "distill", "export"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temp", type=float, default=SOFT_TEMP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hard", action="store_true",
                        help="hard CE na ação da busca (R1 em escala)")
    parser.add_argument("--margin", action="store_true",
                        help="soft-CE ponderado pela confiança da busca")
    args = parser.parse_args()
    if args.command == "build":
        build(temp=args.temp)
    elif args.command == "value":
        train_value(epochs=max(args.epochs, 15), seed=args.seed)
    elif args.command == "distill":
        distill(epochs=args.epochs, lr=args.lr, seed=args.seed,
                hard=args.hard, margin=args.margin)
        export()
    else:
        export()


if __name__ == "__main__":
    main()
