"""Stage A — train and AUDIT the Crustle value head (dev only, torch).

Consumes the corpus built by value_collect.py and trains the value path
of the network on the game outcome z. The training is the easy half;
the audit is the half that decides whether Stage B is allowed to run at
all, because a search with a bad leaf is worse than one with a noisy
but unbiased leaf (that is exactly why the shipped search rolled out).

THE BASELINE PROBLEM. Brier against a constant 0.5 is 0.25, and a head
that learned nothing except "we usually win this matchup" would beat
that comfortably — our winrate runs 33% to 97% across cells. So three
competitors are scored on the same held-out positions:

    trivial     constant 0.5                       (Brier 0.25)
    base rate   the TRAIN-set marginal, global     (knows we win a lot)
    per-cell    the TRAIN-set marginal OF THAT CELL (knows the matchup)

The last one is the honest bar. It is strictly more informed than
anything the net is told — the net has to INFER the matchup from the
position — so a head that cannot beat it has learned nothing about
positions, only about opponents. The skill score reported at the end is
against the per-cell base rate.

Held-out is by EPISODE: every position of a game lands on one side of
the split, so a head cannot score by memorising a game it has seen the
rest of.

Run from the repo root:
    python -m src.rl_models.value_train --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Final

import numpy as np

from ..ingestion.build_card_model import REPO_ROOT
from .encoding import ENCODING_DIM
from .value_collect import CELLS, OUT_DIR as CORPUS_DIR
from .value_net import VALUE_CKPT, VALUE_METRICS, VALUE_NPZ, export_value_net

CELL_NAMES: Final[tuple[str, ...]] = tuple(c[0] for c in CELLS)
CELL_SHARES: Final[dict[str, float]] = {c[0]: c[3] for c in CELLS}
VAL_MOD: Final[int] = 5


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


class Corpus:
    """All cells in one place. States stay float16 until batched."""

    def __init__(self, corpus_dir: Path = CORPUS_DIR) -> None:
        states: list[np.ndarray] = []
        values: list[np.ndarray] = []
        ours: list[np.ndarray] = []
        episodes: list[np.ndarray] = []
        cells: list[np.ndarray] = []
        plies: list[np.ndarray] = []
        found: list[str] = []
        for name in CELL_NAMES:
            path = corpus_dir / f"cell_{name}.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                n = len(data["values"])
                states.append(data["states"])
                values.append(data["values"])
                ours.append(data["ours"])
                episodes.append(data["episode_ids"])
                plies.append(data["plies"])
                cells.append(np.full(n, CELL_NAMES.index(name), np.int16))
            found.append(name)
        if not states:
            raise SystemExit(f"no corpus in {corpus_dir} — run value_collect")
        self.cells_present = tuple(found)
        self.states = np.concatenate(states, axis=0)
        self.values = np.concatenate(values).astype(np.float32)
        self.ours = np.concatenate(ours).astype(bool)
        self.episodes = np.concatenate(episodes)
        self.cell_ids = np.concatenate(cells)
        self.plies = np.concatenate(plies)

    def __len__(self) -> int:
        return len(self.values)

    def split(self) -> tuple[np.ndarray, np.ndarray]:
        """Train/val indices, held out by GAME (episode id)."""
        is_val = (self.episodes % VAL_MOD) == 0
        idx = np.arange(len(self.values))
        return idx[~is_val], idx[is_val]


def _outcome(z: np.ndarray) -> np.ndarray:
    """z in {-1,0,+1} -> outcome in {0, 0.5, 1} (a draw is half a win)."""
    return (z + 1.0) / 2.0


def _brier(pred_prob: np.ndarray, z: np.ndarray) -> float:
    return float(np.mean((pred_prob - _outcome(z)) ** 2))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _auc(scores: np.ndarray, outcome: np.ndarray) -> float:
    """Rank AUC of ``scores`` against a binary-ish outcome.

    Reported because Brier SKILL is unusable where the base rate is
    extreme: in a cell we win 96% of the time the baseline Brier is
    0.023, so a couple of confident misses read as -52% skill while the
    absolute error moves by 0.012. AUC asks the question the search
    actually cares about — given two positions, does the head put the
    better one higher — and is invariant to the base rate.

    Draws (outcome 0.5) are dropped: they have no side to rank on.
    """
    keep = outcome != 0.5
    s, y = scores[keep], outcome[keep] > 0.5
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties so a constant predictor scores exactly 0.5
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def train(corpus: Corpus, epochs: int, lr: float, batch_size: int,
          seed: int, weight_decay: float, dropout: float = 0.0,
          loss_kind: str = "mse") -> dict:
    import torch
    from torch import nn

    from .value_net import ValueNet

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = corpus.split()
    print(f"corpus: {len(corpus)} positions from "
          f"{len(np.unique(corpus.episodes))} games, "
          f"{len(corpus.cells_present)} cells")
    print(f"split by game: {len(train_idx)} train / {len(val_idx)} val")

    # Normalization from the TRAIN split only, and baked into the export.
    sample = train_idx if len(train_idx) <= 200_000 else rng.choice(
        train_idx, 200_000, replace=False)
    block = corpus.states[np.sort(sample)].astype(np.float32)
    state_mean = block.mean(axis=0)
    state_std = block.std(axis=0)
    state_std[state_std < 1e-6] = 1.0
    del block

    def batch(idx: np.ndarray):
        x = corpus.states[idx].astype(np.float32)
        x = (x - state_mean) / state_std
        return torch.from_numpy(x), torch.from_numpy(corpus.values[idx])

    net = ValueNet(dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=lr,
                            weight_decay=weight_decay)
    mse = nn.MSELoss()

    def loss_fn(out, target):
        """MSE on z, or BCE on the win probability.

        BCE is offered because the cells that failed the first audit are
        the ones we win 91-96% of the time: under MSE the rare losses
        contribute almost no gradient once the net predicts the base
        rate, so it stops discriminating exactly where discrimination is
        scarce. BCE keeps a large gradient on confident mistakes.
        """
        if loss_kind == "mse":
            return mse(out, target)
        p = ((out + 1.0) / 2.0).clamp(1e-6, 1 - 1e-6)
        y = (target + 1.0) / 2.0
        return -(y * p.log() + (1 - y) * (1 - p).log()).mean()

    @torch.no_grad()
    def predict(idx: np.ndarray) -> np.ndarray:
        net.eval()
        out = np.empty(len(idx), np.float32)
        for start in range(0, len(idx), 4096):
            chunk = idx[start:start + 4096]
            x, _ = batch(chunk)
            out[start:start + len(chunk)] = net(x).numpy()
        return out

    # Model selection runs on OUR-SEAT held-out Brier, not the aggregate.
    # The aggregate is inflated by how easily the net can tell which
    # matchup it is looking at, and that signal is constant across the
    # sibling leaves of any real decision — optimising it would pick the
    # checkpoint that is best at something the search cannot use.
    ours_val = val_idx[corpus.ours[val_idx]]
    best = {"brier": float("inf"), "epoch": -1}
    order = train_idx.copy()
    for epoch in range(epochs):
        net.train()
        rng.shuffle(order)
        t0 = time.perf_counter()
        total = 0.0
        nb = 0
        for start in range(0, len(order), batch_size):
            chunk = np.sort(order[start:start + batch_size])
            x, y = batch(chunk)
            loss = loss_fn(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        pred_all = predict(val_idx)
        brier_all = _brier((pred_all + 1.0) / 2.0, corpus.values[val_idx])
        pred_ours = predict(ours_val)
        brier = _brier((pred_ours + 1.0) / 2.0, corpus.values[ours_val])
        flag = ""
        if brier < best["brier"]:
            best = {"brier": brier, "epoch": epoch}
            torch.save(net.state_dict(), VALUE_CKPT)
            flag = " *"
        print(f"epoch {epoch + 1:2d}/{epochs}  mse {total / nb:.4f}  "
              f"val Brier {brier_all:.4f}  our-seat {brier:.4f}  "
              f"({time.perf_counter() - t0:.0f}s){flag}")

    net.load_state_dict(torch.load(VALUE_CKPT, map_location="cpu",
                                   weights_only=True))
    export_value_net(net.state_dict(), state_mean, state_std, VALUE_NPZ)
    print(f"\nbest epoch {best['epoch'] + 1} (Brier {best['brier']:.4f}) "
          f"-> {VALUE_NPZ}")
    return {"best_epoch": best["epoch"] + 1, "val_brier": best["brier"]}


# --------------------------------------------------------------------------- #
# Audit — the part Gate A is read from
# --------------------------------------------------------------------------- #


def report(corpus: Corpus) -> dict:
    from .value_net import NumpyValueNet

    net = NumpyValueNet.load(VALUE_NPZ)
    if net is None:
        raise SystemExit(f"could not load {VALUE_NPZ}")

    train_idx, val_idx = corpus.split()
    z_tr, z_va = corpus.values[train_idx], corpus.values[val_idx]

    # predictions on held-out, batched
    pred = np.empty(len(val_idx), np.float32)
    for start in range(0, len(val_idx), 4096):
        chunk = val_idx[start:start + 4096]
        pred[start:start + len(chunk)] = net.value_batch(
            corpus.states[chunk].astype(np.float32))
    p_model = (pred + 1.0) / 2.0

    # The competitors, all fitted on TRAIN only.
    #
    # per-(cell, seat) is the one that matters. The first audit showed
    # why: recording both seats makes the marginal ~0.5 in every cell,
    # so a per-CELL baseline looks trivial while the model was quietly
    # collecting most of its "skill" from knowing WHICH MATCHUP it was
    # in (+71% in Lucario, where we win 92% of games). That skill is
    # worthless to a search: every sibling leaf of a decision shares the
    # cell AND the seat, so anything constant across them cannot change
    # a ranking. Only discrimination WITHIN (cell, seat) can.
    p_global = float(np.mean(_outcome(z_tr)))
    per_cell_rate: dict[int, float] = {}
    per_cellseat_rate: dict[tuple[int, int], float] = {}
    for cid in np.unique(corpus.cell_ids):
        mask = corpus.cell_ids[train_idx] == cid
        per_cell_rate[int(cid)] = (float(np.mean(_outcome(z_tr[mask])))
                                   if mask.any() else p_global)
        for seat_flag in (0, 1):
            sm = mask & (corpus.ours[train_idx] == bool(seat_flag))
            per_cellseat_rate[(int(cid), seat_flag)] = (
                float(np.mean(_outcome(z_tr[sm]))) if sm.sum() > 20
                else per_cell_rate[int(cid)])
    p_cell = np.asarray([per_cell_rate[int(c)]
                         for c in corpus.cell_ids[val_idx]], np.float32)
    p_cellseat = np.asarray(
        [per_cellseat_rate[(int(c), int(o))]
         for c, o in zip(corpus.cell_ids[val_idx], corpus.ours[val_idx])],
        np.float32)

    b_model = _brier(p_model, z_va)
    b_trivial = _brier(np.full(len(z_va), 0.5, np.float32), z_va)
    b_global = _brier(np.full(len(z_va), p_global, np.float32), z_va)
    b_cell = _brier(p_cell, z_va)
    b_cellseat = _brier(p_cellseat, z_va)
    pearson = (float(np.corrcoef(pred, z_va)[0, 1])
               if pred.std() > 1e-9 else 0.0)

    print("\n" + "=" * 92)
    print("GATE A — value head calibration (held-out by GAME)")
    print("=" * 92)
    print(f"  held-out positions: {len(val_idx)} "
          f"({int(corpus.ours[val_idx].sum())} ours)")
    print(f"  Brier  trivial (const 0.5)         {b_trivial:.4f}")
    print(f"  Brier  base rate (global marginal) {b_global:.4f}   "
          f"(p={p_global:.3f})")
    print(f"  Brier  base rate PER CELL          {b_cell:.4f}")
    print(f"  Brier  base rate PER CELL x SEAT   {b_cellseat:.4f}   "
          f"<- honest bar")
    print(f"  Brier  MODEL                       {b_model:.4f}")
    skill_trivial = 1.0 - b_model / b_trivial
    skill_cell = 1.0 - b_model / b_cell
    skill_cellseat = 1.0 - b_model / b_cellseat
    print(f"  skill vs trivial        {skill_trivial:+.1%}")
    print(f"  skill vs per-cell       {skill_cell:+.1%}")
    print(f"  skill vs per-cell-seat  {skill_cellseat:+.1%}   "
          f"<- the number that counts")
    print(f"  Pearson(value, z)  {pearson:+.4f}")

    # ---- per cell, restricted to OUR seat ---------------------------- #
    #
    # OUR-SEAT columns are the primary read. Those are the positions the
    # search will actually rank against each other, and the baseline
    # they are scored against is that cell-and-seat's own win rate — so
    # "we win this matchup a lot" earns exactly zero credit here.
    print("\n  per cell — OUR-SEAT positions (what the search actually ranks)")
    print(f"  {'cell':20s}{'n_ours':>8s}{'base':>7s}{'Brier0':>9s}"
          f"{'Brier':>9s}{'skill':>8s}{'pearson':>9s}{'AUC':>8s}"
          f"{'|':>3s}{'skill_all':>11s}{'AUC_all':>9s}")
    print("  " + "-" * 103)
    rows = []
    for cid, name in enumerate(CELL_NAMES):
        mask = corpus.cell_ids[val_idx] == cid
        if not mask.any():
            continue
        zc, pc, rc = z_va[mask], p_model[mask], pred[mask]
        om = corpus.ours[val_idx][mask]
        # all-positions view (kept for continuity with the first audit)
        b0_all = _brier(np.full(mask.sum(), per_cell_rate.get(cid, p_global),
                                np.float32), zc)
        bm_all = _brier(pc, zc)
        pear_all = float(np.corrcoef(rc, zc)[0, 1]) if rc.std() > 1e-9 else 0.0
        skill_all = 1.0 - bm_all / b0_all if b0_all > 0 else 0.0
        auc_all = _auc(rc, _outcome(zc))
        # our-seat view, against the (cell, seat) base rate
        if om.sum() > 20:
            base_o = per_cellseat_rate.get((cid, 1), p_global)
            b0_o = _brier(np.full(int(om.sum()), base_o, np.float32), zc[om])
            bm_o = _brier(pc[om], zc[om])
            pear_o = (float(np.corrcoef(rc[om], zc[om])[0, 1])
                      if rc[om].std() > 1e-9 and zc[om].std() > 1e-9 else 0.0)
            skill_o = 1.0 - bm_o / b0_o if b0_o > 0 else 0.0
            auc_o = _auc(rc[om], _outcome(zc[om]))
        else:
            base_o = b0_o = bm_o = skill_o = pear_o = auc_o = float("nan")
        rows.append({"cell": name, "n": int(mask.sum()),
                     "n_ours": int(om.sum()), "base_ours": base_o,
                     "brier_base_ours": b0_o, "brier_ours": bm_o,
                     "skill_ours": skill_o, "pearson_ours": pear_o,
                     "auc_ours": auc_o, "brier_all": bm_all,
                     "skill_all": skill_all, "pearson_all": pear_all,
                     "auc_all": auc_all})
        print(f"  {name:20s}{om.sum():8d}{base_o:7.3f}{b0_o:9.4f}{bm_o:9.4f}"
              f"{skill_o:+8.1%}{pear_o:+9.3f}{auc_o:8.3f}{'|':>3s}"
              f"{skill_all:+11.1%}{auc_all:9.3f}")

    # ---- calibration curve ------------------------------------------ #
    print("\n  calibration (held-out, all positions)")
    print(f"  {'predicted':>20s}{'n':>9s}{'mean pred':>12s}"
          f"{'observed':>11s}{'gap':>9s}")
    print("  " + "-" * 61)
    edges = np.linspace(0.0, 1.0, 11)
    calib = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_model >= lo) & (p_model < hi if hi < 1.0 else p_model <= 1.0)
        if m.sum() < 20:
            continue
        obs = float(np.mean(_outcome(z_va[m])))
        mp = float(np.mean(p_model[m]))
        calib.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                      "mean_pred": mp, "observed": obs})
        print(f"  [{lo:.1f}, {hi:.1f})".rjust(20)
              + f"{m.sum():9d}{mp:12.3f}{obs:11.3f}{obs - mp:+9.3f}")

    # ---- by game phase ---------------------------------------------- #
    print("\n  by game phase (ply of the position within its game)")
    print(f"  {'phase':>20s}{'n':>9s}{'Brier':>10s}{'pearson':>10s}")
    print("  " + "-" * 49)
    phases = [(0, 40), (40, 100), (100, 200), (200, 10 ** 9)]
    phase_rows = []
    for lo, hi in phases:
        m = (corpus.plies[val_idx] >= lo) & (corpus.plies[val_idx] < hi)
        if m.sum() < 50:
            continue
        bm = _brier(p_model[m], z_va[m])
        pear = (float(np.corrcoef(pred[m], z_va[m])[0, 1])
                if pred[m].std() > 1e-9 else 0.0)
        phase_rows.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                           "brier": bm, "pearson": pear})
        label = f"{lo}-{hi if hi < 10 ** 9 else '+'}"
        print(f"  {label:>20s}{m.sum():9d}{bm:10.4f}{pear:+10.3f}")

    # ---- verdict ----------------------------------------------------- #
    ok_rows = [r for r in rows if r["n_ours"] > 20]
    clone_cells = [r for r in ok_rows if "-" in r["cell"]]
    field_cells = [r for r in ok_rows if "-" not in r["cell"]]
    worst = (min(ok_rows, key=lambda r: r["pearson_ours"])
             if ok_rows else None)
    print("\n  " + "-" * 90)
    if field_cells:
        print(f"  corrected-field cells: our-seat AUC "
              f"{min(r['auc_ours'] for r in field_cells):.3f} .. "
              f"{max(r['auc_ours'] for r in field_cells):.3f}")
    if clone_cells:
        # The honesty check that the rollout search failed: a leaf whose
        # quality depended on the opponent being our own model is a leaf
        # that evaporates on the ladder.
        print(f"  CLONE cells (NOT our model): our-seat AUC "
              f"{min(r['auc_ours'] for r in clone_cells):.3f} .. "
              f"{max(r['auc_ours'] for r in clone_cells):.3f}")
    if worst is not None:
        print(f"  weakest cell: {worst['cell']} (our-seat AUC "
              f"{worst['auc_ours']:.3f}, pearson "
              f"{worst['pearson_ours']:+.3f})")
    min_pear = min((r["pearson_ours"] for r in ok_rows), default=0.0)
    min_skill = min((r["skill_ours"] for r in ok_rows), default=0.0)
    min_auc = min((r["auc_ours"] for r in ok_rows), default=0.0)

    # GATE A, stated in terms of what the search can use.
    #
    # Brier skill is kept as a check on CALIBRATION but is not the
    # consistency criterion: with a 96% base rate its denominator is
    # 0.023 and it swings wildly on a handful of positions. AUC is the
    # consistency criterion because ranking sibling leaves is literally
    # the operation the search performs, and AUC is base-rate invariant.
    passed = (b_model < b_trivial and skill_cellseat > 0.05
              and min_auc > 0.60)
    print(f"\n  GATE A: {'PASS' if passed else 'FAIL'}")
    print(f"    Brier {b_model:.4f} < 0.25 trivial             "
          f"{'ok' if b_model < b_trivial else 'FAIL'}")
    print(f"    skill vs per-cell-seat {skill_cellseat:+.1%} > +5%      "
          f"{'ok' if skill_cellseat > 0.05 else 'FAIL'}")
    print(f"    min our-seat AUC       {min_auc:.3f} > 0.60      "
          f"{'ok' if min_auc > 0.60 else 'FAIL'}")
    print(f"    (informational) min our-seat pearson {min_pear:+.3f}, "
          f"min Brier skill {min_skill:+.1%}")
    print("=" * 92)

    metrics = {
        "n_val": int(len(val_idx)), "n_train": int(len(train_idx)),
        "brier_model": b_model, "brier_trivial": b_trivial,
        "brier_base_global": b_global, "brier_base_percell": b_cell,
        "brier_base_percellseat": b_cellseat,
        "skill_vs_trivial": skill_trivial, "skill_vs_percell": skill_cell,
        "skill_vs_percellseat": skill_cellseat,
        "min_pearson_ours": min_pear, "min_skill_ours": min_skill,
        "min_auc_ours": min_auc,
        "pearson": pearson, "cells": rows, "calibration": calib,
        "phases": phase_rows, "gate_a_pass": bool(passed),
    }
    VALUE_METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(VALUE_METRICS, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1)
    print(f"-> {VALUE_METRICS}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="all",
                        choices=("train", "report", "all"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss", choices=("mse", "bce"), default="mse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    args = parser.parse_args()

    corpus = Corpus(args.corpus_dir)
    if args.command in ("train", "all"):
        train(corpus, args.epochs, args.lr, args.batch_size, args.seed,
              args.weight_decay, args.dropout, args.loss)
    if args.command in ("report", "all"):
        report(corpus)


if __name__ == "__main__":
    main()
