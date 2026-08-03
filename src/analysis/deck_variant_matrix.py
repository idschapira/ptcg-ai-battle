"""High-N matrix for the Crustle DECK variants, with the MECHANISM.

Question: do the 4 Rock Fighting Energy slots buy more as bodies (more
Pokémon = more prizes the opponent must take = more turns for the mill to
land) or as mill/consistency, than they do as effect prevention?

Design decisions, and why:

* **Same pilot on our side in every arm** (`crustle-v3`, the ship). Only
  the deck changes, so the difference is the deck lever in isolation.
* **Per-CELL, never a field average.** A field mean mixes an opponent we
  beat 90% with one we lose 65% and reports something true of neither;
  the project has paid for that lesson. Cells are weighted by their
  measured share of OUR REAL ladder games (see deadweight_audit) only for
  the aggregate line, which is reported next to the cells, not instead.
* **The Alakazam cell twice.** Its heuristic pilot is a known-inflated
  model of the real thing (measured ~70% internal vs 35% on ladder), so
  the cell also runs against a behaviour-cloned pilot. A deck change that
  only helps against the inflated model has not been shown to help.
* **Newcombe intervals for the DIFFERENCE vs the ship**, not two Wilson
  intervals eyeballed for overlap. Non-overlap and "difference excludes
  zero" are different questions.
* **Seats broken out.** Seats alternate; a lopsided split is a
  first-player artefact worth seeing rather than averaging away.
* **The mechanism, not just the winrate.** The change is supposed to move
  two specific quantities: the turn the game ends on when we lose, and
  how close the opponent's deck got to zero. If the winrate does not move
  but those do, the direction is right and N is short. If neither moves,
  the thesis is simply wrong. Measured via the ``observer`` hook on
  play_one_game, so the engine loop is not reimplemented.

Sharded over processes because cg.api keeps ONE module-level agent_ptr
with no lock (same reason as search_validation).

Run from the repo root:
    python -m src.analysis.deck_variant_matrix --games 600
    python -m src.analysis.deck_variant_matrix --games 200 --cells alakazam-bc
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

from ..environment_wrapper.ab_test import (newcombe_difference,
                                           wilson_interval)
from ..ingestion.build_card_model import REPO_ROOT

OUT_DIR: Final[Path] = REPO_ROOT / "data" / "processed" / "deck_variants"

# our side: label -> deck path (all flown by the SHIP pilot)
ARMS: Final[dict[str, str]] = {
    "SHIP":  "deck.csv",
    "V1-corpos": "data/decks/variant_crustle_v1_corpos.csv",
    "V2-mill": "data/decks/variant_crustle_v2_mill.csv",
    "V3-corpos-agr": "data/decks/variant_crustle_v3_corpos_agr.csv",
    # CONTROL, added after the first matrix run. V1-V3 all trade Rock
    # Fighting for cards that are not energy, so they change TWO things
    # at once: how much effect prevention we hold AND how much energy the
    # deck runs (10 -> 8 -> 6). Land Collapse costs {C}{C}, so the mill
    # itself is paid for out of that pool, and the damage was monotone in
    # energy removed. V4 removes the SAME 2 Rock Fighting but refills
    # with Basic {F}, holding the energy count at 10 and the body count
    # at 13 — so it isolates the protection clause as the single
    # variable. Flat here means the clause is what was cheap; negative
    # means the clause was carrying the cells.
    "V4-protswap": "data/decks/variant_crustle_v4_protswap.csv",
}
OUR_PILOT: Final[str] = "crustle-v3"

# opponent cells: label -> (deck@arm, share of our real ladder games)
# Shares are the measured archetype mix over 225 real episodes of the
# shipped list (src/analysis/deadweight_audit.py, 2026-07-30).
CELLS: Final[dict[str, tuple[str, float]]] = {
    # THE cell: 26.7% of real games and our worst real winrate (35%).
    "alakazam-heur": ("data/decks/meta_alakazam.csv@heuristic", 0.267),
    "alakazam-bc": (
        "data/decks/meta_alakazam.csv@network,models/bc_majkel.npz,"
        "models/feature_stats.npz", 0.0),          # same cell, honest pilot
    "grimmsnarl-bc": (
        "data/decks/meta_grimmsnarl.csv@network,models/bc_grimmsnarl.npz,"
        "models/feature_stats.npz", 0.160),
    "lucario": ("data/decks/seed_mega_lucario.csv@heuristic", 0.160),
    # 14.2% of real games and, until 31/Jul, a cell with no decklist at
    # all -> silently never tested. List reconstructed from our own 32
    # real episodes by src/analysis/mine_opponent_deck.py. Roughly
    # calibrated: internal 97.0% vs 90.6% real (29/32).
    "archaludon": ("data/decks/meta_archaludon.csv@heuristic", 0.142),
    "kangaskhan": ("data/decks/meta_crustle_kangaskhan.csv@heuristic", 0.058),
    "starmie": ("data/decks/meta_starmie.csv@heuristic", 0.040),
    "spidops": ("data/decks/meta_spidops.csv@heuristic", 0.036),
    "mirror": (f"deck.csv@{OUR_PILOT}", 0.013),
}

LOW_DECK: Final[int] = 5    # "1-2 turns from deck-out"


# --------------------------------------------------------------------------
# mechanism probe
# --------------------------------------------------------------------------

@dataclass
class MechanismTally:
    """Mechanism counters, all poolable by addition."""

    games: int = 0
    losses: int = 0
    loss_turns: list[int] = field(default_factory=list)
    win_turns: list[int] = field(default_factory=list)
    # how close the OPPONENT's deck got to zero, in games we LOST
    opp_low_in_losses: list[int] = field(default_factory=list)
    opp_near_deckout_losses: int = 0    # opponent reached <= LOW_DECK
    our_deckouts: int = 0               # we hit 0 first (self-mill deaths)

    def merge(self, other: "MechanismTally") -> None:
        self.games += other.games
        self.losses += other.losses
        self.loss_turns.extend(other.loss_turns)
        self.win_turns.extend(other.win_turns)
        self.opp_low_in_losses.extend(other.opp_low_in_losses)
        self.opp_near_deckout_losses += other.opp_near_deckout_losses
        self.our_deckouts += other.our_deckouts


class _Probe:
    """Tracks one game's deck lows from OUR seat, then files the result."""

    __slots__ = ("_tally", "_our_seat", "_our_low", "_opp_low")

    def __init__(self, tally: MechanismTally, our_seat: int) -> None:
        self._tally = tally
        self._our_seat = our_seat
        self._our_low: int | None = None
        self._opp_low: int | None = None

    def __call__(self, obs_dict: dict) -> None:
        players = ((obs_dict.get("current") or {}).get("players")) or []
        if len(players) < 2:
            return
        for seat, attr in ((self._our_seat, "_our_low"),
                           (1 - self._our_seat, "_opp_low")):
            count = players[seat].get("deckCount")
            if not isinstance(count, int):
                continue
            current = getattr(self, attr)
            setattr(self, attr, count if current is None
                    else min(current, count))

    def finish(self, result: int, turns: int) -> None:
        tally = self._tally
        tally.games += 1
        won = result == self._our_seat
        if won:
            tally.win_turns.append(turns)
            return
        if result not in (0, 1):
            return                      # draw: neither a loss nor a win
        tally.losses += 1
        tally.loss_turns.append(turns)
        if self._opp_low is not None:
            tally.opp_low_in_losses.append(self._opp_low)
            if self._opp_low <= LOW_DECK:
                tally.opp_near_deckout_losses += 1
        if self._our_low == 0:
            tally.our_deckouts += 1


# --------------------------------------------------------------------------
# shard worker
# --------------------------------------------------------------------------

def _run_shard(payload: tuple[str, str, int, int]) -> dict:
    """Play ``games`` of one (our deck, cell) pairing in a fresh engine."""
    a_text, b_text, games, seed = payload
    from ..deckbuilding.gauntlet import run_pair
    from ..deckbuilding.legality import read_deck_ids
    from ..environment_wrapper.ab_test import ArmMetrics, ArmSpec, arm_factory
    from ..ingestion.build_effect_model import EffectIndex
    from ..ingestion.card_index import CardIndex

    index, effects = CardIndex(), EffectIndex()
    deck_a_text, arm_a = a_text.split("@", 1)
    deck_b_text, arm_b = b_text.split("@", 1)
    deck_a = read_deck_ids(REPO_ROOT / deck_a_text)
    deck_b = read_deck_ids(REPO_ROOT / deck_b_text)
    spec_a, spec_b = ArmSpec.parse(arm_a), ArmSpec.parse(arm_b)
    metrics_a, metrics_b = ArmMetrics(), ArmMetrics()

    tally = MechanismTally()

    def on_game(game_index: int, a_seat: int, _seed: int) -> _Probe:
        return _Probe(tally, a_seat)

    pair = run_pair(arm_factory(spec_a, index, effects, metrics_a, deck_a),
                    arm_factory(spec_b, index, effects, metrics_b, deck_b),
                    deck_a, deck_b, games, seed, on_game=on_game)
    return {
        "a_wins": pair.a_wins,
        "b_wins": pair.b_wins,
        "draws": pair.draws,
        "errors": list(pair.errors),
        "a_wins_seat": list(pair.a_wins_by_seat),
        "decided_seat": list(pair.decided_by_seat),
        "mechanism": asdict(tally),
    }


@dataclass
class CellResult:
    """One (arm, cell) pairing, pooled over shards."""

    arm: str
    cell: str
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    errors: list[str] = field(default_factory=list)
    a_wins_seat: list[int] = field(default_factory=lambda: [0, 0])
    decided_seat: list[int] = field(default_factory=lambda: [0, 0])
    mechanism: MechanismTally = field(default_factory=MechanismTally)

    @property
    def decided(self) -> int:
        return self.a_wins + self.b_wins

    @property
    def winrate(self) -> float:
        return self.a_wins / self.decided if self.decided else 0.5

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.a_wins, self.decided)

    def absorb(self, shard: dict) -> None:
        self.a_wins += shard["a_wins"]
        self.b_wins += shard["b_wins"]
        self.draws += shard["draws"]
        self.errors.extend(shard["errors"])
        for i in (0, 1):
            self.a_wins_seat[i] += shard["a_wins_seat"][i]
            self.decided_seat[i] += shard["decided_seat"][i]
        other = MechanismTally(**shard["mechanism"])
        self.mechanism.merge(other)


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def run_matrix(arms: dict[str, str], cells: dict[str, tuple[str, float]],
               games: int, seed: int, workers: int,
               shard_games: int) -> dict[tuple[str, str], CellResult]:
    payloads: list[tuple[str, str, int, int]] = []
    owners: list[tuple[str, str]] = []
    for arm, deck in arms.items():
        a_text = f"{deck}@{OUR_PILOT}"
        for cell, (b_text, _share) in cells.items():
            remaining, shard_index = games, 0
            while remaining > 0:
                n = min(shard_games, remaining)
                payloads.append((a_text, b_text, n,
                                 seed + 1000 * shard_index))
                owners.append((arm, cell))
                remaining -= n
                shard_index += 1

    results = {(arm, cell): CellResult(arm, cell)
               for arm in arms for cell in cells}
    started = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        for owner, shard in zip(owners,
                                pool.imap(_run_shard, payloads, chunksize=1)):
            results[owner].absorb(shard)
    elapsed = time.perf_counter() - started
    print(f"{len(payloads)} shards, {sum(p[2] for p in payloads)} games "
          f"in {elapsed:.0f}s on {workers} workers\n")
    return results


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def report(results: dict[tuple[str, str], CellResult],
           arms: dict[str, str], cells: dict[str, tuple[str, float]]) -> dict:
    payload: dict = {"cells": {}, "aggregate": {}}
    total_errors = sum(len(r.errors) for r in results.values())

    print("=" * 96)
    print("MATRIZ POR CÉLULA — winrate NOSSO lado, IC95 Wilson; "
          "diferença vs SHIP = Newcombe")
    print("=" * 96)
    for cell in cells:
        ship = results[("SHIP", cell)]
        print(f"\n[{cell}]  oponente = {cells[cell][0].split('@')[0]}"
              f"  (share real {cells[cell][1]:.1%})")
        print(f"  {'arm':15s} {'n':>5s} {'WR':>7s} {'IC95':>16s} "
              f"{'Δ vs SHIP':>10s} {'IC95 da Δ':>17s} {'seat0':>8s} "
              f"{'seat1':>8s}")
        for arm in arms:
            res = results[(arm, cell)]
            lo, hi = res.ci
            if arm == "SHIP":
                delta = "     —"
                delta_ci = "               —"
            else:
                diff, dlo, dhi = newcombe_difference(
                    ship.a_wins, ship.decided, res.a_wins, res.decided)
                delta = f"{diff:+6.1%}"
                delta_ci = f"[{dlo:+5.1%}, {dhi:+5.1%}]"
            seat = []
            for i in (0, 1):
                den = res.decided_seat[i]
                seat.append(f"{res.a_wins_seat[i]/den:.0%}" if den else "-")
            print(f"  {arm:15s} {res.decided:5d} {res.winrate:7.1%} "
                  f"[{lo:5.1%},{hi:5.1%}] {delta:>10s} {delta_ci:>17s} "
                  f"{seat[0]:>8s} {seat[1]:>8s}")
            payload["cells"].setdefault(cell, {})[arm] = {
                "decided": res.decided, "wins": res.a_wins,
                "draws": res.draws, "winrate": res.winrate,
                "ci": [lo, hi],
                "a_wins_seat": res.a_wins_seat,
                "decided_seat": res.decided_seat,
                "errors": len(res.errors),
                "mechanism": _mechanism_payload(res.mechanism),
            }

    print("\n" + "=" * 96)
    print("MECANISMO — o que a mudança DEVERIA mover "
          f"(turno da morte; oponente a <= {LOW_DECK} cartas)")
    print("=" * 96)
    for cell in cells:
        print(f"\n[{cell}]")
        print(f"  {'arm':15s} {'derrotas':>8s} {'turno morte':>12s} "
              f"{'turno vitória':>13s} {'opp<=5 nas L':>13s} "
              f"{'opp deck low':>13s} {'self-deckout':>13s}")
        for arm in arms:
            tally = results[(arm, cell)].mechanism
            death = _median(tally.loss_turns)
            win_turn = _median(tally.win_turns)
            low = _median(tally.opp_low_in_losses)
            near = (tally.opp_near_deckout_losses / tally.losses
                    if tally.losses else None)
            print(f"  {arm:15s} {tally.losses:8d} "
                  f"{death if death is not None else '-':>12} "
                  f"{win_turn if win_turn is not None else '-':>13} "
                  f"{near if near is None else f'{near:.1%}':>13} "
                  f"{low if low is not None else '-':>13} "
                  f"{tally.our_deckouts:>13d}")

    print("\n" + "=" * 96)
    print("AGREGADO PONDERADO pelo campo real (só como resumo — "
          "a decisão é por célula)")
    print("=" * 96)
    weighted = {}
    total_share = sum(share for _, (_, share) in cells.items() if share > 0)
    for arm in arms:
        acc = 0.0
        for cell, (_b, share) in cells.items():
            if share <= 0:
                continue
            acc += share * results[(arm, cell)].winrate
        weighted[arm] = acc / total_share if total_share else 0.0
        print(f"  {arm:15s} {weighted[arm]:.1%}")
    payload["aggregate"] = weighted
    payload["total_errors"] = total_errors
    print(f"\nexceptions TOTAIS: {total_errors} (gate: 0)")
    for (arm, cell), res in results.items():
        for error in res.errors[:2]:
            print(f"  {arm}/{cell}: {error}")
    return payload


def _mechanism_payload(tally: MechanismTally) -> dict:
    return {
        "games": tally.games,
        "losses": tally.losses,
        "median_loss_turn": _median(tally.loss_turns),
        "median_win_turn": _median(tally.win_turns),
        "mean_loss_turn": (sum(tally.loss_turns) / len(tally.loss_turns)
                           if tally.loss_turns else None),
        "opp_near_deckout_share_in_losses": (
            tally.opp_near_deckout_losses / tally.losses
            if tally.losses else None),
        "median_opp_deck_low_in_losses": _median(tally.opp_low_in_losses),
        "self_deckouts": tally.our_deckouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=600,
                        help="games per (arm, cell) — the A/B lesson is "
                             "that ~150 is noise; use 300-600")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-games", type=int, default=50)
    parser.add_argument("--cells", nargs="*", default=None,
                        help="restrict to these cell labels")
    parser.add_argument("--arms", nargs="*", default=None,
                        help="restrict to these arm labels (SHIP is always "
                             "kept: it is the baseline)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cells = (CELLS if not args.cells
             else {k: v for k, v in CELLS.items() if k in args.cells})
    arms = (ARMS if not args.arms
            else {k: v for k, v in ARMS.items()
                  if k in set(args.arms) | {"SHIP"}})
    if not cells or not arms:
        raise SystemExit("nenhuma célula/arm selecionada")

    print(f"arms: {list(arms)}\ncells: {list(cells)}\n"
          f"{args.games} jogos por (arm, célula) = "
          f"{args.games * len(arms) * len(cells)} jogos totais\n")
    results = run_matrix(arms, cells, args.games, args.seed,
                         args.workers, args.shard_games)
    payload = report(results, arms, cells)
    out = args.out or (OUT_DIR / "matrix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\njson: {out}")


if __name__ == "__main__":
    main()
