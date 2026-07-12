"""Board-development diagnostic: CrustleAgent variants vs aggressive decks.

The v2 ladder failure mode (misplay hunt, 474/474 fidelity) is the
BOARD-WIPE: ~11/12 losses died with a nearly full deck and 0-2 pokémon
in play, because the anti-self-mill RELATIVE trigger suppressed
consistency items during setup. Winrate alone can saturate and hide the
fix, so this harness measures the mechanism directly: how much board
each variant develops against decks that punish a thin board.

Per (variant, opponent), over n games (seats alternated, engine loop =
selfplay.play_one_game — never reimplemented; statistics reused from
ab_test.wilson_interval):
  - winrate + Wilson 95% CI over decided games
  - board@t5 / board@t9: mean of OUR pokémon in play at our last
    decision with state turn <= 5 / <= 9 (setup depth)
  - wipe-state rate: games where our field drops to <= 1 pokémon at any
    decision from turn 3 on (the terminal-spiral precursor)
  - board-wipe-loss share: losses whose LAST observed state had <= 2
    pokémon in play AND deck > 20 cards (the ladder loss signature)

Run from the repo root:
    python -m src.analysis.board_development --games 100 --seed 0 \
        --variants v2 v3 --opponents raging_bolt mega_lucario terastal_box
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Callable, Final

from ..agent_heuristics.crustle_agent import CrustleAgent
from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..deckbuilding.gauntlet import discover_decks
from ..deckbuilding.legality import read_deck_ids, validate_deck
from ..environment_wrapper.ab_test import wilson_interval
from ..environment_wrapper.selfplay import RESULT_DRAW, Agent, play_one_game
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex

T_EARLY: Final[int] = 5     # "board@t5": end of the setup turns
T_MID: Final[int] = 9       # "board@t9": developed midgame
WIPE_FROM_TURN: Final[int] = 3   # ignore the pre-setup single active
WIPE_FIELD: Final[int] = 1
LOSS_FIELD: Final[int] = 2       # ladder signature: died with 0-2 in play
LOSS_DECK: Final[int] = 20       # ...and a deck still full


class BoardProbe:
    """Wraps OUR agent; reads the raw obs dict before every decision.

    All reads are None-safe on the raw dict (never trusts a field), so a
    malformed observation degrades to missing metrics, never a crash.
    """

    __slots__ = ("_agent", "board_t5", "board_t9", "wiped", "final_field",
                 "final_deck")

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self.board_t5: int | None = None
        self.board_t9: int | None = None
        self.wiped: bool = False
        self.final_field: int | None = None
        self.final_deck: int | None = None

    def __call__(self, obs_dict: dict) -> list[int]:
        try:
            state = obs_dict.get("current") or {}
            me = state.get("yourIndex")
            turn = state.get("turn")
            players = state.get("players") or []
            if me is not None and 0 <= me < len(players):
                mine = players[me] or {}
                count = (len([p for p in (mine.get("active") or []) if p])
                         + len([p for p in (mine.get("bench") or []) if p]))
                self.final_field = count
                self.final_deck = mine.get("deckCount")
                if turn is not None:
                    if turn <= T_EARLY:
                        self.board_t5 = count
                    if turn <= T_MID:
                        self.board_t9 = count
                    if turn >= WIPE_FROM_TURN and count <= WIPE_FIELD:
                        self.wiped = True
        except Exception:
            pass  # metrics are best-effort; the decision must go through
        return self._agent(obs_dict)


@dataclass
class MatchupStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    errors: list[str] = field(default_factory=list)
    t5: list[int] = field(default_factory=list)
    t9: list[int] = field(default_factory=list)
    wipe_games: int = 0
    boardwipe_losses: int = 0

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    def report(self, label: str) -> None:
        lo, hi = wilson_interval(self.wins, self.decided)
        n = self.decided + self.draws
        mean_t5 = sum(self.t5) / len(self.t5) if self.t5 else 0.0
        mean_t9 = sum(self.t9) / len(self.t9) if self.t9 else 0.0
        bw_share = (self.boardwipe_losses / self.losses
                    if self.losses else 0.0)
        print(f"{label}")
        print(f"  n={n} decided={self.decided} draws={self.draws} "
              f"wins={self.wins} losses={self.losses}")
        print(f"  winrate {self.wins / self.decided if self.decided else 0.0:.1%}"
              f"  IC95 [{lo:.1%}, {hi:.1%}]")
        print(f"  board@t5 {mean_t5:.2f}  board@t9 {mean_t9:.2f}  "
              f"wipe-state rate {self.wipe_games / max(n, 1):.1%}  "
              f"board-wipe losses {self.boardwipe_losses}/{self.losses} "
              f"({bw_share:.1%} of losses)")
        print(f"  exceptions {len(self.errors)} (must be 0)")
        for error in self.errors[:5]:
            print(f"    {error}")


def run_matchup(make_ours: Callable[[int], Agent],
                make_theirs: Callable[[int], Agent],
                our_deck: list[int], their_deck: list[int],
                n_games: int, seed: int) -> MatchupStats:
    """Same seat/deck alternation contract as gauntlet.run_pair, with a
    BoardProbe on OUR side and per-game outcome pairing (run_pair only
    aggregates, and the loss classification needs game-level data)."""
    stats = MatchupStats()
    for game_index in range(n_games):
        probe = BoardProbe(make_ours(seed + game_index))
        opponent = make_theirs(seed + 10_000 + game_index)
        our_seat = game_index % 2
        agents = (probe, opponent) if our_seat == 0 else (opponent, probe)
        decks = ((our_deck, their_deck) if our_seat == 0
                 else (their_deck, our_deck))
        try:
            result, _ = play_one_game(agents, list(decks[0]), list(decks[1]))
        except Exception as exc:  # noqa: BLE001 — exceptions are a gate metric
            stats.errors.append(
                f"game {game_index}: {type(exc).__name__}: {exc}")
            continue
        if probe.board_t5 is not None:
            stats.t5.append(probe.board_t5)
        if probe.board_t9 is not None:
            stats.t9.append(probe.board_t9)
        if probe.wiped:
            stats.wipe_games += 1
        if result == RESULT_DRAW:
            stats.draws += 1
        elif result == our_seat:
            stats.wins += 1
        else:
            stats.losses += 1
            if (probe.final_field is not None
                    and probe.final_field <= LOSS_FIELD
                    and (probe.final_deck or 0) > LOSS_DECK):
                stats.boardwipe_losses += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["v2", "v3"])
    parser.add_argument("--opponents", type=str, nargs="+",
                        default=["raging_bolt", "mega_lucario",
                                 "terastal_box"])
    args = parser.parse_args()

    index = CardIndex()
    effects = EffectIndex()
    decks = discover_decks()
    for name in ["crustle", *args.opponents]:
        if name not in decks:
            raise SystemExit(f"unknown deck '{name}' (field: {sorted(decks)})")
    ids: dict[str, list[int]] = {}
    for name in ["crustle", *args.opponents]:
        deck_ids = read_deck_ids(decks[name])
        report = validate_deck(deck_ids, index)
        if not report.ok:
            raise SystemExit(f"deck '{name}' is ILLEGAL — aborting")
        ids[name] = deck_ids

    t0 = time.perf_counter()
    for opponent in args.opponents:
        for variant in args.variants:
            make_ours = (lambda s, v=variant: CrustleAgent(
                seed=s, index=index, effects=effects, variant=v))
            make_theirs = (lambda s: HeuristicAgent(
                seed=s, index=index, effects=effects))
            stats = run_matchup(make_ours, make_theirs, ids["crustle"],
                                ids[opponent], args.games, args.seed)
            stats.report(f"[board-diag] crustle-{variant} vs {opponent} "
                         f"(heuristic pilot)")
    print(f"(wall time {time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
