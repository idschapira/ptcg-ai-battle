"""Deck gauntlet (Task 4.5c): round-robin of seed decks under ONE pilot.

Measures the deck lever in isolation: the production NetworkAgent
(models/policy_value.npz + models/feature_stats.npz) plays BOTH sides of
every game — only the decks differ — over a full round-robin of
data/decks/ seeds with alternating seats. Also runs the sanity check:
the top candidate deck with the network pilot vs HeuristicAgent and vs
RandomAgent (mirror deck, pilots differ).

Every deck is validated by src/deckbuilding/legality.py before any
simulation; an illegal deck aborts the run. Game loop = play_one_game
from selfplay (never reimplemented); seat/deck alternation mirrors
arena.run_arena. Pilot latency is measured per pairing (µs per
NetworkAgent call, encode+normalize+forward included).

Run from the repo root:
    python -m src.deckbuilding.gauntlet --games 100 --seed 0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..agent_heuristics.random_agent import RandomAgent
from ..environment_wrapper.selfplay import (RESULT_DRAW, Agent,
                                            play_one_game)
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .legality import read_deck_ids, validate_deck

DECKS_DIR: Final[Path] = REPO_ROOT / "data" / "decks"
DECKS: Final[dict[str, Path]] = {
    "mega_lucario": DECKS_DIR / "seed_mega_lucario.csv",
    "iono": DECKS_DIR / "seed_iono.csv",
    "abomasnow": DECKS_DIR / "placeholder_abomasnow.csv",
}
CANDIDATE: Final[str] = "mega_lucario"  # sanity-check deck (60.4% in cabt)


class TimedAgent:
    """Wraps an agent, collecting per-call latency in µs (None-safe)."""

    __slots__ = ("_agent", "times_us")

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self.times_us: list[float] = []

    def __call__(self, obs_dict: dict) -> list[int]:
        t0 = time.perf_counter()
        answer = self._agent(obs_dict)
        self.times_us.append((time.perf_counter() - t0) * 1e6)
        return answer


@dataclass(frozen=True)
class PairResult:
    """A wins/losses over n games of deck_a vs deck_b (seats alternated)."""

    a_wins: int
    b_wins: int
    draws: int
    avg_turns: float
    latency_mean_us: float
    latency_p99_us: float
    errors: tuple[str, ...]

    @property
    def games(self) -> int:
        return self.a_wins + self.b_wins + self.draws

    @property
    def winrate_a(self) -> float:
        decided = self.a_wins + self.b_wins
        return self.a_wins / decided if decided else 0.5


def run_pair(make_a: Callable[[int], Agent], make_b: Callable[[int], Agent],
             deck_a: list[int], deck_b: list[int], n_games: int,
             seed: int, timed: TimedAgent | None = None) -> PairResult:
    """n games of (agent A, deck A) vs (agent B, deck B), seats alternating.

    Same alternation contract as arena.run_arena: A is player 0 in even
    games and the deck follows its agent. `timed` aggregates the pilot's
    per-call latency across the pair (pass the wrapper used inside
    make_a/make_b).
    """
    a_wins = b_wins = draws = 0
    turns_seen: list[int] = []
    errors: list[str] = []
    for game_index in range(n_games):
        agent_a = make_a(seed + game_index)
        agent_b = make_b(seed + 10_000 + game_index)
        a_seat = game_index % 2
        agents = (agent_a, agent_b) if a_seat == 0 else (agent_b, agent_a)
        decks = (deck_a, deck_b) if a_seat == 0 else (deck_b, deck_a)
        try:
            result, turns = play_one_game(agents, list(decks[0]), list(decks[1]))
        except Exception as exc:  # noqa: BLE001 — exceptions are a gate metric
            errors.append(f"game {game_index}: {type(exc).__name__}: {exc}")
            continue
        turns_seen.append(turns)
        if result == RESULT_DRAW:
            draws += 1
        elif result == a_seat:
            a_wins += 1
        else:
            b_wins += 1

    times = sorted(timed.times_us) if timed is not None else []
    return PairResult(
        a_wins=a_wins, b_wins=b_wins, draws=draws,
        avg_turns=sum(turns_seen) / len(turns_seen) if turns_seen else 0.0,
        latency_mean_us=sum(times) / len(times) if times else 0.0,
        latency_p99_us=times[min(len(times) - 1, int(len(times) * 0.99))]
        if times else 0.0,
        errors=tuple(errors),
    )


def _load_validated_decks(index: CardIndex) -> dict[str, list[int]]:
    """Read every gauntlet deck; abort (SystemExit) on any illegal one."""
    decks: dict[str, list[int]] = {}
    for name, path in DECKS.items():
        ids = read_deck_ids(path)
        report = validate_deck(ids, index)
        if not report.ok:
            for error in report.errors:
                print(f"  - {error}")
            raise SystemExit(f"deck '{name}' ({path}) is ILLEGAL — aborting")
        decks[name] = ids
        print(f"deck {name:14s} LEGAL ({path.name})")
    return decks


def _fmt_pair(label: str, res: PairResult) -> str:
    return (f"{label}: {res.a_wins}-{res.b_wins}"
            f"{f'-{res.draws}d' if res.draws else ''} "
            f"({res.winrate_a:.1%} decided), {res.games} games, "
            f"avg turns {res.avg_turns:.1f}, "
            f"pilot {res.latency_mean_us:.0f}us mean / "
            f"{res.latency_p99_us:.0f}us p99, "
            f"exceptions {len(res.errors)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100,
                        help="games per pairing (>=60 recommended)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    index = CardIndex()
    effects = EffectIndex()
    decks = _load_validated_decks(index)

    from ..rl_models.network_agent import NetworkAgent
    pilot = NetworkAgent(index=index, effects=effects)  # production pair
    if pilot._fallback is not None:
        raise SystemExit("production weights missing — gauntlet needs the net")

    names = list(DECKS)
    winrate: dict[tuple[str, str], float] = {}
    print(f"\n=== round-robin: production pilot both sides, "
          f"{args.games} games/pair ===")
    t0 = time.perf_counter()
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            timed = TimedAgent(pilot)  # both seats -> every call is the pilot
            res = run_pair(lambda s: timed, lambda s: timed,
                           decks[name_a], decks[name_b],
                           args.games, args.seed, timed)
            winrate[(name_a, name_b)] = res.winrate_a
            winrate[(name_b, name_a)] = 1.0 - res.winrate_a
            print(_fmt_pair(f"{name_a} vs {name_b}", res))
            for error in res.errors[:5]:
                print(f"  {error}")
    print(f"(round-robin wall time {time.perf_counter() - t0:.1f}s)")

    print("\nwinrate matrix (row vs col, decided games):")
    width = max(len(n) for n in names)
    print(" " * (width + 2) + "  ".join(f"{n:>12s}" for n in names)
          + f"  {'mean':>7s}")
    for name_a in names:
        cells = []
        rates = []
        for name_b in names:
            if name_a == name_b:
                cells.append(f"{'-':>12s}")
            else:
                rate = winrate[(name_a, name_b)]
                rates.append(rate)
                cells.append(f"{rate:>12.1%}")
        mean = sum(rates) / len(rates)
        print(f"{name_a:<{width}s}  " + "  ".join(cells) + f"  {mean:>7.1%}")

    print(f"\n=== sanity: deck {CANDIDATE} mirror, pilots differ, "
          f"{args.games} games each ===")
    candidate = decks[CANDIDATE]
    timed = TimedAgent(pilot)
    res = run_pair(lambda s: timed,
                   lambda s: HeuristicAgent(seed=s, index=index,
                                            effects=effects),
                   candidate, candidate, args.games, args.seed, timed)
    print(_fmt_pair("network vs heuristic", res))
    timed = TimedAgent(pilot)
    res = run_pair(lambda s: timed, lambda s: RandomAgent(seed=s),
                   candidate, candidate, args.games, args.seed, timed)
    print(_fmt_pair("network vs random", res))


if __name__ == "__main__":
    main()
