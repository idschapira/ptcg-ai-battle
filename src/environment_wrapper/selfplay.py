"""Local self-play harness: N games between configurable agents.

Drives the vendored engine binary directly through cg.game (the same path
sample_submission uses) and reports wins, average turns and exceptions.
With --record <dir>, every decision point is serialized to one JSON per
game for viewer/battle_viewer.html (dev/debug tool, not training data).

Run from the repo root:
    python -m src.environment_wrapper.selfplay --games 20
    python -m src.environment_wrapper.selfplay --p0 heuristic --p1 random \
        --games 20 --record viewer/recordings
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final

from cg import game

from ..agent_heuristics.random_agent import RandomAgent, read_deck_csv
from .recorder import GameRecorder

Agent = Callable[[dict], list[int]]

# A random-vs-random game rarely exceeds a few hundred selections; this cap
# only exists so a wrapper bug cannot hang the harness forever.
MAX_SELECTIONS_PER_GAME: Final[int] = 20_000

RESULT_DRAW: Final[int] = 2


@dataclass
class SelfPlayStats:
    games: int = 0
    wins: list[int] = field(default_factory=lambda: [0, 0])
    draws: int = 0
    turns: list[int] = field(default_factory=list)
    exceptions: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def avg_turns(self) -> float:
        return sum(self.turns) / len(self.turns) if self.turns else 0.0


def play_one_game(
    agents: tuple[Agent, Agent],
    deck0: list[int],
    deck1: list[int],
    recorder: GameRecorder | None = None,
) -> tuple[int, int]:
    """Play a single game; returns (result, final turn count).

    result: 0/1 = winning player index, 2 = draw.
    """
    obs_dict, start_data = game.battle_start(deck0, deck1)
    if obs_dict is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start_data.errorPlayer} "
            f"errorType={start_data.errorType}"
        )
    try:
        for _ in range(MAX_SELECTIONS_PER_GAME):
            state = obs_dict["current"]
            result = state["result"]
            if result != -1:
                return result, state["turn"]
            acting_player = state["yourIndex"]
            agent = agents[acting_player]
            answer = agent(obs_dict)
            if recorder is not None:
                recorder.record_step(obs_dict, answer,
                                     getattr(agent, "last_scores", None))
            obs_dict = game.battle_select(answer)
        raise RuntimeError(f"game exceeded {MAX_SELECTIONS_PER_GAME} selections")
    finally:
        game.battle_finish()


def _make_agent(kind: str, seed: int) -> Agent:
    if kind == "heuristic":
        from ..agent_heuristics.heuristic_agent import HeuristicAgent
        return HeuristicAgent(seed=seed)
    return RandomAgent(seed=seed)


def run_selfplay(n_games: int, seed: int = 0, p0: str = "random", p1: str = "random",
                 record_dir: Path | None = None) -> SelfPlayStats:
    deck = read_deck_csv()
    stats = SelfPlayStats()
    index = None
    if record_dir is not None:
        from ..ingestion.card_index import CardIndex
        index = CardIndex()
    for game_index in range(n_games):
        agents: tuple[Agent, Agent] = (
            _make_agent(p0, seed + 2 * game_index),
            _make_agent(p1, seed + 2 * game_index + 1),
        )
        recorder = (GameRecorder(index, (p0, p1))
                    if record_dir is not None and index is not None else None)
        stats.games += 1
        try:
            result, turns = play_one_game(agents, list(deck), list(deck), recorder)
        except Exception as exc:  # noqa: BLE001 — the whole point is counting these
            stats.exceptions += 1
            stats.errors.append(f"game {game_index}: {type(exc).__name__}: {exc}")
            continue
        stats.turns.append(turns)
        if result == RESULT_DRAW:
            stats.draws += 1
        else:
            stats.wins[result] += 1
        if recorder is not None and record_dir is not None:
            recorder.save(record_dir / f"game_{game_index:03d}.json", result, turns)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20, help="number of games to play")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument("--p0", choices=("random", "heuristic"), default="random")
    parser.add_argument("--p1", choices=("random", "heuristic"), default="random")
    parser.add_argument("--record", type=Path, default=None, metavar="DIR",
                        help="write one decoded JSON per game into DIR")
    args = parser.parse_args()

    t0 = time.perf_counter()
    stats = run_selfplay(args.games, args.seed, args.p0, args.p1, args.record)
    elapsed = time.perf_counter() - t0

    print(f"games played:     {stats.games} in {elapsed:.1f}s "
          f"({elapsed / max(stats.games, 1):.2f}s/game)")
    print(f"wins player 0:    {stats.wins[0]}")
    print(f"wins player 1:    {stats.wins[1]}")
    print(f"draws:            {stats.draws}")
    print(f"avg turns:        {stats.avg_turns:.1f}")
    print(f"exceptions:       {stats.exceptions}  (must be 0)")
    for error in stats.errors:
        print(f"  {error}")


if __name__ == "__main__":
    main()
