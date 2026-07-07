"""Gate B arena: HeuristicAgent vs RandomAgent over N games.

Seats alternate every game (heuristic plays as player 0 in even games) to
cancel first-player advantage. Reuses play_one_game from selfplay.

Run from the repo root:  python -m src.environment_wrapper.arena --games 100
"""

from __future__ import annotations

import argparse
import time

from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..agent_heuristics.random_agent import RandomAgent, read_deck_csv
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .selfplay import RESULT_DRAW, Agent, play_one_game

GATE_B_WINRATE: float = 0.65


def run_arena(n_games: int, seed: int = 0) -> tuple[int, int, int, list[int], list[str]]:
    """Returns (heuristic wins, random wins, draws, turns per game, errors)."""
    deck = read_deck_csv()
    index = CardIndex()
    effects = EffectIndex()

    heuristic_wins = random_wins = draws = 0
    turns_seen: list[int] = []
    errors: list[str] = []
    for game_index in range(n_games):
        heuristic: Agent = HeuristicAgent(seed=seed + game_index, index=index, effects=effects)
        rand: Agent = RandomAgent(seed=seed + 10_000 + game_index)
        heuristic_seat = game_index % 2  # alternate who starts as player 0
        agents = (heuristic, rand) if heuristic_seat == 0 else (rand, heuristic)
        try:
            result, turns = play_one_game(agents, list(deck), list(deck))
        except Exception as exc:  # noqa: BLE001 — exceptions are a gate metric
            errors.append(f"game {game_index}: {type(exc).__name__}: {exc}")
            continue
        turns_seen.append(turns)
        if result == RESULT_DRAW:
            draws += 1
        elif result == heuristic_seat:
            heuristic_wins += 1
        else:
            random_wins += 1
    return heuristic_wins, random_wins, draws, turns_seen, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    wins, losses, draws, turns, errors = run_arena(args.games, args.seed)
    elapsed = time.perf_counter() - t0
    decided = wins + losses
    winrate = wins / decided if decided else 0.0

    print(f"games:            {args.games} in {elapsed:.1f}s "
          f"({elapsed / max(args.games, 1):.2f}s/game)")
    print(f"heuristic wins:   {wins}")
    print(f"random wins:      {losses}")
    print(f"draws:            {draws}")
    print(f"winrate (decided): {winrate:.1%}")
    print(f"avg turns:        {sum(turns) / len(turns):.1f}" if turns else "avg turns: n/a")
    print(f"exceptions:       {len(errors)}  (must be 0)")
    for error in errors[:10]:
        print(f"  {error}")
    verdict = "PASS" if winrate > GATE_B_WINRATE and not errors else "FAIL"
    print(f"Gate B (>{GATE_B_WINRATE:.0%} winrate, 0 exceptions): {verdict}")


if __name__ == "__main__":
    main()
