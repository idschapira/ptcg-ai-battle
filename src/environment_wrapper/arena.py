"""Arena: any two agents over N games with alternating seats (quality gates).

Seats alternate every game (agent A plays as player 0 in even games) to
cancel first-player advantage. Reuses play_one_game from selfplay.

Gates (CLAUDE.md): B = heuristic > random at >65% winrate;
C = network >= best baseline. exceptions must always be 0.

Run from the repo root:
    python -m src.environment_wrapper.arena --games 100            # Gate B
    python -m src.environment_wrapper.arena --a network --b heuristic
    python -m src.environment_wrapper.arena --a network --b random
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

from ..agent_heuristics.random_agent import RandomAgent, read_deck_csv
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .selfplay import RESULT_DRAW, Agent, play_one_game

GATE_B_WINRATE: float = 0.65

AGENT_KINDS = ("random", "heuristic", "network")


def _factory(kind: str, index: CardIndex,
             effects: EffectIndex) -> Callable[[int], Agent]:
    """Per-game agent factory (index/effects shared across games)."""
    if kind == "heuristic":
        from ..agent_heuristics.heuristic_agent import HeuristicAgent
        return lambda seed: HeuristicAgent(seed=seed, index=index, effects=effects)
    if kind == "network":
        from ..rl_models.network_agent import NetworkAgent
        network = NetworkAgent(index=index, effects=effects)  # deterministic
        return lambda seed: network
    return lambda seed: RandomAgent(seed=seed)


def run_arena(n_games: int, seed: int = 0, a: str = "heuristic",
              b: str = "random") -> tuple[int, int, int, list[int], list[str]]:
    """Returns (A wins, B wins, draws, turns per game, errors)."""
    deck = read_deck_csv()
    index = CardIndex()
    effects = EffectIndex()
    make_a, make_b = _factory(a, index, effects), _factory(b, index, effects)

    a_wins = b_wins = draws = 0
    turns_seen: list[int] = []
    errors: list[str] = []
    for game_index in range(n_games):
        agent_a = make_a(seed + game_index)
        agent_b = make_b(seed + 10_000 + game_index)
        a_seat = game_index % 2  # alternate who starts as player 0
        agents = (agent_a, agent_b) if a_seat == 0 else (agent_b, agent_a)
        try:
            result, turns = play_one_game(agents, list(deck), list(deck))
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
    return a_wins, b_wins, draws, turns_seen, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--a", choices=AGENT_KINDS, default="heuristic")
    parser.add_argument("--b", choices=AGENT_KINDS, default="random")
    args = parser.parse_args()

    t0 = time.perf_counter()
    wins, losses, draws, turns, errors = run_arena(args.games, args.seed,
                                                   args.a, args.b)
    elapsed = time.perf_counter() - t0
    decided = wins + losses
    winrate = wins / decided if decided else 0.0

    print(f"games:            {args.games} in {elapsed:.1f}s "
          f"({elapsed / max(args.games, 1):.2f}s/game)")
    print(f"{args.a} wins:    {wins}")
    print(f"{args.b} wins:    {losses}")
    print(f"draws:            {draws}")
    print(f"winrate {args.a} (decided): {winrate:.1%}")
    print(f"avg turns:        {sum(turns) / len(turns):.1f}" if turns else "avg turns: n/a")
    print(f"exceptions:       {len(errors)}  (must be 0)")
    for error in errors[:10]:
        print(f"  {error}")
    if (args.a, args.b) == ("heuristic", "random"):
        verdict = "PASS" if winrate > GATE_B_WINRATE and not errors else "FAIL"
        print(f"Gate B (>{GATE_B_WINRATE:.0%} winrate, 0 exceptions): {verdict}")


if __name__ == "__main__":
    main()
