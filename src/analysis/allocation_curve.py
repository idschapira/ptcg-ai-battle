"""Cost curve: how much bank does a search cost, per allocation setting?

Two questions in one measurement, because they trade against each other:

ALLOCATION (how much of the game do we search?)
    ``contested_margin`` decides which decisions are worth searching: a
    decision is DOMINATED when the prior's best option beats the
    runner-up by more than that fraction of the score range. Smaller
    margin = stricter = fewer searches = cheaper. Sweeping it traces the
    cost against the fraction searched.

ROLLOUT OPPONENT MODEL (how expensive is each rollout?)
    The measurements say the search's value tracks how well the rollout
    policy matches the true opponent, which argues for behaviour-cloned
    nets instead of a generic heuristic. A net costs more per selection
    than a heuristic, and "3-4x" was a guess — this measures it.

The output is what a budget decision needs: for each configuration, the
fraction of decisions searched, the worst episode, and that episode
projected onto Kaggle's slower cores as a share of the 600s bank.

Winrate is NOT measured here — this is the cost side alone, run cheap
and on few games. Effect comes from the A/B presets.

Run from the repo root:
    python -m src.analysis.allocation_curve --games 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Final

from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..deckbuilding.legality import read_deck_ids
from ..environment_wrapper.selfplay import play_one_game
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from ..rl_models.runtime_search_agent import (OPPONENT_PILOT_GENERIC,
                                              OPPONENT_PILOT_NETWORK,
                                              RuntimeSearchAgent)

BANK_S: Final[float] = 600.0
KAGGLE_SLOWDOWN: Final[float] = 3.0

# margin -> label. None means "search every eligible decision".
MARGINS: Final[tuple[float | None, ...]] = (None, 0.30, 0.10, 0.05, 0.02)


def measure(margin: float | None, pilot: str, games: int, seed: int,
            index: CardIndex, effects: EffectIndex,
            our_deck: list[int], opp_deck: list[int]) -> dict:
    walls: list[float] = []
    decisions = searched = rollouts = exceptions = 0
    for game in range(games):
        agent = RuntimeSearchAgent(
            index=index, effects=effects, seed=seed + game,
            own_deck_ids=list(our_deck),
            opponent_deck_override=list(opp_deck),
            contested_margin=margin, opponent_pilot=pilot)
        opponent = HeuristicAgent(seed=seed + 500 + game, index=index,
                                  effects=effects)
        t0 = time.perf_counter()
        try:
            play_one_game((agent, opponent), list(our_deck), list(opp_deck))
        except Exception:  # noqa: BLE001 — counted, never fatal to the sweep
            exceptions += 1
        walls.append(time.perf_counter() - t0)
        decisions += agent.stats.decisions
        searched += agent.stats.searched
        rollouts += agent.stats.rollouts
        exceptions += agent.stats.exceptions
    worst = max(walls) if walls else 0.0
    return {
        "share_searched": searched / decisions if decisions else 0.0,
        "searched": searched, "decisions": decisions, "rollouts": rollouts,
        "median_s": sorted(walls)[len(walls) // 2] if walls else 0.0,
        "worst_s": worst,
        "projected_s": worst * KAGGLE_SLOWDOWN,
        "bank_share": worst * KAGGLE_SLOWDOWN / BANK_S,
        "exceptions": exceptions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deck", type=Path,
                        default=Path("data/decks/meta_grimmsnarl.csv"),
                        help="opponent deck (Grimmsnarl is the cell that "
                             "was over budget)")
    args = parser.parse_args()

    index, effects = CardIndex(), EffectIndex()
    our_deck = read_deck_ids(REPO_ROOT / "deck.csv")
    opp_deck = read_deck_ids(REPO_ROOT / args.deck)

    print(f"\n== cost curve vs {args.deck.name}, {args.games} games/point ==")
    print(f"  {'rollout model':14s} {'margin':>7s} {'searched':>9s} "
          f"{'median':>8s} {'worst':>8s} {'x3':>7s} {'bank':>6s}  exc")
    for pilot, label in ((OPPONENT_PILOT_GENERIC, "heuristic"),
                         (OPPONENT_PILOT_NETWORK, "BC net")):
        for margin in MARGINS:
            row = measure(margin, pilot, args.games, args.seed, index,
                          effects, our_deck, opp_deck)
            tag = "off" if margin is None else f"{margin:.2f}"
            flag = "" if row["bank_share"] < 0.5 else "  OVER"
            print(f"  {label:14s} {tag:>7s} {row['share_searched']:8.0%} "
                  f"{row['median_s']:7.1f}s {row['worst_s']:7.1f}s "
                  f"{row['projected_s']:6.0f}s {row['bank_share']:5.0%} "
                  f"{row['exceptions']:4d}{flag}")

    print(f"\n  bank = {BANK_S:.0f}s per agent per episode; x3 is the assumed "
          f"Kaggle slowdown.\n  Target: worst-case projection comfortably "
          f"under 50% of the bank.")


if __name__ == "__main__":
    main()
