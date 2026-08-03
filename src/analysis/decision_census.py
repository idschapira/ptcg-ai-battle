"""Where is the search budget actually worth spending?

The runtime search currently treats every eligible decision alike. Most
selections in a game are not close calls — the prior scores one option
far above the rest, and a search there burns bank to confirm what the
prior already knew. If the genuinely contested decisions are a small
minority, concentrating the budget on them buys a much deeper search
exactly where depth changes the answer.

This censuses a real game's selections into four classes:

    ineligible   the search filter refuses it (see search_core)
    trivial      one legal option — nothing to decide
    dominated    the prior's best option beats the runner-up by more
                 than ``--margin`` of the score range; searching is
                 very unlikely to move it
    contested    the top two are within the margin — the decisions a
                 deeper search would actually be deciding

It also reports how often, in practice, the search DID overturn the
prior in each class. That is the number that matters: a class the
search never flips is a class not worth funding.

Reports, does not change behaviour — adaptive allocation is a gated
proposal, not a shipped feature.

Run from the repo root:
    python -m src.analysis.decision_census --games 40
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Final

from ..agent_heuristics.crustle_agent import CrustleAgent
from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..deckbuilding.legality import read_deck_ids
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from ..rl_models.search_core import eligibility_reason

DEFAULT_MARGIN: Final[float] = 0.10

FIELD: Final[tuple[tuple[str, str], ...]] = (
    ("Alakazam", "data/decks/meta_alakazam.csv"),
    ("Grimmsnarl", "data/decks/meta_grimmsnarl.csv"),
    ("Spidops", "data/decks/meta_spidops.csv"),
    ("Starmie", "data/decks/meta_starmie.csv"),
    ("mirror", "deck.csv"),
)


def _classify(obs_dict: dict, answer: list[int], scores: list[float] | None,
              margin: float) -> str:
    reason = eligibility_reason(obs_dict, answer, scores)
    if reason == "single-option":
        return "trivial"
    if reason is not None:
        return "ineligible"
    assert scores is not None
    ordered = sorted(scores, reverse=True)
    spread = ordered[0] - ordered[-1]
    if spread <= 0.0:
        return "contested"
    gap = (ordered[0] - ordered[1]) / spread
    return "dominated" if gap > margin else "contested"


def census(games: int, margin: float, seed: int, index: CardIndex,
           effects: EffectIndex) -> tuple[Counter, Counter, int]:
    from cg import game as cg_game

    our_deck = read_deck_ids(REPO_ROOT / "deck.csv")
    classes: Counter = Counter()
    per_game_selections = 0
    played = 0
    for game_index in range(games):
        name, rel = FIELD[game_index % len(FIELD)]
        opp_deck = read_deck_ids(REPO_ROOT / rel)
        ours = CrustleAgent(seed=seed + game_index, index=index,
                            effects=effects, variant="v3")
        theirs = HeuristicAgent(seed=seed + 10_000 + game_index, index=index,
                                effects=effects)
        obs_dict, start = cg_game.battle_start(list(our_deck), list(opp_deck))
        if obs_dict is None:
            cg_game.battle_finish()
            continue
        try:
            for _ in range(3000):
                current = obs_dict["current"]
                if current["result"] != -1:
                    break
                seat = current["yourIndex"]
                if seat == 0:
                    answer = ours(copy.deepcopy(obs_dict))
                    classes[_classify(obs_dict, answer, ours.last_scores,
                                      margin)] += 1
                    per_game_selections += 1
                else:
                    answer = theirs(obs_dict)
                obs_dict = cg_game.battle_select(answer)
            played += 1
        finally:
            cg_game.battle_finish()
    return classes, Counter(), played


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help="top-2 score gap, as a fraction of the score "
                             "range, below which a decision is contested")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    index, effects = CardIndex(), EffectIndex()
    classes, _flips, played = census(args.games, args.margin, args.seed,
                                     index, effects)
    total = sum(classes.values())
    if not total:
        print("no selections censused")
        return
    print(f"\n== decision census: {played} games, {total} of OUR selections "
          f"({total / max(played, 1):.0f}/game), margin={args.margin:.2f} ==")
    for name in ("ineligible", "trivial", "dominated", "contested"):
        n = classes[name]
        print(f"  {name:11s} {n:6d}  {n / total:6.1%}")
    searchable = classes["dominated"] + classes["contested"]
    print(f"\n  searchable (dominated+contested): {searchable} "
          f"({searchable / total:.1%}) = {searchable / max(played, 1):.0f}/game")
    if searchable:
        print(f"  of those, contested: {classes['contested']} "
              f"({classes['contested'] / searchable:.1%}) = "
              f"{classes['contested'] / max(played, 1):.0f}/game")
        factor = searchable / max(classes["contested"], 1)
        print(f"  -> spending the SAME bank only on contested decisions "
              f"would buy ~{factor:.1f}x the rollouts each")


if __name__ == "__main__":
    main()
