"""Why does the search fail in the mirror — wrong model, or too long a horizon?

The mirror A/B was flat-to-negative while the field cells were flat-to-
strongly-positive, and there are two stories that both explain it:

    WRONG MODEL   the rollouts play the opponent as a generic heuristic
                  and the real mirror opponent is CrustleAgent v3
    LONG HORIZON  a Crustle mirror is a mill race that runs ~50 turns,
                  so a rollout from a mid-game decision passes through
                  hundreds of subsequent choices before it resolves.
                  Every one of those is made by a rollout policy, not by
                  the agent, so the leaf value measures the ROLLOUT
                  POLICY's game far more than it measures the candidate.

The first was tested directly (opponent_pilot="match") and moved the
mirror only from 44.8% to 48.3%, not statistically. This measures the
second, which is the cheaper and more structural explanation: how DEEP
is a rollout, per matchup?

The instrument is a counting wrapper around the rollout policies, so
nothing in RuntimeSearchAgent changes — the agent under measurement is
byte-for-byte the one that was measured in the A/Bs.

Read it as: a rollout whose value is decided 400 selections away is not
evaluating the candidate, it is sampling the rollout policy. If the
mirror's depth dwarfs the field's, the search is being asked a question
1-ply determinized rollouts cannot answer at that range, and no amount
of fixing the opponent model helps.

Run from the repo root:
    python -m src.analysis.horizon_probe --games 6
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Final

from ..deckbuilding.legality import read_deck_ids
from ..environment_wrapper.selfplay import play_one_game
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from ..rl_models.runtime_search_agent import RuntimeSearchAgent

MATCHUPS: Final[tuple[tuple[str, str, str], ...]] = (
    ("mirror (Crustle)", "deck.csv", "crustle-v3"),
    ("Grimmsnarl", "data/decks/meta_grimmsnarl.csv", "heuristic"),
    ("Alakazam", "data/decks/meta_alakazam.csv", "heuristic"),
)


class _Counter:
    """Wraps a rollout policy and counts the selections it is asked for."""

    __slots__ = ("_agent", "_box")

    def __init__(self, agent, box: list[int]) -> None:
        self._agent = agent
        self._box = box

    def __call__(self, obs_dict: dict) -> list[int]:
        self._box[0] += 1
        return self._agent(obs_dict)


class _ProbedAgent(RuntimeSearchAgent):
    """RuntimeSearchAgent that records how deep each rollout ran.

    Overrides only the two policy factories, so every decision rule,
    filter and budget check stays exactly as shipped.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.rollout_depths: list[int] = []
        self._live = [0]
        super().__init__(*args, **kwargs)

    def _rollout_agent(self):
        if self._live[0]:
            self.rollout_depths.append(self._live[0])
        self._live = [0]
        return _Counter(super()._rollout_agent(), self._live)

    def _opponent_rollout_agent(self, archetype: str):
        return _Counter(super()._opponent_rollout_agent(archetype),
                        self._live)

    def flush(self) -> None:
        if self._live[0]:
            self.rollout_depths.append(self._live[0])
            self._live = [0]


def _pilot(kind: str, index: CardIndex, effects: EffectIndex, seed: int):
    if kind == "crustle-v3":
        from ..agent_heuristics.crustle_agent import CrustleAgent
        return CrustleAgent(seed=seed, index=index, effects=effects,
                            variant="v3")
    from ..agent_heuristics.heuristic_agent import HeuristicAgent
    return HeuristicAgent(seed=seed, index=index, effects=effects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    index, effects = CardIndex(), EffectIndex()
    our_deck = read_deck_ids(REPO_ROOT / "deck.csv")

    print(f"\n== rollout depth by matchup ({args.games} games each) ==")
    print(f"  {'matchup':18s} {'turns':>6s} {'rollouts':>9s} "
          f"{'median':>7s} {'p90':>6s} {'max':>6s}")
    for label, deck_rel, opp_kind in MATCHUPS:
        opp_deck = read_deck_ids(REPO_ROOT / deck_rel)
        depths: list[int] = []
        turns_seen: list[int] = []
        for game in range(args.games):
            agent = _ProbedAgent(index=index, effects=effects,
                                 seed=args.seed + game,
                                 own_deck_ids=list(our_deck),
                                 opponent_deck_override=list(opp_deck))
            opponent = _pilot(opp_kind, index, effects,
                              args.seed + 5000 + game)
            try:
                _result, turns = play_one_game(
                    (agent, opponent), list(our_deck), list(opp_deck))
                turns_seen.append(turns)
            except Exception as exc:  # noqa: BLE001 — probe, never fatal
                print(f"    ({label} game {game} failed: {exc})")
                continue
            agent.flush()
            depths.extend(agent.rollout_depths)
        if not depths:
            print(f"  {label:18s}  (no rollouts recorded)")
            continue
        ordered = sorted(depths)
        print(f"  {label:18s} {statistics.mean(turns_seen):6.1f} "
              f"{len(depths):9d} {statistics.median(ordered):7.0f} "
              f"{ordered[int(len(ordered) * 0.9)]:6d} {ordered[-1]:6d}")

    print("\n  A rollout that runs hundreds of selections before it resolves\n"
          "  is measuring the ROLLOUT POLICY, not the candidate: every one\n"
          "  of those selections is played by the policy, and the candidate\n"
          "  contributes only the first. Depth is the horizon problem made\n"
          "  countable, and no opponent-model fix reaches it.")


if __name__ == "__main__":
    main()
