"""Does the pilot REFUSE to attack, or is attacking never legal?

"attacks/game is too low" is the symptom. It has two very different
causes, and the fix for one is worthless against the other:

  the pilot declines   an attack was on the table and something else
                       outscored it -> the SCORE BANDS are the problem
  never on the table   the engine never offered an attack that turn ->
                       the BOARD is the problem (no attacker in front, or
                       no energy on it), and no band can fix that

Attacking ENDS the turn while every other action keeps the MAIN prompt
open, so a low attack band only REORDERS actions inside a turn. It can
only cost an attack if a development action taken first removes the
ability to attack. That makes the split above measurable rather than
arguable, which is the whole point of this module.

Per turn of the surveyed side it reports: was an attack ever offered,
was one taken, and — when none was offered — who was in front and how
much energy they had.

Run from the repo root:
    python -m src.analysis.attack_census --games 60 \
        --deck data/decks/meta_alakazam.csv --arm heuristic-tempo-scaled
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Final

from cg.api import OptionType, SelectContext

from ..agent_heuristics.crustle_agent import CrustleAgent
from ..deckbuilding.legality import read_deck_ids
from ..environment_wrapper.selfplay import play_one_game
from ..environment_wrapper.wrapper import EnvironmentWrapper
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex

OUR_DECK: Final[Path] = REPO_ROOT / "deck.csv"

# OptionType stringifies to its integer, which makes the "what did it
# prefer" histogram unreadable; name them once, from the enum itself.
_OPTION_NAMES: Final[dict[int, str]] = {
    int(getattr(OptionType, name)): name
    for name in dir(OptionType)
    if not name.startswith("_") and isinstance(
        getattr(OptionType, name, None), OptionType)
}


def _option_name(option_type) -> str:
    try:
        return _OPTION_NAMES.get(int(option_type), str(option_type))
    except (TypeError, ValueError):
        return str(option_type)


class _Census:
    """Wraps the surveyed agent; one entry per turn it was asked to act."""

    def __init__(self, inner, wrapper: EnvironmentWrapper,
                 index: CardIndex) -> None:
        self._inner, self._wrapper, self._index = inner, wrapper, index
        # turn -> [attack offered, attack taken, active id, energies]
        self.turns: dict[int, list] = {}
        self.attacks: Counter = Counter()
        self.declined_for: Counter = Counter()

    def __call__(self, obs_dict: dict) -> list[int]:
        answer = self._inner(obs_dict)
        try:
            self._observe(obs_dict, answer)
        except Exception:
            pass       # a census must never change the game it measures
        return answer

    def _observe(self, obs_dict: dict, answer: list[int]) -> None:
        obs = self._wrapper.parse(obs_dict)
        select = obs.select
        if select is None or select.context != SelectContext.MAIN:
            return
        turn = (obs_dict.get("current") or {}).get("turn") or 0
        options = select.option
        offered = [o for o in options if o.type == OptionType.ATTACK]
        picked = options[answer[0]] if answer else None
        state = obs.current
        me = state.players[state.yourIndex]
        active = me.active[0] if me.active else None

        entry = self.turns.setdefault(turn, [False, False, None, 0])
        entry[0] = entry[0] or bool(offered)
        entry[2] = active.id if active is not None else None
        entry[3] = len(active.energies or []) if active is not None else 0
        if picked is not None and picked.type == OptionType.ATTACK:
            entry[1] = True
            self.attacks[picked.attackId] += 1
        elif offered and picked is not None:
            self.declined_for[_option_name(picked.type)] += 1


def census(deck: Path, arm: str, games: int, seed: int) -> dict:
    from ..environment_wrapper.ab_test import ArmMetrics, ArmSpec, arm_factory

    index, effects = CardIndex(), EffectIndex()
    wrapper = EnvironmentWrapper(index)
    ours = read_deck_ids(OUR_DECK)
    theirs = read_deck_ids(deck)
    make = arm_factory(ArmSpec.parse(arm), index, effects, ArmMetrics(),
                       theirs)

    totals: Counter = Counter()
    attacks: Counter = Counter()
    declined: Counter = Counter()
    empty_active: Counter = Counter()
    idle_body: Counter = Counter()
    for game in range(games):
        probe = _Census(make(seed + game), wrapper, index)
        foe = CrustleAgent(seed=seed + 10_000 + game, index=index,
                           effects=effects, variant="v3")
        # alternate seats, as everywhere else in this repo
        if game % 2 == 0:
            play_one_game((foe, probe), list(ours), list(theirs))
        else:
            play_one_game((probe, foe), list(theirs), list(ours))
        for _turn, (offered, taken, active_id, energies) in probe.turns.items():
            totals["turns"] += 1
            totals["offered"] += int(offered)
            totals["taken"] += int(taken)
            if not offered:
                card = index.get_card(active_id) if active_id else None
                idle_body[card.card_name if card else "vazio"] += 1
                empty_active[energies] += 1
        attacks.update(probe.attacks)
        declined.update(probe.declined_for)

    turns = max(totals["turns"], 1)
    offered = max(totals["offered"], 1)
    print(f"\ndeck={deck.name}  arm={arm}  games={games}")
    print(f"  turnos do lado medido/jogo     : {totals['turns'] / games:.2f}")
    print(f"  ataques/jogo                   : "
          f"{sum(attacks.values()) / games:.2f}")
    print(f"  turnos que OFERECERAM ataque   : "
          f"{totals['offered']}/{totals['turns']} = "
          f"{totals['offered'] / turns:.1%}")
    print(f"  ...e que CONVERTERAM em ataque : "
          f"{totals['taken']}/{totals['offered']} = "
          f"{totals['taken'] / offered:.1%}   <- banda de score")
    print(f"  turnos SEM ataque na mesa      : "
          f"{turns - totals['offered']} = "
          f"{1 - totals['offered'] / turns:.1%}   <- board/energia")
    if declined:
        print("\n  o que preferiram quando havia ataque na mesa:")
        for name, count in declined.most_common(6):
            print(f"    {name:14s} {count}")
    print("\n  turnos sem oferta — quem estava no ativo:")
    for name, count in idle_body.most_common(6):
        print(f"    {name:28s} {count}")
    print("  turnos sem oferta — energias no ativo:")
    for energies, count in sorted(empty_active.items()):
        print(f"    {energies} energia(s): {count}")

    return {"turns": totals["turns"], "offered": totals["offered"],
            "taken": totals["taken"],
            "attacks": sum(attacks.values()), "games": games}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path,
                        default=Path("data/decks/meta_alakazam.csv"))
    parser.add_argument("--arm", type=str, default="heuristic-tempo-scaled")
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    census(args.deck, args.arm, args.games, args.seed)


if __name__ == "__main__":
    main()
