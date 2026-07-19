"""Deck legality validation (Task 4.5a) — 60 card_ids against CardIndex.

Rules enforced (Kaggle/cabt deck construction):
  1. exactly 60 cards, every id known to the CardIndex;
  2. at most 4 copies per card NAME (Basic Energy is unlimited);
  3. at least one Basic Pokémon (a deck must open with an Active);
  4. evolution lines are coherent: no Stage 1/2 (or Mega evolving from a
     listed previous stage) without its previous-stage NAME in the deck.
     Pokémon with an ability are EXEMPT: the engine itself never enforces
     lines, and real ladder lists run lineless ability techs (taksai's
     Mega Starmie deck plays Cinderace with no Raboot — its Explosiveness
     ability places it during setup; deck verified engine-legal via
     battle_start, 2026-07-16);
  5. at most 1 ACE SPEC card in total (across names and copies);
  6. energies support the attackers: every Pokémon that has attacks must
     have at least one attack whose typed cost is coverable by the energy
     types the deck provides (RAINBOW-providing special energy covers any
     type, and TEAM_ROCKET energy counts the same way — it pays any type
     on the Team Rocket's Pokémon that use it; kashiwashira's ladder deck
     runs TR Mimikyu {P}{C} with only {G}+TR energy, verified engine-legal
     via battle_start, 2026-07-16; Colorless is payable by any energy
     card). Pokémon with an ability are EXEMPT — real lists run ability
     techs whose attack is never meant to be used (e.g. Chien-Pao/Snow
     Sink in the city-league Clefairy list, whose {W} attack has no water
     in the deck).

None-safe: unknown ids become errors, never exceptions; all other checks
still run over the known cards. Pure lookup — no engine calls.

Run from the repo root:
    python -m src.deckbuilding.legality deck.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from cg.api import EnergyType

from ..ingestion.card_index import Card, CardIndex

DECK_SIZE: Final[int] = 60
MAX_COPIES: Final[int] = 4
MAX_ACE_SPEC: Final[int] = 1

# stage_code taxonomy (verified against dim_card, Sprint 5D):
STAGE_BASIC_ENERGY: Final[int] = 1
STAGE_SPECIAL_ENERGY: Final[int] = 2
STAGE_ITEM: Final[int] = 3
STAGE_TOOL: Final[int] = 4
STAGE_SUPPORTER: Final[int] = 5
STAGE_STADIUM: Final[int] = 6
STAGE_BASIC: Final[int] = 7
STAGE_STAGE1: Final[int] = 8
STAGE_STAGE2: Final[int] = 9

_ENERGY_STAGES: Final[frozenset[int]] = frozenset(
    {STAGE_BASIC_ENERGY, STAGE_SPECIAL_ENERGY})
_EVOLVED_STAGES: Final[frozenset[int]] = frozenset(
    {STAGE_STAGE1, STAGE_STAGE2})


@dataclass(frozen=True)
class LegalityReport:
    """Outcome of validate_deck: legal iff there are no errors."""

    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _cost_coverable(cost: tuple[tuple[int, int], ...],
                    provided: frozenset[int], has_energy: bool) -> bool:
    """Could this attack cost ever be paid with the deck's energy TYPES?"""
    if not cost:
        return True
    if not has_energy:
        return False
    rainbow = (int(EnergyType.RAINBOW) in provided
               or int(EnergyType.TEAM_ROCKET) in provided)
    for energy_type, _qty in cost:
        if energy_type == int(EnergyType.COLORLESS):
            continue  # any energy card pays colorless
        if energy_type not in provided and not rainbow:
            return False
    return True


def validate_deck(card_ids: Sequence[int],
                  index: CardIndex) -> LegalityReport:
    errors: list[str] = []

    if len(card_ids) != DECK_SIZE:
        errors.append(f"deck has {len(card_ids)} cards (must be {DECK_SIZE})")

    cards: list[Card] = []
    for card_id in card_ids:
        card = index.get_card(int(card_id))
        if card is None:
            errors.append(f"unknown card id {card_id}")
        else:
            cards.append(card)

    # 2. copies per name (basic energy unlimited)
    copies_by_name = Counter(c.card_name for c in cards)
    basic_energy_names = {c.card_name for c in cards
                          if c.stage_code == STAGE_BASIC_ENERGY}
    for name, count in sorted(copies_by_name.items()):
        if count > MAX_COPIES and name not in basic_energy_names:
            errors.append(f"{count} copies of '{name}' (max {MAX_COPIES})")

    # 3. at least one Basic Pokémon
    if not any(c.stage_code == STAGE_BASIC for c in cards):
        errors.append("no Basic Pokémon in deck")

    # 4. evolution coherence by previous-stage name (ability techs exempt)
    deck_names = set(copies_by_name)
    for name in sorted({c.card_name for c in cards
                        if (c.stage_code in _EVOLVED_STAGES
                            or c.previous_stage is not None)
                        and not c.skill_ids}):
        card = next(c for c in cards if c.card_name == name)
        if card.previous_stage and card.previous_stage not in deck_names:
            errors.append(f"'{name}' without its previous stage "
                          f"'{card.previous_stage}' in deck")

    # 5. ACE SPEC total
    ace_count = sum(1 for c in cards if c.is_ace_spec)
    if ace_count > MAX_ACE_SPEC:
        errors.append(f"{ace_count} ACE SPEC cards (max {MAX_ACE_SPEC})")

    # 6. energy types support the attackers
    provided = frozenset(c.type_code for c in cards
                         if c.stage_code in _ENERGY_STAGES
                         and c.type_code is not None)
    has_energy = any(c.stage_code in _ENERGY_STAGES for c in cards)
    # fodder de evolução é isento: um Pokémon cuja evolução está no deck
    # pode ter ataque impagável — ele existe para evoluir (verificado
    # empiricamente via battle_start com a lista real de Grimmsnarl do
    # ladder: Snorunt {W} num deck só-{D}, aceito pelo engine, 17/Jul).
    evolves_in_deck = {c.previous_stage for c in cards if c.previous_stage}
    for name in sorted({c.card_name for c in cards
                        if c.attack_ids and not c.skill_ids
                        and c.card_name not in evolves_in_deck}):
        card = next(c for c in cards if c.card_name == name)
        attacks = [index.get_attack(a) for a in card.attack_ids]
        costs = [a.cost for a in attacks if a is not None]
        if costs and not any(_cost_coverable(cost, provided, has_energy)
                             for cost in costs):
            errors.append(f"no deck energy can pay any attack of '{name}'")

    return LegalityReport(errors=tuple(errors))


def read_deck_ids(path: Path) -> list[int]:
    """One card_id per line (same format as deck.csv). None-safe: skips
    blanks; non-numeric lines become id -1 (reported as unknown)."""
    ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(int(line) if line.lstrip("-").isdigit() else -1)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck", type=Path, help="deck csv (one id per line)")
    args = parser.parse_args()

    index = CardIndex()
    report = validate_deck(read_deck_ids(args.deck), index)
    if report.ok:
        print(f"{args.deck}: LEGAL")
    else:
        print(f"{args.deck}: ILLEGAL ({len(report.errors)} problems)")
        for error in report.errors:
            print(f"  - {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
