"""Unit tests for src/deckbuilding/legality.py (Task 4.5a).

Base fixture = the repo's deck.csv (the official Kaggle sample deck,
Mega Abomasnow): it must validate as LEGAL, and each rule is exercised
by mutating one aspect of a copy of it. Card ids used in mutations
(verified against dim_card): 3=Basic {W} Energy, 5=Basic {P} Energy,
721=Kyogre, 722=Snover, 723=Mega Abomasnow ex (Stage 1, prev Snover),
1080=Unfair Stamp (ACE SPEC), 1158=Maximum Belt (ACE SPEC).

Run from the repo root:  python -m unittest tests.test_legality
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.deckbuilding.legality import (DECK_SIZE, read_deck_ids,
                                       validate_deck)
from src.ingestion.card_index import CardIndex

REPO_ROOT = Path(__file__).resolve().parents[1]

WATER_ENERGY = 3
PSYCHIC_ENERGY = 5
KYOGRE = 721
SNOVER = 722
MEGA_ABOMASNOW = 723
UNFAIR_STAMP = 1080
MAXIMUM_BELT = 1158


class TestDeckLegality(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.legal = read_deck_ids(REPO_ROOT / "deck.csv")

    def _errors(self, deck: list[int]) -> str:
        return "\n".join(validate_deck(deck, self.index).errors)

    def test_official_sample_deck_is_legal(self) -> None:
        report = validate_deck(self.legal, self.index)
        self.assertTrue(report.ok, msg="\n".join(report.errors))
        self.assertEqual(len(self.legal), DECK_SIZE)

    def test_wrong_size_fails(self) -> None:
        self.assertIn("59 cards", self._errors(self.legal[:-1]))
        self.assertIn("61 cards", self._errors(self.legal + [WATER_ENERGY]))

    def test_unknown_id_fails_without_crashing(self) -> None:
        deck = [999_999] + self.legal[1:]
        self.assertIn("unknown card id 999999", self._errors(deck))

    def test_five_copies_by_name_fail(self) -> None:
        # 5th Snover replaces one energy: same size, one name over the cap.
        deck = list(self.legal)
        deck[deck.index(WATER_ENERGY)] = SNOVER
        self.assertIn("5 copies of 'Snover'", self._errors(deck))

    def test_basic_energy_is_exempt_from_copy_cap(self) -> None:
        report = validate_deck(self.legal, self.index)  # 35x {W} energy
        self.assertTrue(report.ok)
        self.assertGreater(self.legal.count(WATER_ENERGY), 4)

    def test_no_basic_pokemon_fails(self) -> None:
        deck = [WATER_ENERGY if self.index.card(i).stage_code in (7, 8, 9)
                else i for i in self.legal]
        self.assertIn("no Basic Pokémon", self._errors(deck))

    def test_stage1_without_base_fails(self) -> None:
        # Remove every Snover: Mega Abomasnow ex loses its previous stage.
        deck = [WATER_ENERGY if i == SNOVER else i for i in self.legal]
        self.assertIn("'Mega Abomasnow ex' without its previous stage "
                      "'Snover'", self._errors(deck))

    def test_second_ace_spec_fails(self) -> None:
        # Deck already runs Maximum Belt; add Unfair Stamp over an energy.
        deck = list(self.legal)
        deck[deck.index(WATER_ENERGY)] = UNFAIR_STAMP
        self.assertIn("2 ACE SPEC cards", self._errors(deck))
        self.assertEqual(deck.count(MAXIMUM_BELT), 1)

    def test_wrong_energy_type_fails(self) -> None:
        # Kyogre attacks need {W}; a mono-Psychic base can't ever pay them.
        deck = [PSYCHIC_ENERGY if i == WATER_ENERGY else i for i in self.legal]
        self.assertIn("no deck energy can pay any attack of 'Kyogre'",
                      self._errors(deck))

    def test_ability_tech_exempt_from_energy_rule(self) -> None:
        # Real city-league Clefairy list runs Chien-Pao (209, {W} attack,
        # Snow Sink ability) with zero water energy — must stay legal.
        deck = read_deck_ids(REPO_ROOT / "data" / "decks" / "seed_clefairy.csv")
        report = validate_deck(deck, self.index)
        self.assertTrue(report.ok, msg="\n".join(report.errors))
        self.assertIn(209, deck)

    def test_no_energy_at_all_fails(self) -> None:
        # Strip every energy card: attacks with any cost become unpayable
        # (the size violation is reported alongside — both must appear).
        deck = [i for i in self.legal if i != WATER_ENERGY]
        errors = self._errors(deck)
        self.assertIn("no deck energy can pay any attack", errors)
        self.assertIn(f"{len(deck)} cards", errors)


if __name__ == "__main__":
    unittest.main()
